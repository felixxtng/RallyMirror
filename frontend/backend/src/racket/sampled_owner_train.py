"""One Runpod command: masked training, high-resolution refinement, honest 80% gate."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .mediapipe_bundle import BODY_NAMES, sha, to_original, write_json
from .real_owner_data import SIDES, RealPairs, scene_bounds
from .real_owner_train import (archive_results, checkpoint, evaluate, fit as learning_fit,
                               gate_records, legacy, previews, warm_model)
from .sampled_owner_data import FORMAT, MaskedFrames, masked_loss, select_queries


def metric_key(metrics):
    return (min(metrics['per_side'][s]['pck20_including_misses'] for s in SIDES),
            metrics['pck20_including_misses'], -metrics['wrong_racket_within20_queries'],
            metrics['pck50_including_misses'],
            -(metrics['mean_error_detected_only'] if metrics['mean_error_detected_only'] is not None else float('inf')))


def acceptance(metrics):
    near, far = [metrics['per_side'][s]['pck20_including_misses'] for s in SIDES]
    return dict(rule='PCK20 >= 0.80 on EACH side separately; missing predictions count as failures',
                near=near, far=far, passed=near >= .80 and far >= .80,
                scope='Held-out source-match located labels; not final-test accuracy or full-video recall')


def verify(bundle):
    bundle = Path(bundle)
    m = json.loads((bundle/'manifest.json').read_text())
    if m['version'] != FORMAT or m['body_names'] != BODY_NAMES:
        raise ValueError('Use the v006 masked ownership bundle')
    for name, expected in m['file_hashes'].items():
        path = (bundle/name).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file() or sha(path) != expected:
            raise ValueError('Missing/changed upload: '+name)
    if sha(bundle/m['initial_checkpoint']) != m['initial_checkpoint_sha256']:
        raise ValueError('Warm checkpoint provenance mismatch')
    records = m['frames']
    if len(records) != len({r['id'] for r in records}):
        raise ValueError('Duplicate frame ID')
    groups = {s: {r['source_group'] for r in records if r['split'] == s} for s in ('train', 'val')}
    if not all(groups.values()) or groups['train'] & groups['val']:
        raise ValueError('Source match leakage')
    if {r['id'] for r in records if r['split'] == 'val'} != set(m['fixed_validation_ids']):
        raise ValueError('Validation denominator changed')
    for r in records:
        if r['split'] not in ('train', 'val') or set(r['players']) != set(SIDES):
            raise ValueError('Invalid split or player slots')
        boxes = [p['owner_box'] for p in r['players'].values() if p['owner_box']]
        if scene_bounds(boxes, m['context_padding']) != r['crop']:
            raise ValueError('Context must depend only on verified body boxes')
        if not r['train_eligible'] or not any(p['valid_target'] for p in r['players'].values()):
            raise ValueError('Bundle contains frame with no usable target')
        for player in r['players'].values():
            if np.asarray(player['pose']).shape != (17, 3) or not np.all(np.isfinite(player['pose'])):
                raise ValueError('Invalid pose features')
            if player['valid_target']:
                if not player['owner_box'] or player['racket_state'] != 'located':
                    raise ValueError('Target needs a verified body and located racket')
                if not np.allclose(to_original(player['target'], r['crop']), player['target_original'], atol=1e-6):
                    raise ValueError('Target transform mismatch')
                if not all(math.isfinite(v) and 0 <= v < 1 for v in player['target']):
                    raise ValueError('Target outside body-derived context')
            elif r['split'] == 'val':
                raise ValueError('Do not silently remove validation targets')
    if {r['source_group'] for r in m['legacy_validation']} & groups['train']:
        raise ValueError('Legacy validation leakage')
    return m


def stage(model, training, validation, m, output, device, epochs, batch, patience, size,
          encoder_lr, head_lr, label, initial_metrics=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stage_m = dict(m, image_size=size)
    encoder, head = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith('encoder.') else head).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=encoder, lr=encoder_lr), dict(params=head, lr=head_lr)],
                                 weight_decay=.0001)
    # Conservative warm-start updates; warmup then smooth decay.
    def schedule(epoch):
        if epoch < 3:
            return (epoch+1)/3
        return .1+.9*(1+math.cos(math.pi*(epoch-3)/max(1, epochs-3)))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    clip_counts = Counter(r['clip'] for r in training.records)
    weights = [1/math.sqrt(clip_counts[r['clip']]) for r in training.records]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loader = DataLoader(training, batch_size=batch, sampler=sampler, num_workers=0)
    best_metrics = initial_metrics or evaluate(model, validation, device, batch)[0]
    best = metric_key(best_metrics)
    best_epoch = 0
    torch.save(checkpoint(model, stage_m), output/'best.pt')
    write_json(output/'best_metrics.json', best_metrics)
    stale = 0
    history = []
    for epoch in range(1, epochs+1):
        started = time.monotonic()
        model.train()
        seen = 0
        total_loss = 0.
        total_rank = 0.
        for step, (images, poses, boxes, targets, masks, indices) in enumerate(loader, 1):
            separation = torch.tensor([training.records[i]['head_separation_original_pixels']
                                       for i in indices.tolist()])
            rgb, pose, box, target, other, ranked = select_queries(images, poses, boxes, targets, masks, separation)
            optimizer.zero_grad(set_to_none=True)
            logits = model(rgb.to(device), pose.to(device), box.to(device))
            loss, localization, ranking = masked_loss(logits, target.to(device), other.to(device), ranked.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite masked training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            seen += len(target)
            total_loss += loss.item()*len(target)
            total_rank += ranking.item()*len(target)
            if step == 1 or step % 50 == 0 or step == len(loader):
                print(f'{label} epoch {epoch}/{epochs} batch {step}/{len(loader)} '
                      f'loss={loss.item():.4f} ownership_loss={ranking.item():.5f} queries={len(target)}', flush=True)
        scheduler.step()
        metrics, _ = evaluate(model, validation, device, batch)
        key = metric_key(metrics)
        history.append(dict(epoch=epoch, loss=total_loss/seen, ownership_loss=total_rank/seen,
                            seconds=time.monotonic()-started, **metrics))
        write_json(output/'history.json', history)
        print(f'{label} EPOCH {epoch}: near={metrics["per_side"]["near_side"]["pck20_including_misses"]:.3f} '
              f'far={metrics["per_side"]["far_side"]["pck20_including_misses"]:.3f} '
              f'PCK20={metrics["pck20_including_misses"]:.3f}; {time.monotonic()-started:.1f}s', flush=True)
        if key > best:
            best = key
            best_metrics = metrics
            best_epoch = epoch
            stale = 0
            torch.save(checkpoint(model, stage_m), output/'best.pt')
            write_json(output/'best_metrics.json', metrics)
        else:
            stale += 1
        if stale >= patience:
            print(f'{label} early stop: {patience} epochs without improvement; best epoch {best_epoch}', flush=True)
            break
    model.load_state_dict(torch.load(output/'best.pt', map_location=device, weights_only=True)['state_dict'])
    return best_metrics, best_epoch


def run(bundle, output, device='cuda', epochs=100, batch=4, patience=25, refine_epochs=30, refine_patience=12,
        sanity_epochs=150, verify_only=False):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    m = verify(bundle)
    print('VERIFIED v006:', json.dumps(m['counts']), flush=True)
    if verify_only:
        return True
    if output.exists():
        raise FileExistsError('Use a new output directory; previous results are preserved')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use a GPU PyTorch pod')
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device(device)
    output.mkdir(parents=True)
    selected_name = 'racket_center_real_owner_v006.pt'
    stage_name = 'baseline'
    full_started = False
    write_json(output/'run_config.json', dict(epochs=epochs, patience=patience, batch_frames=batch,
        refine_epochs=refine_epochs, refine_patience=refine_patience, resolutions=[640, 768, 960],
        manifest_sha256=sha(bundle/'manifest.json'), seed=42, synthetic_training=False,
        unknowns_masked=True, sampler='inverse sqrt TRAIN clip count; no validation sampling',
        selection='Maximize weaker side PCK20, combined PCK20, fewer swaps, PCK50, mean error',
        acceptance='Near >=80% AND far >=80% PCK20; misses fail', final_test_evaluated=False))
    try:
        train = [r for r in m['frames'] if r['split'] == 'train']
        val = [r for r in m['frames'] if r['split'] == 'val']
        model = warm_model(bundle/m['initial_checkpoint'], device)
        write_json(output/'status.json', dict(status='baseline_evaluation', full_training_started=False))
        validation = RealPairs(bundle, val, size=768)
        base_metrics, base_pred = evaluate(model, validation, device, batch)
        write_json(output/'baseline_paired_metrics.json', base_metrics)
        write_json(output/'baseline_paired_predictions.json', base_pred)
        # Preserve the previous operating resolution as a genuine candidate too.
        base640_metrics, _ = evaluate(model, RealPairs(bundle, val, size=640), device, batch)
        baseline_dir = output/'baseline640'
        baseline_dir.mkdir()
        torch.save(checkpoint(model, dict(m, image_size=640)), baseline_dir/'best.pt')
        write_json(output/'baseline640_corrected_validation_metrics.json', base640_metrics)
        baseline_native, _ = evaluate(model, RealPairs(bundle, m['native_v005_validation'], size=640), device, batch)
        write_json(output/'baseline_native_v005_metrics.json', baseline_native)
        native_expected = m['native_v005_expected_metrics']
        if any(baseline_native[k] != native_expected[k] for k in
               ('within_20px', 'within_50px', 'missing', 'wrong_racket_within20_queries')):
            write_json(output/'native_baseline_warning.json',
                       dict(expected=native_expected, observed=baseline_native,
                            note='Hardware/runtime reproduction differs; same-input comparison remains reported.'))
        paired_train = [r for r in train if all(p['valid_target'] for p in r['players'].values())]
        gate = RealPairs(bundle, gate_records(paired_train), size=256)
        stage_name = 'learning_check'
        write_json(output/'status.json', dict(status=stage_name, full_training_started=False))
        gate_metrics, _ = learning_fit(model, gate, gate, m, output/'learning_check', device,
                                      sanity_epochs, min(batch, 4), sanity=True)
        passed = gate_metrics['pck20_including_misses'] >= .95 and gate_metrics['wrong_racket_within20_queries'] == 0
        write_json(output/'learning_gate.json', dict(passed=passed, metrics=gate_metrics,
                                                   scope='Four TRAIN images; not held-out accuracy'))
        if not passed:
            raise RuntimeError('Training-only learning check failed; full training not started')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        torch.manual_seed(42)
        model = warm_model(bundle/m['initial_checkpoint'], device)
        stage_name = 'full_training'
        full_started = True
        write_json(output/'status.json', dict(status=stage_name, full_training_started=True))
        print('FULL TRAINING STARTING: original v005 checkpoint reloaded; learning-check weights discarded', flush=True)
        full_metrics, full_epoch = stage(model, MaskedFrames(bundle, train, training=True, size=768),
            validation, m, output/'full', device, epochs, batch, patience, 768, .00003, .0001, 'FULL',
            initial_metrics=base_metrics)
        selected_metrics, selected_dir, selected_epoch = full_metrics, output/'full', full_epoch
        stage_name = 'refinement'
        write_json(output/'status.json', dict(status=stage_name, full_training_started=True))
        refine_validation = RealPairs(bundle, val, size=960)
        print('REFINEMENT STARTING: full-stage best checkpoint, 960px, lower learning rates', flush=True)
        refine_metrics, refine_epoch = stage(model, MaskedFrames(bundle, train, training=True, size=960),
            refine_validation, m, output/'refine', device, refine_epochs, batch, refine_patience, 960,
            .00001, .00003, 'REFINE')
        if metric_key(refine_metrics) > metric_key(full_metrics):
            selected_metrics, selected_dir, selected_epoch = refine_metrics, output/'refine', refine_epoch
        if metric_key(base640_metrics) > metric_key(selected_metrics):
            selected_metrics, selected_dir, selected_epoch = base640_metrics, baseline_dir, 0
        saved = torch.load(selected_dir/'best.pt', map_location='cpu', weights_only=True)
        saved.update(acceptance_target=.80, annotation_manifest_sha256=m['annotation_manifest_sha256'],
                     training_masks_supported=True)
        torch.save(saved, output/selected_name)
        model.load_state_dict(saved['state_dict'])
        final_validation = RealPairs(bundle, val, size=saved['image_size'])
        metrics, pred = evaluate(model, final_validation, device, batch)
        ablated, _ = evaluate(model, final_validation, device, batch, ablate=True)
        old_metrics, old_pred = legacy(model, bundle, m, device, batch)
        write_json(output/'paired_metrics.json', metrics)
        write_json(output/'paired_predictions.json', pred)
        write_json(output/'conditioning_zeroed_metrics.json', ablated)
        write_json(output/'legacy578_metrics.json', old_metrics)
        write_json(output/'legacy578_predictions.json', old_pred)
        goal = acceptance(metrics)
        write_json(output/'acceptance.json', goal)
        write_json(output/'comparison.json', dict(baseline_same_768_inputs=base_metrics, selected=metrics,
            selected_resolution=saved['image_size'], selected_stage=selected_dir.name, best_epoch=selected_epoch,
            native_v005_unchanged_640_inputs=baseline_native,
            note='Validation targets corrected explicitly in 3 frames. Native v005 diagnostic retains original targets.'))
        # Final side-by-side comparison uses identical resolution/context for both models.
        baseline = warm_model(bundle/m['initial_checkpoint'], device)
        matched_base, matched_pred = evaluate(baseline, final_validation, device, batch)
        write_json(output/'baseline_matched_resolution_metrics.json', matched_base)
        write_json(output/'baseline_matched_resolution_predictions.json', matched_pred)
        previews(bundle, m, matched_pred, pred, output)
        write_json(output/'status.json', dict(status='complete', full_training_started=True,
            selected_stage=selected_dir.name, best_epoch=selected_epoch, image_size=saved['image_size'],
            acceptance_passed=goal['passed'], final_test_evaluated=False))
        print('ACCEPTANCE', json.dumps(goal), flush=True)
        print('FINAL PAIRED METRICS', json.dumps(metrics), flush=True)
        return True
    except Exception as error:
        write_json(output/'status.json', dict(status='failed', stage=stage_name, error=repr(error),
                                              full_training_started=full_started))
        raise
    finally:
        archive_results(output, selected_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('racket_ownership_run_v006'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--refine-epochs', type=int, default=30)
    parser.add_argument('--refine-patience', type=int, default=12)
    parser.add_argument('--sanity-epochs', type=int, default=150)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.refine_epochs, args.refine_patience, args.sanity_epochs) < 1:
        parser.error('Epochs, batch size and patience must be positive')
    run(args.bundle, args.output, args.device, args.epochs, args.batch_size, args.patience,
        args.refine_epochs, args.refine_patience, args.sanity_epochs, args.verify_only)


if __name__ == '__main__':
    main()
