"""Train or fine-tune ShuttleNet; use declared group splits, not random rallies."""

import argparse
import csv
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import random
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ShuttleDataset, read_split_manifest
from .model import ShuttleNet, heatmap_loss

DEFAULT_MANIFEST = Path('data/shuttle/split_manifest.csv')
DEFAULT_ROOT = Path('data/shuttle/clips')
DEFAULTS = dict(batch_size=2, lr=.001, patience=8, min_delta=0.,
                lr_patience=2, lr_factor=.5, min_lr=1e-6)
HISTORY_FIELDS = ('epoch', 'train_loss', 'val_loss', 'lr')


def choose_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_datasets(root, manifest=None):
    if manifest is None and Path(root).resolve() == DEFAULT_ROOT.resolve() and DEFAULT_MANIFEST.is_file():
        manifest = DEFAULT_MANIFEST
    train = ShuttleDataset(root, 'train', augment=True, split_manifest=manifest)
    val = ShuttleDataset(root, 'val', split_manifest=manifest)
    if not len(train) or not len(val):
        raise ValueError('Require nonempty labelled train and validation datasets')
    if train.groups & val.groups:
        raise ValueError(f'Source group overlap: {sorted(train.groups & val.groups)}')
    # Also protect candidate tests when using a metadata-only staging/legacy root.
    guard = manifest or (DEFAULT_MANIFEST if DEFAULT_MANIFEST.is_file() else None)
    if guard:
        assignments = read_split_manifest(guard)
        test_ids = {name for name, row in assignments.items() if row['proposed_split'] == 'test'}
        test_groups = {row['source_group'] for row in assignments.values() if row['proposed_split'] == 'test'}
        loaded_ids = {Path(p).name for p in train.clips + val.clips}
        if loaded_ids & test_ids or (train.groups | val.groups) & test_groups:
            raise ValueError('Reserved candidate test data included in training/validation')
    dimensions = train.dimensions | val.dimensions
    if len(dimensions) != 1:
        raise ValueError('Prepared input dimensions disagree across train/validation')
    if any(not any(row['visible'] == 1 for _, _, row in dataset.samples) for dataset in (train, val)):
        raise ValueError('Both splits need visible shuttle labels')
    # Small metadata/label signatures only: no video hashing or frame scan.
    signature = {}
    for clip in train.clips + val.clips:
        for name in ('meta.json', 'labels.csv'):
            path = Path(clip)/name
            signature[str(path.resolve())] = file_hash(path)
    if manifest:
        signature[str(Path(manifest).resolve())] = file_hash(manifest)
    return train, val, next(iter(dimensions)), signature, manifest


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    batches = loader if max_batches is None else islice(loader, max_batches)
    with torch.set_grad_enabled(optimizer is not None):
        for frames, target in batches:
            frames, target = frames.to(device), target.to(device)
            loss = heatmap_loss(model(frames), target)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss; checkpoint not advanced')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
            total += float(loss.detach().cpu()) * frames.shape[0]
            count += frames.shape[0]
    if not count:
        raise ValueError('No batches processed')
    return total / count


def load_model_checkpoint(path, width, height):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model'), dict):
        raise ValueError('Checkpoint must contain a ShuttleNet model state dict')
    expected = dict(width=width, height=height, base_channels=16)
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Checkpoint dimensions/base_channels must match {expected}')
    if checkpoint.get('architecture', 'ShuttleNet') != 'ShuttleNet' or checkpoint.get('input_channels', 9) != 9:
        raise ValueError('Incompatible checkpoint architecture/input channels')
    model = ShuttleNet(16)
    try:
        model.load_state_dict(checkpoint['model'], strict=True)
    except RuntimeError as error:
        raise ValueError('Checkpoint model tensor names/shapes are incompatible') from error
    return model, checkpoint


def rng_state(device):
    numpy_state = np.random.get_state()
    state = dict(torch=torch.get_rng_state(), python=random.getstate(),
                 numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]])
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state_all()
    elif device.type == 'mps':
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    name, keys, position, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached))
    if device.type == 'cuda' and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
    elif device.type == 'mps' and 'mps' in state:
        torch.mps.set_rng_state(state['mps'])


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def write_history(path, history):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def inference_checkpoint(state):
    return dict(model=state['best_model'], width=state['width'], height=state['height'],
                base_channels=16, architecture='ShuttleNet', input_channels=9,
                epoch=state['best_epoch'], val_loss=state['best_val_loss'],
                method='three-frame RGB heatmap')


def repair_outputs(output, state):
    """last.pt is the committed state if interruption occurred between file writes."""
    write_history(output/'history.csv', state['history'])
    expected = inference_checkpoint(state)
    try:
        current = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
        matches = (current['epoch'] == expected['epoch'] and current['val_loss'] == expected['val_loss']
                   and set(current['model']) == set(expected['model'])
                   and all(torch.equal(v, current['model'][k]) for k, v in expected['model'].items()))
    except (OSError, RuntimeError, KeyError, EOFError):
        matches = False
    if not matches:
        atomic_save(expected, output/'best.pt')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--split-manifest', type=Path, help='Overrides meta.split; default auto-detected for the canonical clip root')
    parser.add_argument('--output', type=Path, default=Path('outputs/shuttle'))
    parser.add_argument('--epochs', type=int, default=40, help='Total target epochs, including completed epochs on resume')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--init-weights', type=Path, help='Model-only initialization; fresh Adam/scheduler/history')
    source.add_argument('--resume', type=Path, help='Resume this output directory from its last.pt')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--patience', type=int, help='Early-stop bad epochs; default 8, 0 disables')
    parser.add_argument('--min-delta', type=float, help='Minimum meaningful loss improvement; default 0')
    parser.add_argument('--lr-patience', type=int, help='ReduceLROnPlateau patience; default 2')
    parser.add_argument('--lr-factor', type=float, help='LR reduction factor; default 0.5')
    parser.add_argument('--min-lr', type=float, help='LR floor; default 1e-6')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--smoke-test', action='store_true', help='CPU only: one train batch + one val batch, batch size 1, one partial cycle')
    args = parser.parse_args(argv)
    if args.epochs <= 0 or (args.smoke_test and args.device not in ('auto', 'cpu')):
        parser.error('epochs must be positive; smoke tests allow only CPU/auto (forced CPU)')
    if args.output.resolve() == Path('outputs/shuttle_v2').resolve():
        parser.error('V2 output is protected; choose a new output directory')
    if args.init_weights and args.output.resolve() == args.init_weights.parent.resolve():
        parser.error('Initialization checkpoint directory cannot be the new output directory')
    if args.resume:
        if args.resume.name != 'last.pt' or args.resume.parent.resolve() != args.output.resolve():
            parser.error('Resume requires last.pt in the same --output directory')
    elif any((args.output/name).exists() for name in ('run.json', 'history.csv', 'best.pt', 'last.pt')):
        parser.error('Output already contains run artifacts; use --resume or a fresh directory')

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    train, val, (width, height), signature, manifest = prepare_datasets(args.root, args.split_manifest)
    if args.resume or args.init_weights:
        model, previous = load_model_checkpoint(args.resume or args.init_weights, width, height)
    else:
        model, previous = ShuttleNet(16), None
    if args.resume:
        required = {'optimizer', 'scheduler', 'rng', 'history', 'training_config', 'data_signature',
                    'best_model', 'best_epoch', 'best_val_loss', 'early_best', 'bad_epochs', 'run_id'}
        if not required <= previous.keys():
            raise ValueError('Legacy/inference checkpoint cannot resume; use --init-weights in a fresh directory')
        if previous['data_signature'] != signature:
            raise ValueError('Training data/manifest changed since checkpoint; refusing resume')
        config = previous['training_config']
        if config['smoke_test'] != args.smoke_test:
            raise ValueError('Smoke checkpoints can only resume in smoke mode; start a new real fine-tune instead')
        for name in DEFAULTS:
            requested = getattr(args, name)
            if requested is not None and requested != config[name]:
                raise ValueError(f'Resume restores {name}={config[name]}; changing it requires a new run')
        run_path = args.output/'run.json'
        if not run_path.is_file() or json.loads(run_path.read_text()).get('run_id') != previous['run_id']:
            raise ValueError('Run provenance missing or mismatched')
    else:
        config = {name: getattr(args, name) if getattr(args, name) is not None else default
                  for name, default in DEFAULTS.items()}
        config['smoke_test'] = args.smoke_test
        if args.smoke_test:
            config['batch_size'] = 1
    if (config['batch_size'] <= 0 or config['lr'] <= 0 or config['patience'] < 0
            or config['lr_patience'] < 0 or not 0 < config['lr_factor'] < 1
            or config['min_delta'] < 0 or not 0 <= config['min_lr'] <= config['lr']
            or any(not math.isfinite(float(v)) for v in config.values())):
        parser.error('Invalid batch size, LR, patience, delta, factor or LR floor')
    device = torch.device('cpu') if args.smoke_test else choose_device(args.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['lr_factor'], patience=config['lr_patience'],
        threshold=config['min_delta'], threshold_mode='abs', min_lr=config['min_lr'])
    if args.resume:
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        state = previous
        restore_rng(state['rng'], device)
        repair_outputs(args.output, state)
    else:
        state = dict(epoch=0, history=[], best_val_loss=float('inf'), best_epoch=0,
                     early_best=float('inf'), bad_epochs=0, run_id=uuid.uuid4().hex)
        args.output.mkdir(parents=True, exist_ok=True)
        summary = dict(train_clips=train.clips, val_clips=val.clips, train_groups=sorted(train.groups),
                       val_groups=sorted(val.groups), train_labels=len(train), val_labels=len(val),
                       width=width, height=height, epochs=args.epochs, **config,
                       init_weights=str(args.init_weights) if args.init_weights else None,
                       init_weights_sha256=file_hash(args.init_weights) if args.init_weights else None,
                       split_manifest=str(manifest) if manifest else None, data_signature=signature,
                       run_id=state['run_id'], heatmap_radius=3, loss='balanced BCE with hard background pixels',
                       source_independence_verified=False)
        with (args.output/'run.json').open('x') as file:
            json.dump(summary, file, indent=2)
            file.write('\n')
    print(f'Device: {device}; train={len(train)}, val={len(val)}; input={width}x{height}', flush=True)
    print('Declared-group separation checked. match09/match10 are not verified single-match identities; no leakage-free claim.', flush=True)
    if args.smoke_test:
        print('SMOKE ONLY: one CPU training batch and one validation batch; not a full epoch.', flush=True)
    if config['patience'] and state['bad_epochs'] >= config['patience']:
        print('Checkpoint already early-stopped; no further steps. Start a new run to change policy.')
        return
    start_epoch = state['epoch']+1
    stop_epoch = min(args.epochs, start_epoch) if args.smoke_test else args.epochs
    train_loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    limit = 1 if args.smoke_test else None
    for epoch in range(start_epoch, stop_epoch+1):
        lr_used = optimizer.param_groups[0]['lr']
        train_loss = run_epoch(model, train_loader, device, optimizer, max_batches=limit)
        val_loss = run_epoch(model, val_loader, device, max_batches=limit)
        improved = val_loss < state['best_val_loss']
        if improved:
            state.update(best_val_loss=val_loss, best_epoch=epoch,
                         best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if val_loss < state['early_best']-config['min_delta']:
            state.update(early_best=val_loss, bad_epochs=0)
        else:
            state['bad_epochs'] += 1
        scheduler.step(val_loss)
        state['history'].append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, lr=lr_used))
        state.update(format_version=2, architecture='ShuttleNet', input_channels=9,
                     model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     width=width, height=height, base_channels=16, epoch=epoch, val_loss=val_loss,
                     method='three-frame RGB heatmap', optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), training_config=config,
                     data_signature=signature, rng=rng_state(device))
        atomic_save(state, args.output/'last.pt')
        if improved:
            atomic_save(inference_checkpoint(state), args.output/'best.pt')
        write_history(args.output/'history.csv', state['history'])
        print(f'Epoch/cycle {epoch}/{args.epochs}: train={train_loss:.5f} val={val_loss:.5f} '
              f'next_lr={optimizer.param_groups[0]["lr"]:.3g} bad_epochs={state["bad_epochs"]}', flush=True)
        if config['patience'] and state['bad_epochs'] >= config['patience']:
            print('Early stopping: validation loss has not improved.', flush=True)
            break
    print(f'Best checkpoint: {args.output / "best.pt"}; resume state: {args.output / "last.pt"}')


if __name__ == '__main__':
    main()
