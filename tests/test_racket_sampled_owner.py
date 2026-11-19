import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src.racket.mediapipe_bundle import BODY_NAMES
from src.racket.real_owner_data import RealPairs
from src.racket.sampled_owner_data import FORMAT, MaskedFrames, masked_loss, select_queries
from src.racket.sampled_owner_package import merged_records
from src.racket.sampled_owner_train import acceptance, metric_key, run, stage, verify
from src.racket.schema import file_hash, save_json
from tests import test_racket_real_owner as real_fixture


class SampledOwnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def records(self):
        a = real_fixture.RealOwnershipTests().record()
        b = copy.deepcopy(a)
        b.update(id='b', split='val', source_group='match02', image='b.jpg')
        for r in (a, b):
            for player in r['players'].values():
                player['racket_state'] = 'located'
        a['players']['far_side'].update(valid_target=False, target=None, target_original=None,
                                         racket_state='unlocatable')
        a['head_separation_original_pixels'] = 0.
        return a, b

    def bundle(self, root):
        a, b = self.records()
        for name in ('a.jpg', 'b.jpg'):
            cv2.imwrite(str(root/name), np.full((64, 64, 3), 90, np.uint8))
        (root/'warm.pt').write_bytes(b'unit-test-hash')
        m = dict(version=FORMAT, run_version='v006', body_names=BODY_NAMES,
                 context_padding=0., image_size=64, frames=[a, b], legacy_validation=[],
                 initial_checkpoint='warm.pt', initial_checkpoint_sha256=file_hash(root/'warm.pt'),
                 fixed_validation_ids=['b'],
                 file_hashes={name: file_hash(root/name) for name in ('warm.pt', 'a.jpg', 'b.jpg')})
        save_json(root/'manifest.json', m)
        return m

    def test_unknown_query_never_forwarded_or_used_as_opponent(self):
        images = torch.zeros(2, 3, 8, 8)
        poses = torch.zeros(2, 2, 17, 3)
        boxes = torch.zeros(2, 2, 4)
        targets = torch.tensor([[[.2, .3], [.7, .8]], [[.1, .2], [.9, .8]]], requires_grad=True)
        masks = torch.tensor([[True, False], [True, True]])
        rgb, pose, box, target, other, ranked = select_queries(images, poses, boxes, targets, masks, torch.tensor([100., 100.]))
        self.assertEqual(len(rgb), 3)
        self.assertEqual(ranked.tolist(), [False, True, True])
        logits = torch.zeros(3, 1, 16, 16, requires_grad=True)
        loss, _, _ = masked_loss(logits, target, other, ranked)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(targets.grad[0, 1], torch.zeros(2)))
        self.assertGreater(targets.grad[0, 0].abs().sum().item(), 0.)

    def test_partial_dataset_retains_positive_and_augments_only_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = self.bundle(root)
            a, b = m['frames']
            ds = MaskedFrames(root, [a], training=True, size=64)
            image, pose, box, target, mask, index = ds[0]
            self.assertEqual(mask.tolist(), [True, False])
            self.assertEqual(image.shape, (3, 64, 64))
            self.assertTrue(torch.isfinite(target).all())
            self.assertEqual(index, 0)
            with self.assertRaisesRegex(ValueError, 'training frames'):
                MaskedFrames(root, [b], training=True)

    def test_export_overrides_are_unique_and_unknown_validation_is_not_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = dict(id='a', clip='rally01', frame=1, players={'near_side': dict(target_valid=True)})
            corrected = dict(original, note='new correction')
            for split in ('train', 'val'):
                (root/(split+'.jsonl')).write_text(json.dumps(original)+'\n' if split == 'train' else '')
                (root/(split+'_with_masks.jsonl')).write_text(json.dumps(corrected)+'\n' if split == 'train' else '')
            merged = merged_records(root)
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]['note'], 'new correction')
            (root/'val_with_masks.jsonl').write_text(json.dumps(dict(original, players={'far_side': dict(target_valid=False)})))
            with self.assertRaisesRegex(ValueError, 'unresolved'):
                merged_records(root)

    def test_verify_rejects_leakage_invalid_masks_and_lost_validation_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = self.bundle(root)
            self.assertEqual(len(verify(root)['frames']), 2)
            original = copy.deepcopy(m)
            m['frames'][1]['source_group'] = 'match01'
            save_json(root/'manifest.json', m)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                verify(root)
            m = copy.deepcopy(original)
            m['frames'][1]['players']['far_side']['valid_target'] = False
            save_json(root/'manifest.json', m)
            with self.assertRaisesRegex(ValueError, 'validation targets'):
                verify(root)
            m = copy.deepcopy(original)
            m['fixed_validation_ids'] = []
            save_json(root/'manifest.json', m)
            with self.assertRaisesRegex(ValueError, 'denominator'):
                verify(root)
            m = copy.deepcopy(original)
            m['frames'][0]['players']['far_side']['valid_target'] = True
            save_json(root/'manifest.json', m)
            with self.assertRaisesRegex(ValueError, 'located racket'):
                verify(root)

    def test_exact_80_percent_both_sides_pass_and_weaker_side_selects(self):
        def metrics(near, far):
            return dict(per_side={s: dict(pck20_including_misses=v) for s, v in
                                  [('near_side', near), ('far_side', far)]},
                        pck20_including_misses=(near+far)/2, wrong_racket_within20_queries=0,
                        pck50_including_misses=.9, mean_error_detected_only=10.)
        self.assertTrue(acceptance(metrics(.80, .80))['passed'])
        self.assertFalse(acceptance(metrics(.99, .79))['passed'])
        self.assertGreater(metric_key(metrics(.81, .81)), metric_key(metrics(.99, .79)))

    def test_one_epoch_partial_training_and_reload_checkpoint(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Conv2d(3, 1, 1)

            def forward(self, images, poses, boxes):
                return self.head(torch.nn.functional.avg_pool2d(images, 4))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = self.bundle(root)
            model = TinyModel()
            result, epoch = stage(model, MaskedFrames(root, [m['frames'][0]], training=True, size=64),
                RealPairs(root, [m['frames'][1]], size=64), m, root/'smoke', torch.device('cpu'),
                epochs=1, batch=1, patience=1, size=64, encoder_lr=.00003, head_lr=.0001, label='SMOKE')
            history = json.loads((root/'smoke/history.json').read_text())
            self.assertTrue(np.isfinite(history[0]['loss']))
            self.assertEqual(history[0]['ownership_loss'], 0.)
            self.assertEqual(result['samples'], 2)
            saved = torch.load(root/'smoke/best.pt', map_location='cpu', weights_only=True)
            self.assertEqual(saved['image_size'], 64)
            self.assertEqual(saved['run_version'], 'v006')
            model.load_state_dict(saved['state_dict'])

    def test_complete_runner_packages_results_even_when_accuracy_target_fails(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Conv2d(3, 1, 1)

            def forward(self, images, poses, boxes):
                return self.head(torch.nn.functional.adaptive_avg_pool2d(images, (16, 16)))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'bundle'
            root.mkdir()
            m = self.bundle(root)
            b = m['frames'][1]
            b['full_image'] = 'b.jpg'
            for i in range(2):
                train = copy.deepcopy(b)
                train.update(id='paired_train_'+str(i), split='train', source_group='train_match_'+str(i))
                m['frames'].append(train)
            m.update(counts={}, native_v005_validation=[b], baseline_label='v005',
                     native_v005_expected_metrics={k: 0 for k in
                        ('within_20px', 'within_50px', 'missing', 'wrong_racket_within20_queries')},
                     annotation_manifest_sha256='synthetic-test')
            save_json(root/'manifest.json', m)
            output = Path(tmp)/'run'
            # Stub only the expensive warm model/gate and legacy diagnostic.
            # Both training stages, evaluation, selection, preview and ZIP run.
            gate = dict(pck20_including_misses=1., wrong_racket_within20_queries=0)
            with patch('src.racket.sampled_owner_train.warm_model', side_effect=lambda *args: TinyModel()), patch(
                    'src.racket.sampled_owner_train.learning_fit', return_value=(gate, 1)), patch(
                    'src.racket.sampled_owner_train.legacy', return_value=({}, [])):
                self.assertTrue(run(root, output, device='cpu', epochs=1, batch=1, patience=1,
                                    refine_epochs=1, refine_patience=1, sanity_epochs=1))
            self.assertEqual(json.loads((output/'status.json').read_text())['status'], 'complete')
            self.assertFalse(json.loads((output/'acceptance.json').read_text())['passed'])
            self.assertTrue((output/'racket_center_real_owner_v006.pt').exists())
            self.assertTrue((output/'previews/v005_vs_v006.mp4').exists())
            self.assertTrue(output.with_name(output.name+'_results.zip').exists())
