"""Tiny CPU batches only: checkpoint/resume, split guards and stopping policy."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src.shuttle.dataset import letterbox_transform, read_split_manifest
from src.shuttle.demo import ShuttleDetector
from src.shuttle.model import ShuttleNet
from src.shuttle.train import load_model_checkpoint, main, prepare_datasets


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clips = self.root/'clips'
        self.manifest = self.root/'split.csv'
        self.rows = []
        for name, split in [('train_clip', 'train'), ('val_clip', 'val'), ('test_clip', 'test')]:
            clip = self.clips/name
            (clip/'frames').mkdir(parents=True)
            meta = dict(source_group=name, split='train', frame_count=2,
                        original_width=32, original_height=32,
                        transform=letterbox_transform(32, 32, 32, 32))
            (clip/'meta.json').write_text(json.dumps(meta))
            unknown = name == 'train_clip'
            (clip/'labels.csv').write_text('frame,x,y,visible\n0,16,16,1\n1,,,'+('' if unknown else '0')+'\n')
            for frame in range(2):
                image = np.tile(np.arange(32, dtype=np.uint8)[None, :, None], (32, 1, 3))
                cv2.imwrite(str(clip/'frames'/f'{frame:06d}.jpg'), image)
            self.rows.append(dict(rally_id=name, source_group=name, leakage_block=name,
                                  proposed_split=split, labelled_frames=1 if unknown else 2,
                                  unknown_labels=int(unknown),
                                  status='reserved_candidate_test_pending_source_verification' if split == 'test' else 'development'))
        self.write_manifest()
        self.init = self.root/'init.pt'
        torch.save(dict(model=ShuttleNet().state_dict(), width=32, height=32, base_channels=16), self.init)

    def write_manifest(self):
        with self.manifest.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def invoke(self, output, *extra):
        argv = ['--root', str(self.clips), '--split-manifest', str(self.manifest),
                '--output', str(output), '--device', 'cpu', *map(str, extra)]
        with contextlib.redirect_stdout(io.StringIO()):
            main(argv)

    def load(self, output):
        return torch.load(output/'last.pt', map_location='cpu', weights_only=True)

    def test_manifest_overrides_metadata_and_excludes_test_unknowns(self):
        train, val, dims, _, _ = prepare_datasets(self.clips, self.manifest)
        self.assertEqual((len(train), len(val), dims), (1, 2, (32, 32)))
        self.assertEqual({Path(p).name for p in train.clips+val.clips}, {'train_clip', 'val_clip'})
        self.assertFalse(train.groups & val.groups)
        # Test frames/labels need not even exist for a train/validation-only staged root.
        import shutil
        shutil.rmtree(self.clips/'test_clip')
        prepare_datasets(self.clips, self.manifest)

    def test_overlap_reserved_reassignment_and_missing_data_fail(self):
        self.rows[1]['source_group'] = 'train_clip'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'multiple splits'):
            read_split_manifest(self.manifest)
        self.rows[1]['source_group'] = 'val_clip'
        self.rows[2]['proposed_split'] = 'train'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Reserved candidate'):
            read_split_manifest(self.manifest)
        self.rows[2]['proposed_split'] = 'test'
        self.write_manifest()
        (self.clips/'val_clip'/'meta.json').unlink()
        with self.assertRaises(FileNotFoundError):
            prepare_datasets(self.clips, self.manifest)

    def test_architecture_and_resolution_checks(self):
        model, checkpoint = load_model_checkpoint(self.init, 32, 32)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k,v in checkpoint['model'].items()))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            load_model_checkpoint(self.init, 64, 32)
        checkpoint['model']['enc1.0.weight'] = torch.zeros(16, 3, 3, 3)
        bad = self.root/'bad.pt'
        torch.save(checkpoint, bad)
        with self.assertRaisesRegex(ValueError, 'tensor names/shapes'):
            load_model_checkpoint(bad, 32, 32)

    def test_real_step_resume_matches_uninterrupted_and_inference(self):
        source_hash = hashlib.sha256(self.init.read_bytes()).hexdigest()
        complete, resumed = self.root/'complete', self.root/'resumed'
        self.invoke(complete, '--init-weights', self.init, '--epochs', 2, '--batch-size', 1, '--lr', .0001)
        self.invoke(resumed, '--init-weights', self.init, '--epochs', 1, '--batch-size', 1, '--lr', .0001)
        first = self.load(resumed)
        self.assertTrue(first['optimizer']['state'])
        self.assertTrue(any(not torch.equal(v, torch.load(self.init, weights_only=True)['model'][k])
                            for k,v in first['model'].items()))
        # Simulate interrupted history/best publication after an authoritative last.pt save.
        (resumed/'history.csv').write_text('interrupted partial write')
        (resumed/'best.pt').unlink()
        self.invoke(resumed, '--resume', resumed/'last.pt', '--epochs', 2)
        actual, expected = self.load(resumed), self.load(complete)
        self.assertEqual(actual['epoch'], 2)
        self.assertEqual(actual['history'], expected['history'])
        self.assertEqual(actual['scheduler'], expected['scheduler'])
        self.assertEqual(actual['bad_epochs'], expected['bad_epochs'])
        for key in expected['model']:
            self.assertTrue(torch.equal(expected['model'][key], actual['model'][key]), key)
        for key, value in expected['optimizer']['state'].items():
            for field in value:
                self.assertTrue(torch.equal(value[field], actual['optimizer']['state'][key][field]))
        with (resumed/'history.csv').open() as file:
            self.assertEqual([int(row['epoch']) for row in csv.DictReader(file)], [1, 2])
        self.assertEqual(hashlib.sha256(self.init.read_bytes()).hexdigest(), source_hash)
        for name in ('best.pt', 'last.pt'):
            detector = ShuttleDetector(resumed/name, device='cpu')
            frame = np.zeros((32, 32, 3), np.uint8)
            detector.predict(frame, frame, frame)
        fresh = self.root/'fresh_finetune'
        self.invoke(fresh, '--init-weights', resumed/'last.pt', '--epochs', 1,
                    '--batch-size', 1, '--lr', .0001)
        reset = self.load(fresh)
        self.assertEqual(len(reset['history']), 1)
        self.assertNotEqual(reset['run_id'], actual['run_id'])
        self.assertTrue(all(value['step'].item() == 1 for value in reset['optimizer']['state'].values()))

    def test_scheduler_early_stop_best_and_resume(self):
        output = self.root/'stopped'
        with patch('src.shuttle.train.run_epoch', side_effect=[1., .4, 1., .5]) as epoch:
            self.invoke(output, '--init-weights', self.init, '--epochs', 10,
                        '--patience', 1, '--lr-patience', 0)
            self.assertEqual(epoch.call_count, 4)
        state = self.load(output)
        self.assertEqual((state['epoch'], state['bad_epochs'], state['best_epoch']), (2, 1, 1))
        self.assertAlmostEqual(state['optimizer']['param_groups'][0]['lr'], .0005)
        best_hash = hashlib.sha256((output/'best.pt').read_bytes()).hexdigest()
        history = (output/'history.csv').read_bytes()
        with patch('src.shuttle.train.run_epoch') as epoch:
            self.invoke(output, '--resume', output/'last.pt', '--epochs', 12)
            epoch.assert_not_called()
        self.assertEqual((output/'history.csv').read_bytes(), history)
        self.assertEqual(hashlib.sha256((output/'best.pt').read_bytes()).hexdigest(), best_hash)

    def test_smoke_is_cpu_one_batch_and_cannot_become_real_training(self):
        output = self.root/'smoke'
        with patch('src.shuttle.train.run_epoch', return_value=.5) as epoch:
            self.invoke(output, '--init-weights', self.init, '--epochs', 40, '--smoke-test')
            self.assertEqual(epoch.call_count, 2)
            self.assertTrue(all(call.kwargs['max_batches'] == 1 for call in epoch.call_args_list))
            self.assertTrue(all(str(call.args[2]) == 'cpu' for call in epoch.call_args_list))
        self.assertEqual(self.load(output)['epoch'], 1)
        with self.assertRaisesRegex(ValueError, 'Smoke checkpoints'):
            self.invoke(output, '--resume', output/'last.pt', '--epochs', 2)

    def test_resume_rejects_changed_data_and_output_reuse(self):
        output = self.root/'run'
        with patch('src.shuttle.train.run_epoch', return_value=.5):
            self.invoke(output, '--init-weights', self.init, '--epochs', 1)
        before = (output/'last.pt').read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(output, '--init-weights', self.init, '--epochs', 2)
        path = self.clips/'train_clip'/'labels.csv'
        path.write_text(path.read_text().replace('16,16', '15,16'))
        with self.assertRaisesRegex(ValueError, 'changed since checkpoint'):
            self.invoke(output, '--resume', output/'last.pt', '--epochs', 2)
        self.assertEqual((output/'last.pt').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
