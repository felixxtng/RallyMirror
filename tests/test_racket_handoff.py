import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.install_racket_model import install
from scripts.racket_demo_utils import predict_batch
from src.racket.real_owner_infer import RealOwnerRacketPredictor


class HandoffTests(unittest.TestCase):
    def test_installer_checks_hash_is_idempotent_and_preserves_existing_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'received.pt'
            source.write_bytes(b'selected-checkpoint')
            card = root/'card.json'
            card.write_text(json.dumps(dict(weights_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
            destination = root/'models/runtime.pt'
            self.assertEqual(install(source, card, destination), destination)
            self.assertEqual(install(source, card, destination), destination)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            destination.write_bytes(b'different-model')
            with self.assertRaises(FileExistsError):
                install(source, card, destination)
            self.assertEqual(destination.read_bytes(), b'different-model')
            source.write_bytes(b'corrupted-upload')
            with self.assertRaisesRegex(ValueError, 'model-card hash'):
                install(source, card, destination)
            self.assertEqual(destination.read_bytes(), b'different-model')

    def test_demo_batch_helper_matches_runtime_for_multiple_and_missing_players(self):
        class Stub(torch.nn.Module):
            def forward(self, images, pose, box):
                logits = torch.full((len(images), 1, 16, 16), -100.)
                logits[:, 0, 8, 6] = 100.
                return logits

        predictor = object.__new__(RealOwnerRacketPredictor)
        predictor.device = torch.device('cpu')
        predictor.size = 64
        predictor.context_padding = .05
        predictor.model = Stub()
        players = [dict(track_id='A', bbox=[70, 100, 130, 220], keypoints=[]),
                   dict(track_id='B', bbox=[160, 40, 200, 120], keypoints=[])]
        original = copy.deepcopy(players)
        frame = np.zeros((400, 400, 3), np.uint8)
        items = [dict(frame=frame, players=copy.deepcopy(players)),
                 dict(frame=frame, players=[]),
                 dict(frame=frame, players=copy.deepcopy(players[:1]))]
        predict_batch(predictor, items)
        for item in items:
            direct = predictor.predict_players(item['frame'], item['players'])
            for expected, actual in zip(direct, item['players']):
                self.assertAlmostEqual(expected['racket']['x'], actual['racket']['x'], places=4)
                self.assertAlmostEqual(expected['racket']['y'], actual['racket']['y'], places=4)
                self.assertEqual(expected['track_id'], actual['track_id'])
        self.assertEqual(players, original)
