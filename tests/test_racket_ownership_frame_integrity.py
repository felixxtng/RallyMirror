import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_racket_ownership_fast as fixture
from src.racket.ownership_full_review import FullReviewLabeler
from src.racket.schema import file_hash


class FrameIntegrityTests(unittest.TestCase):
    def test_seek_mismatch_retries_sequential_decode_and_caches_only_verified_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protected = fixture.FastOwnershipTests().fixture(Path(tmp))
            editor = FullReviewLabeler(root)
            image = np.zeros((100, 160, 3), dtype=np.uint8)
            editor.task['image_sha256'] = hashlib.sha256(image.tobytes()).hexdigest()
            try:
                with patch('src.racket.ownership_fast_labeler.FrameSource') as source, patch(
                        'src.racket.ownership_fast_labeler.read_selected',
                        return_value=iter([(editor.task['frame'], image)])) as sequential:
                    source.return_value.read.return_value = image+1
                    self.assertIs(editor.image(), image)
                    self.assertIs(editor.image(), image)
                    self.assertEqual(editor.cached_id, editor.identity)
                    sequential.assert_called_once_with(editor.meta, [editor.task['frame']])
                    source.return_value.read.assert_called_once()
            finally:
                editor.close()
            self.assertEqual(protected, {p: file_hash(p) for p in protected})

    def test_persistent_mismatch_still_rejected_and_not_cached_as_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = fixture.FastOwnershipTests().fixture(Path(tmp))
            editor = FullReviewLabeler(root)
            image = np.zeros((100, 160, 3), dtype=np.uint8)
            editor.task['image_sha256'] = hashlib.sha256(image.tobytes()).hexdigest()
            try:
                with patch('src.racket.ownership_fast_labeler.FrameSource') as source, patch(
                        'src.racket.ownership_fast_labeler.read_selected',
                        return_value=iter([(editor.task['frame'], image+2)])):
                    source.return_value.read.return_value = image+1
                    with self.assertRaisesRegex(ValueError, 'even after sequential decoding'):
                        editor.image()
                    self.assertIsNone(editor.cached_id)
            finally:
                editor.close()
