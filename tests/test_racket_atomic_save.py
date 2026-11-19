"""Rapid autosave tolerates temporary Windows locks without hiding failures."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.racket.schema import read_json, save_json


class AtomicSaveTests(unittest.TestCase):
    def test_temporary_windows_lock_preserves_old_file_until_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"annotations.json"
            save_json(path, {"old": True})
            original_replace = Path.replace
            attempts = []
            def temporarily_locked(source, target):
                attempts.append(1)
                if len(attempts) < 3:
                    self.assertEqual(read_json(path), {"old": True})
                    error = PermissionError("temporarily locked")
                    error.winerror = 32
                    raise error
                return original_replace(source, target)
            with patch.object(Path, "replace", temporarily_locked), patch("src.racket.schema.time.sleep"):
                save_json(path, {"new": True})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(read_json(path), {"new": True})

    def test_permanent_lock_is_reported_and_keeps_previous_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"annotations.json"
            save_json(path, {"old": True})
            error = PermissionError("locked")
            error.winerror = 5
            with patch.object(Path, "replace", side_effect=error) as replace, patch("src.racket.schema.time.sleep"):
                with self.assertRaises(PermissionError):
                    save_json(path, {"new": True})
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(read_json(path), {"old": True})
            self.assertEqual(read_json(path.with_suffix(".json.tmp")), {"new": True})


if __name__ == "__main__":
    unittest.main()
