"""Offline checks for the model-fetch helper; no model or network is required."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("fetch_models", Path(__file__).resolve().parents[1] / "scripts/fetch-models.py")
fetch_models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_models)


class ModelFetchTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".work/deployment-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.content = b"selected model bytes"
        self.expected = hashlib.sha256(self.content).hexdigest()
        self.path = self.root / "models/example.pt"

    def test_existing_selected_file_is_verified_without_network(self):
        self.path.parent.mkdir()
        self.path.write_bytes(self.content)
        with patch.object(fetch_models.urllib.request, "urlopen") as network:
            self.assertEqual(fetch_models.selected_model("models/example.pt", self.expected, self.root), "verified")
        network.assert_not_called()

    def test_existing_different_file_is_never_overwritten(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b"another checkpoint")
        with self.assertRaisesRegex(ValueError, "left unchanged"):
            fetch_models.selected_model("models/example.pt", self.expected, self.root)
        self.assertEqual(self.path.read_bytes(), b"another checkpoint")

    def test_check_mode_does_not_fetch_missing_file(self):
        with patch.object(fetch_models.urllib.request, "urlopen") as network:
            with self.assertRaises(FileNotFoundError):
                fetch_models.selected_model("models/example.pt", self.expected, self.root, check=True)
        network.assert_not_called()

    def test_download_is_verified_then_installed_without_partial_artifacts(self):
        with patch.object(fetch_models.urllib.request, "urlopen", return_value=io.BytesIO(self.content)):
            self.assertEqual(fetch_models.selected_model("models/example.pt", self.expected, self.root), "downloaded")
        self.assertEqual(self.path.read_bytes(), self.content)
        self.assertEqual(list(self.path.parent.glob("*.part")), [])

    def test_wrong_download_hash_installs_nothing_and_removes_temporary_file(self):
        with patch.object(fetch_models.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "No model was installed"):
                fetch_models.selected_model("models/example.pt", self.expected, self.root)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_paths_cannot_escape_models_directory(self):
        for path in ["", "models", "../other.pt", "models/../../other.pt", "/models/other.pt", "src/other.pt", "models\\..\\other.pt"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                fetch_models.selected_model(path, self.expected, self.root)


if __name__ == "__main__":
    unittest.main()
