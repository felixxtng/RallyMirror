"""Manual labels, coordinate transforms, split safety and learning behaviour."""
import copy
import sys
import tempfile
import unittest
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.court.learning_data import (ARCHITECTURE, CourtDataset, GROUND_POINT_NAMES,
                                     LEGACY_POINT_NAMES, POINT_NAMES, blank_points,
                                     import_sources, load_manifest, save_manifest,
                                     sync_images, target_heatmaps, upgrade_manifest)
from src.court.learning_labeler import CourtLabeler
from src.court.learning_model import CourtNet, decode_points, keypoint_loss
from src.court.learning_train import fit_predictions, load_model
from src.court.court_model import COURT_LENGTH, COURT_POINTS, COURT_WIDTH
from src.court.homography import apply_homography


def record():
    return {"id": "matchA/sample", "image": "sample.jpg", "width": 1280, "height": 720,
            "group": "matchA", "court_id": "courtA", "view_id": "courtA_main",
            "split": "train", "unusable": False, "points": blank_points()}


def manifest(row):
    return {"version": 2, "point_names": list(POINT_NAMES), "images": [row]}


class CourtLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def write_sample(self, root, row):
        cv2.imwrite(str(root / "sample.jpg"), np.full((720, 1280, 3), 100, np.uint8))
        save_manifest(root, manifest(row))

    def test_manual_click_hidden_undo_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_sample(root, record())
            labeler = CourtLabeler(root)
            labeler.draw()  # Also exercises the actual preview layout without opening a window.
            labeler.on_mouse(cv2.EVENT_LBUTTONDOWN, 480, 42 + 270, 0, None)
            saved = load_manifest(root)["images"][0]["points"]
            self.assertEqual(saved[POINT_NAMES[0]]["xy"], [640.0, 360.0])
            labeler.annotate()  # hidden
            self.assertEqual(load_manifest(root)["images"][0]["points"][POINT_NAMES[1]]["status"], "hidden")
            labeler.undo()
            self.assertEqual(load_manifest(root)["images"][0]["points"][POINT_NAMES[1]]["status"], "unlabelled")
            resumed = CourtLabeler(root)
            self.assertEqual(resumed.selected, 1)

    def test_concurrent_labeler_cannot_overwrite_new_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_sample(root, record())
            first, second = CourtLabeler(root), CourtLabeler(root)
            first.annotate([100, 100])
            with self.assertRaisesRegex(RuntimeError, "another session"):
                second.annotate([200, 200])
            self.assertEqual(load_manifest(root)["images"][0]["points"][POINT_NAMES[0]]["xy"], [100, 100])

    def test_dataset_coordinate_conversion_and_hidden_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = record()
            for i, name in enumerate(POINT_NAMES):
                row["points"][name] = {"status": "visible", "xy": [640., 360.]}
            row["points"][POINT_NAMES[-1]] = {"status": "hidden", "xy": None}
            self.write_sample(root, row)
            dataset = CourtDataset(root, "train", width=320, height=184)
            image, target, mask, points = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 184, 320))
            self.assertEqual(tuple(target.shape), (12, 46, 80))
            np.testing.assert_allclose(points[0], [160, 92])
            self.assertEqual(mask[-1], 0)
            self.assertTrue(torch.isnan(points[-1]).all())
            self.assertAlmostEqual(float(target[0].sum()), 1, places=5)
            self.assertEqual(len(CourtDataset(root, "val")), 0)

    def test_rejects_match_leakage_even_for_unfinished_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = manifest(record())
            other = copy.deepcopy(data["images"][0])
            other.update(id="matchA/other", split="val")
            data["images"].append(other)
            save_manifest(root, data)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                CourtDataset(root, "train")

    def test_loss_ignores_hidden_points(self):
        logits = torch.zeros((1, 12, 16, 16), requires_grad=True)
        points = np.full((12, 2), np.nan)
        points[0] = [8, 8]
        target, mask = target_heatmaps(points, 16, 16)
        loss = keypoint_loss(logits, target[None], mask[None])
        loss.backward()
        self.assertGreater(float(logits.grad[:, 0].abs().sum()), 0)
        self.assertEqual(float(logits.grad[:, 1:].abs().sum()), 0)

    def test_random_model_learns_and_own_checkpoint_roundtrips(self):
        torch.manual_seed(42)
        model = CourtNet()
        images = torch.rand((1, 3, 64, 64))
        coords = np.array([[2, 13], [13, 13], [4, 2], [11, 2], [3, 9], [12, 9], [4, 5], [11, 5],
                           [3, 6], [12, 6], [8, 9], [8, 5]], float)
        target, mask = target_heatmaps(coords, 16, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial = float(keypoint_loss(model(images), target[None], mask[None]).detach())
        for _ in range(12):
            optimizer.zero_grad()
            loss = keypoint_loss(model(images), target[None], mask[None])
            loss.backward()
            optimizer.step()
        model.eval()
        logits = model(images)
        self.assertLess(float(keypoint_loss(logits, target[None], mask[None]).detach()), initial)
        self.assertEqual(tuple(decode_points(logits).shape), (1, 12, 2))
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "own.pt"
            torch.save({"model": model.state_dict(), "architecture": ARCHITECTURE,
                        "point_names": list(POINT_NAMES)}, checkpoint)
            restored, _ = load_model(checkpoint, torch.device("cpu"))
            torch.testing.assert_close(restored(images), logits)

    def test_geometry_rejects_collapsed_points_and_handles_an_outlier(self):
        self.assertIsNone(fit_predictions(np.full((12, 2), 100., np.float32), (960, 540), 8))
        court = np.float32([[0, 0], [COURT_WIDTH, 0], [0, COURT_LENGTH], [COURT_WIDTH, COURT_LENGTH]])
        corners = np.float32([[150, 500], [830, 490], [310, 150], [650, 140]])
        H = cv2.getPerspectiveTransform(court, corners)
        points = np.zeros((12, 2))
        for i, name in enumerate(POINT_NAMES):
            if name in GROUND_POINT_NAMES:
                points[i] = apply_homography(H, COURT_POINTS[name])
            else:
                points[i] = [500, 20]  # Deliberately unrelated raised net pixels.
        points[-1] += [100, -60]
        fit = fit_predictions(points, (960, 540), 8)
        self.assertIsNotNone(fit)
        recovered, inliers = fit
        self.assertEqual(inliers, 9)
        np.testing.assert_allclose(apply_homography(recovered, points[:4]), court, atol=1e-5)
        points[8:10] = [[-1e6, -1e6], [1e6, 1e6]]
        changed, _ = fit_predictions(points, (960, 540), 8)
        np.testing.assert_allclose(changed, recovered)

    def test_eight_point_migration_preserves_original_and_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = record()
            old["points"] = {n: {"status": "visible", "xy": [100+i, 200+i]} for i, n in enumerate(LEGACY_POINT_NAMES)}
            data = {"version": 1, "point_names": list(LEGACY_POINT_NAMES), "images": [old]}
            original = json.dumps(data, indent=2).encode()
            (root / "annotations.json").write_bytes(original)
            upgrade_manifest(root)
            self.assertEqual((root / "annotations.eight_points.backup.json").read_bytes(), original)
            upgraded = load_manifest(root)
            for n in LEGACY_POINT_NAMES:
                self.assertEqual(upgraded["images"][0]["points"][n], old["points"][n])
            self.assertEqual(upgraded["images"][0]["points"]["NL-TOP"]["status"], "unlabelled")
            labeler = CourtLabeler(root)
            self.assertEqual(labeler.selected, 8)

    def test_other_match_on_same_court_cannot_leak_into_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = manifest(record())
            other = copy.deepcopy(data["images"][0])
            other.update(id="matchB/other", group="matchB", view_id="courtA_other", split="test")
            data["images"].append(other)
            save_manifest(root, data)
            with self.assertRaisesRegex(ValueError, "court_id"):
                CourtDataset(root, "train")

    def test_one_per_view_does_not_count_repeat_rallies_as_new_courts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = record()
            for name in POINT_NAMES:
                row["points"][name] = {"status": "visible", "xy": [100, 100]}
            data = manifest(row)
            other = copy.deepcopy(row)
            other["id"] = "matchA/another_rally"
            data["images"].append(other)
            save_manifest(root, data)
            self.assertEqual(len(CourtDataset(root, "train", one_per_view=True)), 1)
            self.assertEqual(len(CourtLabeler(root, one_per_view=True).indices), 1)
            self.assertEqual(len(CourtDataset(root, "train", one_per_view=False)), 2)

    def test_drop_folder_sync_preserves_labels_and_skips_duplicate_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            images.mkdir()
            cv2.imwrite(str(images / "first.png"), np.full((80, 128, 3), 100, np.uint8))
            self.assertEqual(sync_images(root), 1)
            labeler = CourtLabeler(root)
            for _ in POINT_NAMES:
                labeler.annotate([10, 20])
            before = (root / "annotations.json").read_bytes()
            self.assertEqual(sync_images(root), 0)
            self.assertEqual((root / "annotations.json").read_bytes(), before)
            cv2.imwrite(str(images / "duplicate.bmp"), np.full((80, 128, 3), 100, np.uint8))
            cv2.imwrite(str(images / "second.png"), np.full((80, 128, 3), 200, np.uint8))
            self.assertEqual(sync_images(root), 1)
            rows = load_manifest(root)["images"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["points"][POINT_NAMES[0]]["xy"], [10, 20])
            self.assertEqual(rows[1]["points"], blank_points())
            self.assertEqual(rows[1]["split"], "train")
            self.assertEqual(rows[1]["court_identity_status"], "unverified_photo")
            self.assertEqual(CourtLabeler(root).index, 1)

    def test_drop_folder_bad_image_does_not_partially_register_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            images.mkdir()
            cv2.imwrite(str(images / "a.png"), np.zeros((80, 128, 3), np.uint8))
            (images / "z.png").write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "Cannot read"):
                sync_images(root)
            self.assertFalse((root / "annotations.json").exists())

    def test_drop_folder_missing_registered_image_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            save_manifest(root, manifest(record()))
            before = (root / "annotations.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "registered image"):
                sync_images(root)
            self.assertEqual((root / "annotations.json").read_bytes(), before)

    def test_photo_roster_import_is_repeatable_and_keeps_manual_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            photo = root / "court.png"
            cv2.imwrite(str(photo), np.full((80, 128, 3), 100, np.uint8))
            csv_path = root / "sources.csv"
            with csv_path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["source", "group", "court_id", "view_id", "split", "frame"])
                writer.writerow(["court.png", "matchA", "venueA_court01", "venueA_court01_main", "train", ""])
            output = root / "dataset"
            import_sources(csv_path, output)
            labeler = CourtLabeler(output)
            labeler.annotate([10, 20])
            import_sources(csv_path, output)
            data = load_manifest(output)
            self.assertEqual(len(data["images"]), 1)
            self.assertEqual(data["images"][0]["points"][POINT_NAMES[0]]["xy"], [10, 20])


if __name__ == "__main__":
    unittest.main()
