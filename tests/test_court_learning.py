"""Tests for the learned court detector's data and geometry code."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.court.augment import (flip_index, flip_name, random_matrix, random_occlusion,
                               resize_matrix, transform_points)
from src.court.court_model import COURT_POINTS
from src.court.export_dataset import DEFAULT_POINTS, angle_of, assign_splits, project_court_points, venue_of
from src.court.fit import calibration_from_points, near_end_is_lower
from src.court.homography import apply_homography, ransac_homography
from src.court.metrics import angle_summary, floor_grid_in_image, half_errors
from src.court.zoom import ZoomView

H_COURT_TO_IMAGE = np.array([[110.0, 35.0, 300.0], [-6.0, -30.0, 900.0], [0.0, 0.0012, 1.0]])
H_IMAGE_TO_COURT = np.linalg.inv(H_COURT_TO_IMAGE)

try:
    import torch
    from src.court.net import CourtNet, decode_heatmaps, make_heatmaps
except Exception:          # torch missing: the network tests are skipped
    torch = None


class AugmentGeometryTests(unittest.TestCase):
    def test_flip_names(self):
        self.assertEqual(flip_name("DL-B0"), "DR-B0")
        self.assertEqual(flip_name("SR-SS1"), "SL-SS1")
        self.assertEqual(flip_name("C-B0"), "C-B0")

    def test_flip_index_is_its_own_inverse(self):
        idx = flip_index(DEFAULT_POINTS)
        self.assertEqual([idx[i] for i in idx], list(range(len(DEFAULT_POINTS))))
        self.assertEqual(DEFAULT_POINTS[idx[0]], "DR-B0")

    def test_zero_strength_is_plain_resize(self):
        rng = np.random.default_rng(0)
        M = random_matrix(rng, 1280, 720, 448, 256, strength=0.0)
        np.testing.assert_allclose(M / M[2, 2], resize_matrix(1280, 720, 448, 256), atol=1e-9)

    def test_transform_marks_points_leaving_the_frame(self):
        M = resize_matrix(1000, 500, 448, 256)
        moved, inside = transform_points(M, [[500, 250], [1200, 250], [-5, 10]], 448, 256)
        self.assertEqual(list(inside), [True, False, False])
        np.testing.assert_allclose(moved[0], [224, 128])


class ExportTests(unittest.TestCase):
    def test_venue_name(self):
        self.assertEqual(venue_of("venueA_03"), "venuea")
        self.assertEqual(venue_of("VenueA_01"), "venuea")        # case does not matter

    def test_splits_by_venue(self):
        mapping, warnings = assign_splits(["a", "b", "c", "d"])
        self.assertEqual(sorted(mapping.values()), ["test", "train", "train", "val"])
        self.assertFalse(warnings)
        mapping, warnings = assign_splits(["venueA", "venueB"])
        self.assertEqual(mapping, {"venuea": "train", "venueb": "test"})
        self.assertTrue(warnings)
        mapping, _ = assign_splits(["a", "b", "c"], val_venues=["a"], test_venues=["b"])
        self.assertEqual(mapping, {"a": "val", "b": "test", "c": "train"})
        mapping, _ = assign_splits(["VenueA", "venueA", "venueB", "VenueC"], test_venues=["VENUEC"])
        self.assertEqual(mapping["venuec"], "test")
        self.assertEqual(sorted(mapping), ["venuea", "venueb", "venuec"])
        with self.assertRaises(ValueError):
            assign_splits(["a", "b"], test_venues=["zzz"])

    def test_projection_visibility(self):
        names = ["DL-B0", "DL-NET", "DR-NET"]
        result = project_court_points(H_IMAGE_TO_COURT, names, (1280, 720))
        self.assertFalse(result["DL-B0"][1])               # lands below the bottom edge
        self.assertTrue(result["DL-NET"][1])
        expected = apply_homography(H_COURT_TO_IMAGE, COURT_POINTS["DL-NET"])
        np.testing.assert_allclose(result["DL-NET"][0], expected, atol=1e-6)


class RansacTests(unittest.TestCase):
    def test_outliers_are_rejected(self):
        names = DEFAULT_POINTS
        court = np.array([COURT_POINTS[n] for n in names])
        pixels = apply_homography(H_COURT_TO_IMAGE, court)
        bad = pixels.copy()
        bad[1] += [150, -90]                                # two wrong detections
        bad[9] += [-120, 60]
        H, inliers = ransac_homography(bad, court, thresh_px=4.0)
        self.assertIsNotNone(H)
        self.assertEqual(int(inliers.sum()), len(names) - 2)
        self.assertFalse(inliers[1] or inliers[9])
        np.testing.assert_allclose(apply_homography(H, pixels[3]), court[3], atol=1e-3)

    def test_too_few_points(self):
        H, mask = ransac_homography([[0, 0], [1, 1], [2, 0]], [[0, 0], [1, 1], [2, 0]], 3.0)
        self.assertIsNone(H)


class OcclusionAndMetricsTests(unittest.TestCase):
    def test_occlusion_changes_pixels_but_not_shape_or_input(self):
        image = np.full((256, 448, 3), 100, np.uint8)
        changed = 0
        for seed in range(20):
            out = random_occlusion(image, np.random.default_rng(seed))
            self.assertEqual(out.shape, image.shape)
            self.assertEqual(out.dtype, np.uint8)
            changed += int((out != image).any())
        self.assertTrue(np.all(image == 100))           # the input is never modified
        self.assertGreater(changed, 10)                  # most seeds paint something

    def test_edge_shifts_can_push_points_out_of_frame(self):
        rng = np.random.default_rng(1)
        outside = 0
        for _ in range(300):
            M = random_matrix(rng, 1280, 720, 448, 256)
            _, inside = transform_points(M, [[640, 700]], 448, 256)   # point near the bottom edge
            outside += int(not inside[0])
        self.assertGreater(outside, 10)
        self.assertLess(outside, 250)

    def test_floor_grid_and_half_errors(self):
        pixels, court = floor_grid_in_image(H_IMAGE_TO_COURT, (1280, 720))
        self.assertGreater(len(court), 20)
        np.testing.assert_allclose(apply_homography(H_IMAGE_TO_COURT, pixels), court, atol=1e-6)
        errors = np.where(court[:, 1] < 6.70, 0.5, 0.1)
        halves = half_errors(errors, court)
        self.assertAlmostEqual(halves["near"][0], 0.5)
        self.assertAlmostEqual(halves["far"][0], 0.1)
        self.assertEqual(half_errors([0.2], [[1.0, 2.0]])["far"], (None, None))


class ZoomViewTests(unittest.TestCase):
    def test_starts_fitted_and_round_trips(self):
        view = ZoomView(1920, 1080, 1280, 720)
        np.testing.assert_allclose(view.to_image(640, 360), [960, 540])
        np.testing.assert_allclose(view.to_display(*view.to_image(100, 50)), [100, 50])

    def test_zoom_keeps_point_under_cursor(self):
        view = ZoomView(1920, 1080, 1280, 720)
        before = view.to_image(900, 500)
        view.zoom_at(900, 500, 4.0)
        np.testing.assert_allclose(view.to_image(900, 500), before, atol=1e-6)
        self.assertAlmostEqual(view.scale, view.base_scale * 4.0)

    def test_zoom_limits_and_reset(self):
        view = ZoomView(1920, 1080, 1280, 720)
        view.zoom_at(10, 10, 0.1)                       # cannot zoom out past the fitted view
        self.assertAlmostEqual(view.scale, view.base_scale)
        view.zoom_at(10, 10, 1000)                      # cannot zoom in forever
        self.assertAlmostEqual(view.scale, view.base_scale * ZoomView.MAX_ZOOM)
        view.reset()
        self.assertEqual((view.origin_x, view.origin_y), (0.0, 0.0))

    def test_pan_is_clamped_to_the_image(self):
        view = ZoomView(1920, 1080, 1280, 720)
        view.zoom_at(0, 0, 2.0)
        view.pan_by_display(10 ** 6, 10 ** 6)
        far = view.to_image(1279, 719)
        self.assertLessEqual(far[0], 1920 + 1e-6)
        self.assertLessEqual(far[1], 1080 + 1e-6)
        view.pan_by_display(-10 ** 6, -10 ** 6)
        self.assertEqual((view.origin_x, view.origin_y), (0.0, 0.0))


class FitSafetyTests(unittest.TestCase):
    NAMES = DEFAULT_POINTS

    def detections(self, names=None):
        names = names or self.NAMES
        court = [COURT_POINTS[n] for n in self.NAMES]
        return apply_homography(H_COURT_TO_IMAGE, court)

    def test_good_detections_are_accepted(self):
        info = {}
        calib = calibration_from_points(self.NAMES, self.detections(), np.ones(12), (1280, 720), info=info)
        self.assertIsNotNone(calib)
        self.assertEqual(info["inliers"], 12)
        self.assertTrue(near_end_is_lower(np.array(calib["H_image_to_court"])))

    def test_too_few_consistent_points_are_rejected(self):
        pixels = self.detections()
        pixels[2:] += np.random.default_rng(0).uniform(-300, 300, (10, 2))   # garbage detections
        info = {}
        self.assertIsNone(calibration_from_points(self.NAMES, pixels, np.ones(12), (1280, 720), info=info))
        self.assertIn("consistent", info["reason"])

    def test_low_confidence_points_are_ignored(self):
        info = {}
        conf = np.array([1.0] * 3 + [0.0] * 9)
        self.assertIsNone(calibration_from_points(self.NAMES, self.detections(), conf, (1280, 720), info=info))
        self.assertIn("confident", info["reason"])

    def test_upside_down_court_is_rejected(self):
        swap = {"B0": "B1", "B1": "B0", "SS0": "SS1", "SS1": "SS0", "LS0": "LS1", "LS1": "LS0"}
        wrong_names = [n.split("-")[0] + "-" + swap[n.split("-")[1]] for n in self.NAMES]
        info = {}
        self.assertIsNone(calibration_from_points(wrong_names, self.detections(), np.ones(12), (1280, 720), info=info))
        self.assertIn("upside down", info["reason"])
        self.assertIsNotNone(calibration_from_points(wrong_names, self.detections(), np.ones(12), (1280, 720),
                                                     check_orientation=False))


class AngleTests(unittest.TestCase):
    def test_angle_tag_from_name(self):
        self.assertEqual(angle_of("venueF_low_04"), "low")
        self.assertEqual(angle_of("VenueC_Phone_01"), "phone")
        self.assertEqual(angle_of("venueA_01"), "standard")
        self.assertEqual(venue_of("venueF_low_04"), "venuef")      # the tag does not change the venue

    def test_angle_summary(self):
        rows = [{"angle": "standard", "fitted": True, "court_error_mean_m": 0.1},
                {"angle": "standard", "fitted": True, "court_error_mean_m": 0.5},
                {"angle": "low", "fitted": False, "court_error_mean_m": None},
                {"angle": "low", "fitted": True, "court_error_mean_m": 0.2}]
        summary = angle_summary(rows, 0.3)
        self.assertEqual(summary["standard"], {"images": 2, "fitted": 2, "rejected": 0, "success": 1,
                                               "mean_error_m": 0.3})
        self.assertEqual((summary["low"]["rejected"], summary["low"]["success"]), (1, 1))

    def test_tilt_augmentation_sometimes_squashes_the_far_edge(self):
        rng = np.random.default_rng(5)
        ratios = []
        for _ in range(400):
            M = random_matrix(rng, 1280, 720, 448, 256)
            top, _ = transform_points(M, [[0, 0], [1280, 0]], 448, 256)
            bottom, _ = transform_points(M, [[0, 720], [1280, 720]], 448, 256)
            ratios.append((top[1, 0] - top[0, 0]) / (bottom[1, 0] - bottom[0, 0]))
        ratios = np.array(ratios)
        self.assertGreater(int((ratios < 0.78).sum()), 10)       # strong squash happens
        self.assertGreater(int((ratios > 0.9).sum()), 150)       # most samples stay near normal
        self.assertTrue(np.all(ratios > 0.3))


@unittest.skipIf(torch is None, "torch not installed")
class NetworkTests(unittest.TestCase):
    def test_forward_shape(self):
        model = CourtNet(n_points=12, base=8)
        out = model(torch.zeros(2, 3, 256, 448))
        self.assertEqual(tuple(out.shape), (2, 12, 64, 112))

    def test_heatmap_decode_round_trip(self):
        points = np.array([[200.3, 100.7], [30.0, 240.0], [440.0, 10.0]])
        heat = make_heatmaps(points, [True, True, False])
        self.assertEqual(float(heat[2].max()), 0.0)          # invisible point: empty map
        clipped = np.clip(heat, 1e-4, 1 - 1e-4)
        logits = torch.from_numpy(np.log(clipped / (1 - clipped)))[None]
        decoded, conf = decode_heatmaps(logits)
        np.testing.assert_allclose(decoded[0, :2], points[:2], atol=0.75)
        self.assertGreater(conf[0, 0], 0.9)


if __name__ == "__main__":
    unittest.main()
