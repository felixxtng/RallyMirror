"""Check full-court fitting independently of the real video used for tuning."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.court.court_model import COURT_LENGTH, COURT_WIDTH
from src.court.detector import detect_court
from src.court.draw import overlay_court


class AutomaticDetectorTests(unittest.TestCase):
    def render(self, corners, occlude=False):
        frame = np.full((540, 960, 3), (95, 30, 70), np.uint8)
        cv2.fillConvexPoly(frame, np.int32(corners[[0, 1, 3, 2]]), (90, 140, 70))
        model = np.float32([[0, 0], [COURT_WIDTH, 0], [0, COURT_LENGTH], [COURT_WIDTH, COURT_LENGTH]])
        H = cv2.getPerspectiveTransform(model, corners)
        overlay_court(frame, np.linalg.inv(H), color=(245, 245, 245), thickness=3)
        if occlude:
            cv2.rectangle(frame, (460, 275), (490, 370), (80, 20, 10), -1)
        return frame

    def test_recovers_two_perspectives_with_occlusion(self):
        cases = [np.float32([[150, 500], [830, 490], [310, 150], [650, 140]]),
                 np.float32([[100, 505], [780, 510], [260, 170], [610, 175]])]
        for corners in cases:
            with self.subTest(corners=corners.tolist()):
                result, _, error = detect_court(self.render(corners, occlude=True))
                self.assertIsNone(error)
                self.assertIsNotNone(result)
                errors = np.linalg.norm(np.array(result["corners_image"]) - corners, axis=1)
                self.assertLess(errors.max(), 10)

    def test_rejects_green_floor_without_court_lines(self):
        frame = np.full((540, 960, 3), (90, 140, 70), np.uint8)
        with self.assertRaisesRegex(ValueError, "No white line"):
            detect_court(frame)

    def test_reports_original_resolution_coordinates(self):
        corners = np.float32([[150, 500], [830, 490], [310, 150], [650, 140]])
        frame = cv2.resize(self.render(corners), (1920, 1080))
        result, _, error = detect_court(frame)
        self.assertIsNone(error)
        self.assertEqual(result["image_size"], [1920, 1080])
        errors = np.linalg.norm(np.array(result["corners_image"]) - corners * 2, axis=1)
        self.assertLess(errors.max(), 20)

    def test_rejects_non_court_image(self):
        with self.assertRaisesRegex(ValueError, "No large green"):
            detect_court(np.zeros((540, 960, 3), np.uint8))


if __name__ == "__main__":
    unittest.main()
