"""Dependency-free checks of geometry; not a real model/inference test."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

source = Path(__file__).resolve().parents[1] / "src/pose/estimator.py"
tree = ast.parse(source.read_text())
# Load just the actual geometry and extraction definitions without vision packages.
selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name in {"crop_bounds", "PlayerPose"}]
namespace = {"math": math}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)

class Frame:
    shape = (100, 200, 3)
    def __getitem__(self, slices):
        self.slices = slices
        return self

class GeometryTests(unittest.TestCase):
    def test_padding_and_clipping(self):
        bounds = namespace["crop_bounds"]
        self.assertEqual(bounds(Frame(), [50, 20, 150, 80]), (35, 11, 165, 89))
        self.assertEqual(bounds(Frame(), [-10, -10, 210, 110]), (0, 0, 200, 100))
        for box in ([10, 20, 10, 50], [300, 20, 400, 50], [0, 0, math.nan, 5]):
            self.assertIsNone(bounds(Frame(), box))

    def test_mapping_and_missing_pose(self):
        namespace.update(NAMES=["nose"], REMOVED_LANDMARKS=set(),
            cv2=NS(COLOR_BGR2RGB=1, cvtColor=lambda crop, code: crop),
            mp=NS(ImageFormat=NS(SRGB=1), Image=lambda **kwargs: kwargs))
        pose = namespace["PlayerPose"].__new__(namespace["PlayerPose"])
        pose.model = NS(detect=lambda image: NS(pose_landmarks=[[
            NS(x=0.25, y=0.75, visibility=0.9, presence=0.8)]]))
        points = pose.extract(Frame(), {"bbox": [50, 20, 150, 80]})
        self.assertEqual(points[0]["x"], 67.5)
        self.assertEqual(points[0]["y"], 69.5)
        self.assertEqual(points[0]["name"], "nose")
        self.assertEqual(pose.extract(Frame(), None), [])
        pose.model = NS(detect=lambda image: NS(pose_landmarks=[]))
        self.assertEqual(pose.extract(Frame(), {"bbox": [50, 20, 150, 80]}), [])

    def test_removed_landmarks_are_excluded(self):
        namespace.update(
            NAMES=[f"point_{i}" for i in range(33)],
            REMOVED_LANDMARKS={1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                               17, 18, 19, 20, 21, 22},
            cv2=NS(COLOR_BGR2RGB=1, cvtColor=lambda crop, code: crop),
            mp=NS(ImageFormat=NS(SRGB=1), Image=lambda **kwargs: kwargs),
        )
        landmark = NS(x=0.5, y=0.5, visibility=1.0, presence=1.0)
        pose = namespace["PlayerPose"].__new__(namespace["PlayerPose"])
        pose.model = NS(detect=lambda image: NS(pose_landmarks=[[landmark] * 33]))

        points = pose.extract(Frame(), {"bbox": [50, 20, 150, 80]})

        self.assertEqual(len(points), 17)
        self.assertEqual(
            {point["name"] for point in points},
            {f"point_{i}" for i in range(33)} -
            {f"point_{i}" for i in namespace["REMOVED_LANDMARKS"]},
        )

if __name__ == "__main__":
    unittest.main()
