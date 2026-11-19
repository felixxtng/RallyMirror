from pathlib import Path
import tempfile
import unittest

from tests import test_racket_ownership_fast as fixture
from src.racket.ownership_box_review import SampledBoxLabeler, body_digest, proposal_risk
from src.racket.ownership_fast import SampleStore, initial_row
from src.racket.ownership_fast_labeler import FastLabeler
from src.racket.schema import file_hash


class SampledBoxesTests(unittest.TestCase):
    def test_new_frame_boxes_only_then_two_point_workflow_preserves_review_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protected = fixture.FastOwnershipTests().fixture(Path(tmp))
            editor = SampledBoxLabeler(root)
            try:
                self.assertEqual(set(editor.queue), {'train01_00002', 'partial01_00003'})
                self.assertNotIn('val01_00001', editor.queue)
                # Highest-priority case has no boxes; no point annotations are requested.
                self.assertEqual(editor.identity, 'partial01_00003')
                editor.handle_key(ord('1')); editor.handle_key(ord('b'))
                editor.click_original([45., 50.]); editor.click_original([90., 95.])
                self.assertEqual(editor.identity, 'partial01_00003')
                editor.handle_key(ord('2')); editor.handle_key(ord('b'))
                editor.click_original([90., 10.]); editor.click_original([125., 46.])
                editor.handle_key(32)
                self.assertEqual(editor.identity, 'train01_00002')
                saved = SampleStore(root).row('partial01_00003')
                self.assertEqual(saved['decision'], 'pending')
                self.assertTrue(all(p['point'] is None and p['racket_state'] == 'pending' for p in saved['players'].values()))
                self.assertTrue(all(p['body_state'] == 'verified' for p in saved['players'].values()))
                editor.handle_key(32)  # existing proposals are correct in the fixture
                self.assertTrue(editor.completed)
                self.assertEqual(editor.remaining(), [])
                self.assertEqual(editor.draw().shape[0], editor.view_height+240)
            finally:
                editor.close()
            points = FastLabeler(root, 'partial01_00003')
            try:
                points.click_original([70., 80.]); points.click_original([130., 35.])
                self.assertNotEqual(points.identity, 'partial01_00003')  # body checks already done
            finally:
                points.close()
            resumed = SampledBoxLabeler(root)
            try:
                self.assertEqual(resumed.remaining(), [])  # point edits cannot undo body review
            finally:
                resumed.close()
            self.assertEqual(protected, {p: file_hash(p) for p in protected})

    def test_h_only_marks_selected_body_unknown_and_requires_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = fixture.FastOwnershipTests().fixture(Path(tmp))
            editor = SampledBoxLabeler(root)
            try:
                editor.handle_key(ord('1')); editor.handle_key(ord('h'))
                self.assertEqual(editor.identity, 'partial01_00003')
                editor.handle_key(ord('2')); editor.handle_key(ord('h')); editor.handle_key(32)
                row = SampleStore(root).row('partial01_00003')
                self.assertEqual(row['decision'], 'pending')
                self.assertTrue(all(p['body_state'] == 'unlocatable' for p in row['players'].values()))
                self.assertTrue(all(p['racket_state'] == 'pending' for p in row['players'].values()))
            finally:
                editor.close()

    def test_priority_flags_official_geometry_without_altering_boxes(self):
        meta = dict(width=1000, height=600)
        task = dict(proposals=dict(near_side=[400., 300., 550., 550.], far_side=[820., 140., 900., 350.]))
        row = initial_row(task)
        before = body_digest(row)
        risk = proposal_risk(task, row, meta)
        self.assertGreater(risk['score'], 0)
        self.assertEqual(risk['selected'], 'far_side')
        self.assertIn('possible official', ' '.join(risk['reasons']))
        self.assertEqual(body_digest(row), before)


if __name__ == '__main__':
    unittest.main()
