from pathlib import Path
import tempfile
import unittest

from tests import test_racket_ownership_fast as fixture
from src.racket.ownership_fast import SampleStore, export_labels
from src.racket.ownership_full_review import FullReviewLabeler
from src.racket.schema import file_hash, read_json


class FullReviewTests(unittest.TestCase):
    def test_racket_click_auto_selects_far_and_manual_selection_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = fixture.FastOwnershipTests().fixture(Path(tmp))
            e = FullReviewLabeler(root)
            try:
                e.click_original([70., 80.])
                self.assertEqual(e.step, ('far_side', 'racket'))
                self.assertEqual(e.identity, 'train01_00002')
                e.handle_key(ord('1'))
                self.assertEqual(e.step, ('near_side', 'racket'))
                e.click_original([72., 81.])
                self.assertEqual(e.step, ('far_side', 'racket'))
                e.handle_key(ord('2'))
                e.click_original([129., 36.])
                self.assertEqual(e.identity, 'partial01_00003')
                self.assertEqual(e.step, ('near_side', 'racket'))
                row = SampleStore(root).row('train01_00002')
                self.assertEqual(row['players']['near_side']['point'], [72., 81.])
                self.assertEqual(row['players']['far_side']['point'], [129., 36.])
            finally:
                e.close()

    def test_h_auto_selects_far_only_in_racket_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = fixture.FastOwnershipTests().fixture(Path(tmp))
            e = FullReviewLabeler(root)
            try:
                e.handle_key(ord('b'))
                e.click_original([45., 50.]); e.click_original([90., 95.])
                self.assertEqual(e.step, ('near_side', 'body'))
                e.handle_key(ord('h'))
                self.assertEqual(e.step, ('near_side', 'body'))
                self.assertEqual(e.current['players']['near_side']['body_state'], 'unlocatable')
                e.handle_key(ord('p')); e.handle_key(ord('h'))
                self.assertEqual(e.step, ('far_side', 'racket'))
                e.handle_key(ord('h'))
                row = SampleStore(root).row('train01_00002')
                for player in row['players'].values():
                    self.assertEqual(player['racket_state'], 'unlocatable')
                self.assertEqual(e.identity, 'partial01_00003')
            finally:
                e.close()

    def test_all_frames_original_order_points_and_boxes_editable_and_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protected = fixture.FastOwnershipTests().fixture(Path(tmp))
            e = FullReviewLabeler(root)
            try:
                self.assertEqual(e.queue, ['train01_00002', 'partial01_00003', 'val01_00001'])
                e.handle_key(ord('1')); e.handle_key(ord('b'))
                e.click_original([45., 50.]); e.click_original([90., 95.])
                e.handle_key(ord('2')); e.handle_key(ord('b'))
                e.click_original([90., 10.]); e.click_original([125., 46.])
                self.assertEqual(e.identity, 'train01_00002')
                e.handle_key(ord('1')); e.click_original([70., 80.])
                e.click_original([130., 35.])
                self.assertEqual(e.identity, 'partial01_00003')
                self.assertEqual(SampleStore(root).row('train01_00002')['decision'], 'complete')
                e.handle_key(ord('a'))
                e.handle_key(ord('1')); e.click_original([72., 81.])
                self.assertEqual(e.identity, 'train01_00002')
                self.assertEqual(e.current['players']['near_side']['point'], [72., 81.])
                self.assertEqual(e.draw().shape[0], e.view_height+240)
            finally:
                e.close()
            self.assertEqual(protected, {p: file_hash(p) for p in protected})

    def test_original_reference_can_be_corrected_with_provenance_and_frozen_export_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protected = fixture.FastOwnershipTests().fixture(Path(tmp))
            e = FullReviewLabeler(root, 'val01_00001')
            try:
                e.handle_key(ord('1')); e.click_original([70., 80.])
                self.assertEqual(e.current['players']['near_side']['reference_override']['original_point'], [130., 45.])
                e.handle_key(ord('2')); e.click_original([130., 35.])
                row = SampleStore(root).row('val01_00001')
                self.assertEqual(row['decision'], 'complete')
                self.assertEqual(row['players']['near_side']['point'], [70., 80.])
                out = Path(tmp)/'export'
                export_labels(root, out)
                self.assertNotEqual((out/'val.jsonl').read_text(), (out/'historical_frozen/val.jsonl').read_text())
                import json
                new = next(json.loads(s) for s in (out/'val.jsonl').read_text().splitlines() if json.loads(s)['id']=='val01_00001')
                self.assertEqual(new['players']['near_side']['reference_override']['original_point'], [130., 45.])
                self.assertEqual(protected, {p: file_hash(p) for p in protected})
            finally:
                e.close()

    def test_near_h_selects_far_without_advance_and_far_h_confirms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = fixture.FastOwnershipTests().fixture(Path(tmp))
            e = FullReviewLabeler(root, 'val01_00001')
            try:
                e.handle_key(ord('1')); e.handle_key(ord('h'))
                self.assertEqual(e.identity, 'val01_00001')
                self.assertEqual(e.current['players']['near_side']['racket_state'], 'unlocatable')
                self.assertIsNotNone(e.current['players']['far_side']['point'])
                e.handle_key(ord('2')); e.handle_key(ord('h'))
                self.assertEqual(SampleStore(root).row('val01_00001')['decision'], 'complete')
            finally:
                e.close()

    def test_resume_and_direct_id_do_not_filter_out_the_original_first_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = fixture.FastOwnershipTests().fixture(Path(tmp))
            e = FullReviewLabeler(root)
            try:
                e.handle_key(ord('1')); e.click_original([70., 80.])
                e.handle_key(ord('2')); e.handle_key(ord('h'))
            finally:
                e.close()
            resumed = FullReviewLabeler(root)
            try:
                self.assertEqual(resumed.identity, 'partial01_00003')
                resumed.handle_key(ord('a'))
                self.assertEqual(resumed.identity, 'train01_00002')
            finally:
                resumed.close()
            direct = FullReviewLabeler(root, 'train01_00002')
            try:
                self.assertEqual(direct.identity, 'train01_00002')
            finally:
                direct.close()

    def test_confirmation_status_auto_next_near_only_and_edit_invalidates_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = fixture.FastOwnershipTests().fixture(Path(tmp))
            e = FullReviewLabeler(root)
            try:
                self.assertEqual(e.review_status, 'DRAFT - NOT CONFIRMED')
                e.handle_key(32)
                self.assertEqual(e.identity, 'train01_00002')
                self.assertIn('NOT CONFIRMED', e.message)
                self.assertIsNone(e.last_confirmation)
                e.handle_key(ord('2')); e.handle_key(ord('b')); e.handle_key(ord('h'))
                e.handle_key(ord('1')); e.click_original([70., 80.])
                e.handle_key(ord('2')); e.handle_key(ord('h'))
                self.assertEqual(e.identity, 'partial01_00003')
                self.assertIn('CONFIRMED: train01 frame 2', e.last_confirmation)
                self.assertIn('Automatically moved', e.message)
                row = SampleStore(root).row('train01_00002')
                self.assertEqual(row['players']['far_side']['racket_state'], 'unlocatable')
                self.assertEqual(row['players']['far_side']['body_state'], 'unlocatable')
                self.assertEqual(row['players']['near_side']['racket_state'], 'located')
                e.handle_key(ord('a'))
                self.assertEqual(e.review_status, 'CONFIRMED')
                e.handle_key(ord('1')); e.click_original([72., 81.])
                self.assertEqual(e.review_status, 'DRAFT - NOT CONFIRMED')
                e.handle_key(32); e.handle_key(ord('a'))
                self.assertEqual(e.review_status, 'CONFIRMED')
            finally:
                e.close()

    def test_all_click_h_pairs_auto_next_and_preserve_independent_labels(self):
        for near, far in [(True, True), (True, False), (False, True), (False, False)]:
            with self.subTest(near=near, far=far), tempfile.TemporaryDirectory() as tmp:
                root, _ = fixture.FastOwnershipTests().fixture(Path(tmp))
                e = FullReviewLabeler(root)
                try:
                    for visible, point in [(near, [70., 80.]), (far, [130., 35.])]:
                        if visible:
                            e.click_original(point)
                        else:
                            e.handle_key(ord('h'))
                    self.assertEqual(e.identity, 'partial01_00003')
                    row = SampleStore(root).row('train01_00002')
                    self.assertEqual(row['decision'], 'complete')
                    for side, visible in zip(('near_side', 'far_side'), (near, far)):
                        player = row['players'][side]
                        self.assertEqual(player['racket_state'], 'located' if visible else 'unlocatable')
                        self.assertEqual(player['point'] is not None, visible)
                finally:
                    e.close()

    def test_missing_body_blocks_auto_next_but_keeps_points_then_space_accepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = fixture.FastOwnershipTests().fixture(Path(tmp))
            e = FullReviewLabeler(root, 'partial01_00003')
            try:
                e.click_original([70., 80.]); e.click_original([130., 35.])
                self.assertEqual(e.identity, 'partial01_00003')
                self.assertEqual(e.review_status, 'DRAFT - NOT CONFIRMED')
                self.assertIn('body missing', e.message)
                self.assertEqual(SampleStore(root).row(e.identity)['players']['far_side']['point'], [130., 35.])
                for side, box in [('1', ([45., 50.], [90., 95.])), ('2', ([90., 10.], [125., 46.]))]:
                    e.handle_key(ord(side)); e.handle_key(ord('b'))
                    e.click_original(box[0]); e.click_original(box[1])
                self.assertEqual(e.identity, 'partial01_00003')
                e.handle_key(32)
                self.assertEqual(e.identity, 'val01_00001')
                self.assertEqual(SampleStore(root).row('partial01_00003')['decision'], 'complete')
            finally:
                e.close()


if __name__ == '__main__':
    unittest.main()
