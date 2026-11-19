import copy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tests.racket_fixtures import ownership_history
from src.racket.ownership import OwnershipStore
from src.racket.ownership_fast import FORMAT, SIDES, SampleStore, court_coverage, export_labels, farthest_samples, initial_row, read_selected
from src.racket.ownership_fast_labeler import FastLabeler
from src.racket.schema import file_hash, save_json, read_json


class FastOwnershipTests(unittest.TestCase):
    def fixture(self, base):
        source, previous, oldroot, before, _ = ownership_history(base)
        old = OwnershipStore(oldroot)
        tasks = [
            dict(id='train01_00002', clip='train01', frame=2, split='train', source_group='match01',
                 kind='new', references={}, candidates=[[50., 50., 90., 95.], [90., 10., 120., 45.]],
                 proposals=dict(near_side=[50., 50., 90., 95.], far_side=[90., 10., 120., 45.])),
            dict(id='partial01_00003', clip='partial01', frame=3, split='train', source_group='match01',
                 kind='new', references={}, candidates=[], proposals=dict(near_side=None, far_side=None)),
            dict(id='val01_00001', clip='val01', frame=1, split='val', source_group='match02',
                 kind='correction', references=dict(near_side=[130., 45.], far_side=[65., 75.]),
                 candidates=[], proposals=dict(near_side=[50., 50., 90., 95.], far_side=[90., 10., 120., 45.]),
                 retained_verified_sides=['near_side'])
        ]
        clips = {}
        for c in ['train01', 'partial01', 'val01']:
            clips[c] = dict(meta=read_json(source/'clips'/c/'meta.json'))
        protected = {str(p): file_hash(p) for folder in (source, previous, oldroot) for p in folder.rglob('*.json')}
        root = base/'fast'
        save_json(root/'manifest.json', dict(format=FORMAT, tasks=tasks, clips=clips,
            protected_sources=protected, previous_root=str(oldroot), acceptance=dict(near_at_least=.85, far_at_least=.85)))
        return root, protected

    def test_click_pair_and_boxes_auto_advance_resume_and_no_source_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protected = self.fixture(Path(tmp))
            editor = FastLabeler(root)
            try:
                self.assertEqual(editor.step, ('near_side', 'racket'))
                editor.click_original([70., 80.]); editor.click_original([130., 35.])
                self.assertEqual(editor.step, ('near_side', 'body'))
                editor.click_original([60., 60.]); editor.click_original([100., 20.])
                self.assertEqual(editor.identity, 'partial01_00003')
                row = SampleStore(root).row('train01_00002')
                self.assertEqual(row['decision'], 'complete')
                # Revisiting a saved body mistake never requires relabelling rackets.
                editor.handle_key(ord('a')); editor.handle_key(ord('e'))
                self.assertEqual(editor.step, ('near_side', 'body'))
                self.assertEqual(editor.current['players']['near_side']['point'], [70., 80.])
                self.assertIsInstance(editor.draw(), np.ndarray)
            finally:
                editor.close()
            self.assertEqual(protected, {p: file_hash(p) for p in protected})
            self.assertEqual(FastLabeler(root).identity, 'train01_00002')

    def test_all_four_click_h_combinations_keep_independent_positive_masks(self):
        for near_visible, far_visible in [(True, True), (True, False), (False, True), (False, False)]:
            with self.subTest(near=near_visible, far=far_visible), tempfile.TemporaryDirectory() as tmp:
                root, _ = self.fixture(Path(tmp))
                editor = FastLabeler(root)
                try:
                    for visible, point in [(near_visible, [70., 80.]), (far_visible, [130., 35.])]:
                        if visible: editor.click_original(point)
                        else: editor.handle_key(ord('h'))
                    editor.handle_key(32); editor.handle_key(32)
                    row = SampleStore(root).row('train01_00002')
                    self.assertEqual(row['decision'], 'complete')
                    for side, visible in zip(SIDES, [near_visible, far_visible]):
                        p = row['players'][side]
                        self.assertEqual(p['racket_state'], 'located' if visible else 'unlocatable')
                        self.assertEqual(p['point'] is not None, visible)
                    out = Path(tmp)/'export'
                    report = export_labels(root, out)
                    records = [__import__('json').loads(line) for line in (out/'train_with_masks.jsonl').read_text().splitlines()]
                    new = next(r for r in records if r['id'] == 'train01_00002')
                    self.assertEqual([new['players'][s]['target_valid'] for s in SIDES], [near_visible, far_visible])
                    self.assertTrue(report['partial_queries_require_mask_aware_trainer'])
                    self.assertTrue((out/'historical_frozen/val.jsonl').exists())
                finally:
                    editor.close()

    def test_correction_points_locked_and_validation_stays_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protected = self.fixture(Path(tmp))
            editor = FastLabeler(root, 'val01_00001')
            try:
                self.assertEqual(editor.step, ('far_side', 'body'))
                editor.handle_key(ord('2'))
                self.assertIn('locked', editor.message)
                editor.handle_key(ord('b'))
                editor.click_original([90., 10.]); editor.click_original([125., 46.])
                self.assertEqual(SampleStore(root).row('val01_00001')['decision'], 'complete')
                out = Path(tmp)/'out'
                export_labels(root, out)
                self.assertNotIn('val01', (out/'train.jsonl').read_text())
                original = (out/'historical_frozen/val.jsonl').read_text()
                corrected = (out/'val.jsonl').read_text()
                self.assertNotEqual(original, corrected)
                self.assertEqual(protected, {p: file_hash(p) for p in protected})
                changed = editor.store.row('val01_00001')
                changed['players']['far_side']['point'] = [20., 20.]
                with self.assertRaisesRegex(ValueError, 'locked'):
                    editor.store.save('val01_00001', changed)
            finally:
                editor.close()

    def test_undo_and_body_h_preserve_located_point_but_exclude_invalid_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.fixture(Path(tmp))
            editor = FastLabeler(root)
            try:
                editor.click_original([70., 80.])
                editor.handle_key(8)
                self.assertEqual(editor.step, ('near_side', 'racket'))
                editor.click_original([70., 80.]); editor.handle_key(ord('h'))
                editor.handle_key(ord('h')); editor.handle_key(ord('h'))
                row = SampleStore(root).row('train01_00002')
                self.assertEqual(row['players']['near_side']['point'], [70., 80.])
                self.assertEqual(row['players']['near_side']['body_state'], 'unlocatable')
                self.assertEqual(editor.store.status()['positive_queries'], 0)
            finally:
                editor.close()

    def test_distinct_temporal_visual_sampling_and_exact_video_indexing(self):
        frames = list(range(200))
        features = [[np.sin(i/7), np.cos(i/5)] for i in frames]
        picked = farthest_samples(frames, features, 20, 200)
        self.assertEqual(len(set(picked)), 20)
        self.assertGreater(max(picked)-min(picked), 170)
        self.assertGreaterEqual(min(abs(a-b) for i, a in enumerate(picked) for b in picked[i+1:]), 3)
        self.assertEqual(picked, farthest_samples(frames, features, 20, 200))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (80, 60))
            for i in range(12): writer.write(np.full((60, 80, 3), i*15, np.uint8))
            writer.release()
            meta = dict(source=str(path), source_kind='video', width=80, height=60, frame_count=8, start_frame=2)
            got = list(read_selected(meta, [0, 3, 7]))
            self.assertEqual([i for i, _ in got], [0, 3, 7])
            self.assertTrue(all(abs(image.mean()-(i+2)*15) < 4 for i, image in got))

    def test_floor_filter_accepts_grey_courts_and_rejects_dark_cutaways(self):
        grey = np.full((180, 320, 3), 125, np.uint8)
        green = np.full((180, 320, 3), (40, 140, 40), np.uint8)
        cutaway = np.full((180, 320, 3), 15, np.uint8)
        cutaway[:, 130:190] = (20, 20, 170)  # a red shirt is not broad court coverage
        self.assertGreater(court_coverage(grey), .5)
        self.assertGreater(court_coverage(green), .5)
        self.assertLess(court_coverage(cutaway), .5)

    def test_duplicate_body_owner_is_rejected_before_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.fixture(Path(tmp))
            editor = FastLabeler(root)
            try:
                editor.click_original([70., 80.]); editor.click_original([130., 35.])
                editor.body_action([50., 50., 90., 95.])
                with self.assertRaisesRegex(ValueError, 'same person'):
                    editor.body_action([50., 50., 90., 95.])
                self.assertEqual(editor.identity, 'train01_00002')
                self.assertEqual(editor.step, ('far_side', 'body'))
                self.assertEqual(SampleStore(root).row(editor.identity)['decision'], 'pending')
                editor.body_action([90., 10., 120., 45.])
                self.assertEqual(editor.identity, 'partial01_00003')
            finally:
                editor.close()

    def test_hard_validation_case_cannot_disappear_from_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.fixture(Path(tmp))
            editor = FastLabeler(root, 'val01_00001')
            try:
                editor.handle_key(ord('h'))  # target remains located, but owner body cannot be supplied
                out = Path(tmp)/'export'
                report = export_labels(root, out)
                self.assertEqual(report['unresolved_validation_corrections'], ['val01_00001'])
                original = [__import__('json').loads(s)['id'] for s in (out/'historical_frozen/val.jsonl').read_text().splitlines()]
                current = [__import__('json').loads(s)['id'] for s in (out/'val.jsonl').read_text().splitlines()]
                self.assertEqual(current, original)
            finally:
                editor.close()

    def test_clear_boxes_and_direct_near_redraw_keep_locked_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protected = self.fixture(Path(tmp))
            editor = FastLabeler(root, 'val01_00001')
            try:
                original = {s: copy.deepcopy(editor.current['players'][s]['point']) for s in SIDES}
                self.assertEqual(editor.step, ('far_side', 'body'))
                editor.handle_key(ord('c'))
                self.assertEqual(editor.step, ('near_side', 'body'))
                self.assertTrue(editor.box_mode)
                self.assertFalse(editor.show_candidates)
                self.assertTrue(all(p['bbox'] is None for p in editor.current['players'].values()))
                editor.handle_key(ord('n'))
                editor.click_original([45., 50.]); editor.click_original([90., 95.])
                self.assertEqual(editor.step, ('far_side', 'body'))
                editor.handle_key(ord('f'))
                editor.click_original([90., 10.]); editor.click_original([125., 46.])
                saved = SampleStore(root).row('val01_00001')
                self.assertEqual(saved['decision'], 'complete')
                self.assertEqual({s: saved['players'][s]['point'] for s in SIDES}, original)
                self.assertEqual(protected, {p: file_hash(p) for p in protected})
            finally:
                editor.close()

    def test_near_body_can_be_adjusted_before_racket_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.fixture(Path(tmp))
            editor = FastLabeler(root)
            try:
                editor.handle_key(ord('n'))
                self.assertEqual(editor.step, ('near_side', 'body'))
                editor.click_original([45., 50.]); editor.click_original([90., 95.])
                self.assertEqual(editor.step, ('near_side', 'racket'))
                self.assertEqual(editor.current['players']['near_side']['body_state'], 'verified')
            finally:
                editor.close()


if __name__ == '__main__':
    unittest.main()
