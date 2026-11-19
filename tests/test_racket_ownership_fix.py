import copy
from pathlib import Path
import tempfile
import unittest

from tests import test_racket_ownership_fast as fixture
from src.racket.ownership_fast import SampleStore
from src.racket.ownership_fix_labeler import CorrectionLabeler
from src.racket.schema import file_hash, read_json, save_json


class CorrectionQueueTests(unittest.TestCase):
    def test_familiar_controls_only_fix_queue_space_advances_and_resume_preserves_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protected = fixture.FastOwnershipTests().fixture(Path(tmp))
            m = read_json(root/'manifest.json')
            second = copy.deepcopy(m['tasks'][-1])
            second.update(id='val01_00000', frame=0)
            m['tasks'].append(second)
            save_json(root/'manifest.json', m)
            editor = CorrectionLabeler(root)
            try:
                self.assertEqual(editor.queue, ['val01_00001', 'val01_00000'])
                points = {s: p['point'] for s, p in editor.current['players'].items()}
                editor.handle_key(ord('1'))
                self.assertEqual(editor.step, ('near_side', 'body'))
                editor.handle_key(ord('b'))
                editor.click_original([45., 50.]); editor.click_original([90., 95.])
                self.assertEqual(editor.identity, 'val01_00001')
                self.assertEqual(editor.current['decision'], 'pending')
                self.assertEqual({s: p['point'] for s, p in editor.current['players'].items()}, points)
                editor.handle_key(32)
                self.assertEqual(editor.identity, 'val01_00000')
                self.assertEqual(SampleStore(root).row('val01_00001')['decision'], 'complete')
                self.assertEqual(editor.draw().shape[0], editor.view_height+240)
            finally:
                editor.close()
            resumed = CorrectionLabeler(root)
            try:
                self.assertEqual(resumed.identity, 'val01_00000')
                resumed.handle_key(ord('h'))
                self.assertTrue(resumed.completed)
                self.assertFalse(resumed.remaining())
                self.assertTrue(all(k.startswith('val01') for k in resumed.queue))
                resumed.handle_key(ord('a'))
                self.assertEqual(resumed.identity, 'val01_00001')
            finally:
                resumed.close()
            self.assertEqual(protected, {p: file_hash(p) for p in protected})


if __name__ == '__main__':
    unittest.main()
