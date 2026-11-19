from pathlib import Path
import tempfile
import unittest

from tests import test_racket_ownership_all as all_fixture
from src.racket.ownership import OwnershipStore
from src.racket.ownership_labeler import OwnershipLabeler
from src.racket.ownership_review import prepare_review,CorrectionReview
from src.racket.schema import file_hash


class CorrectionReviewTests(unittest.TestCase):
    def test_saved_pairs_preserved_space_advances_and_resume_skips_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);_,_,root,before,_=all_fixture.AllOwnershipTests().dataset(base)
            store=OwnershipStore(root);identity='train01_00001'
            row=store.row(identity);row['decision']='approved'
            for p in row['players'].values():p['ownership_verified']=True
            store.save(identity,row)
            path=base/'review.json'
            flags=[dict(id=k,side='near_side',visual_review='Fix body association')
                   for k in [identity,'partial01_00000']]
            prepare_review(store,dict(outside_body_derived_context=flags),path)
            saved={p:file_hash(p) for p in root.rglob('*.json')}
            editor=OwnershipLabeler(root,review=path)
            try:
                self.assertEqual(editor.queue,[identity,'partial01_00000'])
                self.assertEqual(editor.current,row);self.assertEqual(editor.active,'near')
                self.assertEqual(saved,{p:file_hash(p) for p in saved})
                editor.handle_key(ord(' '));self.assertEqual(editor.identity,'partial01_00000')
            finally:editor.close()
            editor=OwnershipLabeler(root,review=path)
            try:
                self.assertEqual(editor.identity,'partial01_00000')
                editor.handle_key(ord(' ')) # incomplete row must not be counted as reviewed
                self.assertEqual(len(editor.review.remaining(editor.queue)),1)
                editor.handle_key(ord('h'))
                self.assertEqual(editor.review.remaining(editor.queue),[])
                self.assertEqual(OwnershipStore(root).row(identity),row)
            finally:editor.close()
            with self.assertRaisesRegex(ValueError,'All correction frames reviewed'):
                OwnershipLabeler(root,review=path)
            # Old source files and imported ownership reviews stay untouched.
            self.assertEqual(before,{p:file_hash(p) for p in before})
            # Editing a reviewed record makes it unfinished again, even if still approved.
            store=OwnershipStore(root);changed=store.row(identity)
            changed['players']['near_side']['bbox'][0]+=1;store.save(identity,changed)
            review=CorrectionReview(store,path)
            self.assertEqual(review.remaining(list(review.entries)),[identity])

    def test_cannot_reopen_imported_or_validation_or_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);_,_,root,_,_=all_fixture.AllOwnershipTests().dataset(base);store=OwnershipStore(root)
            for number,ids in enumerate([['train01_00000'],['val01_00002'],['train01_00001','train01_00001']]):
                flags=[dict(id=k,side='far_side',visual_review='Review') for k in ids]
                with self.assertRaises(ValueError):
                    prepare_review(store,dict(outside_body_derived_context=flags),base/f'bad{number}.json')


if __name__=='__main__':unittest.main()
