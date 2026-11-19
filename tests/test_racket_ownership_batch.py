import copy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tests import test_racket_ownership as ownership_fixture
from src.racket.ownership import OwnershipStore
from src.racket.ownership_batch import curate,load_batch,batch_status,pending_order,usable_pair,_rank
from src.racket.ownership_labeler import OwnershipLabeler
from src.racket.schema import digest,file_hash,save_json


class CuratedOwnershipTests(unittest.TestCase):
    def fixture(self,base,reviewed_train=False):
        source,root=ownership_fixture.OwnershipTests().dataset(base)
        store=OwnershipStore(root)
        identity='train01_00000' if reviewed_train else 'val01_00000'
        row=self.approved(store,identity);store.save(identity,row)
        path=root/'batches/round2.json'
        curate(root,base/'bundle',path,primary=1,reserve=0 if reviewed_train else 1,target=1,minimum_per_group=1,baseline_bundle=base/'none')
        return source,root,path

    def approved(self,store,identity):
        row=store.row(identity);row.update(locked_owner='near_side',other_point=[130.,45.],decision='approved')
        for p in row['players'].values():p['ownership_verified']=True
        return row

    def test_fresh_training_only_deterministic_and_completed_rows_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source,root,path=self.fixture(base,True);store=OwnershipStore(root);batch=load_batch(store,path)
            self.assertEqual([r['id'] for r in batch['entries']],['train01_00001'])
            self.assertIn('train01_00000',batch['protected_existing_rows'])
            before={p:file_hash(p) for p in source.rglob('*.json')}
            editor=OwnershipLabeler(root,batch=path)
            try:
                editor.handle_key(ord('n'));editor.mouse_event(cv2.EVENT_LBUTTONDOWN,round(130*editor.scale),round(45*editor.scale),0,None)
                editor.handle_key(13);self.assertEqual(editor.current['decision'],'approved')
                self.assertTrue(batch_status(editor.store,batch)['ready_for_package'])
                editor.move(-1);self.assertEqual(editor.identity,'train01_00001')
                self.assertIsInstance(editor.draw(),np.ndarray)
                self.assertEqual(before,{p:file_hash(p) for p in before})
                self.assertEqual(digest(editor.store.row('train01_00000')),batch['protected_existing_rows']['train01_00000'])
            finally:editor.close()
            with self.assertRaises(FileExistsError):curate(root,base/'bundle',path,1,0,1,1,base/'none')

    def test_unknown_primary_activates_reserve_and_resume_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,root,path=self.fixture(Path(tmp));editor=OwnershipLabeler(root,batch=path)
            try:
                batch=editor.batch;primary=editor.identity
                editor.handle_key(ord('h'));self.assertNotEqual(editor.identity,primary)
                self.assertEqual(next(e['stage'] for e in batch['entries'] if e['id']==editor.identity),'reserve')
                self.assertFalse(batch_status(editor.store,batch)['ready_for_package'])
                editor.handle_key(ord('f'))
            finally:editor.close()
            resumed=OwnershipLabeler(root,batch=path)
            try:
                self.assertEqual(resumed.current['locked_owner'],'far_side')
                resumed.current['other_point']=[130.,45.];resumed.accept()
                progress=batch_status(resumed.store,batch)
                self.assertEqual(progress['new_usable_pairs'],1);self.assertEqual(progress['primary']['unlocatable'],1)
                self.assertTrue(progress['ready_for_package']);self.assertEqual(pending_order(resumed.store,batch),[])
            finally:resumed.close()

    def test_outside_context_does_not_count_as_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,root,path=self.fixture(Path(tmp));store=OwnershipStore(root);batch=load_batch(store,path);identity=batch['entries'][0]['id']
            row=self.approved(store,identity)
            row['players']['near_side']['bbox']=[5.,5.,15.,15.];row['players']['far_side']['bbox']=[20.,5.,30.,15.]
            store.save(identity,row);self.assertFalse(usable_pair(store.tasks[identity],row))
            status=batch_status(store,batch)
            self.assertEqual(status['primary']['approved_outside_context'],1)
            self.assertEqual(status['new_usable_pairs'],0);self.assertFalse(status['ready_for_package'])
            self.assertTrue(pending_order(store,batch))

    def test_changed_protected_rows_or_val_task_or_duplicate_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,root,path=self.fixture(Path(tmp));store=OwnershipStore(root);original=load_batch(store,path)
            batch=copy.deepcopy(original);batch['entries'][0]['id']='val01_00001'
            batch['entries'][0]['task_sha256']=digest(store.tasks['val01_00001']);save_json(path,batch)
            with self.assertRaisesRegex(ValueError,'TRAIN'):load_batch(store,path)
            batch=copy.deepcopy(original);batch['entries'].append(batch['entries'][0]);save_json(path,batch)
            with self.assertRaisesRegex(ValueError,'duplicated'):load_batch(store,path)
            save_json(path,original);row=store.row('val01_00000');row['other_point']=[135.,45.];store.save('val01_00000',row)
            with self.assertRaisesRegex(ValueError,'changed'):load_batch(store,path)

    def test_ready_stops_reserves_and_pilot_cannot_mix_with_curated_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,root,path=self.fixture(Path(tmp));store=OwnershipStore(root);batch=load_batch(store,path)
            identity=batch['entries'][0]['id'];store.save(identity,self.approved(store,identity))
            self.assertTrue(batch_status(store,batch)['ready_for_package']);self.assertEqual(pending_order(store,batch),[])
            with self.assertRaisesRegex(ValueError,'not both'):OwnershipLabeler(root,pilot=True,batch=path)
            np.testing.assert_allclose(_rank([2,2,2]),[.5,.5,.5])


if __name__=='__main__':unittest.main()
