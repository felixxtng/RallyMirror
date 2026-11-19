import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tests import test_racket_ownership as ownership_fixture
from src.racket.ownership import OwnershipStore,export
from src.racket.ownership_all import consolidate,all_status
from src.racket.ownership_labeler import OwnershipLabeler
from src.racket.schema import blank,save_json,file_hash


class AllOwnershipTests(unittest.TestCase):
    def dataset(self,base,exclusion=False):
        source,previous=ownership_fixture.OwnershipTests().dataset(base)
        # This clip has one approved location, one approved unknown and one
        # unapproved location; other frames have never been touched.
        images=base/'partial_images';images.mkdir()
        for f in range(6):cv2.imwrite(str(images/f'{f:03}.png'),np.full((100,160,3),100,np.uint8))
        meta=dict(schema_version=1,keypoint_names=['head_center'],annotation_mode='selected_racket',
            source_kind='images',source=str(images),images=[f'{f:03}.png' for f in range(6)],source_hash='partial-fixture',
            source_group='match01',split='train',width=160,height=100,frame_count=6,fps=30,start_frame=0)
        rows={}
        for f in [0,1,2]:
            row=blank(f,['head_center']);row.update(state='visible',approved=f!=2)
            row['keypoints'][0].update(x=65.,y=75.,v=2);rows[str(f)]=row
        rows['1']['state']='unavailable';rows['1']['keypoints'][0].update(x=None,y=None,v=0)
        save_json(source/'clips/partial01/meta.json',meta);save_json(source/'clips/partial01/annotations.json',dict(frames=rows))
        old=OwnershipStore(previous)
        row=old.row('train01_00000');row.update(locked_owner='near_side',other_point=[130.,45.],decision='approved')
        for p in row['players'].values():p['ownership_verified']=True
        old.save('train01_00000',row)
        draft=old.row('train01_00001');draft.update(locked_owner='far_side',other_point=[130.,45.]);old.save('train01_00001',draft)
        unknown=old.row('val01_00000');unknown.update(locked_owner='near_side',decision='unlocatable');old.save('val01_00000',unknown)
        far=copy.deepcopy(row);far.update(frame=1,locked_owner='far_side',locked_reference_sha256=old.tasks['val01_00001']['locked_reference']['row_sha256'])
        old.save('val01_00001',far)
        review=base/'known_exclusions.json'
        save_json(review,[dict(id='train01_00000',reason='Fixture: known bad body association')] if exclusion else [])
        output=base/'all';before={p:file_hash(p) for root in [source,previous] for p in root.rglob('*.json')}
        report=consolidate(source,previous,output,base/'bundle',review)
        return source,previous,output,before,report

    def test_partial_approved_points_import_drafts_preserve_reviews_and_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,previous,root,before,report=self.dataset(Path(tmp));old=OwnershipStore(previous);new=OwnershipStore(root)
            self.assertIn('partial01_00000',new.tasks);self.assertNotIn('partial01_00001',new.tasks);self.assertNotIn('partial01_00002',new.tasks)
            self.assertFalse(any(t['split']=='test' for t in new.tasks.values()))
            self.assertEqual(report['imported_completed_reviews_locked'],3);self.assertEqual(report['imported_pending_drafts'],1)
            self.assertEqual(report['pending_by_split'],dict(train=2,val=1));self.assertEqual(report['remaining'],3)
            self.assertEqual(new.manifest['fixed_original_paired_validation_ids'],['val01_00001'])
            for identity in ['train01_00000','train01_00001','val01_00000','val01_00001']:
                self.assertEqual(new.row(identity),old.row(identity))
            self.assertNotIn('train01_00000',new.queue());self.assertIn('train01_00001',new.queue())
            order=[new.tasks[k]['split'] for k in new.queue()];self.assertEqual(order,sorted(order,key=lambda s:s!='train'))
            self.assertEqual(before,{p:file_hash(p) for p in before})
            with self.assertRaises(FileExistsError):consolidate(source,previous,root,Path(tmp)/'bundle')

    def test_imported_completed_pair_cannot_be_edited_even_via_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,root,_,_=self.dataset(Path(tmp));store=OwnershipStore(root)
            original=store.row('train01_00000');store.save('train01_00000',original)
            changed=copy.deepcopy(original);changed['other_point']=[135.,45.]
            with self.assertRaisesRegex(ValueError,'locked'):store.save('train01_00000',changed)
            self.assertEqual(store.row('train01_00000'),original)
            with self.assertRaises(ValueError):OwnershipLabeler(root,clip='train01',frame=0)
            # External modifications to imported output are also detected.
            path=root/'clips/train01/ownership.json';ann=json.loads(path.read_text());ann['frames']['0']=changed;save_json(path,ann)
            with self.assertRaisesRegex(ValueError,'changed'):OwnershipStore(root)

    def test_single_queue_manual_boxes_advances_to_preserved_draft_then_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,root,before,_=self.dataset(Path(tmp));editor=OwnershipLabeler(root)
            try:
                self.assertEqual(editor.identity,'partial01_00000');self.assertEqual(len(editor.queue),3)
                self.assertIsNone(editor.current['locked_owner'])
                editor.handle_key(ord('n'));editor.current['other_point']=[130.,45.]
                editor.current['players']['near_side']['bbox']=[50.,50.,90.,95.]
                editor.current['players']['far_side']['bbox']=[90.,10.,120.,45.]
                editor.accept();self.assertEqual(editor.identity,'train01_00001')
                self.assertEqual(editor.current['locked_owner'],'far_side');self.assertEqual(editor.current['other_point'],[130.,45.])
                editor.accept();self.assertEqual(editor.identity,'val01_00002')
                editor.handle_key(ord('h'));self.assertTrue(all_status(root)['all_existing_reviews_complete'])
                self.assertIsInstance(editor.draw(),np.ndarray)
            finally:editor.close()
            self.assertEqual(before,{p:file_hash(p) for p in before})

    def test_export_carries_prior_exclusion_and_never_moves_val_into_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,root,_,_=self.dataset(Path(tmp),True);store=OwnershipStore(root)
            row=store.row('partial01_00000');row.update(locked_owner='near_side',other_point=[130.,45.],decision='approved')
            row['players']['near_side'].update(bbox=[50.,50.,90.,95.],ownership_verified=True)
            row['players']['far_side'].update(bbox=[90.,10.,120.,45.],ownership_verified=True)
            store.save('partial01_00000',row);out=Path(tmp)/'export';report=export(root,out)
            self.assertEqual(report['counts'],dict(train=1,val=1));self.assertEqual(report['excluded']['prior_training_quality_review'],1)
            train=[json.loads(s) for s in (out/'train.jsonl').read_text().splitlines()]
            val=[json.loads(s) for s in (out/'val.jsonl').read_text().splitlines()]
            self.assertEqual(train[0]['id'],'partial01_00000');self.assertEqual(val[0]['id'],'val01_00001')
            self.assertEqual(val[0]['locked_owner'],'far_side')
            self.assertEqual(val[0]['players']['far_side']['head_center'],store.tasks['val01_00001']['locked_reference']['point'])


if __name__=='__main__':unittest.main()
