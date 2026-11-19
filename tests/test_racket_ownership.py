"""Protection of old points, explicit real ownership, resumability and splits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.racket.ownership import OwnershipStore, prepare, export, status, import_v003
from src.racket.ownership_labeler import OwnershipLabeler
from src.racket.schema import blank, file_hash, save_json


class OwnershipTests(unittest.TestCase):
    def dataset(self, base):
        source=base/'manual';cache=[]
        for name,split,group in [('train01','train','match01'),('val01','val','match02'),('test01','test','match03')]:
            images=base/(name+'_images');images.mkdir()
            for f in range(3):cv2.imwrite(str(images/f'{f:03}.png'),np.full((100,160,3),70+f,np.uint8))
            meta=dict(schema_version=1,keypoint_names=['head_center'],annotation_mode='selected_racket',
                source_kind='images',source=str(images),images=[f'{f:03}.png' for f in range(3)],source_hash='fixture',
                source_group=group,split=split,width=160,height=100,frame_count=3,fps=30,start_frame=0)
            frames={}
            for f in range(3):
                row=blank(f,['head_center']);row.update(state='visible',approved=True)
                row['keypoints'][0].update(x=65.,y=75.,v=2);frames[str(f)]=row
                cache.append(dict(id=f'{name}_{f:05d}',frame=f,target_original=[65.,75.],
                    owner_box=[50.,50.,90.,95.],candidates=[dict(bbox=[90.,10.,120.,45.],confidence=.9)]))
            save_json(source/'clips'/name/'meta.json',meta)
            save_json(source/'clips'/name/'annotations.json',dict(frames=frames))
        # A completed unknown near point cannot form a two-positive ownership pair.
        path=source/'clips/train01/annotations.json';ann=json.loads(path.read_text())
        ann['frames']['2']['state']='unavailable';ann['frames']['2']['keypoints'][0].update(x=None,y=None,v=0)
        save_json(path,ann)
        bundle=base/'bundle';save_json(bundle/'manifest.json',dict(samples=cache))
        root=base/'ownership';prepare(source,root,bundle,per_clip=1)
        return source,root

    def test_prepare_freezes_near_excludes_test_and_unknown_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source,root=self.dataset(base)
            store=OwnershipStore(root)
            self.assertEqual(len(store.queue()),5)
            self.assertEqual(len(store.queue(pilot=True)),2)
            self.assertFalse(any('test01' in k for k in store.tasks))
            self.assertNotIn('train01_00002',store.tasks)
            with self.assertRaises(FileExistsError):prepare(source,root,base/'bundle')
            with self.assertRaises(ValueError):prepare(source,source/'another')
            report=status(root);self.assertEqual(report['all']['pending'],5)

    def test_ui_cannot_edit_near_points_and_requires_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,root=self.dataset(Path(tmp));hashes={p:file_hash(p) for p in source.rglob('*.json')}
            editor=OwnershipLabeler(root)
            try:
                before=copy.deepcopy(editor.task['locked_reference'])
                # Selecting NEAR changes only which BODY BOX Tab/B controls.
                editor.handle_key(ord('n'))
                editor.handle_key(ord('1'))
                editor.mouse_event(cv2.EVENT_LBUTTONDOWN,round(65*editor.scale),round(75*editor.scale),0,None)
                self.assertEqual(editor.current['decision'],'pending')
                self.assertFalse(editor.current['players']['far_side']['ownership_verified'])
                self.assertEqual(editor.task['locked_reference'],before)
                # Label the far centre OUTSIDE its body box: legitimate racket reach.
                editor.mouse_event(cv2.EVENT_LBUTTONDOWN,round(130*editor.scale),round(45*editor.scale),0,None)
                editor.handle_key(13)
                self.assertEqual(editor.store.row('train01_00000')['decision'],'approved')
                self.assertEqual(editor.identity,'train01_00001')
                editor.handle_key(ord('h'))
                editor.move(-1);editor.handle_key(ord('u'))
                self.assertEqual(hashes,{p:file_hash(p) for p in hashes})
                self.assertIsInstance(editor.draw(),np.ndarray)
            finally:editor.close()

    def test_invalid_or_same_owner_boxes_are_not_accepted_and_manual_boxes_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,root=self.dataset(Path(tmp));editor=OwnershipLabeler(root)
            try:
                editor.handle_key(ord('n'))
                editor.current['other_point']=[130.,45.]
                editor.current['players']['far_side']['bbox']=copy.deepcopy(editor.current['players']['near_side']['bbox'])
                editor.handle_key(13)
                self.assertEqual(editor.current['decision'],'pending')
                self.assertIn('same region',editor.message)
                editor.handle_key(ord('2'));editor.handle_key(ord('b'))
                for x,y in [(90,10),(120,45)]:
                    editor.mouse_event(cv2.EVENT_LBUTTONDOWN,round(x*editor.scale),round(y*editor.scale),0,None)
                editor.handle_key(13)
                self.assertEqual(editor.store.row('train01_00000')['decision'],'approved')
            finally:editor.close()

    def test_source_change_or_tampered_near_snapshot_blocks_save_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,root=self.dataset(Path(tmp));store=OwnershipStore(root)
            identity=store.queue()[0];row=store.row(identity)
            row['near_point']=[1,2]
            with self.assertRaises(ValueError):store.save(identity,row)
            path=source/'clips/train01/annotations.json';path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'changed externally'):store.save(identity,store.row(identity))
            with self.assertRaises(ValueError):export(root,Path(tmp)/'export')
        with tempfile.TemporaryDirectory() as tmp:
            _,root=self.dataset(Path(tmp));path=root/'manifest.json';m=json.loads(path.read_text())
            m['tasks'][0]['locked_reference']['point']['x']=1;save_json(path,m)
            with self.assertRaisesRegex(ValueError,'reference'):OwnershipStore(root)

    def test_resume_export_identity_split_unknown_and_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source,root=self.dataset(base);hashes={p:file_hash(p) for p in source.rglob('*.json')}
            editor=OwnershipLabeler(root)
            try:
                editor.handle_key(ord('n'))
                editor.mouse_event(cv2.EVENT_LBUTTONDOWN,round(130*editor.scale),round(45*editor.scale),0,None)
            finally:editor.close()
            resumed=OwnershipLabeler(root)
            try:
                self.assertIsNotNone(resumed.current['other_point'])
                resumed.handle_key(13);resumed.handle_key(ord('x'))
                self.assertEqual(resumed.identity,'val01_00000')
                resumed.handle_key(ord('n'))
                resumed.mouse_event(cv2.EVENT_LBUTTONDOWN,round(130*resumed.scale),round(45*resumed.scale),0,None)
                resumed.handle_key(13);resumed.handle_key(ord('h'))
            finally:resumed.close()
            output=base/'export';report=export(root,output)
            self.assertEqual(report['counts'],{'train':1,'val':1})
            self.assertEqual(report['excluded'],{'quarantined':1,'unlocatable':1,'pending':1})
            exclusions=[json.loads(s) for s in (output/'exclusions.jsonl').read_text().splitlines()]
            self.assertEqual([r['id'] for r in exclusions if r['exclude_original_source_row_from_future_training']],['train01_00001'])
            for split in ['train','val']:
                rows=[json.loads(s) for s in (output/(split+'.jsonl')).read_text().splitlines()]
                self.assertEqual(rows[0]['split'],split)
                self.assertEqual(set(rows[0]['players']),{'near_side','far_side'})
                self.assertEqual(rows[0]['players']['near_side']['head_center']['x'],65.)
                self.assertTrue(rows[0]['players']['far_side']['ownership_verified'])
            self.assertEqual(hashes,{p:file_hash(p) for p in hashes})
            with self.assertRaises(FileExistsError):export(root,output)

    def test_far_owned_locked_point_exports_as_far_without_changing_coordinate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source,root=self.dataset(base);hashes={p:file_hash(p) for p in source.rglob('*.json')}
            editor=OwnershipLabeler(root)
            try:
                ref=copy.deepcopy(editor.task['locked_reference'])
                # No implicit near role; a click cannot be saved before N/F.
                editor.mouse_event(cv2.EVENT_LBUTTONDOWN,round(130*editor.scale),round(45*editor.scale),0,None)
                self.assertIsNone(editor.current['other_point'])
                self.assertIsNone(editor.current['locked_owner'])
                editor.handle_key(13)
                self.assertEqual(editor.current['decision'],'pending')
                editor.handle_key(ord('f'))
                self.assertEqual(editor.other_owner,'near_side')
                editor.mouse_event(cv2.EVENT_LBUTTONDOWN,round(130*editor.scale),round(45*editor.scale),0,None)
                editor.handle_key(13)
                self.assertEqual(editor.store.row('train01_00000')['locked_owner'],'far_side')
                self.assertEqual(ref,editor.store.tasks['train01_00000']['locked_reference'])
            finally:editor.close()
            export(root,base/'out')
            record=json.loads((base/'out/train.jsonl').read_text().splitlines()[0])
            self.assertEqual(record['players']['far_side']['head_center'],ref['point'])
            self.assertEqual(record['players']['near_side']['head_center']['x'],130.)
            self.assertEqual(hashes,{p:file_hash(p) for p in hashes})

    def test_owner_switch_invalidates_other_point_and_duplicate_racket_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,root=self.dataset(Path(tmp));editor=OwnershipLabeler(root)
            try:
                editor.handle_key(ord('n'))
                editor.mouse_event(cv2.EVENT_LBUTTONDOWN,round(130*editor.scale),round(45*editor.scale),0,None)
                editor.handle_key(ord('f'))
                self.assertIsNone(editor.current['other_point'])
                ref=editor.task['locked_reference']['point']
                editor.current['other_point']=[ref['x'],ref['y']]
                editor.handle_key(13)
                self.assertIn('coincide',editor.message)
                self.assertEqual(editor.current['decision'],'pending')
            finally:editor.close()

    def test_import_v003_preserves_draft_verbatim_and_reopens_owner_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source,root=self.dataset(base);legacy=base/'legacy'
            save_json(legacy/'manifest.json',dict(format='racket_real_ownership_v003',source_root=str(source.resolve())))
            old=dict(frame=0,far=dict(point=[65.,75.],bbox=[90.,10.,120.,45.],track_id=None),
                     near_association=dict(bbox=[50.,50.,90.,95.],track_id=None),decision='approved')
            path=legacy/'clips/train01/far_side.json';save_json(path,dict(clip='train01',frames={'0':old}))
            original_hash=file_hash(path);report=import_v003(legacy,root)
            self.assertEqual(report['imported_pending'],1)
            self.assertEqual(file_hash(path),original_hash)
            self.assertEqual(json.loads((root/'legacy_v003_import.json').read_text())['originals'][0]['data']['frames']['0'],old)
            row=OwnershipStore(root).row('train01_00000')
            self.assertEqual(row['decision'],'pending');self.assertIsNone(row['locked_owner']);self.assertIsNone(row['other_point'])

    def test_clear_selected_body_box_preserves_both_points_and_other_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,root=self.dataset(Path(tmp));hashes={p:file_hash(p) for p in source.rglob('*.json')}
            editor=OwnershipLabeler(root)
            try:
                editor.handle_key(ord('F'))
                editor.mouse_event(cv2.EVENT_LBUTTONDOWN,round(130*editor.scale),round(45*editor.scale),0,None)
                reference=copy.deepcopy(editor.task['locked_reference']);other=copy.deepcopy(editor.current['other_point'])
                far_box=copy.deepcopy(editor.current['players']['far_side']['bbox']);identity=editor.identity
                editor.handle_key(ord('1'));editor.handle_key(ord('C'))
                self.assertIsNone(editor.current['players']['near_side']['bbox'])
                self.assertEqual(editor.current['players']['far_side']['bbox'],far_box)
                self.assertEqual(editor.current['other_point'],other)
                self.assertEqual(editor.task['locked_reference'],reference)
                self.assertEqual(editor.current['locked_owner'],'far_side')
                editor.handle_key(13)
                self.assertEqual(editor.identity,identity)
                self.assertEqual(editor.current['decision'],'pending')
                self.assertIn('BOTH boxes',editor.message)
                resumed=OwnershipStore(root).row(identity)
                self.assertIsNone(resumed['players']['near_side']['bbox'])
                self.assertEqual(hashes,{p:file_hash(p) for p in hashes})
            finally:editor.close()


if __name__=='__main__':unittest.main()
