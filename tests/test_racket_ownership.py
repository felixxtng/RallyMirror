"""Protection of old points, explicit real ownership, resumability and splits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.racket.ownership import OwnershipStore, prepare, export, status, import_v003
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



if __name__=='__main__':unittest.main()
