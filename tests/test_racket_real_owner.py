import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.racket.mediapipe_bundle import BODY_NAMES,sha,to_original
from src.racket.mediapipe_model import spatial_loss
from src.racket.real_owner_data import (FORMAT,SIDES,RealPairs,augment_pair,flatten_queries,
    real_ownership_loss,scene_bounds,selected_pose,summarize_real)
from src.racket.real_owner_infer import RealOwnerRacketPredictor
from src.racket.real_owner_train import checkpoint,warm_model
from src.racket.owner_model import SelectedPlayerFusion


class RealOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def record(self):
        boxes=[[70,100,130,220],[160,40,200,120]];crop=scene_bounds(boxes)
        record=dict(id='a',clip='rally01',frame=1,split='train',source_group='match01',
            train_eligible=True,image='a.jpg',original_size=[400,400],crop=crop,players={},
            head_separation_original_pixels=100,body_box_iou=0.)
        for side,box,xy in zip(SIDES,boxes,[[100,180],[180,80]]):
            pose=np.zeros((17,3),np.float32);target=[(xy[k]-crop[k])/crop[2] for k in range(2)]
            pose[0]=[*target,1]
            record['players'][side]=dict(owner_box=box,pose=pose.tolist(),target=target,
                target_original=xy,valid_target=True,pose_status='available')
        return record

    def test_identical_rgb_two_different_selected_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=self.record();cv2.imwrite(str(root/'a.jpg'),np.full((96,96,3),80,np.uint8))
            ds=RealPairs(root,[r]);image,poses,boxes,targets,_=ds[0]
            rgb,p,b,t=flatten_queries(image[None],poses[None],boxes[None],targets[None])
            self.assertTrue(torch.equal(rgb[0],rgb[1]));self.assertFalse(torch.equal(b[0],b[1]))
            self.assertFalse(torch.equal(t[0],t[1]));self.assertFalse(torch.equal(p[0],p[1]))
            for i,s in enumerate(SIDES):np.testing.assert_allclose(to_original(t[i].tolist(),r['crop']),r['players'][s]['target_original'],atol=1e-4)
            r['split']='val'
            with self.assertRaises(ValueError):RealPairs(root,[r],training=True)

    def test_body_only_context_and_both_target_geometry(self):
        r=self.record();bounds=scene_bounds([p['owner_box'] for p in r['players'].values()])
        r['players']['near_side']['target_original']=[9999,-9999]
        self.assertEqual(bounds,scene_bounds([p['owner_box'] for p in r['players'].values()]))
        for seed in range(20):
            poses=np.zeros((2,17,3),np.float32);targets=np.array([[.3,.6],[.7,.3]],np.float32)
            poses[:,0,:2]=targets;poses[:,0,2]=1
            boxes=np.array([[.1,.2,.4,.8],[.6,.1,.9,.6]],np.float32)
            _,moved,box,xy=augment_pair(np.zeros((96,96,3),np.uint8),poses,boxes,targets,np.random.default_rng(seed))
            np.testing.assert_allclose(moved[:,0,:2],xy,atol=1e-6)
            self.assertTrue(np.all(box[:,2:]>box[:,:2]));self.assertTrue(np.all((xy>=0)&(xy<1)))

    def test_padding_covers_overhead_racket_and_is_saved_for_prediction(self):
        boxes=[[882.8114624,616.3390503,1124.4553223,1027.5328369],
               [885.0665283,179.5813599,994.5229492,483.4066162]]
        point=[902.62,21.26];old=scene_bounds(boxes);new=scene_bounds(boxes,.05)
        self.assertLess((point[1]-old[1])/old[2],0)
        self.assertTrue(all(0<=(point[k]-new[k])/new[2]<1 for k in range(2)))
        self.assertEqual(scene_bounds(boxes),scene_bounds(boxes,0.))
        with self.assertRaises(ValueError):scene_bounds(boxes,float('nan'))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'padded.pt';model=SelectedPlayerFusion()
            torch.save(checkpoint(model,dict(image_size=64,context_padding=.05,
                run_version='v005',initial_checkpoint_sha256='test')),path)
            predictor=RealOwnerRacketPredictor(path)
            self.assertEqual(predictor.context_padding,.05)
            class Stub(torch.nn.Module):
                def forward(self,images,pose,box):
                    logits=torch.full((2,1,16,16),-100.)
                    logits[:,0,8,4]=100
                    return logits
            predictor.model=Stub()
            players=[dict(bbox=b,keypoints=[],track_id=i) for i,b in enumerate(boxes)]
            original=copy.deepcopy(players)
            result=predictor.predict_players(np.zeros((1200,2200,3),np.uint8),players)
            xy=to_original([4.5/16,8.5/16],new)
            self.assertAlmostEqual(result[0]['racket']['x'],xy[0],places=3)
            self.assertAlmostEqual(result[0]['racket']['y'],xy[1],places=3)
            self.assertEqual(players,original)

    def test_wrong_racket_is_not_every_large_error_and_misses_count(self):
        rows=[]
        for i,(e,other,sep) in enumerate([(100,2,100),(100,200,100),(None,None,100),(5,100,100),(25,10,30)]):
            for side in SIDES:
                rows.append(dict(pair_id=str(i),side=side,source_group='m',clip='r',
                    error_original_pixels=e,opponent_error_original_pixels=other,
                    head_separation_original_pixels=sep,point_original=None if e is None else [i,0]))
        m=summarize_real(rows)
        self.assertEqual(m['samples'],10);self.assertEqual(m['missing'],2)
        self.assertEqual(m['within_20px'],2);self.assertEqual(m['ownership_eligible_queries'],8)
        self.assertEqual(m['wrong_racket_within20_queries'],2);self.assertEqual(m['wrong_racket_rate'],.25)
        self.assertEqual(m['both_correct20'],1);self.assertEqual(m['collapsed_selection_frames'],3)
        self.assertEqual(summarize_real([])['samples'],0)

    def test_ranking_pushes_selected_up_and_opponent_down(self):
        logits=torch.zeros(2,1,16,16,requires_grad=True)
        target=torch.tensor([[.28125,.28125],[.71875,.71875]])
        other=target.flip(0);loss,ce,rank=real_ownership_loss(logits,target,other,torch.tensor([True,True]))
        ranking_grad=torch.autograd.grad(rank,logits,retain_graph=True)[0]
        self.assertLess(ranking_grad[0,0,4,4].item(),0);self.assertGreater(ranking_grad[0,0,11,11].item(),0)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        excluded,_,zero=real_ownership_loss(logits,target,other,torch.tensor([False,False]))
        self.assertEqual(zero.item(),0);self.assertAlmostEqual(excluded.item(),spatial_loss(logits,target).item())

    def test_wrong_person_pose_rejected_without_changing_body(self):
        points=[dict(name='nose',x=180,y=60,visibility=1,presence=1)]
        original=copy.deepcopy(points)
        features,status=selected_pose(points,[70,100,130,220],[[160,40,200,120]],[0,0,400])
        self.assertEqual(status,'rejected_clear_other_body_head');self.assertFalse(features.any());self.assertEqual(points,original)

    def test_runtime_adapter_keeps_body_and_uses_common_context(self):
        class Stub(torch.nn.Module):
            def forward(self,images,pose,box):
                self.images=images;self.boxes=box
                logits=torch.full((2,1,16,16),-100.)
                logits[0,0,8,4]=100;logits[1,0,5,11]=100
                return logits
        predictor=object.__new__(RealOwnerRacketPredictor);predictor.device=torch.device('cpu');predictor.size=64;predictor.model=Stub()
        players=[dict(bbox=[70,100,130,220],keypoints=[],track_id='alice'),dict(bbox=[160,40,200,120],keypoints=[],track_id='bob')]
        original=copy.deepcopy(players);result=predictor.predict_players(np.zeros((400,400,3),np.uint8),players)
        self.assertEqual(players,original);self.assertEqual(result[0]['bbox'],players[0]['bbox'])
        self.assertEqual(result[1]['track_id'],'bob');self.assertEqual(result[0]['keypoints'],[])
        self.assertTrue(torch.equal(predictor.model.images[0],predictor.model.images[1]))
        self.assertFalse(torch.equal(predictor.model.boxes[0],predictor.model.boxes[1]))
        crop=scene_bounds([p['bbox'] for p in players]);xy=to_original([4.5/16,8.5/16],crop)
        self.assertAlmostEqual(result[0]['racket']['x'],xy[0],places=4)
        self.assertAlmostEqual(result[0]['racket']['y'],xy[1],places=4)

    def test_checkpoint_format_and_warm_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'models').mkdir();path=root/'models/baseline_v002.pt'
            model=SelectedPlayerFusion();torch.save(dict(format='racket_owner_v002',body_names=BODY_NAMES,state_dict=model.state_dict()),path)
            warm=warm_model(path,torch.device('cpu'))
            self.assertTrue(torch.equal(warm.owner_gain,model.owner_gain))
            with self.assertRaises(ValueError):RealOwnerRacketPredictor(path)
            torch.save(checkpoint(model,dict(image_size=640,initial_checkpoint_sha256='test')),root/'new.pt')
            loaded=RealOwnerRacketPredictor(root/'new.pt');self.assertEqual(loaded.size,640)



if __name__=='__main__':unittest.main()
