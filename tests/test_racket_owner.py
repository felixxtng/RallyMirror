import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
import torch

from src.racket.mediapipe_bundle import BODY_NAMES
from src.racket.owner_model import SelectedPlayerFusion,owner_maps
from src.racket.owner_data import OwnerSamples,make_pairs,pair_only_dataset,augment_owner
from src.racket.mediapipe_model import spatial_loss


class OwnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self,root):
        records=[]
        for i in range(2):
            image=np.full((96,96,3),30+i*150,np.uint8);path=f'{i}.jpg';cv2.imwrite(str(root/path),image)
            pose=np.zeros((17,3),np.float32);pose[0]=[.4,.4,1.];pose[5]=[.2,.5,1.]
            records.append(dict(id=str(i),clip=str(i),source_group='match'+str(i),split='train',train_eligible=True,
                image=path,pose=pose.tolist(),crop=[100,200,300],owner_box=[175,260,325,440],target=[.2,.5],
                target_original=[160,350],original_size=[640,640],valid_input=True,target_in_crop=True,frame=i))
        return records

    def test_same_rgb_different_owner_and_target_with_inverse_transform(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);records=self.fixture(root);pairs=make_pairs(records,1)
            dataset=pair_only_dataset(root,records,pairs,96)
            a=dataset.raw(0);b=dataset.raw(1)
            np.testing.assert_array_equal(a[0],b[0])
            self.assertFalse(np.array_equal(a[2],b[2]));self.assertFalse(np.array_equal(a[3],b[3]))
            for i,item in enumerate([a,b]):
                _,pose,box,target,record,transform=item
                inverse=(target-transform['offset'])/transform['scale']
                np.testing.assert_allclose(inverse,record['target'],atol=1e-6)
                ref,reference=dataset.reference(i)
                self.assertEqual(ref['id'],record['id']);self.assertEqual(reference,transform)
                np.testing.assert_allclose(pose[5,:2],target,atol=1e-6)

    def test_validation_cannot_enter_ownership_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);records=self.fixture(root);pairs=make_pairs(records,1)
            records[0]['split']='val'
            with self.assertRaises(ValueError):make_pairs(records,1)
            with self.assertRaises(ValueError):OwnerSamples(root,records,pairs)

    def test_augmentation_keeps_marker_and_target_geometry_together(self):
        for seed in range(12):
            pose=np.zeros((17,3),np.float32);pose[0]=[.3,.5,1.]
            _,moved,box,target=augment_owner(np.zeros((96,96,3),np.uint8),pose,np.array([.1,.2,.7,.8],np.float32),np.array([.3,.5],np.float32),np.random.default_rng(seed))
            np.testing.assert_allclose(moved[0,:2],target,atol=1e-6)
            self.assertGreater(box[2],box[0]);self.assertGreater(box[3],box[1])

    def test_box_is_available_when_body_pose_is_missing(self):
        maps=owner_maps(torch.zeros(1,17,3),torch.tensor([[.2,.3,.6,.8]]),24)
        self.assertEqual(tuple(maps.shape),(1,20,24,24))
        self.assertGreater(float(maps[:,0].sum()),0.)
        self.assertEqual(float(maps[:,1:].sum()),0.)

    def test_conditioning_changes_output_and_gets_gradients(self):
        torch.manual_seed(42);model=SelectedPlayerFusion();model.train()
        image=torch.randn(1,3,96,96).repeat(2,1,1,1)
        pose=torch.zeros(2,17,3);box=torch.tensor([[.1,.1,.3,.7],[.6,.2,.9,.9]])
        output=model(image,pose,box)
        self.assertEqual(tuple(output.shape),(2,1,24,24))
        self.assertFalse(torch.allclose(output[0],output[1]))
        loss=spatial_loss(output,torch.tensor([[.25,.4],[.8,.5]]));loss.backward()
        self.assertGreater(float(model.condition_stem[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.condition_head[-1].weight.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
