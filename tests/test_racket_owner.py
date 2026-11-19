import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
import torch

from src.racket.mediapipe_bundle import BODY_NAMES
from src.racket.owner_model import SelectedPlayerFusion,owner_maps
from src.racket.mediapipe_model import spatial_loss


class OwnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)





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
