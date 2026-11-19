"""Meaningful safeguards for geometry, masked pose input and honest metrics."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch

from src.racket.mediapipe_bundle import BODY_NAMES,pose_array,square_bounds,square_image,to_original,to_unit
from src.racket.mediapipe_model import decode,gaussian_maps,RacketPoseFusion,spatial_loss
from src.racket.legacy_validation import summarize


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_edge_crop_keeps_original_coordinates_and_padding(self):
        box=[-10,5,20,85];crop=square_bounds(box);point=[3.,76.]
        np.testing.assert_allclose(to_original(to_unit(point,crop),crop),point)
        image=np.zeros((100,100,3),np.uint8)
        result=square_image(image,crop,128)
        self.assertEqual(result.shape,(128,128,3))
        self.assertTrue(np.any(np.all(result==114,axis=2)))


    def test_missing_pose_maps_are_zero_and_outside_landmarks_masked(self):
        pose=pose_array([dict(name='left_wrist',x=500,y=500,visibility=1.,presence=1.)],[0,0,100])
        self.assertTrue(all(row[2]==0 for row in pose))
        maps=gaussian_maps(torch.tensor([pose],dtype=torch.float32),32)
        self.assertEqual(float(maps.sum()),0.)


    def test_misses_stay_in_metric_denominator(self):
        metrics=summarize([dict(error_original_pixels=10.),dict(error_original_pixels=80.),dict(error_original_pixels=None)])
        self.assertEqual(metrics['samples'],3)
        self.assertEqual(metrics['missing'],1)
        self.assertAlmostEqual(metrics['pck20_including_misses'],1/3)
        self.assertEqual(metrics['mean_error_detected_only'],45.)

    def test_spatial_loss_rewards_correct_peak_and_decodes_coordinates(self):
        logits=torch.full((1,1,32,32),-10.)
        logits[0,0,8,23]=10.
        truth=torch.tensor([[(23.5)/32,(8.5)/32]])
        wrong=torch.tensor([[.1,.9]])
        self.assertLess(float(spatial_loss(logits,truth)),float(spatial_loss(logits,wrong)))
        predicted,_=decode(logits)
        torch.testing.assert_close(predicted,truth,atol=.0001,rtol=0)

    def test_fusion_backprop_and_frozen_batchnorm_statistics(self):
        model=RacketPoseFusion();model.train()
        before=model.encoder.bn1.running_mean.clone()
        output=model(torch.randn(2,3,96,96),torch.zeros(2,17,3))
        loss=spatial_loss(output,torch.tensor([[.2,.4],[.8,.6]]));loss.backward()
        self.assertEqual(tuple(output.shape),(2,1,24,24))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()),0.)
        torch.testing.assert_close(before,model.encoder.bn1.running_mean)




if __name__=='__main__':unittest.main()
