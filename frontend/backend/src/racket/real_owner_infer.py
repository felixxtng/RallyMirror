"""Real-ownership adapter: shared scene context, separately selected body inputs.

This adapter adds a racket result without replacing the body bbox/keypoints.
Callers supply both tracked players; near/far names are not model classes.
"""
import math
from pathlib import Path

import torch

from .mediapipe_bundle import BODY_NAMES,square_image,to_original
from .mediapipe_model import decode,image_tensor
from .owner_model import SelectedPlayerFusion
from .real_owner_data import FORMAT,scene_bounds,selected_pose,unit_box


class RealOwnerRacketPredictor:
    def __init__(self,weights,device='cpu'):
        self.device=torch.device(device)
        saved=torch.load(Path(weights),map_location='cpu',weights_only=True)
        if saved.get('format')!=FORMAT or saved.get('body_names')!=BODY_NAMES:
            raise ValueError('Use a real-ownership checkpoint with this adapter')
        self.size=saved['image_size'];self.context_padding=saved.get('context_padding',0.)
        self.model=SelectedPlayerFusion().to(self.device)
        self.model.load_state_dict(saved['state_dict']);self.model.eval()

    @torch.inference_mode()
    def predict_players(self,frame,players):
        """Players contain bbox and original-pixel keypoints; return new dicts.

        All players see the same body-derived scene. Track IDs, boxes and raw
        body points pass through unchanged. No racket labels enter inference.
        Single-player frames are supported but have different context geometry
        from this trial's two-player validation cohort.
        """
        if not players:return []
        crop=scene_bounds([p['bbox'] for p in players],getattr(self,'context_padding',0.));image=square_image(frame,crop,self.size)
        poses=[];statuses=[]
        for i,p in enumerate(players):
            features,status=selected_pose(p.get('keypoints') or [],p['bbox'],
                [other['bbox'] for j,other in enumerate(players) if i!=j],crop)
            poses.append(features.tolist());statuses.append(status)
        pose=torch.tensor(poses,dtype=torch.float32,device=self.device)
        boxes=torch.tensor([unit_box(p['bbox'],crop).tolist() for p in players],dtype=torch.float32,device=self.device)
        images=image_tensor(image)[None].repeat(len(players),1,1,1).to(self.device)
        point,mass=decode(self.model(images,pose,boxes));h,w=frame.shape[:2];results=[]
        for player,xy,concentration,status in zip(players,point.cpu().tolist(),mass.cpu().tolist(),statuses):
            original=to_original(xy,crop)
            valid=all(math.isfinite(v) for v in original) and 0<=original[0]<w and 0<=original[1]<h
            racket=dict(name='head_center',x=original[0] if valid else None,y=original[1] if valid else None,
                located=valid,coordinate_system='original_pixels',reason='ok' if valid else 'outside_image',
                concentration=concentration,score_type='uncalibrated_heatmap_peak_mass',
                pose_feature_status=status,model_format=FORMAT)
            result=dict(player);result['racket']=racket;results.append(result)
        return results


class CombinedRealOwnedPlayerPose:
    def __init__(self,weights,device='cpu',body_model='models/pose_landmarker_lite.task'):
        from src.pose.estimator import PlayerPose
        self.body=PlayerPose(body_model);self.racket=RealOwnerRacketPredictor(weights,device)

    def extract_players(self,frame,players):
        prepared=[]
        for player in players:
            item=dict(player)
            if 'keypoints' not in item:item['keypoints']=self.body.extract(frame,player)
            prepared.append(item)
        return self.racket.predict_players(frame,prepared)

    def close(self):self.body.close()
