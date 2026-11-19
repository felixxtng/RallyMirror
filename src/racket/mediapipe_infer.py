"""Optional fusion adapter: preserves body keypoints and original person box.

No shared pose/temporal/detection module is modified or automatically wired to
this experimental predictor. Call with a teammate's selected/tracked player.
"""
from pathlib import Path
import math

import torch

from .mediapipe_bundle import VERSION,BODY_NAMES,pose_array,square_bounds,square_image,to_original
from .mediapipe_model import RacketPoseFusion,decode,image_tensor


class FusionRacketPredictor:
    def __init__(self,weights,device='cpu',constraint='none'):
        self.device=torch.device(device)
        if constraint not in ('none','body_box','expanded_box','wrist_region'):
            raise ValueError('Unknown player constraint')
        self.constraint=constraint
        saved=torch.load(Path(weights),map_location='cpu',weights_only=True)
        if saved.get('format')!=VERSION or saved.get('body_names')!=BODY_NAMES:
            raise ValueError('Use this trial\'s racket_center_fusion.pt, not YOLO or MediaPipe .task weights')
        self.size=saved['image_size']
        self.model=RacketPoseFusion().to(self.device)
        self.model.load_state_dict(saved['state_dict']);self.model.eval()

    @torch.inference_mode()
    def predict(self,frame,body_box,body_keypoints):
        crop=square_bounds(body_box)
        image=square_image(frame,crop,self.size)
        pose=torch.tensor(pose_array(body_keypoints,crop),dtype=torch.float32,device=self.device)[None]
        logits=self.model(image_tensor(image)[None].to(self.device),pose)
        if self.constraint=='none':
            normalized,score=decode(logits)
        else:
            from .player_constraint import decode_player
            normalized,score,valid=decode_player(logits,[body_box],[crop],pose,self.constraint)
            if not bool(valid[0]):
                return dict(name='head_center',x=None,y=None,located=False,reason='empty_player_region')
        point=to_original(normalized[0].cpu().tolist(),crop)
        h,w=frame.shape[:2]
        if not all(math.isfinite(x) for x in point) or not 0<=point[0]<w or not 0<=point[1]<h:
            return dict(name='head_center',x=None,y=None,located=False,reason='prediction_outside_image')
        return dict(name='head_center',x=point[0],y=point[1],located=True,
                    concentration=float(score[0]),score_type='uncalibrated_heatmap_peak_mass',
                    coordinate_system='original_pixels',constraint=self.constraint)


class CombinedPlayerPose:
    """Frozen body pose plus trained extra point; works with either supplied box.

    The caller owns tracking identity and near/far selection. Far-side performance
    is unvalidated. An unlabelled far racket is never treated as an absence.
    """
    def __init__(self,weights,device='cpu',body_model='models/pose_landmarker_lite.task',constraint='none'):
        from src.pose.estimator import PlayerPose
        self.body=PlayerPose(body_model)
        self.racket=FusionRacketPredictor(weights,device,constraint)

    def extract(self,frame,target):
        if target is None:return None
        points=self.body.extract(frame,target)
        result=dict(target)
        result.update(keypoints=points,racket=self.racket.predict(frame,target['bbox'],points))
        return result

    def close(self):self.body.close()
