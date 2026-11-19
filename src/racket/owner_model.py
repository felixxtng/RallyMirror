"""v002: explicitly selected-player image/MediaPipe conditioning.

The original v001 image backbone and head are reusable. Selected-body masks
and body/wrist maps enter before the backbone's residual stages and at output.
No numeric athlete ID is used as a visual class. MediaPipe remains frozen.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .mediapipe_bundle import BODY_NAMES
from .mediapipe_model import RacketPoseFusion,gaussian_maps

FORMAT='racket_owner_v002'


def owner_maps(pose,box,size):
    axis=(torch.arange(size,device=pose.device,dtype=pose.dtype)+.5)/size
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    body=((xx[None]>=box[:,0,None,None])&(xx[None]<=box[:,2,None,None])&
          (yy[None]>=box[:,1,None,None])&(yy[None]<=box[:,3,None,None])).to(pose.dtype)[:,None]
    joints=gaussian_maps(pose,size,2.)
    wrists=pose[:,[BODY_NAMES.index('left_wrist'),BODY_NAMES.index('right_wrist')]]
    broad=gaussian_maps(wrists,size,max(2.,size*.05))
    return torch.cat([body,joints,broad],dim=1)


class SelectedPlayerFusion(RacketPoseFusion):
    def __init__(self,warm_start=None):
        super().__init__()
        if warm_start:
            saved=torch.load(warm_start,map_location='cpu',weights_only=True)
            if saved.get('format')!='racket_mediapipe_v001':
                raise ValueError('Warm-start requires the v001 selected checkpoint')
            self.load_state_dict(saved['state_dict'],strict=True)
        channels=1+len(BODY_NAMES)+2
        self.condition_stem=nn.Sequential(nn.Conv2d(channels,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU(),
                                           nn.Conv2d(32,64,3,padding=1),nn.GroupNorm(8,64),nn.ReLU())
        self.condition_head=nn.Sequential(nn.Conv2d(channels,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU(),nn.Conv2d(32,1,1))
        self.owner_gain=nn.Parameter(torch.tensor(.1))

    def forward(self,image,pose,box):
        e=self.encoder
        x=e.maxpool(e.relu(e.bn1(e.conv1(image))))
        conditions=owner_maps(pose,box,x.shape[-1])
        x=x+self.owner_gain*self.condition_stem(conditions)
        a=e.layer1(x);b=e.layer2(a);c=e.layer3(b);d=e.layer4(c)
        features=[layer(value) for layer,value in zip(self.lateral,[a,b,c,d])]
        x=features[-1]
        for skip in reversed(features[:-1]):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)+skip
        x=self.smooth(x);size=x.shape[-1]
        axis=(torch.arange(size,device=x.device,dtype=x.dtype)+.5)/size
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        grid=torch.stack([xx,yy])[None].expand(image.shape[0],-1,-1,-1)
        logits=self.head(torch.cat([x,conditions[:,1:18].to(x.dtype),grid],dim=1))
        return logits+self.condition_head(conditions)
