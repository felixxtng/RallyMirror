"""Trainable image + frozen-body-landmark fusion, separate from MediaPipe weights."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from .mediapipe_bundle import BODY_NAMES


def gaussian_maps(points, size, sigma=2.):
    """points: B,K,3 normalized x,y,confidence. Zero confidence means no pose."""
    axis=(torch.arange(size,device=points.device,dtype=points.dtype)+.5)/size
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    distance=(xx-points[:,:,0,None,None])**2+(yy-points[:,:,1,None,None])**2
    return torch.exp(-distance/(2*(sigma/size)**2))*points[:,:,2,None,None]


def target_distribution(target, size, sigma=1.5):
    points=torch.cat([target[:,None,:],torch.ones_like(target[:,None,:1])],dim=-1)
    maps=gaussian_maps(points,size,sigma).flatten(1)
    return maps/maps.sum(1,keepdim=True).clamp_min(1e-12)


def spatial_loss(logits, target):
    # A single spatial probability distribution avoids the former sparse-BCE
    # failure where a nearly uniform background could dominate the loss.
    distribution=target_distribution(target,logits.shape[-1])
    return -(distribution*F.log_softmax(logits.flatten(1),dim=1)).sum(1).mean()


def decode(logits):
    """Local soft-argmax around peak; do not average two distant rackets."""
    b,_,h,w=logits.shape
    index=logits.flatten(1).argmax(1)
    px,py=index%w,index//w
    yy,xx=torch.meshgrid(torch.arange(h,device=logits.device),torch.arange(w,device=logits.device),indexing='ij')
    local=(xx[None]-px[:,None,None]).abs()<=2
    local=local&((yy[None]-py[:,None,None]).abs()<=2)
    weights=F.softmax(logits[:,0].masked_fill(~local,-1e9).flatten(1),dim=1).view(b,h,w)
    x=(weights*(xx+.5)).sum((1,2))/w
    y=(weights*(yy+.5)).sum((1,2))/h
    # Concentration is diagnostic, not a calibrated correctness probability.
    concentration=F.softmax(logits.flatten(1),dim=1).max(1).values
    return torch.stack([x,y],dim=1),concentration


class RacketPoseFusion(nn.Module):
    def __init__(self, backbone_path=None):
        super().__init__()
        self.encoder=resnet18(weights=None)
        if backbone_path:
            self.encoder.load_state_dict(torch.load(backbone_path,map_location='cpu',weights_only=True))
        self.encoder.fc=nn.Identity()
        self.lateral=nn.ModuleList(nn.Conv2d(channels,64,1) for channels in [64,128,256,512])
        self.smooth=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.ReLU())
        self.head=nn.Sequential(nn.Conv2d(64+len(BODY_NAMES)+2,64,3,padding=1),
                                nn.GroupNorm(8,64),nn.ReLU(),
                                nn.Conv2d(64,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,1,1))

    def train(self,mode=True):
        super().train(mode)
        # Small batches: keep pretrained BatchNorm running statistics fixed.
        for module in self.encoder.modules():
            if isinstance(module,nn.BatchNorm2d): module.eval()
        return self

    def forward(self,image,pose):
        e=self.encoder
        x=e.maxpool(e.relu(e.bn1(e.conv1(image))))
        a=e.layer1(x); b=e.layer2(a); c=e.layer3(b); d=e.layer4(c)
        features=[layer(value) for layer,value in zip(self.lateral,[a,b,c,d])]
        x=features[-1]
        for skip in reversed(features[:-1]):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)+skip
        x=self.smooth(x); size=x.shape[-1]
        body=gaussian_maps(pose,size)
        axis=(torch.arange(size,device=x.device,dtype=x.dtype)+.5)/size
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        grid=torch.stack([xx,yy])[None].expand(image.shape[0],-1,-1,-1)
        return self.head(torch.cat([x,body.to(x.dtype),grid],dim=1))


def image_tensor(image):
    rgb=image[:,:,::-1].copy().astype(np.float32)/255.
    rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(rgb.transpose(2,0,1).copy())


def augment(image,pose,target,rng):
    """Apply identical geometry to pixels, body landmarks and racket target."""
    import cv2
    image=image.copy(); pose=pose.copy(); target=target.copy()
    size=image.shape[0]
    if rng.random()<.5:
        image=np.ascontiguousarray(image[:,::-1])
        edge=(size-1)/size
        pose[:,0]=edge-pose[:,0];target[0]=edge-target[0]
        # Swap anatomical left/right channels after mirroring.
        swaps=[BODY_NAMES.index(name.replace('left_','right_')) if name.startswith('left_') else
               BODY_NAMES.index(name.replace('right_','left_')) if name.startswith('right_') else i
               for i,name in enumerate(BODY_NAMES)]
        pose=pose[swaps]
    matrix=cv2.getRotationMatrix2D((size/2,size/2),rng.uniform(-12,12),rng.uniform(.85,1.1))
    matrix[:,2]+=rng.uniform(-.04,.04,2)*size
    moved=np.append(target*size,1)@matrix.T/size
    if np.all((moved>=.01)&(moved<=.99)):
        image=cv2.warpAffine(image,matrix,(size,size),borderValue=(114,114,114))
        pose[:,:2]=np.c_[pose[:,:2]*size,np.ones(len(pose))]@matrix.T/size
        target=moved.astype(np.float32)
    pose[:,2]*=((pose[:,0]>=0)&(pose[:,0]<1)&(pose[:,1]>=0)&(pose[:,1]<1))
    # Training on small/downsampled images may help, but does not prove transfer
    # to distant players; far-side clips remain explicitly unscored.
    if rng.random()<.3:
        small=int(size*rng.uniform(.35,.7))
        image=cv2.resize(cv2.resize(image,(small,small)),(size,size))
    image=np.clip(image.astype(np.float32)*rng.uniform(.8,1.2)+rng.uniform(-12,12),0,255).astype(np.uint8)
    if rng.random()<.2: pose[:,2]=0 # image fallback when MediaPipe fails
    return image,pose,target
