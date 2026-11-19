"""Ground-truth-independent limits on the racket output heatmap.

The person box remains a body box. Expand only the permitted racket region;
never clamp a predicted point onto the edge of a body box. No tracker, court or
temporal interface is changed. These are development-time geometric heuristics.
"""
import torch

from .mediapipe_bundle import BODY_NAMES
from .mediapipe_model import decode


def allowed_region(logits,boxes,crops,pose=None,policy='expanded_box'):
    """Batched boxes in original pixels; crops=[origin_x,origin_y,square_side].

    expanded_box: pad .45 body heights horizontally, .25 vertically.
    wrist_region: disks of radius .5 body heights around reliable wrists;
      falls back to expanded_box when wrists are missing. This experimental
      comparison can exclude valid extended/poor-pose cases.
    body_box: comparison only; most real rackets extend beyond this rectangle.
    """
    device,dtype=logits.device,logits.dtype
    boxes=torch.as_tensor(boxes,device=device,dtype=dtype)
    crops=torch.as_tensor(crops,device=device,dtype=dtype)
    if boxes.shape!=(logits.shape[0],4) or crops.shape!=(logits.shape[0],3):
        raise ValueError('One original-pixel person box and square crop per heatmap')
    if not torch.isfinite(boxes).all() or not torch.isfinite(crops).all() or (crops[:,2]<=0).any():
        raise ValueError('Invalid player/crop geometry')
    height=boxes[:,3]-boxes[:,1]
    if (height<=0).any() or (boxes[:,2]<=boxes[:,0]).any():raise ValueError('Empty body box')
    h,w=logits.shape[-2:]
    yy,xx=torch.meshgrid((torch.arange(h,device=device,dtype=dtype)+.5)/h,
                          (torch.arange(w,device=device,dtype=dtype)+.5)/w,indexing='ij')
    x=xx[None]*crops[:,2,None,None]+crops[:,0,None,None]
    y=yy[None]*crops[:,2,None,None]+crops[:,1,None,None]
    pad_x=0 if policy=='body_box' else .45
    pad_y=0 if policy=='body_box' else .25
    permitted=((x>=boxes[:,0,None,None]-pad_x*height[:,None,None])&
               (x<=boxes[:,2,None,None]+pad_x*height[:,None,None])&
               (y>=boxes[:,1,None,None]-pad_y*height[:,None,None])&
               (y<=boxes[:,3,None,None]+pad_y*height[:,None,None]))
    if policy in ('body_box','expanded_box'):return permitted
    if policy!='wrist_region':raise ValueError('Unknown player-region policy')
    if pose is None:return permitted
    pose=torch.as_tensor(pose,device=device,dtype=dtype)
    indices=[BODY_NAMES.index(name) for name in ['left_wrist','right_wrist']]
    wrists=pose[:,indices]
    reliable=torch.isfinite(wrists).all(-1)&(wrists[:,:,2]>=.15)
    wx=wrists[:,:,0]*crops[:,2,None]+crops[:,0,None]
    wy=wrists[:,:,1]*crops[:,2,None]+crops[:,1,None]
    distance=(x[:,None]-wx[:,:,None,None])**2+(y[:,None]-wy[:,:,None,None])**2
    disks=((distance<=(.5*height[:,None,None,None])**2)&reliable[:,:,None,None]).any(1)
    return torch.where(reliable.any(1)[:,None,None],disks,permitted)


def decode_player(logits,boxes,crops,pose=None,policy='expanded_box'):
    allowed=allowed_region(logits,boxes,crops,pose,policy)
    valid=allowed.flatten(1).any(1)
    filtered=logits.masked_fill(~allowed[:,None],-1e9)
    point,_=decode(filtered)
    # Keep the original model's probability mass at the selected peak rather
    # than inflate confidence by renormalizing over a small allowed region.
    index=filtered.flatten(1).argmax(1)
    mass=logits.flatten(1).softmax(1).gather(1,index[:,None])[:,0]
    return point,mass,valid
