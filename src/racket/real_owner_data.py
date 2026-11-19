"""Real two-racket frames: identical RGB context, two selected-player queries."""
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .mediapipe_bundle import BODY_NAMES,pose_array,square_bounds
from .mediapipe_model import image_tensor,spatial_loss

FORMAT='racket_owner_real_v004'
SIDES=('near_side','far_side')


def scene_bounds(boxes,padding=0.):
    """Union of body-derived expanded squares; never reads racket labels."""
    if not boxes:raise ValueError('At least one body box is required')
    if not math.isfinite(padding) or not 0<=padding<=.5:raise ValueError('Invalid scene padding')
    expanded=[square_bounds(box) for box in boxes]
    left=min(c[0] for c in expanded);top=min(c[1] for c in expanded)
    right=max(c[0]+c[2] for c in expanded);bottom=max(c[1]+c[2] for c in expanded)
    side=math.ceil(math.ceil(max(right-left,bottom-top))*(1+2*padding))
    return [math.floor((left+right-side)/2),math.floor((top+bottom-side)/2),side]


def selected_pose(points,box,other_boxes,crop):
    """Drop clearly wrong-person pose features; leave the shared body output intact."""
    nose=next((p for p in points if p['name']=='nose'),None)
    if not points:return np.zeros((17,3),np.float32),'missing'
    if nose and nose.get('visibility',0)*nose.get('presence',0)>=.5:
        x,y=nose['x'],nose['y'];margin=.1*(box[3]-box[1])
        own=box[0]-margin<=x<=box[2]+margin and box[1]-margin<=y<=box[3]+margin
        other=any(b[0]<=x<=b[2] and b[1]<=y<=b[3] for b in other_boxes)
        if other and not own:return np.zeros((17,3),np.float32),'rejected_clear_other_body_head'
    return np.asarray(pose_array(points,crop),np.float32),'available'


def unit_box(box,crop):
    return np.asarray([(box[k]-crop[k%2])/crop[2] for k in range(4)],np.float32)


def augment_pair(image,poses,boxes,targets,rng):
    """Apply one transform to the image, BOTH poses, BOTH boxes and BOTH heads."""
    image=image.copy();poses=poses.copy();boxes=boxes.copy();targets=targets.copy();size=image.shape[0]
    if rng.random()<.5:
        image=np.ascontiguousarray(image[:,::-1]);edge=(size-1)/size
        poses[:,:,0]=edge-poses[:,:,0];targets[:,0]=edge-targets[:,0]
        boxes[:,[0,2]]=edge-boxes[:,[2,0]]
        swaps=[BODY_NAMES.index(n.replace('left_','right_')) if n.startswith('left_') else
               BODY_NAMES.index(n.replace('right_','left_')) if n.startswith('right_') else i for i,n in enumerate(BODY_NAMES)]
        poses=poses[:,swaps,:]
    matrix=cv2.getRotationMatrix2D((size/2,size/2),rng.uniform(-8,8),rng.uniform(.92,1.08))
    matrix[:,2]+=rng.uniform(-.025,.025,2)*size
    moved=np.c_[targets*size,np.ones(2)]@matrix.T/size
    if np.all((moved>=.005)&(moved<=.995)):
        image=cv2.warpAffine(image,matrix,(size,size),borderValue=(114,114,114))
        flat=poses[:,:,:2].reshape(-1,2)
        poses[:,:,:2]=(np.c_[flat*size,np.ones(len(flat))]@matrix.T/size).reshape(2,17,2)
        for i,b in enumerate(boxes):
            corners=np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]])
            corners=np.c_[corners*size,np.ones(4)]@matrix.T/size
            boxes[i]=np.r_[corners.min(0),corners.max(0)]
        targets=moved.astype(np.float32)
    poses[:,:,2]*=((poses[:,:,0]>=0)&(poses[:,:,0]<1)&(poses[:,:,1]>=0)&(poses[:,:,1]<1))
    image=np.clip(image.astype(np.float32)*rng.uniform(.9,1.1)+rng.uniform(-8,8),0,255).astype(np.uint8)
    if rng.random()<.15:
        small=round(size*rng.uniform(.6,.9));image=cv2.resize(cv2.resize(image,(small,small)),(size,size))
    for i in range(2):
        if rng.random()<.1:poses[i,:,2]=0
    return image,poses,boxes,targets


class RealPairs(Dataset):
    def __init__(self,bundle,records,training=False,size=None,seed=42):
        self.bundle=Path(bundle);self.records=records;self.training=training;self.size=size;self.rng=np.random.default_rng(seed)
        if training and any(r['split']!='train' or not r['train_eligible'] for r in records):
            raise ValueError('Only eligible TRAIN frames may receive training augmentation')

    def __len__(self):return len(self.records)

    def raw(self,index):
        r=self.records[index];image=cv2.imread(str(self.bundle/r['image']))
        if image is None:raise ValueError('Unreadable real pair: '+r['image'])
        if self.size:image=cv2.resize(image,(self.size,self.size))
        poses=np.asarray([r['players'][s]['pose'] for s in SIDES],np.float32)
        boxes=np.asarray([unit_box(r['players'][s]['owner_box'],r['crop']) for s in SIDES],np.float32)
        targets=np.asarray([r['players'][s]['target'] for s in SIDES],np.float32)
        return image,poses,boxes,targets

    def __getitem__(self,index):
        image,poses,boxes,targets=self.raw(index)
        if self.training:image,poses,boxes,targets=augment_pair(image,poses,boxes,targets,self.rng)
        return image_tensor(image),torch.from_numpy(poses.copy()),torch.from_numpy(boxes.copy()),torch.from_numpy(targets.copy()),index


def flatten_queries(images,poses,boxes,targets):
    return images.repeat_interleave(2,dim=0),poses.flatten(0,1),boxes.flatten(0,1),targets.flatten(0,1)


def real_ownership_loss(logits,targets,opponents,separated,weight=.25):
    localization=spatial_loss(logits,targets)
    grid=torch.stack([targets,opponents],dim=1).unsqueeze(1)*2-1
    values=F.grid_sample(logits,grid,align_corners=False).flatten(1)
    raw=F.softplus(values[:,1]-values[:,0]+1.)
    ranking=raw[separated].mean() if separated.any() else logits.sum()*0
    return localization+weight*ranking,localization,ranking


def accuracy(rows):
    errors=[r['error_original_pixels'] for r in rows if r['error_original_pixels'] is not None];n=len(rows)
    eligible=[r for r in rows if r['head_separation_original_pixels']>40]
    swaps=sum(r['error_original_pixels'] is not None and r['error_original_pixels']>20 and
              r['opponent_error_original_pixels']<=20 for r in eligible)
    return dict(samples=n,missing=n-len(errors),within_20px=sum(e<=20 for e in errors),within_50px=sum(e<=50 for e in errors),
        pck20_including_misses=sum(e<=20 for e in errors)/n if n else 0.,
        pck50_including_misses=sum(e<=50 for e in errors)/n if n else 0.,
        mean_error_detected_only=float(np.mean(errors)) if errors else None,
        median_error_detected_only=float(np.median(errors)) if errors else None,
        ownership_eligible_queries=len(eligible),wrong_racket_within20_queries=swaps,
        wrong_racket_rate=swaps/len(eligible) if eligible else None,
        coordinate_system='original_video_pixels',scope='verified located two-racket frames; unknown/presence cases excluded')


def summarize_real(rows):
    report=accuracy(rows)
    for key in ['side','source_group','clip']:
        report['per_'+key]={value:accuracy([r for r in rows if r[key]==value]) for value in sorted({r[key] for r in rows})}
    pairs={}
    for row in rows:pairs.setdefault(row['pair_id'],{})[row['side']]=row
    report['paired_frames']=len(pairs);report['both_correct20']=0;report['collapsed_selection_frames']=0
    for pair in pairs.values():
        if set(pair)!=set(SIDES):raise ValueError('Every real frame must retain both selections')
        a,b=pair['near_side'],pair['far_side']
        if all(r['error_original_pixels'] is not None and r['error_original_pixels']<=20 for r in [a,b]):report['both_correct20']+=1
        if a['point_original'] and b['point_original'] and a['head_separation_original_pixels']>40 and math.dist(a['point_original'],b['point_original'])<=20:
            report['collapsed_selection_frames']+=1
    report['ownership_definition']='Head centres >40px apart; wrong-racket selection is >20px from selected head and <=20px from other head. Missing/other errors reported separately.'
    return report
