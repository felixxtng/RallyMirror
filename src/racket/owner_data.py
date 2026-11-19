"""Reuse approved crops; train ownership with matched two-person examples.

Each synthetic canvas is seen twice with different box/pose conditioning and
different existing manual targets. Both components are TRAINING crops only.
They are not far-side ground truth or new manual annotations.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .mediapipe_bundle import BODY_NAMES
from .mediapipe_model import image_tensor


def normalized_box(record):
    x,y,side=record['crop']
    return np.asarray([(record['owner_box'][0]-x)/side,(record['owner_box'][1]-y)/side,
                       (record['owner_box'][2]-x)/side,(record['owner_box'][3]-y)/side],np.float32)


def pair_geometry(pair):
    # Non-overlapping square half-size crops, with offsets fixed per canvas.
    if pair['orientation']=='horizontal':return [[0.,pair['jitter_a']],[.5,pair['jitter_b']]]
    return [[pair['jitter_a'],0.],[pair['jitter_b'],.5]]


def make_pairs(records,count=192,seed=42):
    if any(r['split']!='train' or not r['train_eligible'] for r in records):
        raise ValueError('Ownership pairs require eligible training samples only')
    if len(records)<2:raise ValueError('Need at least two training samples')
    rng=np.random.default_rng(seed);pairs=[]
    for index in range(count):
        a=int(rng.integers(len(records)))
        candidates=[i for i,r in enumerate(records) if i!=a and
                    (r['clip']==records[a]['clip'] if index%2==0 else r['source_group']!=records[a]['source_group'])]
        if not candidates:candidates=[i for i in range(len(records)) if i!=a]
        b=int(rng.choice(candidates))
        pairs.append(dict(id=f'owner_pair_{index:04d}',a=records[a]['id'],b=records[b]['id'],
                          orientation='horizontal' if index%2==0 else 'vertical',
                          jitter_a=float(rng.uniform(0,.5)),jitter_b=float(rng.uniform(0,.5))))
    return pairs


def compose_pair(images,pair):
    size=images[0].shape[0];half=size//2
    canvas=np.full((size,size,3),114,np.uint8)
    offsets=pair_geometry(pair)
    # Use actual raster positions as the coordinate transform.
    offsets=[[round(x*size)/size,round(y*size)/size] for x,y in offsets]
    for image,(x,y) in zip(images,offsets):
        xx,yy=round(x*size),round(y*size)
        canvas[yy:yy+half,xx:xx+half]=cv2.resize(image,(half,half))
    return canvas,offsets


def augment_owner(image,pose,box,target,rng):
    image=image.copy();pose=pose.copy();box=box.copy();target=target.copy()
    size=image.shape[0]
    if rng.random()<.5:
        image=np.ascontiguousarray(image[:,::-1]);edge=(size-1)/size
        pose[:,0]=edge-pose[:,0];target[0]=edge-target[0]
        box[[0,2]]=edge-box[[2,0]]
        swaps=[BODY_NAMES.index(name.replace('left_','right_')) if name.startswith('left_') else
               BODY_NAMES.index(name.replace('right_','left_')) if name.startswith('right_') else i
               for i,name in enumerate(BODY_NAMES)]
        pose=pose[swaps]
    matrix=cv2.getRotationMatrix2D((size/2,size/2),rng.uniform(-10,10),rng.uniform(.9,1.1))
    matrix[:,2]+=rng.uniform(-.04,.04,2)*size
    moved=np.append(target*size,1)@matrix.T/size
    if np.all((moved>=.01)&(moved<=.99)):
        image=cv2.warpAffine(image,matrix,(size,size),borderValue=(114,114,114))
        pose[:,:2]=np.c_[pose[:,:2]*size,np.ones(len(pose))]@matrix.T/size
        corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]])
        corners=np.c_[corners*size,np.ones(4)]@matrix.T/size
        box=np.r_[corners.min(0),corners.max(0)].astype(np.float32);target=moved.astype(np.float32)
    pose[:,2]*=((pose[:,0]>=0)&(pose[:,0]<1)&(pose[:,1]>=0)&(pose[:,1]<1))
    if rng.random()<.2:
        small=int(size*rng.uniform(.4,.75));image=cv2.resize(cv2.resize(image,(small,small)),(size,size))
    image=np.clip(image.astype(np.float32)*rng.uniform(.85,1.15)+rng.uniform(-10,10),0,255).astype(np.uint8)
    if rng.random()<.1:pose[:,2]=0 # box stays available even if MediaPipe fails
    return image,pose,box,target


class OwnerSamples(Dataset):
    def __init__(self,bundle,records,pairs=None,training=False,image_size=None):
        self.bundle=Path(bundle);self.records=records;self.by_id={r['id']:r for r in records}
        self.examples=[dict(kind='natural',id=r['id']) for r in records]
        self.training=training;self.image_size=image_size;self.rng=np.random.default_rng(42)
        for pair in pairs or []:
            if any(self.by_id[pair[k]]['split']!='train' for k in ['a','b']):
                raise ValueError('A validation sample cannot enter an ownership pair')
            for selected in [0,1]:self.examples.append(dict(kind='pair',pair=pair,selected=selected))

    def __len__(self):return len(self.examples)

    def image(self,record):
        image=cv2.imread(str(self.bundle/record['image']))
        if image is None:raise ValueError(f"Unreadable crop: {record['image']}")
        if self.image_size:image=cv2.resize(image,(self.image_size,self.image_size))
        return image

    def reference(self,index):
        example=self.examples[index]
        if example['kind']=='natural':return self.by_id[example['id']],dict(offset=[0.,0.],scale=1.)
        pair=example['pair'];selected=example['selected'];size=self.image_size or 512
        offset=[round(v*size)/size for v in pair_geometry(pair)[selected]]
        return self.by_id[pair['a' if selected==0 else 'b']],dict(offset=offset,scale=.5)

    def raw(self,index):
        example=self.examples[index]
        if example['kind']=='natural':
            record=self.by_id[example['id']];image=self.image(record)
            offset=[0.,0.];scale=1.;pose=np.array(record['pose'],np.float32)
            target=np.array(record['target'],np.float32);box=normalized_box(record)
        else:
            pair=example['pair'];sources=[self.by_id[pair[k]] for k in ['a','b']]
            image,offsets=compose_pair([self.image(r) for r in sources],pair)
            record=sources[example['selected']];offset=offsets[example['selected']];scale=.5
            pose=np.array(record['pose'],np.float32);pose[:,:2]=pose[:,:2]*scale+offset
            target=np.array(record['target'],np.float32)*scale+offset
            box=normalized_box(record)*scale+np.array(offset*2,np.float32)
        return image,pose,box,target,record,dict(offset=offset,scale=scale)

    def __getitem__(self,index):
        image,pose,box,target,_,_=self.raw(index)
        if self.training:image,pose,box,target=augment_owner(image,pose,box,target,self.rng)
        return image_tensor(image),torch.from_numpy(pose.copy()),torch.from_numpy(box.copy()),torch.from_numpy(target.copy()),index


def pair_only_dataset(bundle,records,pairs,image_size=None):
    dataset=OwnerSamples(bundle,records,pairs,False,image_size)
    dataset.examples=[e for e in dataset.examples if e['kind']=='pair']
    return dataset
