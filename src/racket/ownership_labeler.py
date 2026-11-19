"""UI for ownership of two rackets; the existing point has no edit path."""
import copy
from pathlib import Path

import cv2
import numpy as np

from .ownership import OwnershipStore, box_valid
from .schema import save_json
from .source import FrameSource


def screen_layout():
    """Fit image AND controls in the Windows work area (or a small fallback)."""
    width,height=1280,720
    try:
        import ctypes
        from ctypes import wintypes
        rect=wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x30,0,ctypes.byref(rect),0):
            width,height=rect.right-rect.left,rect.bottom-rect.top
    except (AttributeError,OSError):pass
    return min(960,max(600,width-80)),min(400,max(180,height-340))


class OwnershipLabeler:
    def __init__(self, root, pilot=False, clip=None, frame=None, batch=None, review=None):
        self.store=OwnershipStore(root);self.batch=None;self.review=None
        if review:
            if batch or pilot:raise ValueError('Choose a correction review or a new-label batch, not both')
            from .ownership_review import CorrectionReview
            self.review=CorrectionReview(self.store,review)
            self.queue=[k for k in self.review.entries if clip is None or self.store.tasks[k]['clip']==clip]
        elif batch:
            if pilot:raise ValueError('Choose a curated --batch or --pilot, not both')
            from .ownership_batch import load_batch
            self.batch=load_batch(self.store,batch)
            self.queue=[e['id'] for e in self.batch['entries'] if clip is None or self.store.tasks[e['id']]['clip']==clip]
        else:self.queue=self.store.queue(pilot,clip)
        if not self.queue:raise ValueError('No eligible frames for this selection')
        if frame is not None and clip is None:raise ValueError('--frame requires --clip')
        if frame is not None:
            identity=f'{clip}_{frame:05d}'
            if identity not in self.queue:raise ValueError('Frame is not in this queue; omit --pilot for the full eligible queue')
            self.position=self.queue.index(identity)
        else:
            if self.review:
                pending=[self.queue.index(k) for k in self.review.remaining(self.queue)]
            elif self.batch:
                from .ownership_batch import pending_order
                wanted=pending_order(self.store,self.batch,self.queue)
                pending=[self.queue.index(k) for k in wanted]
            else:pending=[i for i,k in enumerate(self.queue) if self.store.row(k)['decision']=='pending']
            if not pending:
                if self.review:raise ValueError('All correction frames reviewed. Use --clip and --frame to revisit a correction.')
                if self.batch:
                    from .ownership_batch import batch_status
                    progress=batch_status(self.store,self.batch)
                    if progress['ready_for_package']:
                        raise ValueError('Curated coverage target met. Use status, or --clip and --frame to inspect a NEW batch label')
                    raise ValueError(f'No pending curated frames; only {progress["new_usable_pairs"]} usable pairs. Check status for match/coverage deficits before packaging')
                raise ValueError('This batch is finished. Use status, or --clip and --frame to inspect a saved far-side label')
            self.position=pending[0]
        self.pilot=pilot;self.source=None;self.source_clip=None;self.cached=None;self.cached_id=None
        self.view_width,self.view_height=screen_layout()
        self.window='RallyMirror - Racket ownership | EXISTING POINT LOCKED'
        if self.store.manifest.get('workflow')=='all_existing_centres':
            self.window='RallyMirror - ALL existing racket labels | ORIGINAL POINT LOCKED'
        if self.review:self.window='RallyMirror - Correct flagged ownership frames | ORIGINAL POINT LOCKED'
        self.zoom=1.;self.offset=np.zeros(2);self.active='far';self.corner=None;self.box_mode=False
        self.message='First assign the GREEN locked point: N = near player, F = far player. Then label the OTHER racket.'
        self.load_row()

    @property
    def identity(self):return self.queue[self.position]

    @property
    def task(self):return self.store.tasks[self.identity]

    @property
    def meta(self):return self.store.clips[self.task['clip']]['meta']

    @property
    def scale(self):return min(self.view_width/self.meta['width'],self.view_height/self.meta['height'])

    def load_row(self):
        self.current=self.store.row(self.identity);self.corner=None;self.box_mode=False;self.active='far'
        if self.current['locked_owner']=='far_side':self.active='near'
        self.message='Assign GREEN locked point owner: N=near, F=far. Being far-side is valid; the point stays unchanged.'
        if self.review:
            self.active=self.review.entries[self.identity]['side'].split('_')[0]
            self.message='Saved points retained. Fix/check the selected BODY box; Space confirms and advances.'

    @property
    def other_owner(self):
        owner=self.current['locked_owner']
        return ('far_side' if owner=='near_side' else 'near_side') if owner else None

    def assign_owner(self,owner):
        previous=self.current['locked_owner'];self.draft()
        # An already drawn point must not silently change which player it labels.
        if owner!=previous:self.current['other_point']=None
        self.current['locked_owner']=owner;self.active=self.other_owner.split('_')[0]
        self.save();self.message=f'Locked point assigned to {owner.upper()}; now click the {self.other_owner.upper()} string-bed centre.'

    def save(self):
        self.store.save(self.identity,self.current)

    def draft(self):
        self.current['decision']='pending';self.current['locked_reference_issue']=None
        for player in self.current['players'].values():player['ownership_verified']=False

    def advance(self):
        if self.review:
            self.review.confirm(self.identity)
            pending=self.review.remaining(self.queue)
            if pending:self.position=self.queue.index(pending[0]);self.load_row()
            else:self.message='ALL FLAGGED FRAMES REVIEWED. Q quits; A/D revisits these correction frames only.'
            self.zoom=1.;self.offset[:]=0
            return
        if self.batch:
            from .ownership_batch import pending_order,batch_status
            pending=pending_order(self.store,self.batch,self.queue)
            if pending:self.position=self.queue.index(pending[0]);self.load_row()
            else:
                status=batch_status(self.store,self.batch)
                self.message=('CURATED BATCH READY for package checks. Q quits; A/D reviews NEW batch frames.' if status['ready_for_package'] else
                    'No pending frames in this selection. Check curated status for usable-pair or match deficits.')
            self.zoom=1.;self.offset[:]=0
            return
        choices=list(range(self.position+1,len(self.queue)))+list(range(self.position))
        pending=[i for i in choices if self.store.row(self.queue[i])['decision']=='pending']
        if pending:self.position=pending[0];self.load_row()
        else:self.message='BATCH FINISHED. Q quits. A/D inspect earlier records; original points stay locked.'
        self.zoom=1.;self.offset[:]=0

    def move(self,direction):
        new=self.position+direction
        if 0<=new<len(self.queue):
            self.position=new;self.load_row();self.zoom=1.;self.offset[:]=0
            self.message='Review the OTHER point only. N/F assigns the locked point owner; original coordinates cannot move.'
        else:self.message='No more frames in that direction.'

    def accept(self):
        proposed=copy.deepcopy(self.current)
        proposed['decision']='approved';proposed['locked_reference_issue']=None
        for player in proposed['players'].values():player['ownership_verified']=True
        self.store.validate(self.task,proposed)
        self.store.save(self.identity,proposed);self.current=proposed
        self.message='Saved verified ownership pair.'
        self.advance()

    def exclude(self,decision):
        self.current['decision']=decision;self.current['other_point']=None
        for player in self.current['players'].values():player['ownership_verified']=False
        self.current['locked_reference_issue']='Locked point coordinate is incorrect or ownership cannot be resolved; exclude without editing source.' if decision=='quarantined' else None
        self.save();self.message='Pair excluded; original coordinate unchanged.';self.advance()

    def cycle_box(self):
        boxes=self.task['candidates']
        if not boxes:
            self.message='No cached body detections. Press B and click two opposite corners around this player.';return
        field=self.active+'_side'
        current=self.current['players'][field]['bbox']
        index=next((i for i,b in enumerate(boxes) if b==current),-1)
        self.draft();self.current['players'][field]['bbox']=copy.deepcopy(boxes[(index+1)%len(boxes)])
        self.save();self.message='Box proposal changed. Confirm it identifies the correct player; Space verifies BOTH boxes.'

    def clear_box(self):
        field=self.active+'_side';self.draft()
        self.current['players'][field].update(bbox=None,track_id=None,ownership_verified=False)
        self.corner=None;self.box_mode=False;self.save()
        self.message=f'{self.active.upper()} BODY box cleared; racket points unchanged. H if the other racket is off-screen/unlocatable.'

    def mouse_event(self,event,x,y,flags,_):
        if not 0<=x<self.view_width or not 0<=y<self.view_height:return
        point=np.array([x,y],float)/(self.scale*self.zoom)+self.offset
        if event==cv2.EVENT_MOUSEWHEEL:
            zoom=min(8.,max(1.,self.zoom*(1.4 if flags>0 else 1/1.4)))
            self.offset=point-np.array([x,y])/(self.scale*zoom);self.zoom=zoom
            if zoom==1:self.offset[:]=0
        elif event==cv2.EVENT_LBUTTONDOWN:
            if not (0<=point[0]<self.meta['width'] and 0<=point[1]<self.meta['height']):return
            if self.box_mode:
                if self.corner is None:self.corner=point;self.message='Now click the opposite corner of this BODY box.';return
                lo=np.minimum(self.corner,point);hi=np.maximum(self.corner,point)
                box=[round(float(v),2) for v in np.r_[lo,hi]]
                if not box_valid(box,self.meta['width'],self.meta['height']):
                    self.message='Box has no area. Press B and draw two opposite corners again.';self.corner=None;return
                self.draft();field=self.active+'_side'
                self.current['players'][field]['bbox']=box;self.corner=None;self.box_mode=False;self.save()
                self.message='Body box saved. Assign locked owner (N/F), click OTHER centre, then Space to confirm.'
            else:
                if not self.current['locked_owner']:
                    self.message='First press N or F to assign the GREEN locked point owner. Then click the OTHER racket.';return
                # There is deliberately no locked-point selector or setter.
                self.draft();self.current['other_point']=[round(float(v),2) for v in point]
                self.save();self.message=f'NEW {self.other_owner.upper()} centre saved. Check body boxes, then Space confirms and advances.'

    def handle_key(self,key):
        if ord('A')<=key<=ord('Z'):key+=32
        try:
            if key in (ord(' '),10,13):self.accept()
            elif key in (ord('n'),ord('f')):self.assign_owner('near_side' if key==ord('n') else 'far_side')
            elif key==ord('h'):self.exclude('unlocatable')
            elif key==ord('x'):self.exclude('quarantined')
            elif key==ord('u'):
                self.draft();self.current['other_point']=None;self.save()
                self.message='NEW point cleared. ORIGINAL locked point unchanged.'
            elif key in (ord('a'),ord('d')):self.move(-1 if key==ord('a') else 1)
            elif key in (ord('1'),ord('2')):
                self.active='near' if key==ord('1') else 'far';self.corner=None;self.box_mode=False
                self.message=f'{self.active.upper()} BODY BOX selected. Tab cycles; B draws; C clears this box only.'
            elif key==9:self.cycle_box()
            elif key==ord('c'):self.clear_box()
            elif key==ord('b'):
                self.box_mode=not self.box_mode;self.corner=None
                self.message=f'BOX MODE: click two opposite corners around the {self.active.upper()} player BODY.' if self.box_mode else 'Point mode: click the OTHER racket centre.'
            elif key==ord('r'):self.zoom=1.;self.offset[:]=0
        except ValueError as error:self.message=str(error)

    def image(self):
        if self.cached_id!=self.identity:
            if self.source_clip!=self.task['clip']:
                if self.source:self.source.close()
                self.source=FrameSource(self.meta);self.source_clip=self.task['clip']
            self.cached=self.source.read(self.task['frame']);self.cached_id=self.identity
        return self.cached

    def draw(self):
        scale=self.scale*self.zoom
        matrix=np.float32([[scale,0,-self.offset[0]*scale],[0,scale,-self.offset[1]*scale]])
        view=cv2.warpAffine(self.image(),matrix,(self.view_width,self.view_height))
        def at(point):return tuple(np.rint((np.array(point)-self.offset)*scale).astype(int))
        for field,label,color in [('near_side','NEAR BODY',(255,255,0)),('far_side','FAR BODY',(0,255,255))]:
            box=self.current['players'][field]['bbox']
            if box:
                cv2.rectangle(view,at(box[:2]),at(box[2:]),color,2)
                cv2.putText(view,label,at([box[0],max(14,box[1]-8)]),cv2.FONT_HERSHEY_SIMPLEX,.52,color,2,cv2.LINE_AA)
        ref=self.task['locked_reference']['point'];near=at([ref['x'],ref['y']])
        cv2.circle(view,near,8,(0,255,0),2,cv2.LINE_AA)
        owner=(self.current['locked_owner'] or 'OWNER UNSET').replace('_side','').upper()
        cv2.putText(view,'LOCKED: '+owner,(near[0]+10,near[1]+16),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,255,0),2,cv2.LINE_AA)
        point=self.current['other_point']
        if point:
            far=at(point);cv2.drawMarker(view,far,(255,0,255),cv2.MARKER_CROSS,19,2)
            cv2.putText(view,'NEW: '+self.other_owner.replace('_side','').upper(),(far[0]+10,far[1]+16),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,0,255),2,cv2.LINE_AA)
        if self.corner is not None:cv2.circle(view,at(self.corner),6,(0,180,255),2)
        canvas=np.vstack([view,np.zeros((250,self.view_width,3),np.uint8)])
        counts={k:self.store.row(k)['decision'] for k in self.queue}
        pending=sum(value=='pending' for value in counts.values())
        stage='PILOT' if self.pilot else 'ALL ELIGIBLE PAIRED FRAMES'
        if self.store.manifest.get('workflow')=='all_existing_centres':stage='ALL EXISTING CENTRES | imported reviews locked'
        statusline=f'{stage} | {self.position+1}/{len(self.queue)} | {pending} unfinished | ORIGINAL POINT LOCKED'
        if self.batch:
            from .ownership_batch import batch_status
            progress=batch_status(self.store,self.batch)
            phase=next(e['stage'] for e in self.batch['entries'] if e['id']==self.identity)
            stage='CURATED READY' if progress['ready_for_package'] else f'CURATED {phase.upper()}'
            statusline=(f'{stage} | {self.position+1}/{len(self.queue)} | new usable {progress["new_usable_pairs"]}/{progress["target_usable_pairs"]}'
                        f' | primary left {progress["primary"].get("pending",0)} | reserve left {progress["reserve"].get("pending",0)}')
        if self.review:
            pending=len(self.review.remaining(self.queue))
            statusline=f'FLAGGED CORRECTIONS | {self.position+1}/{len(self.queue)} | {pending} to review | ORIGINAL POINT LOCKED'
        lines=[f'{self.task["clip"]} | source frame {self.task["frame"]} | {self.task["split"].upper()} {self.task["source_group"]} | {self.current["decision"].upper()}',
               statusline,
               'N: locked point belongs to NEAR player | F: locked point belongs to FAR player (required each frame)',
               'Then click the OTHER string-bed centre | Space: confirm both player boxes, save and advance',
               f'1/2: near/far BODY BOX | Tab: cycle | B: draw | C: CLEAR selected box [{self.active.upper()}]',
               'H: OTHER racket unlocatable | X: original coordinate wrong / ownership unclear -> exclude pair',
               'A/D: previous/next | U: clear NEW point only | Wheel: zoom | R: reset | Q/Esc: save/quit',
               'Cyan: near body | Yellow: far body | Green: ORIGINAL locked point | Magenta: OTHER racket',
               'Original points can belong to EITHER player. Being far-side is not an error. Never label the same racket twice.',
               self.message]
        if self.review:
            entry=self.review.entries[self.identity]
            hint=('Crop margin: check actual body boxes and racket centres; do not enlarge bodies just to include a racket.'
                  if 'crop' in entry['reason'].lower() else
                  f'Fix {entry["side"].replace("_side", "").upper()} BODY: box currently selects an official. B then two corners around the actual player.')
            lines[-2]=hint
        for i,line in enumerate(lines):
            font=min(.43,.43*self.view_width/1000)
            cv2.putText(canvas,line,(9,self.view_height+20+i*24),cv2.FONT_HERSHEY_SIMPLEX,font,
                        (0,255,0) if i==1 else (240,240,240),1,cv2.LINE_AA)
        return canvas

    def close(self):
        if self.source:self.source.close();self.source=None
        self.store.verify_sources()

    def run(self):
        cv2.namedWindow(self.window,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(self.window,self.mouse_event)
        try:
            while True:
                cv2.imshow(self.window,self.draw());key=cv2.waitKey(30)&0xff
                if ord('A')<=key<=ord('Z'):key+=32
                if key in (ord('q'),27) or cv2.getWindowProperty(self.window,cv2.WND_PROP_VISIBLE)<1:break
                self.handle_key(key)
        finally:
            self.close();cv2.destroyWindow(self.window)
