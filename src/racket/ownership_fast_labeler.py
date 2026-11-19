"""Click/H per racket, then click/H per body; all transitions autosave."""
import copy

import cv2
import numpy as np

from .ownership import box_valid
from .ownership_fast import SampleStore, SIDES, read_selected, steps
from .ownership_labeler import screen_layout
from .source import FrameSource

COLORS = {'near_side': (255, 255, 0), 'far_side': (0, 255, 255)}


class FastLabeler:
    def __init__(self, root, identity=None):
        self.store = SampleStore(root)
        self.queue = list(self.store.tasks)
        if identity is not None and identity not in self.store.tasks:
            raise ValueError('Frame is not in this sampled queue')
        pending = [i for i, k in enumerate(self.queue) if self.store.row(k)['decision'] == 'pending']
        if identity is None and not pending:
            raise ValueError('Sampled queue complete. Use status or --id rallyNN_00000 to revisit.')
        self.position = self.queue.index(identity) if identity else pending[0]
        self.view_width, self.view_height = screen_layout()
        self.window = 'RallyMirror - SAMPLED near / far | CLICK or H | autosave'
        self.source = None
        self.source_clip = None
        self.cached_id = None
        self.cached = None
        self.zoom = 1.
        self.offset = np.zeros(2)
        self.message = ''
        self.load()

    @property
    def identity(self):
        return self.queue[self.position]

    @property
    def task(self):
        return self.store.tasks[self.identity]

    @property
    def meta(self):
        return self.store.clips[self.task['clip']]['meta']

    @property
    def scale(self):
        return min(self.view_width/self.meta['width'], self.view_height/self.meta['height'])

    def load(self):
        self.current = self.store.row(self.identity)
        self.box_mode = False
        self.corner = None
        self.undo_stack = []
        self.body_override = None
        self.show_candidates = True
        self.completed = self.current['decision'] != 'pending'
        self.message = self.task.get('reason', '')
        self.zoom = 1.
        self.offset[:] = 0

    @property
    def step(self):
        if self.body_override is not None and self.current['decision'] == 'pending':
            return self.body_override, 'body'
        pending = steps(self.current)
        return pending[0] if pending else None

    def mutate(self, function, action):
        before = copy.deepcopy(self.current)
        self.current['decision'] = 'pending'
        function()
        self.current['actions'].append(action)
        if all(p['racket_state'] == 'unlocatable' for p in self.current['players'].values()):
            for p in self.current['players'].values():
                if p['body_state'] == 'pending':
                    p['body_state'] = 'not_required'
        if not steps(self.current):
            self.current['decision'] = 'complete'
        try:
            self.store.save(self.identity, self.current)
        except ValueError:
            self.current = before
            raise
        self.undo_stack.append(before)
        self.body_override = None
        self.box_mode = False
        self.corner = None
        if self.current['decision'] == 'complete':
            self.advance()

    def advance(self):
        choices = list(range(self.position+1, len(self.queue)))+list(range(self.position))
        pending = [i for i in choices if self.store.row(self.queue[i])['decision'] == 'pending']
        if pending:
            self.position = pending[0]
            self.load()
        else:
            self.completed = True
            self.message = 'ALL SAMPLES SAVED. Q quits. A/D revisits; E reopens a saved frame.'

    def point_action(self, point):
        if not self.step:
            return
        side, phase = self.step
        if phase != 'racket':
            raise ValueError('Current step checks a BODY box, not a racket point')
        def update():
            self.current['players'][side].update(point=list(point), racket_state='located')
        self.mutate(update, dict(side=side, action='manual_point', point=list(point)))

    def hidden_action(self):
        if not self.step:
            return
        side, phase = self.step
        def update():
            if phase == 'racket':
                self.current['players'][side].update(point=None, racket_state='unlocatable')
            else:
                self.current['players'][side].update(bbox=None, body_state='unlocatable')
        self.mutate(update, dict(side=side, action=phase+'_unlocatable'))

    def body_action(self, box):
        if not self.step:
            return
        side, phase = self.step
        if phase != 'body':
            raise ValueError('Finish near/far racket clicks first')
        if not box_valid(box, self.meta['width'], self.meta['height']):
            raise ValueError('Draw opposite corners inside the original image')
        def update():
            self.current['players'][side].update(bbox=copy.deepcopy(box), body_state='verified')
        self.mutate(update, dict(side=side, action='verify_body', bbox=box))

    def click_original(self, point):
        if self.completed or not self.step:
            self.message = 'Saved frame. E reopens it; A/D navigates.'
            return
        side, phase = self.step
        if phase == 'racket':
            self.point_action([round(float(v), 2) for v in point])
            return
        if self.box_mode:
            if self.corner is None:
                self.corner = np.asarray(point)
                self.message = 'Click the opposite BODY corner; second click confirms automatically.'
                return
            a = np.minimum(self.corner, point)
            b = np.maximum(self.corner, point)
            self.body_action([round(float(v), 2) for v in np.r_[a, b]])
            return
        # The highlighted box wins if clicked. Tab cycles when player boxes overlap.
        active = self.current['players'][side]['bbox']
        contains = lambda b: b is not None and b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]
        if contains(active):
            self.body_action(active)
            return
        boxes = [b for b in self.task['candidates'] if contains(b)]
        if boxes:
            chosen = min(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            self.body_action(chosen)
        else:
            self.message = 'No box here. B, then click two BODY corners; H if this body cannot be located.'

    def undo(self):
        if self.undo_stack:
            self.current = self.undo_stack.pop()
            self.store.save(self.identity, self.current)
            self.completed = False
            self.body_override = None
            self.box_mode = False
            self.corner = None
            self.message = 'Last action undone.'
        else:
            self.message = 'After auto-advance: A returns to the previous frame; E reopens its editable actions.'

    def reopen(self):
        self.body_override = None
        self.current['decision'] = 'pending'
        for side in SIDES:
            p = self.current['players'][side]
            if side not in self.task.get('retained_verified_sides', []):
                p['body_state'] = 'pending'
        self.current['actions'].append(dict(action='reopen'))
        self.store.save(self.identity, self.current)
        self.completed = False
        self.box_mode = False
        self.corner = None
        self.message = 'Reopened BODY checks; all racket points retained. 1/2 edits a NEW near/far point.'

    def edit_point(self, side):
        if side in self.task.get('references', {}):
            raise ValueError('Original point is locked. X excludes an incorrect original without changing it.')
        self.undo_stack.append(copy.deepcopy(self.current))
        self.current['decision'] = 'pending'
        self.current['players'][side].update(point=None, racket_state='pending')
        self.store.save(self.identity, self.current)
        self.completed = False
        self.body_override = None
        self.box_mode = False
        self.corner = None

    def select_body(self, side, clear_both=False):
        """Edit either body, including a retained correction body, without touching points."""
        self.undo_stack.append(copy.deepcopy(self.current))
        self.current['decision'] = 'pending'
        selected = SIDES if clear_both else (side,)
        for target in selected:
            self.current['players'][target]['body_state'] = 'pending'
            if clear_both:
                self.current['players'][target]['bbox'] = None
        self.current['actions'].append(dict(action='clear_body_boxes' if clear_both else 'select_body', side=side))
        self.store.save(self.identity, self.current)
        self.completed = False
        self.body_override = side
        self.box_mode = True
        self.corner = None
        if clear_both:
            self.show_candidates = False
        self.message = (('Both body boxes cleared; racket points kept. ' if clear_both else '')+
                        side.replace('_side', '').upper()+' BODY: click two opposite corners. F/N switches side.')

    def handle_key(self, key):
        if ord('A') <= key <= ord('Z'):
            key += 32
        try:
            if key == ord('h') and not self.completed:
                self.hidden_action()
            elif key in (8, ord('u')):
                self.undo()
            elif key in (ord('a'), ord('d')):
                new = self.position+(-1 if key == ord('a') else 1)
                if 0 <= new < len(self.queue):
                    self.position = new
                    self.load()
            elif key == ord('e'):
                self.reopen()
            elif key in (ord('n'), ord('f')):
                self.select_body(SIDES[0] if key == ord('n') else SIDES[1])
            elif key == ord('c'):
                self.select_body(SIDES[0], clear_both=True)
            elif key == ord('g'):
                self.show_candidates = not self.show_candidates
            elif key in (ord('1'), ord('2')):
                self.edit_point(SIDES[0] if key == ord('1') else SIDES[1])
            elif key == ord('x'):
                self.current['decision'] = 'excluded'
                self.current['actions'].append(dict(action='exclude_ambiguous_or_wrong_existing_point'))
                self.store.save(self.identity, self.current)
                self.advance()
            elif key == ord('r'):
                self.zoom = 1.
                self.offset[:] = 0
            elif self.step and self.step[1] == 'body' and not self.completed:
                side = self.step[0]
                if key == ord('b'):
                    self.box_mode = not self.box_mode
                    self.corner = None
                    self.message = 'Click two opposite corners of the BODY only.' if self.box_mode else 'Click a detected BODY to verify it.'
                elif key == 9:
                    self.box_mode = False
                    self.corner = None
                    boxes = self.task['candidates']
                    if boxes:
                        old = self.current['players'][side]['bbox']
                        i = next((i for i, b in enumerate(boxes) if b == old), -1)
                        self.current['players'][side]['bbox'] = copy.deepcopy(boxes[(i+1) % len(boxes)])
                        self.store.save(self.identity, self.current)
                elif key in (32, 10, 13):
                    self.body_action(self.current['players'][side]['bbox'])
        except ValueError as error:
            self.message = str(error)

    def mouse(self, event, x, y, flags, _):
        if not (0 <= x < self.view_width and 0 <= y < self.view_height):
            return
        point = np.array([x, y], float)/(self.scale*self.zoom)+self.offset
        if event == cv2.EVENT_MOUSEWHEEL:
            z = min(12., max(1., self.zoom*(1.4 if flags > 0 else 1/1.4)))
            self.offset = point-np.array([x, y])/(self.scale*z)
            self.zoom = z
            if z == 1:
                self.offset[:] = 0
        elif event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= point[0] < self.meta['width'] and 0 <= point[1] < self.meta['height']:
                try:
                    self.click_original(point)
                except ValueError as error:
                    self.message = str(error)

    def image(self):
        if self.cached_id != self.identity:
            if self.source_clip != self.task['clip']:
                if self.source:
                    self.source.close()
                self.source = FrameSource(self.meta)
                self.source_clip = self.task['clip']
            self.cached = self.source.read(self.task['frame'])
            # Guard against accidentally reviewing a different video or shifted frame.
            import hashlib
            expected = self.task.get('image_sha256')
            if expected and hashlib.sha256(self.cached.tobytes()).hexdigest() != expected:
                # Some video decoders return different pixels after a keyframe
                # seek. Queue preparation decoded in order; retry that exact
                # path before treating this as changed source material.
                self.cached = next(read_selected(self.meta, [self.task['frame']]))[1]
                if hashlib.sha256(self.cached.tobytes()).hexdigest() != expected:
                    raise ValueError('Source image differs from prepared frame even after sequential decoding: '+self.identity)
            self.cached_id = self.identity
        return self.cached

    def draw(self):
        factor = self.scale*self.zoom
        matrix = np.float32([[factor, 0, -self.offset[0]*factor], [0, factor, -self.offset[1]*factor]])
        view = cv2.warpAffine(self.image(), matrix, (self.view_width, self.view_height))
        at = lambda p: tuple(np.rint((np.asarray(p)-self.offset)*factor).astype(int))
        active = self.step[0] if self.step else None
        body_phase = self.step and self.step[1] == 'body'
        if body_phase and self.show_candidates:
            for b in self.task['candidates']:
                cv2.rectangle(view, at(b[:2]), at(b[2:]), (90, 90, 90), 1)
        for side in SIDES:
            p = self.current['players'][side]
            color = COLORS[side]
            short = side.replace('_side', '').upper()
            if p['bbox'] is not None:
                box = p['bbox']
                cv2.rectangle(view, at(box[:2]), at(box[2:]), color, 3 if side == active and body_phase else 1)
                label = short+' BODY '+('OK' if p['body_state'] == 'verified' else 'PROPOSAL')
                cv2.putText(view, label, at([box[0], max(15, box[1]-8)]), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
            if p['point'] is not None:
                locked = side in self.task.get('references', {}) and not getattr(self, 'allow_reference_edit', False)
                cv2.circle(view, at(p['point']), 7, (0, 255, 0) if locked else color, 2, cv2.LINE_AA)
                xy = at(p['point'])
                cv2.putText(view, short+(' LOCKED' if locked else ''), (xy[0]+8, xy[1]+13), cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
        if self.corner is not None:
            cv2.circle(view, at(self.corner), 6, (0, 160, 255), 2)
        canvas = np.vstack([view, np.zeros((240, self.view_width, 3), np.uint8)])
        pending = sum(self.store.row(k)['decision'] == 'pending' for k in self.queue)
        phase = 'SAVED - E reopens' if self.completed else (
            f'{self.step[0].replace("_side", "").upper()} {self.step[1].upper()}' if self.step else 'SAVED')
        action = ('Click string-bed centre OR H if unlocatable. Next racket step opens automatically.'
                  if self.step and self.step[1] == 'racket' else
                  'Click correct BODY box OR Space accepts highlight. B + two corners redraws. H = body unlocatable.')
        lines = [
            f'{self.task["clip"]} frame {self.task["frame"]} | {self.task["split"].upper()} {self.task["source_group"]} | {self.task["kind"].upper()}',
            f'{self.position+1}/{len(self.queue)} | {pending} left | NOW: {phase} | every action autosaves',
            action,
            'N: redraw NEAR body | F: redraw FAR body | C: clear BOTH boxes, keep points | G: show/hide proposals',
            'Racket: click/click, click/H, H/click or H/H. H/H skips body checks. Bodies use click/H too.',
            'H means UNKNOWN, never a negative label. Body boxes enclose people; racket can lie outside the body.',
            'Tab: next box | B: redraw | Backspace/U: undo | A/D: prev/next | E: redo boxes | 1/2: edit NEW near/far point',
            'Wheel: zoom at cursor | R: reset zoom | X: exclude ambiguous frame / incorrect locked point | Q/Esc: quit',
            'Cyan = near, yellow = far, green point = locked. Near/far refer to court sides, not shirt or screen size.',
            self.message[:160],
        ]
        for i, line in enumerate(lines):
            scale = min(.43, (self.view_width-20)/max(1, cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1., 1)[0][0]))
            cv2.putText(canvas, line, (9, self.view_height+18+i*23), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 255, 0) if i == 1 else (235, 235, 235), 1, cv2.LINE_AA)
        return canvas

    def close(self):
        if self.source:
            self.source.close()
        self.store.verify()

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.mouse)
        try:
            while True:
                cv2.imshow(self.window, self.draw())
                key = cv2.waitKey(30) & 0xff
                if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                self.handle_key(key)
        finally:
            self.close()
            cv2.destroyWindow(self.window)
