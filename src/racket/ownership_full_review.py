"""One full-frame editor for both racket points and both body boxes."""
import copy

import cv2
import numpy as np

from .ownership import box_valid
from .ownership_fast import SIDES
from .ownership_fast_labeler import FastLabeler
from .schema import digest, read_json, save_json


class FullReviewLabeler(FastLabeler):
    def __init__(self, root, identity=None):
        from pathlib import Path
        manifest = read_json(Path(root)/'manifest.json')
        self.allow_reference_edit = True
        self.last_confirmation = None
        # Explicit first ID prevents the old fast editor from filtering completed frames.
        super().__init__(root, identity or manifest['tasks'][0]['id'])
        self.queue = [t['id'] for t in manifest['tasks']]
        self.progress_path = self.store.root/'full_review_progress.json'
        self.progress = read_json(self.progress_path) if self.progress_path.exists() else dict(reviewed={})
        pending = self.remaining()
        self.position = self.queue.index(identity or (pending[0] if pending else self.queue[0]))
        self.window = 'RallyMirror - FULL FRAME REVIEW | 1 NEAR / 2 FAR | points + boxes | SPACE saves'
        self.load()
        if not pending and identity is None:
            self.completed = True
            self.message = 'FULL REVIEW COMPLETE. A/D revisits any frame; both points and boxes remain editable.'

    def load(self):
        super().load()
        self.active_side = 'near_side'
        self.mode = 'racket'
        self.completed = False
        self.show_candidates = False
        self.message = 'Check boxes first. NEAR click/H selects FAR; FAR click/H saves and auto-next if valid. 1/2 selects manually.'

    @property
    def step(self):
        return self.active_side, self.mode

    def remaining(self):
        return [k for k in self.queue if self.progress['reviewed'].get(k) != digest(self.store.row(k))]

    @property
    def review_status(self):
        reviewed = getattr(self, 'progress', {}).get('reviewed', {})
        if reviewed.get(self.identity) == digest(self.current):
            return 'EXCLUDED' if self.current['decision'] == 'excluded' else 'CONFIRMED'
        return 'DRAFT - NOT CONFIRMED'

    def reference_provenance(self, side):
        if side in self.task.get('references', {}):
            ref = self.task['references'][side]
            self.current['players'][side]['reference_override'] = dict(
                original_point=copy.deepcopy(ref), provenance='explicit_manual_correction',
                original_reference_sha256=digest(dict(id=self.identity, side=side, point=ref)))

    def edit(self, function, action):
        before = copy.deepcopy(self.current)
        self.current['decision'] = 'pending'
        function()
        self.current['actions'].append(action)
        try:
            self.store.save(self.identity, self.current)
        except ValueError:
            self.current = before
            raise
        self.undo_stack.append(before)
        self.completed = False

    def click_original(self, point):
        side = self.active_side
        if self.mode == 'racket':
            point = [round(float(v), 2) for v in point]
            def update():
                self.reference_provenance(side)
                self.current['players'][side].update(point=point, racket_state='located')
            self.edit(update, dict(action='manual_racket_correction', side=side, point=point))
            self.after_racket_action(side, 'saved')
        elif self.box_mode:
            if self.corner is None:
                self.corner = np.asarray(point)
                self.message = 'Click the opposite BODY corner. Frame stays open until SPACE.'
                return
            a, b = np.minimum(self.corner, point), np.maximum(self.corner, point)
            box = [round(float(v), 2) for v in np.r_[a, b]]
            if not box_valid(box, self.meta['width'], self.meta['height']):
                self.corner = None
                raise ValueError('Invalid body rectangle. Click two opposite corners again.')
            self.edit(lambda: self.current['players'][side].update(bbox=box, body_state='pending'),
                      dict(action='manual_body_correction', side=side, bbox=box))
            self.corner = None
            self.box_mode = False
            self.message = side.upper()+' body drawn. P returns to racket mode; 1/2 changes player; SPACE accepts.'
        else:
            contains = lambda b: b is not None and b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]
            current = self.current['players'][side]['bbox']
            boxes = [b for b in self.task['candidates'] if contains(b)]
            chosen = current if contains(current) else min(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if boxes else None
            if chosen is None:
                self.message = 'No body box here. Press B and draw two corners.'
            else:
                self.edit(lambda: self.current['players'][side].update(bbox=copy.deepcopy(chosen), body_state='pending'),
                          dict(action='select_body_proposal', side=side))
                self.message = 'Body proposal selected. Check the person, then SPACE accepts the frame.'

    def after_racket_action(self, side, result):
        if side == 'near_side':
            self.active_side = 'far_side'
            self.mode = 'racket'
            self.box_mode = False
            self.corner = None
            self.message = 'NEAR racket '+result+'. FAR racket selected automatically: click its centre or H. 1/2 selects manually.'
        else:
            try:
                self.accept()
            except ValueError as error:
                self.message = 'Saved draft; NOT CONFIRMED: '+str(error)+'. Fix then SPACE, or edit FAR racket to retry.'

    def mark_unlocatable(self):
        side = self.active_side
        def update():
            if self.mode == 'racket':
                self.reference_provenance(side)
                self.current['players'][side].update(point=None, racket_state='unlocatable')
            else:
                self.current['players'][side].update(bbox=None, body_state='unlocatable')
        self.edit(update, dict(action=self.mode+'_unlocatable', side=side))
        self.corner = None
        self.box_mode = False
        if self.mode == 'racket':
            self.after_racket_action(side, 'marked UNLOCATABLE')
        else:
            self.message = side.upper()+' BODY marked UNLOCATABLE. 1/2 changes player; SPACE accepts.'

    def accept(self):
        row = copy.deepcopy(self.current)
        if any(p['racket_state'] == 'pending' for p in row['players'].values()):
            raise ValueError('Label BOTH rackets: 1/2 selects player; click centre or H if unlocatable.')
        both_unknown = all(p['racket_state'] == 'unlocatable' for p in row['players'].values())
        for side, p in row['players'].items():
            if p['body_state'] == 'unlocatable':
                continue
            if both_unknown:
                if p['body_state'] == 'pending':
                    p['body_state'] = 'not_required'
                continue
            if p['bbox'] is not None:
                p['body_state'] = 'verified'
            else:
                raise ValueError(side.upper()+' body missing. Select 1/2, B draws; H in BODY mode marks unlocatable.')
        row['decision'] = 'complete'
        self.store.save(self.identity, row)
        self.current = row
        self.advance()

    def advance(self):
        self.progress['reviewed'][self.identity] = digest(self.store.row(self.identity))
        save_json(self.progress_path, self.progress)
        status = 'EXCLUDED' if self.current['decision'] == 'excluded' else 'CONFIRMED'
        self.last_confirmation = f'{status}: {self.task["clip"]} frame {self.task["frame"]} - saved'
        pending = set(self.remaining())
        order = self.queue[self.position+1:]+self.queue[:self.position]
        following = next((k for k in order if k in pending), None)
        if following:
            self.position = self.queue.index(following)
            self.load()
            self.message = self.last_confirmation+'. Automatically moved to the next unconfirmed frame.'
        else:
            self.completed = True
            self.message = 'FULL REVIEW COMPLETE. Q quits. A/D still reaches every frame.'

    def handle_key(self, key):
        if ord('A') <= key <= ord('Z'):
            key += 32
        try:
            if key in (ord('1'), ord('2'), ord('n'), ord('f')):
                self.active_side = SIDES[0] if key in (ord('1'), ord('n')) else SIDES[1]
                self.mode = 'racket'
                self.box_mode = False
                self.corner = None
                self.completed = False
                self.message = self.active_side.upper()+' RACKET selected. Click its centre, or H if unlocatable. B edits its body.'
            elif key == ord('p'):
                self.mode = 'racket'; self.box_mode = False; self.corner = None
            elif key == ord('b'):
                self.mode = 'body'; self.box_mode = True; self.corner = None
                self.message = self.active_side.upper()+' BODY: click two opposite corners. SPACE saves the whole frame.'
            elif key == ord('h'):
                self.mark_unlocatable()
            elif key in (32, 10, 13):
                self.accept()
            elif key == ord('c'):
                side = self.active_side
                def clear():
                    if self.mode == 'racket':
                        self.reference_provenance(side)
                        self.current['players'][side].update(point=None, racket_state='pending')
                    else:
                        self.current['players'][side].update(bbox=None, body_state='pending')
                self.edit(clear, dict(action='clear_'+self.mode, side=side))
                self.corner = None
                self.message = 'Selected '+self.mode+' cleared. Redraw it or H marks it unlocatable.'
            elif key == 9:
                boxes = self.task['candidates']
                if boxes:
                    side = self.active_side
                    old = self.current['players'][side]['bbox']
                    i = next((i for i, b in enumerate(boxes) if b == old), -1)
                    self.edit(lambda: self.current['players'][side].update(
                        bbox=copy.deepcopy(boxes[(i+1) % len(boxes)]), body_state='pending'),
                        dict(action='cycle_body', side=side))
                    self.mode = 'body'; self.box_mode = False; self.corner = None
            elif key in (ord('a'), ord('d')):
                pos = self.position+(-1 if key == ord('a') else 1)
                if 0 <= pos < len(self.queue):
                    self.position = pos
                    self.load()
            elif key in (8, ord('u')):
                self.undo()
            elif key == ord('g'):
                self.show_candidates = not self.show_candidates
            elif key == ord('r'):
                self.zoom = 1.; self.offset[:] = 0
            elif key == ord('x'):
                self.edit(lambda: None, dict(action='exclude_ambiguous_full_review_frame'))
                self.current['decision'] = 'excluded'
                self.store.save(self.identity, self.current)
                self.advance()
        except ValueError as error:
            self.message = 'NOT CONFIRMED: '+str(error)

    def draw(self):
        canvas = FastLabeler.draw(self)
        canvas[self.view_height:] = 0
        status = self.review_status
        lines = [
            f'{status} | {self.position+1}/{len(self.queue)} | {len(self.remaining())} left | {self.task["clip"]} frame {self.task["frame"]} | {self.task["split"].upper()}',
            'LAST SAVE: '+(self.last_confirmation or 'No frame saved in this session yet.'),
            'NOW: '+self.active_side.upper()+' '+self.mode.upper()+' | BOTH racket points AND BOTH body boxes are editable',
            '1: NEAR player | 2: FAR player | Click: set selected racket centre | H: selected racket unlocatable',
            'B: draw selected BODY with two corners | P: return to racket-point mode | Tab: cycle body proposal',
            'FAR racket click/H: save and AUTO-NEXT if valid | SPACE / Enter: accept existing labels and AUTO-NEXT',
            'A/D: previous/next frame | C: clear selected point/box | U/Backspace: undo | X: exclude sample',
            'Check body boxes FIRST. NEAR click/H selects FAR; FAR click/H confirms. Invalid frames stay open.',
            'Wheel: zoom | R: reset | G: grey proposals | Q/Esc: save/quit | Original annotation files preserved',
            self.message[:180],
        ]
        for i, line in enumerate(lines):
            font = min(.43, (self.view_width-20)/max(1, cv2.getTextSize(line, 0, 1., 1)[0][0]))
            color = (235, 235, 235)
            if i == 0:
                color = (0, 255, 0) if status == 'CONFIRMED' else (0, 190, 255)
            elif i in (1, 2):
                color = (0, 255, 0) if i == 1 and self.last_confirmation else (235, 235, 235)
            cv2.putText(canvas, line, (9, self.view_height+20+i*22), 0, font, color, 1, cv2.LINE_AA)
        return canvas
