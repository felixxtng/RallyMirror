"""Corrections only, using the earlier explicit Space-to-accept box controls."""
import copy

import cv2

from .ownership_fast import SIDES
from .ownership_fast_labeler import FastLabeler
from .schema import digest, read_json, save_json


class CorrectionLabeler(FastLabeler):
    def __init__(self, root):
        super().__init__(root, self.first_correction(root))
        self.queue = [k for k, t in self.store.tasks.items() if t['kind'] == 'correction']
        self.progress_path = self.store.root/'corrections_review_progress.json'
        self.progress = read_json(self.progress_path) if self.progress_path.exists() else dict(reviewed={})
        remaining = self.remaining()
        self.position = self.queue.index(remaining[0]) if remaining else 0
        self.window = 'RallyMirror - FIX BOXES ONLY | racket points locked | SPACE saves'
        self.load()
        if not remaining:
            self.completed = True
            self.message = 'ALL CORRECTIONS REVIEWED. Q quits. A/D can revisit these six frames only.'

    @staticmethod
    def first_correction(root):
        from pathlib import Path
        m = read_json(Path(root)/'manifest.json')
        return next(t['id'] for t in m['tasks'] if t['kind'] == 'correction')

    def remaining(self):
        return [k for k in self.queue if self.progress['reviewed'].get(k) != digest(self.store.row(k))]

    def load(self):
        super().load()
        self.review_side = self.task.get('correction_side', 'far_side')
        self.completed = False
        self.message = self.task.get('reason', '')+' | Check BOTH owners; Space accepts.'

    @property
    def step(self):
        return None if self.completed else (self.review_side, 'body')

    def mutate(self, function, action):
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
        self.box_mode = False
        self.corner = None
        self.message = 'Box saved as draft. Check both players; SPACE accepts and advances.'

    def advance(self):
        self.progress['reviewed'][self.identity] = digest(self.store.row(self.identity))
        save_json(self.progress_path, self.progress)
        remaining = self.remaining()
        if remaining:
            self.position = self.queue.index(remaining[0])
            self.load()
        else:
            self.completed = True
            self.message = 'ALL CORRECTIONS REVIEWED. Q quits; this queue never opens new training frames.'

    def accept(self):
        row = copy.deepcopy(self.current)
        for p in row['players'].values():
            if p['bbox'] is None:
                raise ValueError('A body box is missing. 1/2 selects the body; B draws it, or H skips this sample.')
            p['body_state'] = 'verified'
        row['decision'] = 'complete'
        self.store.save(self.identity, row)
        self.current = row
        self.advance()

    def handle_key(self, key):
        if ord('A') <= key <= ord('Z'):
            key += 32
        try:
            if key in (32, 10, 13):
                if not self.completed:
                    self.accept()
            elif key in (ord('1'), ord('2'), ord('n'), ord('f')):
                self.review_side = SIDES[0] if key in (ord('1'), ord('n')) else SIDES[1]
                self.completed = False
                self.box_mode = False
                self.corner = None
                self.message = self.review_side.upper()+' BODY selected. B redraws; Tab cycles; Space accepts both.'
            elif key == ord('b'):
                self.completed = False
                self.box_mode = not self.box_mode
                self.corner = None
                self.message = 'Click two opposite BODY corners, then press SPACE.' if self.box_mode else 'Box selection mode.'
            elif key == 9:
                boxes = self.task['candidates']
                if boxes:
                    side = self.review_side
                    old = self.current['players'][side]['bbox']
                    i = next((i for i, b in enumerate(boxes) if b == old), -1)
                    self.completed = False
                    self.mutate(lambda: self.current['players'][side].update(
                        bbox=copy.deepcopy(boxes[(i+1) % len(boxes)]), body_state='pending'),
                        dict(action='cycle_body', side=side))
            elif key == ord('c'):
                side = self.review_side
                self.completed = False
                self.mutate(lambda: self.current['players'][side].update(bbox=None, body_state='pending'),
                            dict(action='clear_selected_body', side=side))
                self.show_candidates = False
            elif key in (ord('h'), ord('x')):
                self.current['decision'] = 'excluded'
                self.current['actions'].append(dict(action='skip_correction_without_erasing_points'))
                self.store.save(self.identity, self.current)
                self.advance()
            elif key in (ord('a'), ord('d')):
                new = self.position+(-1 if key == ord('a') else 1)
                if 0 <= new < len(self.queue):
                    self.position = new
                    self.load()
            elif key in (8, ord('u')):
                self.undo()
            elif key == ord('r'):
                self.zoom = 1.
                self.offset[:] = 0
            elif key == ord('g'):
                self.show_candidates = not self.show_candidates
        except ValueError as error:
            self.message = str(error)

    def draw(self):
        canvas = super().draw()
        canvas[self.view_height:] = 0
        lines = [
            f'FIXES ONLY | {self.position+1}/{len(self.queue)} | {len(self.remaining())} left | {self.task["clip"]} frame {self.task["frame"]} | {self.task["split"].upper()}',
            'Existing racket points LOCKED. Only correct/check which BODY holds each racket.',
            f'1: NEAR body | 2: FAR body | Selected: {self.review_side.upper()}',
            'B: redraw selected body with TWO corner clicks | Tab: another detected box | C: clear selected box',
            'SPACE / Enter: confirm both boxes, save and NEXT | Drawing never advances automatically',
            'A/D: previous/next correction | H/X: skip/exclude sample | U: undo last box edit',
            'Wheel: zoom | R: reset zoom | G: show/hide proposals | Q/Esc: save and quit',
            self.task.get('reason', ''),
            self.message,
        ]
        for i, line in enumerate(lines):
            font = min(.43, (self.view_width-20)/max(1, cv2.getTextSize(line, 0, 1., 1)[0][0]))
            cv2.putText(canvas, line, (9, self.view_height+20+i*25), 0, font,
                        (0, 255, 0) if i == 0 else (235, 235, 235), 1, cv2.LINE_AA)
        return canvas
