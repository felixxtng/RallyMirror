"""Review the sampled frames' body proposals before collecting racket points."""
from pathlib import Path
import copy
import csv

import cv2

from .ownership import iou
from .ownership_fast import SampleStore, SIDES, steps
from .ownership_fast_labeler import FastLabeler
from .ownership_fix_labeler import CorrectionLabeler
from .schema import digest, file_hash, read_json, save_json


def body_digest(row):
    return digest({s: {k: row['players'][s][k] for k in ('bbox', 'body_state')} for s in SIDES})


def proposal_risk(task, row, meta):
    """Only review-priority heuristics, never automatic ground truth or ownership."""
    w, h = meta['width'], meta['height']
    near, far = [row['players'][s]['bbox'] for s in SIDES]
    scores, reasons = {s: 0 for s in SIDES}, []
    for side, box in zip(SIDES, (near, far)):
        if box is None:
            scores[side] += 6
            reasons.append(side+': missing body proposal')
            continue
        x, y = (box[0]+box[2])/2/w, box[3]/h
        height = (box[3]-box[1])/h
        # Deliberately broad: camera-angle exceptions are manually checked, not rejected.
        if (side == 'far_side' and not .24 <= x <= .76) or not .10 <= x <= .90:
            scores[side] += 10
            reasons.append(side+': outside central playing area; possible official')
        if height < .055:
            scores[side] += 5
            reasons.append(side+': unusually small body')
        if height > .65:
            scores[side] += 5
            reasons.append(side+': large close-up / possible camera cut')
        if side == 'far_side' and y > .72:
            scores[side] += 7
            reasons.append(side+': unusually low far-player box')
    if near and far:
        if iou(near, far) > .45:
            scores['far_side'] += 12
            reasons.append('overlapping body proposals: possible same person twice')
        if far[3] >= near[3]:
            scores['far_side'] += 10
            reasons.append('far box ends below near box: possible swapped owners')
        if far[3]-far[1] > .95*(near[3]-near[1]):
            scores['far_side'] += 4
            reasons.append('far body is as large as near body: verify owner')
    selected = max(SIDES, key=lambda s: scores[s])
    return dict(score=sum(scores.values()), selected=selected,
                reasons=reasons or ['No geometric warning; still verify both players.'])


def prepare_box_review(store):
    path = store.root/'sampled_box_review.json'
    if path.exists():
        plan = read_json(path)
        if plan['manifest_sha256'] != file_hash(store.root/'manifest.json'):
            raise ValueError('Sampled frame manifest changed; preserve and reconcile this box-review plan')
        if set(plan['ids']) != {k for k, t in store.tasks.items() if t['kind'] == 'new'}:
            raise ValueError('Box queue no longer matches the sampled frames')
        return plan
    risks, preserved = {}, {}
    for identity, task in store.tasks.items():
        if task['kind'] != 'new':
            continue
        row = store.row(identity)
        risks[identity] = proposal_risk(task, row, store.clips[task['clip']]['meta'])
        if row['decision'] == 'excluded' or all(p['body_state'] in ('verified', 'unlocatable') for p in row['players'].values()):
            preserved[identity] = body_digest(row)
    ids = sorted(risks, key=lambda k: (-risks[k]['score'], store.tasks[k]['clip'], store.tasks[k]['frame']))
    plan = dict(format='sampled_body_review_v001', manifest_sha256=file_hash(store.root/'manifest.json'),
                ids=ids, risks=risks, preserved_body_hashes=preserved,
                note='All NEW sampled frames; geometric suspects first, not confirmed errors. Six historical corrections excluded.')
    save_json(path, plan)
    with (store.root/'sampled_box_review.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'clip', 'frame', 'priority_score', 'reason', 'already_verified'])
        for k in ids:
            t, r = store.tasks[k], risks[k]
            writer.writerow([k, t['clip'], t['frame'], r['score'], '; '.join(r['reasons']), k in preserved])
    return plan


class SampledBoxLabeler(CorrectionLabeler):
    def __init__(self, root):
        root = Path(root)
        m = read_json(root/'manifest.json')
        first = next(t['id'] for t in m['tasks'] if t['kind'] == 'new')
        FastLabeler.__init__(self, root, first)
        self.plan = prepare_box_review(self.store)
        self.queue = self.plan['ids']
        self.progress_path = self.store.root/'sampled_box_review_progress.json'
        self.progress = read_json(self.progress_path) if self.progress_path.exists() else dict(reviewed={})
        remaining = self.remaining()
        self.position = self.queue.index(remaining[0]) if remaining else 0
        self.window = 'RallyMirror - SAMPLED BODY BOXES ONLY | 1/2 B SPACE | no racket clicks'
        self.load()
        if not remaining:
            self.completed = True
            self.message = 'ALL SAMPLED BOXES REVIEWED. Q quits. Racket annotation is a separate command.'

    def load(self):
        super().load()
        if hasattr(self, 'plan'):
            risk = self.plan['risks'][self.identity]
            self.review_side = risk['selected']
            self.message = 'PRIORITY CHECK: '+'; '.join(risk['reasons'])

    def remaining(self):
        result = []
        for k in self.queue:
            row = self.store.row(k)
            if row['decision'] == 'excluded':
                continue
            actual = body_digest(row)
            if actual not in (self.progress['reviewed'].get(k), self.plan['preserved_body_hashes'].get(k)):
                result.append(k)
        return result

    def accept(self):
        row = copy.deepcopy(self.current)
        for p in row['players'].values():
            if p['body_state'] == 'unlocatable':
                continue
            if p['bbox'] is None:
                raise ValueError('Missing body: 1/2 selects it, B draws, or H marks that body unlocatable. Then Space.')
            p['body_state'] = 'verified'
        # Body review must not pretend that unlabelled racket points are complete.
        row['decision'] = 'pending'
        if not steps(row):
            row['decision'] = 'complete'
        self.store.save(self.identity, row)
        self.current = row
        self.advance()

    def advance(self):
        self.progress['reviewed'][self.identity] = body_digest(self.store.row(self.identity))
        save_json(self.progress_path, self.progress)
        remaining = self.remaining()
        if remaining:
            self.position = self.queue.index(remaining[0])
            self.load()
        else:
            self.completed = True
            self.message = 'ALL SAMPLED BOXES REVIEWED. Q quits. No racket points requested in this pass.'

    def handle_key(self, key):
        if ord('A') <= key <= ord('Z'):
            key += 32
        if key == ord('h'):
            side = self.review_side
            self.completed = False
            self.mutate(lambda: self.current['players'][side].update(bbox=None, body_state='unlocatable'),
                        dict(action='body_unlocatable', side=side))
            self.message = side.upper()+' body marked UNLOCATABLE. Check the other body; SPACE saves and advances.'
        else:
            super().handle_key(key)

    def draw(self):
        canvas = FastLabeler.draw(self)
        canvas[self.view_height:] = 0
        risk = self.plan['risks'][self.identity]
        lines = [
            f'SAMPLED BOX REVIEW ONLY | {self.position+1}/{len(self.queue)} | {len(self.remaining())} left | {self.task["clip"]} frame {self.task["frame"]}',
            'Check the CYAN near-player body and YELLOW far-player body. No racket clicks in this pass.',
            f'1: NEAR body | 2: FAR body | Selected: {self.review_side.upper()} | priority score {risk["score"]}',
            'B: redraw selected box with two corners | Tab: another proposal | C: clear selected box',
            'SPACE / Enter: verify both boxes, save and NEXT | A/D: previous/next',
            'H: selected BODY unlocatable (then Space) | X: exclude entire sample | U: undo',
            'Wheel: zoom | R: reset | G: show/hide grey proposals | Q/Esc: save/quit',
            ('Heuristic warning, NOT a confirmed error: '+'; '.join(risk['reasons']))[:180],
            self.message[:180],
        ]
        for i, line in enumerate(lines):
            font = min(.43, (self.view_width-20)/max(1, cv2.getTextSize(line, 0, 1., 1)[0][0]))
            cv2.putText(canvas, line, (9, self.view_height+20+i*25), 0, font,
                        (0, 255, 0) if i == 0 else (235, 235, 235), 1, cv2.LINE_AA)
        return canvas
