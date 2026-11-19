"""Local three-point annotation with autosave, zoom, keyframes and proposal review."""
import argparse
import copy
from pathlib import Path

import cv2
import numpy as np

from .interpolate import review_queue
from .schema import CENTER_NAMES, approved, blank, copy_annotation, load_clip, point_errors, save_json
from .source import FrameSource


class Labeler:
    def __init__(self, clip, interval=10, review=False, start=0, end=None, proposals_only=False, auto_advance=False, player=None):
        self.clip = Path(clip)
        self.meta, self.annotations = load_clip(clip)
        self.generic_player = self.meta.get("annotation_mode") == "selected_racket"
        self.dual = self.meta.get("schema_version") == 2
        self.only_player = player
        if player is not None and (not self.dual or player not in ("player_1", "player_2")):
            raise ValueError("--player requires a two-player head-centre clip and player_1 or player_2")
        self.player = player or self.annotations.get("session", {}).get("player_id", "player_1")
        if self.dual:
            from .players import PLAYER_IDS, PlayerRows
            self.players = PLAYER_IDS
            if self.player not in self.players:
                self.player = self.players[0]
            self.rows = PlayerRows(self.annotations["frames"], self.player)
        else:
            self.rows = self.annotations["frames"]
        self.names = self.meta["keypoint_names"]
        self.proposals_only, self.auto_advance = proposals_only, auto_advance
        self.source = FrameSource(self.meta)
        self.interval, self.review = interval, review
        self.start = start
        self.end = self.meta["frame_count"] if end is None else end
        if not 0 <= self.start < self.end <= self.meta["frame_count"]:
            self.source.close()
            raise ValueError("Require 0 <= start-frame < end-frame <= clip frame count")
        self.keyframes = sorted(set(range(self.start, self.end, interval)) | {self.end-1})
        candidates = self.review_frames() if review else [f for f in self.keyframes if not self.frame_done(f)]
        if review and not candidates:
            self.source.close()
            raise ValueError("No pending review frames in this range. All proposals may already be reviewed.")
        session = self.annotations.get("session", {})
        saved_frame = session.get("frame")
        self.index = saved_frame if session.get("review") == review and saved_frame in candidates else (candidates or [self.start])[0]
        if self.dual:
            self.select_pending_player()
        missing = [i for i, p in enumerate(self.row()["keypoints"]) if p["v"] is None]
        self.active = (missing or [0])[0]
        self.zoom = 1.0
        self.offset = np.array([0.0, 0.0])
        self.cached = None
        self.image = None
        self.mouse = (0, 0)
        self.message = "Annotate the FOREGROUND player's racket" if len(self.names) == 3 else "Click the FOREGROUND racket's STRING-BED CENTRE (not hand or box centre)"
        if self.generic_player:
            self.message = "Click one visible racket's STRING-BED CENTRE; H if it cannot be located reliably"
        self.window = "RallyMirror - Player racket centre" if self.generic_player else "RallyMirror - Racket keypoints"
        self.view_width, self.view_height = 1100, 640
        self.scale = min(self.view_width/self.meta["width"], self.view_height/self.meta["height"])

    def new_blank(self, frame):
        if self.generic_player:
            from .selected import blank_selected
            return blank_selected(frame)
        if self.dual:
            from .players import blank_player
            return blank_player(frame, self.player)
        return blank(frame, self.names)

    def frame_done(self, frame):
        if self.dual and not self.only_player:
            from .players import blank_frame, frame_approved
            return frame_approved(self.annotations["frames"].get(str(frame), blank_frame(frame)), self.meta)
        return approved(self.rows.get(str(frame), self.new_blank(frame)), self.meta, self.rows)

    def select_player(self, player):
        if self.only_player and player != self.only_player:
            return
        self.player = player
        self.rows.player = player
        self.active = 0

    def select_pending_player(self):
        from .players import PlayerRows, blank_player
        if self.only_player:
            self.select_player(self.only_player)
            return not approved(self.row(), self.meta, self.rows)
        for player in (self.player,) + tuple(p for p in self.players if p != self.player):
            rows = PlayerRows(self.annotations["frames"], player)
            row = rows.get(str(self.index), blank_player(self.index, player))
            if not approved(row, self.meta, rows) and (not self.proposals_only or "proposal" in row):
                self.select_player(player)
                return True
        return False

    def after_approval(self):
        if not self.auto_advance:
            return
        if (self.only_player or self.generic_player) and not self.review:
            pending = [f for f in self.keyframes if f > self.index and not self.frame_done(f)]
            if pending:
                self.index = pending[0]
            else:
                self.message = "No more unfinished frames ahead for this player. Q to save/quit; A to inspect earlier frames."
            self.save()
        elif self.dual and not self.only_player and self.select_pending_player():
            self.save()
        else:
            self.move(1, True)

    def review_frames(self):
        if self.dual:
            from .players import PlayerRows, blank_player
            frames = set()
            for player in ((self.only_player,) if self.only_player else self.players):
                rows = PlayerRows(self.annotations["frames"], player)
                for frame in range(self.start, self.end):
                    row = rows.get(str(frame), blank_player(frame, player))
                    if not approved(row, self.meta, rows) and (not self.proposals_only or "proposal" in row):
                        frames.add(frame)
            return sorted(frames)
        return [r["frame"] for r in review_queue(self.meta, self.annotations)
                if self.start <= r["frame"] < self.end and
                (not self.proposals_only or "proposal" in self.rows.get(str(r["frame"]), {}))]

    def row(self):
        return self.rows.get(str(self.index), self.new_blank(self.index))

    def save(self):
        self.annotations["session"] = {"frame": self.index, "review": self.review}
        if self.dual:
            self.annotations["session"]["player_id"] = self.player
            self.annotations["session"]["only_player"] = self.only_player
        save_json(self.clip / "annotations.json", self.annotations)

    def commit(self, row):
        if self.generic_player:
            previous = self.rows.get(str(self.index))
            row.update(label="player", racket_id="selected_racket")
            for key, default in (("player_id", None), ("track_id", None), ("player_bbox", None), ("association_status", "unassigned")):
                row[key] = previous.get(key, default) if previous else default
            # Crop candidates belong to this frame. Preserve them through N/U
            # and point corrections, rather than dropping the trial review entry.
            if previous and "crop_proposal" in previous:
                row["crop_proposal"] = copy.deepcopy(previous["crop_proposal"])
            else:
                row.pop("crop_proposal", None)
            # A changed coordinate may now refer to the other player's racket.
            # Never silently reuse a previously verified crop for that change.
            if previous and previous.get("keypoints") != row["keypoints"]:
                row.update(player_id=None, track_id=None, player_bbox=None, association_status="unassigned")
                if "crop_proposal" in row:
                    row["crop_proposal"]["decision"] = "pending"
        if self.dual:
            previous = self.rows.get(str(self.index))
            row.update(player_id=self.player, racket_id=f"{self.player}_racket")
            # N/U/copy must not erase current-frame tracker associations or copy
            # a previous frame's box into this frame.
            for key in ("track_id", "player_bbox"):
                row[key] = previous.get(key) if previous else None
        self.rows[str(self.index)] = row
        self.save()

    def edit(self):
        row = copy.deepcopy(self.row())
        if row.get("provenance") in {"copied", "interpolated", "model"}:
            row["needs_confirmation"] = True
        row.update(provenance="manual", approved=False, state="draft")
        row.pop("anchors", None)
        row.pop("review_reasons", None)
        return row

    def finish_point(self, row):
        if all(p["v"] is not None for p in row["keypoints"]):
            row["state"] = "visible" if any(p["v"] for p in row["keypoints"]) else "invisible"
            if self.names == CENTER_NAMES and not any(p["v"] for p in row["keypoints"]):
                row["state"] = "unavailable"
            row["approved"] = (not row.get("needs_confirmation", False)
                               and not point_errors(row, self.meta["width"], self.meta["height"], self.names))
        self.commit(row)
        remaining = [i for i, p in enumerate(row["keypoints"]) if p["v"] is None]
        self.active = remaining[0] if remaining else (self.active+1) % len(self.names)
        if row["approved"]:
            self.after_approval()

    def move(self, direction, keyframes=False):
        if self.review:
            frames = self.review_frames()
        else:
            frames = self.keyframes if keyframes else list(range(self.start, self.end))
        available = [f for f in frames if (f-self.index)*direction > 0]
        if available:
            self.index = min(available) if direction > 0 else max(available)
        else:
            self.message = "No more frames in this direction"
        self.active = 0
        if self.dual:
            self.select_pending_player()
        self.save()

    def mouse_event(self, event, x, y, flags, parameter):
        if not (0 <= x < self.view_width and 0 <= y < self.view_height):
            return
        point = np.array([x, y])/(self.scale*self.zoom) + self.offset
        self.mouse = (x, y)
        if event == cv2.EVENT_MOUSEWHEEL:
            new_zoom = min(8.0, max(1.0, self.zoom * (1.4 if flags > 0 else 1/1.4)))
            self.offset = point - np.array([x, y])/(self.scale*new_zoom)
            self.zoom = new_zoom
            if self.zoom == 1:
                self.offset[:] = 0
        elif event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = point
            if 0 <= ox < self.meta["width"] and 0 <= oy < self.meta["height"]:
                row = self.edit()
                row["keypoints"][self.active].update(x=round(float(ox), 2), y=round(float(oy), 2), v=2)
                self.finish_point(row)

    def draw(self):
        if self.cached != self.index:
            self.image = self.source.read(self.index)
            self.cached = self.index
        transform = np.array([[self.scale*self.zoom, 0, -self.offset[0]*self.scale*self.zoom],
                              [0, self.scale*self.zoom, -self.offset[1]*self.scale*self.zoom]])
        image = cv2.warpAffine(self.image, transform, (self.view_width, self.view_height))
        row = self.row()
        colors = [(0, 100, 255), (0, 255, 255), (255, 160, 0)]
        locations = []
        points = row["keypoints"]
        if self.dual:
            from .players import blank_frame
            frame = self.annotations["frames"].get(str(self.index), blank_frame(self.index))
            points = [frame["rackets"][player]["keypoints"][0] for player in self.players]
        for i, p in enumerate(points):
            location = None if not p["v"] else tuple(np.round((np.array([p["x"], p["y"]])-self.offset)*self.scale*self.zoom).astype(int))
            locations.append(location)
            if location is not None:
                cv2.circle(image, location, 5, colors[i], -1 if p["v"] == 2 else 1)
                label = "player" if self.generic_player else self.players[i] if self.dual else str(i+1)
                cv2.putText(image, label, (location[0]+7, location[1]-7), cv2.FONT_HERSHEY_SIMPLEX, .5, colors[i], 1)
                if self.dual and self.players[i] == self.player:
                    cv2.circle(image, location, 10, (255, 255, 255), 1)
        for a, b in ([] if self.dual else zip(range(len(locations)-1), range(1, len(locations)))):
            if locations[a] is not None and locations[b] is not None:
                cv2.line(image, locations[a], locations[b], (0, 220, 0), 1)
        canvas = np.zeros((self.view_height+190, self.view_width, 3), dtype=np.uint8)
        canvas[:self.view_height] = image
        accepted = approved(row, self.meta, self.rows)
        proposal = row.get("proposal", {})
        progress = sum(approved(r, self.meta, self.rows) for r in self.rows.values())
        lines = [f"Frame {self.index}/{self.meta['frame_count']-1} | source {self.index+self.meta['start_frame']} | {row['state']} | {row['provenance']} | {'APPROVED' if accepted else 'NEEDS REVIEW'}",
                 f"Active {self.active+1}: {row['keypoints'][self.active]['name']} | interval {self.interval} | {'REVIEW QUEUE' if self.review else 'KEYFRAMES'} | zoom {self.zoom:.1f}x",
                 "Click: visible point | 1/2/3: select | H: point unlocatable | O: toggle visible/occluded | N: whole racket invisible",
                 "A/D: prev/next frame | J/L: prev/next keyframe | C: copy preceding labelled frame | Enter: accept | U: clear frame",
                 "Mouse wheel: zoom at cursor | R: reset zoom | Q/Esc: save and quit | edits autosave; no automatic advance",
                 self.message]
        if len(self.names) == 1:
            lines[2] = "Click: head centre | H: cannot locate (excluded from training) | N: head absent (negative example)"
            lines[3] = "A/D or J/L: previous/next | Enter: approve | U: clear | O: occluded but location known"
            lines[1] = f"HEAD CENTRE ONLY | reviewed {progress}/{self.meta['frame_count']} | {'MODEL REVIEW' if self.review else 'MANUAL'} | zoom {self.zoom:.1f}x"
            if proposal and not accepted:
                lines[-1] = f"Model score {proposal['score']:.3f} (not calibrated) | {proposal['reason']} | Correct/click then Enter; H if uncertain"
        if self.generic_player:
            lines[1] = f"PLAYER | reviewed {progress}/{self.meta['frame_count']} | one racket centre per frame | zoom {self.zoom:.1f}x"
            lines[2] = "Click: one visible racket's string-bed centre -> save/next frame | H or N: cannot label -> skip/exclude"
            lines[3] = "A/D: previous/next frame | U: clear | Enter: approve current point | R: reset zoom"
            lines[4] = "Q/Esc: save and quit | one click per frame | no player numbers | no interpolation"
            if not proposal or accepted:
                lines[-1] = self.message if self.message.startswith("No more") else "Label one clearly visible racket. This does not claim which person owns it."
        if self.dual:
            statuses = " | ".join(f"{p}: {frame['rackets'][p]['state']}{' OK' if approved(frame['rackets'][p], self.meta, {}) else ' PENDING'}" for p in self.players)
            lines[1] = f"ACTIVE {self.player} | 1/2 or Tab: switch player | {statuses}"
            lines[2] = "Click: active player's head centre | H: unknown/exclude | N: absent | Enter: accept selected player"
            lines[4] = "Auto advance: finish both players, then next frame | R: reset zoom | Q/Esc: save/quit"
            if not proposal or accepted:
                lines[-1] = "Player 1 starts NEAR; player 2 starts FAR. Follow identities; do not swap on movement."
            if self.only_player:
                lines[1] = f"{self.player.upper()} ONLY | reviewed {progress}/{self.meta['frame_count']} | other player's labels are untouched"
                lines[2] = "Click: this player's string-bed centre | H: cannot locate | N: definitely absent | A/D: inspect frames"
                lines[4] = "Auto advance after this player's label; no player switching | R: reset zoom | Q/Esc: save/quit"
                if not proposal or accepted:
                    lines[-1] = self.message if self.message.startswith("No more") else "Follow the same player throughout the clip. No interpolation."
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (10, self.view_height+25+i*28), cv2.FONT_HERSHEY_SIMPLEX, .47, (240, 240, 240), 1)
        return canvas

    def handle_key(self, key):
        if self.dual and key in (ord("1"), ord("2"), 9):
            if self.only_player:
                self.message = f"Fixed to {self.only_player}. Reopen with a different --player to label the other player."
                return
            self.select_player(self.players[1-self.players.index(self.player)] if key == 9 else self.players[key-ord("1")])
            self.save()
        elif key in (ord("1"), ord("2"), ord("3")):
            if key-ord("1") < len(self.names):
                self.active = key-ord("1")
        elif key in (ord("a"), ord("d"), ord("j"), ord("l")):
            self.move(-1 if key in (ord("a"), ord("j")) else 1, key in (ord("j"), ord("l")))
        elif key == ord("h"):
            row = self.edit()
            row["keypoints"][self.active].update(x=None, y=None, v=0)
            self.finish_point(row)
        elif key == ord("o"):
            row = self.edit()
            p = row["keypoints"][self.active]
            if p["v"] in (1, 2):
                p["v"] = 3-p["v"]
                self.finish_point(row)
        elif key == ord("n"):
            previous_proposal = self.row().get("proposal")
            row = self.new_blank(self.index)
            if previous_proposal:
                row["proposal"] = copy.deepcopy(previous_proposal)
            for p in row["keypoints"]:
                p["v"] = 0
            row.update(state="unavailable" if self.generic_player else "invisible", approved=True)
            self.commit(row)
            self.after_approval()
        elif key == ord("u"):
            self.commit(self.new_blank(self.index))
        elif key == ord("c"):
            candidates = [int(f) for f, r in self.rows.items() if int(f) < self.index
                          and r["state"] in ("visible", "invisible")]
            if candidates:
                self.commit(copy_annotation(self.rows[str(max(candidates))], self.index))
                self.message = "Copied preceding annotation; correct points or press Enter to accept"
        elif key in (10, 13):
            row = copy.deepcopy(self.row())
            errors = point_errors(row, self.meta["width"], self.meta["height"], self.names)
            if row.get("provenance") == "model" and row.get("state") == "unavailable":
                errors.append("Model has no reliable centre: click it, H to exclude, or N to confirm absence")
            from .schema import anchors_current
            if not anchors_current(row, self.rows):
                errors.append("Stale interpolation: regenerate or manually correct")
            if errors:
                self.message = "; ".join(errors)
            else:
                row["approved"] = True
                row.pop("needs_confirmation", None)
                self.commit(row)
                self.message = "Accepted; move forward to continue review"
                self.after_approval()
        elif key == ord("r"):
            self.zoom, self.offset = 1.0, np.zeros(2)

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.mouse_event)
        try:
            while True:
                cv2.imshow(self.window, self.draw())
                key = cv2.waitKey(30) & 0xff
                if key in (ord("q"), 27) or cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                self.handle_key(key)
        finally:
            self.save()
            self.source.close()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="Exclusive local-frame endpoint")
    parser.add_argument("--proposals-only", action="store_true", help="Review only pending model predictions, not every unfinished frame")
    parser.add_argument("--auto-advance", action="store_true", help="Advance after approval or a completed manual annotation")
    parser.add_argument("--player", choices=["player_1", "player_2"], help="Label only this player; auto-advance without waiting for the other racket")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("Interval must be positive")
    if args.proposals_only and not args.review:
        parser.error("--proposals-only requires --review")
    Labeler(args.clip, args.interval, args.review, args.start_frame, args.end_frame,
            args.proposals_only, args.auto_advance, args.player).run()


if __name__ == "__main__":
    main()
