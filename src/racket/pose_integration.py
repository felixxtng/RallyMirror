"""Append a racket-centre keypoint to per-player pose records.

This is the single seam between the racket ownership model
(:class:`src.racket.real_owner_infer.RealOwnerRacketPredictor`) and the pose /
temporal data path. It owns the two conventions the rest of the system relies
on, so they live in exactly one place:

1. Keypoint name. The racket point is appended to each player's ``keypoints``
   list under the name ``RACKET_KEYPOINT_NAME`` (``"racket_center"``). This name
   must also be present in ``src.temporal.contracts.KEYPOINT_NAMES`` so the
   temporal feature builder keeps the point (anything not in that list is
   dropped by ``select_keypoints_by_name``).

2. Score + placeholder convention. The racket model returns ``located`` plus an
   ``(x, y)`` that is ``None`` when it could not place the racket. MediaPipe
   keypoints carry ``visibility``/``presence`` in [0, 1]; the racket point has
   neither, so we map ``located`` onto them:

     * located    -> x, y from the model;        visibility = presence = 1.0
     * not located-> x, y = the player's bbox CENTRE (a filled placeholder);
                     visibility = presence = 0.0

   A filled placeholder (not ``None``) is required because
   ``contracts.load_pose_jsonl`` normalizes every named keypoint against the
   bbox and would otherwise drop the whole record. bbox centre is a neutral
   in-box location; the 0.0 scores tell every downstream consumer (drawing,
   the model via the feature vector's scoreless coords, visibility gates) that
   the point is unknown this frame, exactly like a low-confidence MediaPipe
   landmark.

The racket model needs ALL tracked players for one frame at once (the scene crop
is the union of their boxes and each player's pose is validated against the
others), so :meth:`annotate` takes the whole per-frame player list, not one
player.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.racket.real_owner_infer import RealOwnerRacketPredictor

# The name the racket point is written under. Keep in lockstep with the entry
# appended to src.temporal.contracts.KEYPOINT_NAMES.
RACKET_KEYPOINT_NAME = "racket_center"


def _bbox_centre(bbox: List[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _racket_keypoint(racket: Dict[str, Any], bbox: List[float]) -> Dict[str, Any]:
    """Build the ``racket_center`` keypoint dict from a racket result.

    Located -> model x/y with scores 1.0; otherwise a bbox-centre placeholder
    with scores 0.0 (an explicit "unknown" the whole pipeline can read).
    """
    if racket.get("located") and racket.get("x") is not None and racket.get("y") is not None:
        return {
            "name": RACKET_KEYPOINT_NAME,
            "x": float(racket["x"]),
            "y": float(racket["y"]),
            "visibility": 1.0,
            "presence": 1.0,
        }
    cx, cy = _bbox_centre(bbox)
    return {
        "name": RACKET_KEYPOINT_NAME,
        "x": float(cx),
        "y": float(cy),
        "visibility": 0.0,
        "presence": 0.0,
    }


class RacketKeypointAnnotator:
    """Add a ``racket_center`` keypoint to each player's pose for a frame.

    Wraps :class:`RealOwnerRacketPredictor` and loads the torch model ONCE on
    construction; reuse the instance across the whole video.
    """

    def __init__(self, weights: str = "models/racket_center_v006.pt",
                 device: str = "cpu"):
        self.predictor = RealOwnerRacketPredictor(weights, device=device)

    def annotate(self, frame, players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return NEW player dicts whose ``keypoints`` end with ``racket_center``.

        ``players`` is the list of this frame's tracked players, each a dict with
        at least ``bbox`` ([x1, y1, x2, y2], full-frame pixels) and ``keypoints``
        (MediaPipe-style list of ``{name, x, y, visibility, presence}`` in
        full-frame pixels). All other fields (``track_id`` etc.) pass through.

        The returned ``keypoints`` are the input keypoints with one extra entry
        appended, so the body-keypoint order is unchanged and the racket point is
        always last.
        """
        if not players:
            return []
        predicted = self.predictor.predict_players(frame, players)
        out: List[Dict[str, Any]] = []
        for player in predicted:
            record = dict(player)
            racket = record.pop("racket")
            base_kps = list(record.get("keypoints") or [])
            record["keypoints"] = base_kps + [_racket_keypoint(racket, record["bbox"])]
            out.append(record)
        return out
