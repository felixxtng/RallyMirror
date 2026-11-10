"""Target selection for player tracking.

Owned by the detection module. Turns the raw per-frame track list from
``PlayerTracker.track_people`` into the subset we actually want to feed
downstream (pose estimation -> temporal phase recognition).

The core problem this guards against: naive selection can lock onto a
*spectator* or anyone closer to the camera than a player, whose pose then
pollutes the temporal sequence and churns ``player_id``. To contain that, the
selection runs as an explicit filter *pipeline*:

    tracks
      -> confidence gate        (drop flickery / low-confidence detections)
      -> geometry sanity        (drop boxes with implausible size / aspect)
      -> court ROI              (keep only people standing on the court)  [WIP]
      -> rank + take top N      (singles: both players -> N = 2)

Each stage is independent and config-driven, so stages can be tuned (or turned
off) without touching the others. The court-ROI stage is the real spectator
fix; it depends on Person 3's court-coordinate detection, which is still WIP, so
it ships here as a *pass-through seam*: provide a ``court_polygon`` and it
filters on the bbox bottom-centre; omit it and the stage is a no-op. When
Person 3 pushes, only ``_court_roi_filter`` / the polygon wiring changes.

Defaults are intentionally permissive so dropping this in does not change
current behaviour until the thresholds are tuned (suggestions 2-4). ``N`` still
defaults to 1 via ``select_target`` for the existing single-target call sites;
``select_targets`` is the new multi-player entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

Track = Dict[str, Any]
Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SelectionConfig:
    """Tunable knobs for the selection pipeline.

    Defaults are permissive: with these values the pipeline reduces to "rank by
    bbox bottom edge and take the top ``max_targets``", matching the historic
    single-target behaviour when ``max_targets == 1``. Tighten these to enable
    the spectator safeguards (suggestions 2-4).
    """

    # How many players to return. Singles with both ends covered -> 2.
    max_targets: int = 2

    # (2) Confidence gate. Tracks below this are dropped before ranking.
    # 0.0 disables the gate.
    min_confidence: float = 0.0

    # (3) Geometry sanity. Bounds are in pixels / ratio; None disables a check.
    # A standing player's box is taller than it is wide and occupies a
    # plausible slice of the frame. Spectators (seated, partial, or very close
    # to camera) tend to fall outside these bands.
    min_bbox_height: Optional[float] = None       # px
    max_bbox_height: Optional[float] = None        # px
    min_aspect_ratio: Optional[float] = None       # height / width, lower bound
    max_aspect_ratio: Optional[float] = None       # height / width, upper bound

    # (4) Court ROI. Polygon of (x, y) image points enclosing the playing area.
    # When set, a track is kept only if its bbox bottom-centre is inside it.
    # Left None until Person 3's court detection is wired in.
    court_polygon: Optional[Sequence[Point]] = None

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "SelectionConfig":
        """Build a ``SelectionConfig`` from a loaded RallyMirror config dict.

        Reads ``config["detection"]["target_selection"]`` and overrides the
        dataclass defaults with whatever keys are present; unknown keys are
        ignored and missing keys keep their default. Passing ``None`` loads
        ``configs/default.yaml`` via the shared loader. A missing section
        yields plain defaults, so this never raises just because the YAML has
        not been updated yet.
        """
        if config is None:
            # Imported lazily so the detection module has no hard dependency on
            # the temporal package just to construct a default config.
            from src.temporal.config import load_config

            config = load_config()

        section = (config.get("detection") or {}).get("target_selection") or {}

        known = {f.name for f in fields(cls)}
        overrides = {k: v for k, v in section.items() if k in known}

        # Normalize a polygon from nested lists (YAML) into (x, y) tuples.
        poly = overrides.get("court_polygon")
        if poly:
            overrides["court_polygon"] = [tuple(pt) for pt in poly]

        return cls(**overrides)


DEFAULT_CONFIG = SelectionConfig()


def load_selection_config(config: Optional[Dict[str, Any]] = None) -> SelectionConfig:
    """Convenience: build the configured :class:`SelectionConfig`.

    Thin wrapper over :meth:`SelectionConfig.from_config` for call sites that
    just want "the selection config the YAML describes".
    """
    return SelectionConfig.from_config(config)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_bottom_centre(bbox: Sequence[float]) -> Point:
    """Return the bottom-centre point ``((x1 + x2) / 2, y2)`` of a bbox.

    This is the ground-contact proxy the court-ROI test uses: where the player
    is standing, rather than where the whole box sits. Person 3's ROI filter
    keys off exactly this point.
    """
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test. ``polygon`` is an ordered list of
    (x, y) vertices. Returns True for points strictly inside; boundary points
    are treated inclusively enough for our pixel-scale use.
    """
    if polygon is None or len(polygon) < 3:
        # No usable polygon -> treat everything as inside (pass-through).
        return True
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _confidence_gate(tracks: List[Track], cfg: SelectionConfig) -> List[Track]:
    """(2) Drop tracks whose confidence is below ``cfg.min_confidence``.

    Tracks without a ``confidence`` field are kept (we cannot judge them).
    """
    if cfg.min_confidence <= 0.0:
        return tracks
    return [
        t for t in tracks
        if t.get("confidence") is None or t["confidence"] >= cfg.min_confidence
    ]


def _geometry_filter(tracks: List[Track], cfg: SelectionConfig) -> List[Track]:
    """(3) Drop tracks whose bbox size / aspect is implausible for a player."""
    checks = (
        cfg.min_bbox_height, cfg.max_bbox_height,
        cfg.min_aspect_ratio, cfg.max_aspect_ratio,
    )
    if all(c is None for c in checks):
        return tracks

    kept: List[Track] = []
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        aspect = height / width
        if cfg.min_bbox_height is not None and height < cfg.min_bbox_height:
            continue
        if cfg.max_bbox_height is not None and height > cfg.max_bbox_height:
            continue
        if cfg.min_aspect_ratio is not None and aspect < cfg.min_aspect_ratio:
            continue
        if cfg.max_aspect_ratio is not None and aspect > cfg.max_aspect_ratio:
            continue
        kept.append(t)
    return kept


def _court_roi_filter(tracks: List[Track], cfg: SelectionConfig) -> List[Track]:
    """(4) Keep only tracks standing inside the court polygon. [WIP seam]

    This is the primary spectator safeguard. Until Person 3's court-coordinate
    detection is wired in, ``cfg.court_polygon`` is ``None`` and this stage is a
    pass-through. Once a polygon is supplied, a track survives iff its bbox
    bottom-centre (``bbox_bottom_centre``) lies within the polygon.
    """
    polygon = cfg.court_polygon
    if not polygon:
        return tracks
    return [t for t in tracks if point_in_polygon(bbox_bottom_centre(t["bbox"]), polygon)]


def _rank(tracks: List[Track]) -> List[Track]:
    """Rank surviving tracks by how far down the frame they stand.

    Bbox bottom edge (``y2``) is the historic proxy for "nearest the camera /
    on the near court"; ranking descending keeps continuity with the previous
    ``max(..., key=bbox[3])`` behaviour while letting us take the top N.
    """
    return sorted(tracks, key=lambda t: t["bbox"][3], reverse=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_targets(
    tracks: Optional[Sequence[Track]],
    config: Optional[SelectionConfig] = None,
) -> List[Track]:
    """Run the selection pipeline and return up to ``config.max_targets`` tracks.

    The returned list is ranked (strongest candidate first) and may be empty.
    This is the multi-player entry point; singles with both players covered uses
    the default ``max_targets == 2``.
    """
    if not tracks:
        return []
    cfg = config or DEFAULT_CONFIG

    stage: List[Track] = list(tracks)
    stage = _confidence_gate(stage, cfg)
    stage = _geometry_filter(stage, cfg)
    stage = _court_roi_filter(stage, cfg)
    stage = _rank(stage)

    if cfg.max_targets is not None and cfg.max_targets >= 0:
        stage = stage[: cfg.max_targets]
    return stage


def select_target(
    tracks: Optional[Sequence[Track]],
    config: Optional[SelectionConfig] = None,
) -> Optional[Track]:
    """Back-compat single-target entry point.

    Returns the single best candidate (or ``None``) by running the full
    pipeline and taking the top-ranked survivor. Existing single-player call
    sites keep working unchanged.
    """
    cfg = config or DEFAULT_CONFIG
    if cfg.max_targets != 1:
        # Don't mutate the caller's config; just cap to one for this call.
        cfg = SelectionConfig(**{**cfg.__dict__, "max_targets": 1})
    targets = select_targets(tracks, cfg)
    return targets[0] if targets else None
