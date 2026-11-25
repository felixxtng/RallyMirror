"""Causal, full-frame candidate ranking; all distances use original-image pixels."""
from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrackerConfig:
    # Spatial parameters are fractions of the image diagonal; times use seconds.
    motion_radius: float = 0.06
    motion_penalty: float = 0.4
    minimum_quality: float = 0.6
    reacquire_misses: int = 3
    hotspot_radius: float = 0.004
    hotspot_grace_seconds: float = 0.27
    hotspot_window_seconds: float = 2.0
    hotspot_penalty: float = 0.65

    def __post_init__(self):
        if (self.motion_radius <= 0 or self.hotspot_radius <= 0
                or self.hotspot_grace_seconds <= 0 or self.hotspot_window_seconds <= 0
                or self.reacquire_misses < 1
                or not 0 <= self.minimum_quality <= 1
                or not 0 <= self.motion_penalty <= 1 or not 0 <= self.hotspot_penalty <= 1):
            raise ValueError("Invalid tracker configuration")


class TemporalTracker:
    """Soft motion ranking plus expiring hotspot evidence from *all* candidates.

    No synthesized coordinates are emitted. After short losses motion costs are
    removed, allowing global reacquisition. Hotspots expire rather than becoming
    permanent masks. A brief stationary apex is protected by a grace period.
    """
    def __init__(self, width, height, fps=30, config=None):
        if min(width, height, fps) <= 0:
            raise ValueError("Positive dimensions and FPS required")
        self.config = config or TrackerConfig()
        self.diagonal = math.hypot(width, height)
        self.fps = fps
        self.history = deque(maxlen=3)
        self.hotspots = []
        self.misses = 0
        self.last_frame = -1
        self.reason = "initial"

    def update(self, candidates, frame):
        if frame <= self.last_frame:
            raise ValueError("Tracker frame indices must strictly increase")
        if self.last_frame >= 0:
            self.misses += frame - self.last_frame - 1
        self.last_frame = frame
        c = self.config
        window = max(1, round(c.hotspot_window_seconds * self.fps))
        grace = max(2, round(c.hotspot_grace_seconds * self.fps))
        radius = c.hotspot_radius * self.diagonal
        for spot in self.hotspots:
            while spot['hits'] and spot['hits'][0] <= frame - window:
                spot['hits'].popleft()
        self.hotspots = [s for s in self.hotspots if s['hits']]
        ranked = []
        for candidate in candidates:
            x, y, score = (candidate[k] for k in ('x', 'y', 'score'))
            if not all(math.isfinite(v) for v in (x, y, score)) or not 0 <= score <= 1:
                raise ValueError("Candidates require finite coordinates and score in [0, 1]")
            spot = next((s for s in self.hotspots
                         if math.hypot(x-s['x'], y-s['y']) <= radius), None)
            if spot is None:
                spot = dict(x=x, y=y, hits=deque())
                self.hotspots.append(spot)
            if not spot['hits'] or spot['hits'][-1] != frame:
                spot['hits'].append(frame)
            background = c.hotspot_penalty * min(1, max(0, (len(spot['hits']) - grace) / grace))
            motion = 0.0
            if self.history and self.misses < c.reacquire_misses:
                last_frame, lx, ly = self.history[-1]
                vx = vy = 0.0
                if len(self.history) > 1:
                    pf, px, py = self.history[-2]
                    vx, vy = (lx-px)/(last_frame-pf), (ly-py)/(last_frame-pf)
                dt = frame - last_frame
                scale = c.motion_radius*self.diagonal + 1.5*math.hypot(vx, vy)*dt
                # Use both inertia and the last point to tolerate reversals/deceleration.
                distance = min(math.hypot(x-lx-vx*dt, y-ly-vy*dt),
                               math.hypot(x-lx, y-ly))
                motion = c.motion_penalty * (1-math.exp(-0.5*(distance/scale)**2))
            quality = score - motion - background
            ranked.append((quality, candidate, motion, background))
        # Bound memory even for pathological heatmaps; retain strongest evidence.
        self.hotspots.sort(key=lambda s: (len(s['hits']), s['hits'][-1]), reverse=True)
        self.hotspots = self.hotspots[:256]
        best = max(ranked, key=lambda item: item[0], default=None)
        if best is None or best[0] < c.minimum_quality:
            self.misses += 1
            self.reason = 'no_candidates' if best is None else 'low_quality'
            return None
        quality, candidate, motion, background = best
        self.reason = 'reacquired' if self.misses >= c.reacquire_misses else 'accepted'
        if self.misses >= c.reacquire_misses:
            self.history.clear()
        self.history.append((frame, candidate['x'], candidate['y']))
        self.misses = 0
        return dict(candidate, tracking_quality=quality, motion_cost=motion,
                    background_cost=background)
