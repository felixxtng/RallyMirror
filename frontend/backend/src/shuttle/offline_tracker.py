"""Offline Viterbi selection with whole-video, non-semantic hotspot evidence.

Input is a contiguous sequence of full-frame candidate lists, one per frame.
Output contains original candidate coordinates or None; never interpolated points.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class OfflineConfig:
    confidence_reference: float = 0.5
    switch_cost: float = 0.12
    motion_radius: float = 0.07  # Image diagonals per 1/30 second.
    motion_penalty: float = 0.35
    hotspot_radius: float = 0.003  # Fraction of image diagonal.
    hotspot_episode_gap_seconds: float = 0.25
    hotspot_min_episodes: int = 3
    hotspot_min_span_seconds: float = 1.0
    hotspot_stationary_seconds: float = 1.0
    hotspot_penalty: float = 0.65
    support_radius: float = 0.12  # Moving-neighbour search radius at 30 FPS.
    hotspot_support_discount: float = 0.8

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('motion_radius', 'hotspot_radius', 'hotspot_episode_gap_seconds',
                     'hotspot_min_span_seconds', 'hotspot_stationary_seconds', 'support_radius'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if (not 0 <= self.confidence_reference <= 1
                or not 0 <= self.hotspot_support_discount <= 1
                or type(self.hotspot_min_episodes) is not int or self.hotspot_min_episodes < 2):
            raise ValueError('Invalid confidence, support discount or episode count')


class OfflineTracker:
    """Exact first-order DP over all candidates plus a missing state per frame.

    Reward = confidence - reference - discounted hotspot cost. Transitions pay
    a saturating displacement cost or a detection/missing switch cost. Missing
    has zero emission and releases spatial memory. Backtracking uses future as
    well as past evidence. Scores are heuristic utilities, not probabilities.
    """
    def __init__(self, width, height, fps=30, config=None):
        if not all(math.isfinite(v) and v > 0 for v in (width, height, fps)):
            raise ValueError('Positive finite dimensions and FPS required')
        self.width, self.height, self.fps = width, height, fps
        self.diagonal = math.hypot(width, height)
        self.config = config or OfflineConfig()
        self.hotspots = []
        self.path_score = 0.0

    def _locations(self, frames):
        """Fixed-anchor spatial clustering using a grid; one observation/frame.

        Distinct episodes accumulate even when arbitrarily far apart in time.
        Long continuous occupancy also counts, while short pauses alone do not.
        """
        c = self.config
        radius = c.hotspot_radius * self.diagonal
        grid, clusters, assignments = {}, [], []
        for frame, candidates in enumerate(frames):
            indices = []
            for candidate in candidates:
                x, y, score = (candidate[k] for k in ('x', 'y', 'score'))
                if (not all(math.isfinite(v) for v in (x, y, score)) or not 0 <= score <= 1
                        or not (0 <= x < self.width and 0 <= y < self.height)):
                    raise ValueError('Candidates require original-image coordinates and scores in [0, 1]')
                gx, gy = math.floor(x/radius), math.floor(y/radius)
                nearby = [i for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                          for i in grid.get((gx+dx, gy+dy), ())
                          if math.hypot(x-clusters[i]['x'], y-clusters[i]['y']) <= radius]
                if nearby:
                    index = min(nearby, key=lambda i: (math.hypot(
                        x-clusters[i]['x'], y-clusters[i]['y']), i))
                else:
                    index = len(clusters)
                    clusters.append(dict(x=x, y=y, frames=[]))
                    grid.setdefault((gx, gy), []).append(index)
                hits = clusters[index]['frames']
                if not hits or hits[-1] != frame:
                    hits.append(frame)
                indices.append(index)
            assignments.append(indices)
        gap = max(1, round(c.hotspot_episode_gap_seconds*self.fps))
        for cluster in clusters:
            hits = cluster['frames']
            episodes = 1 + sum(b-a > gap for a, b in zip(hits, hits[1:]))
            longest = run = 1
            for a, b in zip(hits, hits[1:]):
                run = run+1 if b == a+1 else 1
                longest = max(longest, run)
            span = (hits[-1]-hits[0])/self.fps
            recurring = (min(1.0, (episodes-1)/c.hotspot_min_episodes)
                         if episodes >= c.hotspot_min_episodes and span >= c.hotspot_min_span_seconds else 0.0)
            stationary = min(1.0, max(0.0, longest/self.fps/c.hotspot_stationary_seconds-1))
            cluster.update(observations=len(hits), episodes=episodes, span_seconds=span,
                           longest_run=longest, strength=max(recurring, stationary))
        self.hotspots = clusters
        return assignments

    def _moving_support(self, frames, frame, candidate):
        # Only real candidates on BOTH sides can discount a hotspot penalty.
        if frame == 0 or frame == len(frames)-1:
            return False
        c = self.config
        minimum = 2*c.hotspot_radius*self.diagonal
        maximum = c.support_radius*self.diagonal*30/self.fps
        def vectors(neighbours, sign):
            result = []
            for neighbour in neighbours:
                dx, dy = sign*(neighbour['x']-candidate['x']), sign*(neighbour['y']-candidate['y'])
                distance = math.hypot(dx, dy)
                if minimum < distance <= maximum:
                    result.append((dx, dy, distance))
            return result
        before = vectors(frames[frame-1], -1)
        after = vectors(frames[frame+1], 1)
        return any((ax*bx+ay*by)/(ad*bd) >= -0.5 and max(ad, bd)/min(ad, bd) <= 4
                   for ax, ay, ad in before for bx, by, bd in after)

    def refine(self, frames):
        c = self.config
        assignments = self._locations(frames)
        self.path_score = 0.0
        if not frames:
            return []
        costs = []
        for frame, candidates in enumerate(frames):
            row = []
            for index, candidate in enumerate(candidates):
                hotspot = self.hotspots[assignments[frame][index]]
                supported = self._moving_support(frames, frame, candidate) if hotspot['strength'] else False
                penalty = c.hotspot_penalty*hotspot['strength']*(
                    1-c.hotspot_support_discount if supported else 1)
                row.append((penalty, supported))
            costs.append(row)
        previous = [0.0]
        backpointers = []
        motion_scale = c.motion_radius*self.diagonal*30/self.fps
        for frame, candidates in enumerate(frames):
            prior_candidates = frames[frame-1] if frame else []
            # Missing-first ordering resolves exact ties conservatively and deterministically.
            missing_options = [previous[0]] + [v-c.switch_cost for v in previous[1:]]
            missing_parent = max(range(len(missing_options)), key=missing_options.__getitem__)
            current = [missing_options[missing_parent]]
            parents = [missing_parent]
            for index, candidate in enumerate(candidates):
                options = [previous[0]-c.switch_cost]
                for p, old in enumerate(prior_candidates, 1):
                    distance = math.hypot(candidate['x']-old['x'], candidate['y']-old['y'])
                    motion = c.motion_penalty*(1-math.exp(-0.5*(distance/motion_scale)**2))
                    options.append(previous[p]-motion)
                parent = max(range(len(options)), key=options.__getitem__)
                current.append(options[parent]+candidate['score']-c.confidence_reference-costs[frame][index][0])
                parents.append(parent)
            previous = current
            backpointers.append(parents)
        # Charge the same exit cost at the end as for an interior segment.
        terminal = [previous[0]] + [v-c.switch_cost for v in previous[1:]]
        state = max(range(len(terminal)), key=terminal.__getitem__)
        self.path_score = terminal[state]
        path = [None]*len(frames)
        for frame in range(len(frames)-1, -1, -1):
            if state:
                index = state-1
                penalty, supported = costs[frame][index]
                path[frame] = dict(frames[frame][index], offline_candidate_index=index,
                                   offline_hotspot_id=assignments[frame][index],
                                   offline_hotspot_cost=penalty, offline_moving_support=supported)
            state = backpointers[frame][state]
        return path

    def diagnostics(self):
        return dict(config=asdict(self.config), path_score=self.path_score,
                    hotspots=[dict(id=i, **{k: v for k, v in spot.items() if k != 'frames'})
                              for i, spot in enumerate(self.hotspots) if spot['strength'] > 0])
