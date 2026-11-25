/*
 * Scroll segmentation, quintic easing and staged camera interpolation adapted
 * from talizen/lunar-rover-3d-scroll engine/scene.ts and engine/util.ts at
 * 411003681be8ed29f4e57764c570a768844aba85. Copyright (c) Talizen, MIT.
 * Full license: frontend/THIRD_PARTY_NOTICES.md.
 */

const clamp = (value: number, low = 0, high = 1) => Math.min(high, Math.max(low, value));
const mix = (from: number, to: number, amount: number) => from + (to - from) * amount;

/** Root-relative native scroll, with no dependency on unrelated page sections. */
export function courtScrollProgress(top: number, height: number, viewportHeight: number) {
  if (![top, height, viewportHeight].every(Number.isFinite)) return 0;
  const travel = height - viewportHeight;
  if (travel <= 0 || viewportHeight <= 0) return 0;
  return clamp(-top / travel);
}

/** Lunar Rover's quintic interpolation keeps camera velocity smooth at joins. */
function easedSegment(progress: number, from: number, to: number) {
  const t = clamp((progress - from) / (to - from));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

export type CourtCameraSample = {
  orbit: number;
  focus: number;
  distance: number;
  offset: readonly [number, number, number];
};

/**
 * Three acts from the source: the full court, a complete turn with a dolly,
 * then a close view. RallyMirror locks the final target onto the shuttle
 * instead of the rover's eye. The caller supplies that target in world space.
 */
export function sampleCourtCamera(progress: number, aspect = 1): CourtCameraSample {
  const p = Number.isFinite(progress) ? clamp(progress) : 0;
  const safeAspect = Number.isFinite(aspect) && aspect > 0 ? clamp(aspect, 0.65, 3) : 1;
  const wide = easedSegment(p, 0.05, 0.6);
  const close = easedSegment(p, 0.6, 0.94);
  const turn = easedSegment(p, 0.07, 0.62);
  const initialDistance = Math.hypot(15, 13.5, 20);
  const tangent = Math.tan((37 * Math.PI) / 360);
  const wideDistance = Math.max(21, 7 / (tangent * safeAspect));
  const closeDistance = Math.max(9.6, 2.8 / (tangent * safeAspect));
  const distance = mix(mix(initialDistance, wideDistance, wide), closeDistance, close);
  const x = mix(15 / initialDistance, 0.54, close);
  const y = mix(13.5 / initialDistance, 0.33, close);
  const z = mix(20 / initialDistance, 0.774, close);
  const normal = Math.hypot(x, y, z);
  return {
    orbit: -0.24 + Math.PI * 2 * turn,
    focus: close,
    distance,
    offset: [(x / normal) * distance, (y / normal) * distance, (z / normal) * distance],
  };
}

/** Frame-rate-independent damping used by the source's camera progress. */
export function dampCourtProgress(current: number, target: number, deltaSeconds: number) {
  const start = Number.isFinite(current) ? clamp(current) : 0;
  const end = Number.isFinite(target) ? clamp(target) : 0;
  const dt = Number.isFinite(deltaSeconds) ? Math.max(0, deltaSeconds) : 0;
  const next = start + (end - start) * (1 - Math.exp(-dt * 7));
  return Math.abs(next - end) < 1e-5 ? end : next;
}
