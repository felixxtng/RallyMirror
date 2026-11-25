/*
 * Adapted from ThreeUI Community Gallery.tsx and the pointer interpolation in
 * animated-top-dock/glassParticleField.ts, commit
 * 68802d5428071ada5c20db8094b1649e6bb770ed. Copyright (c) 2026 Meng To.
 * MIT licensed; full notice is in frontend/THIRD_PARTY_NOTICES.md.
 *
 * Gallery's renderer/resize/visibility/disposal structure is retained. The
 * gallery, images and materials are replaced with RallyMirror court geometry.
 * Added lazy loading, a 30fps draw cap, reduced motion, a non-WebGL fallback,
 * context-loss handling and frame-rate-independent pointer interpolation.
 * Scroll camera acts additionally adapt Talizen's lunar-rover-3d-scroll;
 * see courtScroll.ts and frontend/THIRD_PARTY_NOTICES.md for attribution.
 */
import { useEffect, useRef, useState, type RefObject } from 'react';
import type { BufferGeometry } from 'three';
import { useReducedMotion } from './Motion';
import { courtScrollProgress, dampCourtProgress, sampleCourtCamera } from './courtScroll';

type Three = typeof import('three');
export type CourtSceneProps = {
  reducedMotion?: boolean;
  className?: string;
  scrollRoot?: RefObject<HTMLElement | null>;
};

export default function CourtScene({ reducedMotion, className = '', scrollRoot }: CourtSceneProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [three, setThree] = useState<Three | null>(null);
  const [ready, setReady] = useState(false);
  const systemReducedMotion = useReducedMotion();
  const reduced = reducedMotion ?? systemReducedMotion;

  // Resolve the module before the effect that owns the GL context. An aborted
  // asynchronous import must never create a renderer after its cleanup ran.
  useEffect(() => {
    let cancelled = false;
    import('three')
      .then((module) => {
        if (!cancelled) setThree(module);
      })
      .catch(() => {
        /* The static court remains available if loading fails. */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const host = hostRef.current;
    const canvas = canvasRef.current;
    if (!host || !canvas || !three) return;
    const THREE = three;
    setReady(false);

    let context: WebGL2RenderingContext | null = null;
    try {
      context = canvas.getContext('webgl2', {
        alpha: true,
        antialias: true,
        powerPreference: 'low-power',
      });
    } catch {
      return;
    }
    if (!context) return;

    let renderer: InstanceType<Three['WebGLRenderer']>;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        context,
        alpha: true,
        antialias: true,
        powerPreference: 'low-power',
      });
    } catch {
      return;
    }
    renderer.setClearColor(0x000000, 0);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(37, 1, 0.1, 100);
    const lookAt = new THREE.Vector3(0, 1, 0);
    camera.position.set(15, 14.5, 20);
    camera.lookAt(lookAt);
    scene.add(new THREE.AmbientLight(0xe5f8e6, 2.4));
    const key = new THREE.DirectionalLight(0xffffff, 3.4);
    key.position.set(-5, 12, 8);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xc9ff6d, 1.6);
    rim.position.set(8, 7, -8);
    scene.add(rim);

    const stage = new THREE.Group();
    stage.rotation.y = -0.24;
    scene.add(stage);
    const geometries = new Set<BufferGeometry>();
    const materials = new Set<InstanceType<Three['Material']>>();
    const material = (color: number, options: Record<string, unknown> = {}) => {
      const value = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.7,
        metalness: 0.08,
        ...options,
      });
      materials.add(value);
      return value;
    };
    const mesh = (
      geometry: BufferGeometry,
      surface: InstanceType<Three['Material']>,
      parent = stage,
    ) => {
      geometries.add(geometry);
      const value = new THREE.Mesh(geometry, surface);
      parent.add(value);
      return value;
    };
    const box = (
      w: number,
      h: number,
      d: number,
      surface: InstanceType<Three['Material']>,
      x = 0,
      y = 0,
      z = 0,
    ) => {
      const value = mesh(new THREE.BoxGeometry(w, h, d), surface);
      value.position.set(x, y, z);
      return value;
    };
    const points = (coordinates: number[], color: number, opacity = 1, parent = stage) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(coordinates, 3));
      geometries.add(geometry);
      const surface = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
      materials.add(surface);
      const value = new THREE.LineSegments(geometry, surface);
      parent.add(value);
      return value;
    };

    const dark = material(0x101e19);
    const green = material(0x296149);
    const pale = material(0xe4f1dc);
    const lime = material(0xcdff77, { emissive: 0x69852f, emissiveIntensity: 0.5 });
    box(8.8, 0.28, 15.7, dark, 0, -0.25);
    box(6.1, 0.12, 13.4, green);
    // The dimensions and floor-line arrangement are a court illustration, not
    // inference output or a substitute for a calibrated result.
    const courtLines: number[] = [];
    const floorLine = (x1: number, z1: number, x2: number, z2: number) =>
      courtLines.push(x1, 0.075, z1, x2, 0.075, z2);
    for (const x of [-3.05, -2.59, 2.59, 3.05]) floorLine(x, -6.7, x, 6.7);
    for (const z of [-6.7, -5.94, -1.98, 1.98, 5.94, 6.7]) floorLine(-3.05, z, 3.05, z);
    floorLine(0, -6.7, 0, -1.98);
    floorLine(0, 1.98, 0, 6.7);
    points(courtLines, 0xe7f5d4, 0.95);

    // A real raised net gives the court depth while keeping draw calls small.
    const net: number[] = [];
    for (let i = 0; i <= 36; i++) {
      const x = -3.3 + (i / 36) * 6.6;
      net.push(x, 0.43, 0, x, 1.58, 0);
    }
    for (let i = 0; i <= 7; i++) {
      const y = 0.43 + (i / 7) * 1.15;
      net.push(-3.3, y, 0, 3.3, y, 0);
    }
    points(net, 0xd7e9cf, 0.42);
    box(6.68, 0.055, 0.055, pale, 0, 1.6);
    box(0.075, 1.62, 0.075, pale, -3.3, 0.82);
    box(0.075, 1.62, 0.075, pale, 3.3, 0.82);

    // Two modest floor markers communicate player positions without implying
    // that this decorative hero is an analysed video.
    for (const [x, z] of [
      [-0.9, 3.35],
      [0.85, -3.5],
    ]) {
      const disc = mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.025, 24), lime);
      disc.position.set(x, 0.13, z);
      const halo = mesh(
        new THREE.RingGeometry(0.35, 0.38, 32),
        material(0xc8ff77, { transparent: true, opacity: 0.5, side: THREE.DoubleSide }),
      );
      halo.rotation.x = -Math.PI / 2;
      halo.position.set(x, 0.1, z);
    }

    const flight = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-1.1, 0.7, 3.1),
      new THREE.Vector3(-0.3, 3.5, 1.3),
      new THREE.Vector3(1.6, 4.5, -1.8),
      new THREE.Vector3(2.2, 2.5, -3.7),
    ]);
    const arcGeometry = new THREE.BufferGeometry().setFromPoints(flight.getPoints(65));
    geometries.add(arcGeometry);
    const arcMaterial = new THREE.LineDashedMaterial({
      color: 0xc7fa75,
      transparent: true,
      opacity: 0.52,
      dashSize: 0.14,
      gapSize: 0.13,
    });
    materials.add(arcMaterial);
    const arc = new THREE.Line(arcGeometry, arcMaterial);
    arc.computeLineDistances();
    stage.add(arc);

    // One merged mesh for all sixteen feather vanes, rather than sixteen
    // independent materials and draw calls. No remote models or textures.
    const shuttle = new THREE.Group();
    stage.add(shuttle);
    const cork = mesh(new THREE.SphereGeometry(0.26, 16, 12), material(0xece6cd), shuttle);
    cork.scale.set(1, 0.87, 1);
    const band = mesh(
      new THREE.CylinderGeometry(0.25, 0.25, 0.13, 16),
      material(0xa8d95c),
      shuttle,
    );
    band.position.y = 0.18;
    const featherVertices: number[] = [];
    for (let i = 0; i < 16; i++) {
      const angle = (i / 16) * Math.PI * 2;
      const c = Math.cos(angle),
        s = Math.sin(angle);
      const a = [c * 0.16 - s * 0.025, 0.24, s * 0.16 + c * 0.025];
      const b = [c * 0.16 + s * 0.025, 0.24, s * 0.16 - c * 0.025];
      const d = [c * 0.76 - s * 0.12, 1.65, s * 0.76 + c * 0.12];
      const e = [c * 0.76 + s * 0.12, 1.65, s * 0.76 - c * 0.12];
      featherVertices.push(...a, ...b, ...d, ...b, ...e, ...d);
    }
    const featherGeometry = new THREE.BufferGeometry();
    featherGeometry.setAttribute('position', new THREE.Float32BufferAttribute(featherVertices, 3));
    featherGeometry.computeVertexNormals();
    mesh(featherGeometry, material(0xf8faee, { side: THREE.DoubleSide, roughness: 0.9 }), shuttle);
    shuttle.position.set(2.1, 2.5, -3.2);
    shuttle.rotation.set(-0.2, -0.3, -0.58);
    shuttle.scale.setScalar(1.14);

    const pointer = new THREE.Vector2();
    const pointerTarget = new THREE.Vector2();
    const shuttleFocus = new THREE.Vector3();
    const story = scrollRoot?.current;
    const scrollEnabled = Boolean(story && !reduced);
    let disposed = false;
    let frame = 0;
    let scrollFrame = 0;
    let scrollTarget = 0;
    let scrollCurrent = 0;
    let scrollInitialized = false;
    let elapsed = 0;
    let previousTime = 0;
    let lastDraw = 0;
    let hostVisible = true;
    let documentVisible = !document.hidden;
    let contextLost = false;
    let bounds = host.getBoundingClientRect();

    const readScroll = () => {
      if (disposed) return;
      const rect = scrollEnabled && story ? story.getBoundingClientRect() : null;
      scrollTarget = rect ? courtScrollProgress(rect.top, rect.height, window.innerHeight) : 0;
      if (!scrollInitialized || !scrollEnabled) scrollCurrent = scrollTarget;
      scrollInitialized = true;
    };

    const render = (time = performance.now()) => {
      if (disposed || contextLost) return;
      const delta = previousTime ? Math.min((time - previousTime) / 1000, 0.05) : 0;
      if (!reduced) elapsed += delta;
      previousTime = time;
      // ThreeUI's pointer lerp, adjusted for a 30fps rendering budget.
      pointer.lerp(pointerTarget, reduced ? 1 : 1 - Math.exp(-delta * 5));
      scrollCurrent = scrollEnabled ? dampCourtProgress(scrollCurrent, scrollTarget, delta) : 0;
      const cameraSample = sampleCourtCamera(scrollCurrent, camera.aspect);
      stage.rotation.y = cameraSample.orbit + pointer.x * 0.09;
      stage.rotation.x = -pointer.y * 0.025;
      stage.position.y = reduced ? 0 : Math.sin(elapsed * 0.7) * 0.04;
      shuttle.position.y = 2.5 + (reduced ? 0 : Math.sin(elapsed * 1.2) * 0.2);
      shuttle.rotation.y = -0.3 + (reduced ? 0 : elapsed * 0.08);
      if (scrollEnabled) {
        // The source's wide-to-eye target interpolation becomes a court-to-
        // shuttle move. World coordinates remain correct throughout the orbit.
        stage.updateMatrixWorld(true);
        shuttle.localToWorld(shuttleFocus.set(0, 0.7, 0));
        lookAt.set(0, 1, 0).lerp(shuttleFocus, cameraSample.focus);
        camera.position.set(
          lookAt.x + cameraSample.offset[0] + pointer.x * cameraSample.distance * 0.012,
          lookAt.y + cameraSample.offset[1] - pointer.y * cameraSample.distance * 0.008,
          lookAt.z + cameraSample.offset[2],
        );
        camera.lookAt(lookAt);
      }
      host.dataset.scrollProgress = scrollCurrent.toFixed(4);
      host.dataset.scrollMode = scrollEnabled ? 'cinematic' : 'static';
      host.dataset.cameraDistance = cameraSample.distance.toFixed(2);
      renderer.render(scene, camera);
    };

    const tick = (time: number) => {
      if (disposed || contextLost || !hostVisible || !documentVisible) {
        frame = 0;
        previousTime = 0;
        return;
      }
      if (time - lastDraw >= 1000 / 30) {
        render(time);
        lastDraw = time;
      }
      frame = window.requestAnimationFrame(tick);
    };

    const start = () => {
      if (disposed || contextLost || !hostVisible || !documentVisible) return;
      if (reduced) {
        render(0);
        return;
      }
      if (!frame) frame = window.requestAnimationFrame(tick);
    };
    const stop = () => {
      if (frame) window.cancelAnimationFrame(frame);
      frame = 0;
      previousTime = 0;
    };
    const onScroll = () => {
      if (disposed || !scrollEnabled || scrollFrame) return;
      // Native scrolling remains in charge. Read layout at most once per frame,
      // outside React; the rendering loop eases towards this measured target.
      scrollFrame = window.requestAnimationFrame(() => {
        scrollFrame = 0;
        readScroll();
        start();
      });
    };
    const resize = () => {
      if (disposed || contextLost) return;
      bounds = host.getBoundingClientRect();
      const width = Math.max(1, Math.round(bounds.width));
      const height = Math.max(1, Math.round(bounds.height));
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, width < 600 ? 1.25 : 1.5));
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.fov = camera.aspect < 1 ? 44 : 37;
      camera.updateProjectionMatrix();
      readScroll();
      render();
    };
    const onPointerMove = (event: PointerEvent) => {
      if (reduced || event.pointerType === 'touch') return;
      pointerTarget.set(
        ((event.clientX - bounds.left) / Math.max(1, bounds.width)) * 2 - 1,
        -(((event.clientY - bounds.top) / Math.max(1, bounds.height)) * 2 - 1),
      );
    };
    const onPointerLeave = () => pointerTarget.set(0, 0);
    const handleVisibility = () => {
      documentVisible = !document.hidden;
      if (documentVisible) start();
      else stop();
    };
    const onContextLost = (event: Event) => {
      event.preventDefault();
      contextLost = true;
      stop();
      setReady(false);
    };
    const onContextRestored = () => {
      contextLost = false;
      resize();
      setReady(true);
      start();
    };
    const resizeObserver =
      typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize);
    const intersection =
      typeof IntersectionObserver === 'undefined'
        ? null
        : new IntersectionObserver(([entry]) => {
            hostVisible = entry?.isIntersecting ?? true;
            if (hostVisible) start();
            else stop();
          });
    resizeObserver?.observe(host);
    intersection?.observe(host);
    if (scrollEnabled) window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', resize, { passive: true });
    host.addEventListener('pointermove', onPointerMove, { passive: true });
    host.addEventListener('pointerleave', onPointerLeave);
    document.addEventListener('visibilitychange', handleVisibility);
    canvas.addEventListener('webglcontextlost', onContextLost);
    canvas.addEventListener('webglcontextrestored', onContextRestored);
    resize();
    start();
    setReady(true);

    return () => {
      disposed = true;
      stop();
      if (scrollFrame) window.cancelAnimationFrame(scrollFrame);
      resizeObserver?.disconnect();
      intersection?.disconnect();
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', resize);
      host.removeEventListener('pointermove', onPointerMove);
      host.removeEventListener('pointerleave', onPointerLeave);
      document.removeEventListener('visibilitychange', handleVisibility);
      canvas.removeEventListener('webglcontextlost', onContextLost);
      canvas.removeEventListener('webglcontextrestored', onContextRestored);
      scene.clear();
      geometries.forEach((geometry) => geometry.dispose());
      materials.forEach((surface) => surface.dispose());
      renderer.dispose();
    };
  }, [three, reduced, scrollRoot]);

  return (
    <div
      ref={hostRef}
      className={`court-scene ${className}`.trim()}
      role="img"
      aria-label="Illustrative three-dimensional badminton court with a raised net, player markers and a floating shuttlecock"
      data-renderer={ready ? 'webgl' : 'static'}
      style={{ position: 'relative', isolation: 'isolate' }}
    >
      <svg
        viewBox="0 0 560 430"
        aria-hidden="true"
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          opacity: ready ? 0 : 1,
          transition: 'opacity 250ms ease',
          pointerEvents: 'none',
        }}
      >
        <defs>
          <linearGradient id="rm-court-floor" x2="0.7" y2="1">
            <stop stopColor="#285b40" />
            <stop offset="1" stopColor="#142e24" />
          </linearGradient>
        </defs>
        <path d="M76 198 318 94 500 263 245 380Z" fill="#101e19" />
        <path
          d="M110 202 314 115 465 260 247 357Z"
          fill="url(#rm-court-floor)"
          stroke="#bcdb9b"
          strokeWidth="1.3"
        />
        <g fill="none" stroke="#d5e9bf" strokeWidth="1" opacity=".8">
          <path d="M125 215 330 129M231 341 450 246M137 225 343 142M219 329 437 234M177 164 350 310M285 128 440 271M183 271 397 176M174 160 347 311M211 244 305 201M323 281 259 220" />
          <path d="m175 159 0-53 173 148v57M175 106 348 254M175 117 348 265M175 130 348 278M175 143 348 291" />
        </g>
        <g fill="#cdff77">
          <ellipse cx="234" cy="295" rx="8" ry="4" />
          <ellipse cx="326" cy="182" rx="8" ry="4" />
        </g>
        <path
          d="M233 291Q260 122 387 139"
          stroke="#c7fa75"
          fill="none"
          strokeDasharray="4 6"
          opacity=".7"
        />
        <g transform="translate(390 132) rotate(-30)">
          <ellipse cy="20" rx="8" ry="7" fill="#eee5cd" />
          <path d="m-6 16-14-48 40 0-14 48Z" fill="#f8faee" opacity=".9" />
          <path d="M-14-30-4 15M-6-30-2 15M3-30 2 15M12-30 4 15" stroke="#b5c4af" strokeWidth="1" />
        </g>
      </svg>
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        style={{
          position: 'absolute',
          inset: 0,
          display: 'block',
          width: '100%',
          height: '100%',
          opacity: ready ? 1 : 0,
          transition: 'opacity 250ms ease',
        }}
      />
    </div>
  );
}
