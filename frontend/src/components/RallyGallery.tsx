/*
 * Spatial gallery / index presentation adapted from Talizen's MIT-licensed
 * Aren Vale Composer, revision a15386c5c97e3d94e5c04cd86afbe39d51229bed:
 *   src/lib/site/stage.ts (normalised pointer sway and inward-facing images)
 *   src/lib/site/home.ts + src/page/Projects.tsx (gallery and project index)
 * Copyright (c) Talizen. See frontend/THIRD_PARTY_NOTICES.md for the license.
 *
 * RallyMirror uses ordinary image links and CSS perspective instead of the
 * source's WebGL cylinder, motion-blur passes, GSAP ticker and touch controls.
 * The browser animates layout changes; pointer updates request one frame only.
 */
import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { LayoutGrid, List } from 'lucide-react';
import { useReducedMotion } from './Motion';
import './rally-gallery.css';

export type RallyGalleryItem = {
  id: string;
  title: string;
  detail: string;
  image: string;
  href: string;
};

export type RallyGalleryProps = {
  items: RallyGalleryItem[];
  reducedMotion?: boolean;
  className?: string;
};

type View = 'gallery' | 'list';
type PositionStyle = CSSProperties & Record<`--${string}`, string | number>;
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

function position(index: number, total: number, columns: number): PositionStyle {
  const row = Math.floor(index / columns);
  const column = index % columns;
  const rowCount = Math.min(columns, total - row * columns);
  const across = rowCount <= 1 ? 0 : (column / (rowCount - 1)) * 2 - 1;
  // Stage's images face back towards the cylinder centre. Keep that spatial
  // relationship, but constrain it to a legible range for image evidence.
  return {
    '--gallery-card-yaw': `${-across * 7}deg`,
    '--gallery-card-depth': `${(1 - Math.abs(across)) * 52 - 20}px`,
    '--gallery-card-lift': `${Math.abs(across) * 20 + (row % 2 ? 8 : 0)}px`,
    '--gallery-card-roll': `${across * 0.7}deg`,
  };
}

export default function RallyGallery({
  items,
  reducedMotion = false,
  className = '',
}: RallyGalleryProps) {
  const [view, setView] = useState<View>('gallery');
  const rootRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const previousRects = useRef(new Map<string, DOMRect>());
  const animations = useRef<Animation[]>([]);
  const systemReduced = useReducedMotion();
  const reduced = reducedMotion || systemReduced;
  const listId = useId();
  const columns = Math.min(3, items.length === 4 ? 2 : Math.max(1, items.length));

  function changeView(next: View) {
    if (next === view) return;
    previousRects.current = new Map(
      [...(listRef.current?.querySelectorAll<HTMLElement>('[data-gallery-id]') || [])].map(
        (node) => [node.dataset.galleryId || '', node.getBoundingClientRect()],
      ),
    );
    animations.current.forEach((animation) => animation.cancel());
    animations.current = [];
    setView(next);
  }

  // FLIP keeps the same focusable links in the DOM while they move between the
  // spatial view and index. There is no animation ticker after the transition.
  useLayoutEffect(() => {
    const previous = previousRects.current;
    previousRects.current = new Map();
    if (reduced || !previous.size || !listRef.current) return;
    const nodes = listRef.current.querySelectorAll<HTMLElement>('[data-gallery-id]');
    for (const node of nodes) {
      const before = previous.get(node.dataset.galleryId || '');
      const after = node.getBoundingClientRect();
      if (!before || !after.width || !after.height || !node.animate) continue;
      const x = before.left - after.left;
      const y = before.top - after.top;
      const scaleX = before.width / after.width;
      const scaleY = before.height / after.height;
      if (Math.abs(x) + Math.abs(y) < 1 && Math.abs(scaleX - 1) + Math.abs(scaleY - 1) < 0.01)
        continue;
      animations.current.push(
        node.animate(
          [
            {
              transform: `translate3d(${x}px, ${y}px, 0) scale(${scaleX}, ${scaleY})`,
              transformOrigin: 'top left',
              opacity: 0.72,
            },
            { transform: 'none', transformOrigin: 'top left', opacity: 1 },
          ],
          { duration: 520, easing: 'cubic-bezier(.2,.75,.2,1)' },
        ),
      );
    }
  }, [view, reduced, items.length]);

  useEffect(() => {
    return () => {
      animations.current.forEach((animation) => animation.cancel());
      animations.current = [];
    };
  }, []);

  useEffect(() => {
    const root = rootRef.current;
    const scene = sceneRef.current;
    const list = listRef.current;
    if (!root || !scene || !list) return;
    const finePointer = matchMedia('(hover: hover) and (pointer: fine)');
    let frame = 0;
    let visible = true;
    let x = 0;
    let y = 0;
    const draw = () => {
      frame = 0;
      list.style.setProperty('--gallery-yaw', `${x.toFixed(3)}deg`);
      list.style.setProperty('--gallery-pitch', `${y.toFixed(3)}deg`);
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(draw);
    };
    const reset = () => {
      x = 0;
      y = 0;
      if (frame) cancelAnimationFrame(frame);
      draw();
    };
    const move = (event: PointerEvent) => {
      if (
        view !== 'gallery' ||
        reduced ||
        !finePointer.matches ||
        !visible ||
        document.hidden ||
        root.clientWidth < 680 ||
        event.pointerType === 'touch'
      )
        return;
      const bounds = scene.getBoundingClientRect();
      // The original Stage.sway normalises the pointer around the host centre.
      const nx = clamp(((event.clientX - bounds.left) / Math.max(1, bounds.width)) * 2 - 1, -1, 1);
      const ny = clamp(-((event.clientY - bounds.top) / Math.max(1, bounds.height)) * 2 + 1, -1, 1);
      x = -nx * 3.5;
      y = ny * 2;
      schedule();
    };
    const onVisibility = () => {
      if (document.hidden) {
        reset();
        animations.current.forEach((animation) => animation.cancel());
      }
    };
    const observer =
      typeof IntersectionObserver === 'undefined'
        ? null
        : new IntersectionObserver(([entry]) => {
            visible = entry?.isIntersecting ?? true;
            if (!visible) reset();
          });
    observer?.observe(scene);
    scene.addEventListener('pointermove', move, { passive: true });
    scene.addEventListener('pointerleave', reset);
    finePointer.addEventListener('change', reset);
    window.addEventListener('resize', reset, { passive: true });
    document.addEventListener('visibilitychange', onVisibility);
    reset();
    if (reduced) animations.current.forEach((animation) => animation.cancel());
    return () => {
      if (frame) cancelAnimationFrame(frame);
      observer?.disconnect();
      scene.removeEventListener('pointermove', move);
      scene.removeEventListener('pointerleave', reset);
      finePointer.removeEventListener('change', reset);
      window.removeEventListener('resize', reset);
      document.removeEventListener('visibilitychange', onVisibility);
      list.style.removeProperty('--gallery-yaw');
      list.style.removeProperty('--gallery-pitch');
    };
  }, [view, reduced, items.length]);

  if (!items.length) return null;

  return (
    <div
      ref={rootRef}
      className={`rally-gallery ${className}`.trim()}
      data-view={view}
      data-reduced-motion={reduced ? 'true' : 'false'}
      style={{ '--gallery-columns': columns } as PositionStyle}
    >
      <div className="rally-gallery__toolbar">
        <span className="rally-gallery__count">
          {items.length} project {items.length === 1 ? 'example' : 'examples'}
        </span>
        <div className="rally-gallery__view-controls" role="group" aria-label="Example layout">
          <button
            type="button"
            aria-pressed={view === 'gallery'}
            aria-controls={listId}
            onClick={() => changeView('gallery')}
          >
            <LayoutGrid size={16} aria-hidden="true" /> Gallery view
          </button>
          <button
            type="button"
            aria-pressed={view === 'list'}
            aria-controls={listId}
            onClick={() => changeView('list')}
          >
            <List size={16} aria-hidden="true" /> List view
          </button>
        </div>
      </div>
      <div ref={sceneRef} className="rally-gallery__scene">
        <ul ref={listRef} id={listId} className="rally-gallery__items" role="list">
          {items.map((item, index) => (
            <li
              key={item.id}
              data-gallery-id={item.id}
              className="rally-gallery__item"
              style={position(index, items.length, columns)}
            >
              <a
                className="rally-gallery__link"
                href={item.href.startsWith('#/') ? item.href : '#/project'}
              >
                <span className="rally-gallery__image">
                  <img
                    src={item.image}
                    alt=""
                    width="960"
                    height="540"
                    loading="lazy"
                    decoding="async"
                  />
                </span>
                <span className="rally-gallery__copy">
                  <strong className="rally-gallery__title">{item.title}</strong>
                  {item.detail && <span className="rally-gallery__detail">{item.detail}</span>}
                  <span className="rally-gallery__open" aria-hidden="true">
                    Open example
                  </span>
                </span>
              </a>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
