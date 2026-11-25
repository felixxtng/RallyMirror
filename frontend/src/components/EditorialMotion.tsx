/*
 * Adapted from talizen/blue-design-studio, MIT, copyright (c) Talizen:
 * commit 5aa997d4ca198220c30709d047ba135a53b2df62
 * src/components/motion.tsx (Chars/useCharsReveal/Reveal), Hero.tsx and
 * FeaturedWorks.tsx. https://github.com/talizen/blue-design-studio
 *
 * Retains the staged blur/rise and viewport-entry composition. Native Web
 * Animations and IntersectionObserver replace GSAP/Lenis; text enters by word
 * rather than by glyph. Content is visible by default, and no scroll loop or
 * permanent animation layer is installed. The full MIT notice is retained in
 * the project's third-party notices.
 */
import { useEffect, useLayoutEffect, useRef, type ReactNode, type RefObject } from 'react';
import './editorial-motion.css';

export interface EditorialTextProps {
  text: string;
  className?: string;
  as?: 'span' | 'h2' | 'p';
  /** Entrance delay in milliseconds. */
  delay?: number;
  reducedMotion?: boolean;
}

export interface EditorialRevealProps {
  children: ReactNode;
  className?: string;
  /** Entrance delay in milliseconds. */
  delay?: number;
  reducedMotion?: boolean;
}

const useBrowserLayoutEffect = typeof window === 'undefined' ? useEffect : useLayoutEffect;

/**
 * A one-shot entrance. Only WAAPI effects hide content, so unavailable APIs,
 * skipped observers, disabled motion and animation failures remain readable.
 */
function useEditorialEntrance(
  ref: RefObject<HTMLElement | null>,
  kind: 'words' | 'section',
  delay: number,
  reducedMotion: boolean,
  identity: unknown,
) {
  const seen = useRef({ identity, value: false });

  useBrowserLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const host: HTMLElement = element;
    if (seen.current.identity !== identity) seen.current = { identity, value: false };
    if (seen.current.value) return;

    const query =
      typeof window.matchMedia === 'function'
        ? window.matchMedia('(prefers-reduced-motion: reduce)')
        : null;
    const motionDisabled = () =>
      reducedMotion || !!query?.matches || !!host.closest('[data-motion="off"]');
    if (
      motionDisabled() ||
      typeof IntersectionObserver === 'undefined' ||
      typeof host.animate !== 'function'
    ) {
      seen.current.value = true;
      return;
    }

    let disposed = false;
    let started = false;
    let completed = false;
    let observer: IntersectionObserver | null = null;
    let preferenceObserver: MutationObserver | null = null;
    const animations: Animation[] = [];
    const oldHints = new Map<HTMLElement, string>();
    const viewportContainsHost = () => {
      const box = host.getBoundingClientRect();
      return (
        box.bottom > 0 &&
        box.top < window.innerHeight - 24 &&
        box.right > 0 &&
        box.left < window.innerWidth
      );
    };
    const restore = () => {
      animations.forEach((animation) => animation.cancel());
      oldHints.forEach((value, element) => {
        element.style.willChange = value;
      });
      oldHints.clear();
    };
    const disconnect = () => {
      observer?.disconnect();
      preferenceObserver?.disconnect();
      host.removeEventListener('focusin', complete);
      document.removeEventListener('visibilitychange', onVisibilityChange);
      query?.removeEventListener?.('change', onPreferenceChange);
    };
    function complete() {
      if (disposed || completed) return;
      completed = true;
      seen.current.value = true;
      restore();
      disconnect();
    }
    function start() {
      if (disposed || completed || started || document.hidden) return;
      if (motionDisabled()) {
        complete();
        return;
      }
      started = true;
      const words =
        kind === 'words'
          ? Array.from(host.querySelectorAll<HTMLElement>('[data-editorial-word]'))
          : [];
      // Headings are short. Bound work and total stagger if callers supply
      // paragraph-length copy: one section entrance is enough beyond 60 words.
      const wordMode = words.length > 0 && words.length <= 60;
      const targets = wordMode ? words : [host];
      const startDelay = Number.isFinite(delay) ? Math.max(0, Math.min(delay, 1500)) : 0;
      try {
        targets.forEach((target, index) => {
          oldHints.set(target, target.style.willChange);
          target.style.willChange = 'opacity, transform, filter';
          const effect = target.animate(
            [
              {
                opacity: 0,
                filter: wordMode ? 'blur(12px)' : 'blur(4px)',
                transform: wordMode
                  ? 'translate3d(0, .6em, 0) rotateX(8deg)'
                  : 'translate3d(0, 28px, 0)',
              },
              { opacity: 1, filter: 'blur(0px)', transform: 'translate3d(0, 0, 0) rotateX(0deg)' },
            ],
            {
              duration: wordMode ? 820 : 760,
              delay: startDelay + (wordMode ? Math.min(index * 65, 780) : 0),
              easing: 'cubic-bezier(.22, 1, .36, 1)',
              fill: 'backwards',
            },
          );
          animations.push(effect);
        });
        // Handle cancellation explicitly; no rejected `finished` promise escapes.
        void Promise.all(
          animations.map((animation) => animation.finished.catch(() => undefined)),
        ).then(complete);
      } catch {
        // A browser may expose animate() but reject an effect/property.
        complete();
      }
    }
    function onVisibilityChange() {
      if (disposed || completed) return;
      if (motionDisabled()) {
        complete();
        return;
      }
      if (document.hidden) {
        animations.forEach((animation) => animation.pause());
      } else if (started) {
        if (viewportContainsHost()) animations.forEach((animation) => animation.play());
        else complete();
      } else if (viewportContainsHost()) {
        start();
      }
    }
    function onPreferenceChange() {
      if (motionDisabled()) complete();
    }

    try {
      observer = new IntersectionObserver(
        ([entry]) => {
          if (!entry || disposed || completed) return;
          if (entry.isIntersecting) start();
          else if (started) complete();
        },
        { threshold: 0.04, rootMargin: '0px 0px -24px 0px' },
      );
      observer.observe(host);
      if (typeof MutationObserver !== 'undefined') {
        preferenceObserver = new MutationObserver(onPreferenceChange);
        preferenceObserver.observe(document.documentElement, {
          attributes: true,
          attributeFilter: ['data-motion'],
          subtree: true,
        });
      }
      query?.addEventListener?.('change', onPreferenceChange);
      // Keyboard navigation must never leave focused controls inside a faded
      // section. Its content becomes fully visible as soon as focus enters.
      host.addEventListener('focusin', complete);
      document.addEventListener('visibilitychange', onVisibilityChange);
      // Start above-fold copy before the first paint instead of briefly showing
      // it and hiding it when the initial async IO notification arrives.
      if (viewportContainsHost()) start();
    } catch {
      complete();
    }

    return () => {
      // Do not mark an interrupted mount as played. React StrictMode performs
      // an initial setup/cleanup cycle, and the real mount must still animate.
      disposed = true;
      restore();
      disconnect();
    };
  }, [ref, kind, delay, reducedMotion, identity]);
}

/** Full phrase for assistive technology; visual words animate independently. */
export function EditorialText({
  text,
  className = '',
  as: Tag = 'span',
  delay = 0,
  reducedMotion = false,
}: EditorialTextProps) {
  const ref = useRef<HTMLElement>(null);
  useEditorialEntrance(ref, 'words', delay, reducedMotion, text);
  const lines = text.split('\n');
  return (
    <Tag
      ref={(node) => {
        ref.current = node;
      }}
      className={`editorial-text ${className}`.trim()}
    >
      <span className="editorial-accessible">{text.replace(/\s+/g, ' ').trim()}</span>
      <span className="editorial-visual" aria-hidden="true">
        {lines.map((line, lineIndex) => (
          <span className="editorial-line" key={lineIndex}>
            {line.split(/(\s+)/).map((part, partIndex) =>
              /^\s*$/.test(part) ? (
                part
              ) : (
                <span className="editorial-word" data-editorial-word key={partIndex}>
                  {part}
                </span>
              ),
            )}
            {lineIndex < lines.length - 1 && <br />}
          </span>
        ))}
      </span>
    </Tag>
  );
}

/** A subtle one-shot section entrance with no permanent scroll/render loop. */
export function EditorialReveal({
  children,
  className = '',
  delay = 0,
  reducedMotion = false,
}: EditorialRevealProps) {
  const ref = useRef<HTMLDivElement>(null);
  useEditorialEntrance(ref, 'section', delay, reducedMotion, 'section');
  return (
    <div ref={ref} className={`editorial-reveal ${className}`.trim()}>
      {children}
    </div>
  );
}
