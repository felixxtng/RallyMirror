import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';

/** Respect the operating-system preference and react when it changes. */
export function useReducedMotion() {
  const [reduced, setReduced] = useState(
    () =>
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );

  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(query.matches);
    update();
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);

  return reduced;
}

/** A single entrance, without a permanent animation loop or scroll listener. */
export function Reveal({
  children,
  className = '',
  delay = 0,
  style,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
  style?: CSSProperties;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const reduced = useReducedMotion();

  useEffect(() => {
    const host = hostRef.current;
    if (!host || reduced || !('IntersectionObserver' in window)) {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.08 },
    );
    observer.observe(host);
    return () => observer.disconnect();
  }, [reduced]);

  return (
    <div
      ref={hostRef}
      className={className}
      style={{
        ...style,
        opacity: visible || reduced ? 1 : 0,
        transform: visible || reduced ? 'none' : 'translateY(18px)',
        transition: reduced
          ? 'none'
          : `opacity 650ms ease ${delay}ms, transform 650ms cubic-bezier(.2,.75,.3,1) ${delay}ms`,
      }}
    >
      {children}
    </div>
  );
}
