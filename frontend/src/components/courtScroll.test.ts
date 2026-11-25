import { describe, expect, it } from 'vitest';
import { courtScrollProgress, dampCourtProgress, sampleCourtCamera } from './courtScroll';

describe('court story scroll geometry', () => {
  it('uses only the story travel and clamps before and after the section', () => {
    expect(courtScrollProgress(120, 1850, 1000)).toBe(0);
    expect(courtScrollProgress(-425, 1850, 1000)).toBe(0.5);
    expect(courtScrollProgress(-1000, 1850, 1000)).toBe(1);
  });

  it('remains static for a section without scroll travel or invalid measurements', () => {
    for (const values of [
      [0, 500, 1000],
      [0, 1000, 1000],
      [0, 2000, 0],
      [NaN, 2000, 1000],
      [0, Infinity, 1000],
    ]) {
      expect(courtScrollProgress(...(values as [number, number, number]))).toBe(0);
    }
  });
});

describe('lunar camera adaptation', () => {
  it('preserves the existing wide-camera opening', () => {
    const initial = sampleCourtCamera(0);
    expect(initial.orbit).toBe(-0.24);
    expect(initial.focus).toBe(0);
    expect(initial.offset[0]).toBeCloseTo(15);
    expect(initial.offset[1]).toBeCloseTo(13.5);
    expect(initial.offset[2]).toBeCloseTo(20);
  });

  it('turns the court fully and finishes closer to the shuttle', () => {
    const initial = sampleCourtCamera(0, 1.4);
    const middle = sampleCourtCamera(0.45, 1.4);
    const final = sampleCourtCamera(1, 1.4);
    expect(middle.orbit).toBeGreaterThan(initial.orbit + Math.PI);
    expect(final.orbit - initial.orbit).toBeCloseTo(Math.PI * 2);
    expect(middle.focus).toBe(0);
    expect(final.focus).toBe(1);
    expect(final.distance).toBeLessThan(initial.distance / 2);
    expect(Math.hypot(...final.offset)).toBeCloseTo(final.distance);
  });

  it('keeps all camera values finite and bounded for malformed progress and aspect', () => {
    for (const p of [-10, 0, 0.05, 0.6, 0.94, 1, 20, NaN, Infinity]) {
      for (const aspect of [0, 0.1, 1, 4, NaN, Infinity]) {
        const sample = sampleCourtCamera(p, aspect);
        expect(
          [sample.orbit, sample.focus, sample.distance, ...sample.offset].every(Number.isFinite),
        ).toBe(true);
        expect(sample.focus).toBeGreaterThanOrEqual(0);
        expect(sample.focus).toBeLessThanOrEqual(1);
        expect(sample.distance).toBeGreaterThan(0);
      }
    }
  });

  it('has the same progress damping at 30fps and 60fps', () => {
    const atRate = (rate: number) => {
      let value = 0;
      for (let i = 0; i < rate; i++) value = dampCourtProgress(value, 1, 1 / rate);
      return value;
    };
    expect(atRate(30)).toBeCloseTo(atRate(60), 8);
    expect(atRate(30)).toBeGreaterThan(0.99);
    expect(dampCourtProgress(0.5, 1, 0)).toBe(0.5);
    expect(dampCourtProgress(NaN, NaN, NaN)).toBe(0);
  });
});
