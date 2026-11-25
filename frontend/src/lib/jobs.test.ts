import { describe, expect, it } from 'vitest';
import { JOBS_STORAGE_KEY, MAX_SAVED_JOBS, clearJobs, loadJobs, removeJob, saveJob } from './jobs';
import type { StoredJob } from './types';

function memoryStorage() {
  const data = new Map<string, string>();
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value);
    },
    removeItem: (key: string) => {
      data.delete(key);
    },
  };
}
const job = (id = 'job-1', updatedAt = 10): StoredJob => ({
  id,
  token: 'private-job-token',
  name: 'rally.mp4',
  size: 1000,
  createdAt: 1,
  updatedAt,
  state: 'queued',
  stage: 'queued',
});

describe('private browser analysis history', () => {
  it('persists selected modes and migrates legacy records to combined', () => {
    const storage = memoryStorage();
    storage.setItem(JOBS_STORAGE_KEY, JSON.stringify([job('legacy')]));
    expect(loadJobs(storage)[0].analysis_mode).toBe('combined');
    saveJob({ ...job('court', 20), analysis_mode: 'court' }, storage);
    expect(loadJobs(storage).map((item) => item.analysis_mode)).toEqual(['court', 'combined']);
  });

  it('round-trips and updates a job without duplicate records', () => {
    const storage = memoryStorage();
    saveJob(job(), storage);
    saveJob({ ...job(), state: 'succeeded', updatedAt: 20 }, storage);
    expect(loadJobs(storage)).toMatchObject([{ id: 'job-1', state: 'succeeded', updatedAt: 20 }]);
  });

  it('persists only the explicit history fields, never the team key or signed URLs', () => {
    const storage = memoryStorage();
    saveJob(
      {
        ...job(),
        teamKey: 'must-not-persist',
        videoUrl: 'https://private.example/signed',
      } as StoredJob,
      storage,
    );
    const raw = storage.getItem(JOBS_STORAGE_KEY)!;
    expect(raw).not.toContain('must-not-persist');
    expect(raw).not.toContain('videoUrl');
    expect(raw).toContain('private-job-token');
  });

  it('bounds history and keeps the most recently updated analyses', () => {
    const storage = memoryStorage();
    for (let i = 1; i <= MAX_SAVED_JOBS + 5; i++) saveJob(job('job-' + i, i), storage);
    const results = loadJobs(storage);
    expect(results).toHaveLength(MAX_SAVED_JOBS);
    expect(results[0].id).toBe('job-' + (MAX_SAVED_JOBS + 5));
    expect(results.at(-1)?.id).toBe('job-6');
  });

  it('handles malformed, stale and duplicate records without crashing', () => {
    const storage = memoryStorage();
    storage.setItem(JOBS_STORAGE_KEY, '{broken');
    expect(loadJobs(storage)).toEqual([]);
    storage.setItem(
      JOBS_STORAGE_KEY,
      JSON.stringify([
        job('same', 1),
        { token: 'invalid' },
        job('same', 2),
        { ...job('unknown'), state: 'invented' },
      ]),
    );
    expect(loadJobs(storage)).toMatchObject([{ id: 'same', updatedAt: 2 }]);
  });

  it('preserves unrelated localStorage entries when removing history', () => {
    const storage = memoryStorage();
    storage.setItem('other-app', 'preserve');
    saveJob(job('a'), storage);
    saveJob(job('b'), storage);
    expect(removeJob('a', storage).map((item) => item.id)).toEqual(['b']);
    clearJobs(storage);
    expect(loadJobs(storage)).toEqual([]);
    expect(storage.getItem('other-app')).toBe('preserve');
  });

  it('keeps a new result usable in memory if persistence is blocked', () => {
    const storage = {
      getItem: () => {
        throw Error('blocked');
      },
      setItem: () => {
        throw Error('quota');
      },
      removeItem: () => {
        throw Error('blocked');
      },
    };
    expect(loadJobs(storage)).toEqual([]);
    expect(saveJob(job(), storage)).toHaveLength(1);
    expect(() => clearJobs(storage)).not.toThrow();
  });
});
