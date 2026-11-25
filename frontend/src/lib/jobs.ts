import { isAnalysisMode, isJobState, type StoredJob } from './types';

/** Tokens grant access to private results. Keep this storage on the user's browser. */
export const JOBS_STORAGE_KEY = 'rallymirror.jobs.v1';
export const MAX_SAVED_JOBS = 20;

type StorageLike = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

function browserStorage(): StorageLike | undefined {
  try {
    return typeof window !== 'undefined' ? window.localStorage : undefined;
  } catch {
    return undefined;
  }
}

function validStoredJob(value: unknown): value is StoredJob {
  if (!value || typeof value !== 'object') return false;
  const v = value as Partial<StoredJob>;
  return (
    typeof v.id === 'string' &&
    v.id.length > 0 &&
    v.id.length <= 128 &&
    typeof v.token === 'string' &&
    v.token.length > 0 &&
    v.token.length <= 1024 &&
    typeof v.name === 'string' &&
    v.name.length <= 512 &&
    typeof v.size === 'number' &&
    Number.isFinite(v.size) &&
    v.size >= 0 &&
    typeof v.createdAt === 'number' &&
    Number.isFinite(v.createdAt) &&
    typeof v.updatedAt === 'number' &&
    Number.isFinite(v.updatedAt) &&
    isJobState(v.state) &&
    (v.analysis_mode == null || isAnalysisMode(v.analysis_mode)) &&
    typeof v.stage === 'string' &&
    v.stage.length <= 128 &&
    (v.error == null || typeof v.error === 'string')
  );
}

/** Whitelist fields, so keys, signed URLs or reports cannot accidentally persist. */
function storageRecord(job: StoredJob): StoredJob {
  return {
    id: job.id,
    analysis_mode: job.analysis_mode ?? 'combined',
    token: job.token,
    name: job.name.slice(0, 512),
    size: job.size,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    state: job.state,
    stage: job.stage.slice(0, 128),
    error: job.error?.slice(0, 1000) ?? null,
  };
}

export function loadJobs(storage: StorageLike | undefined = browserStorage()): StoredJob[] {
  if (!storage) return [];
  try {
    const raw: unknown = JSON.parse(storage.getItem(JOBS_STORAGE_KEY) || '[]');
    if (!Array.isArray(raw)) return [];
    const seen = new Set<string>();
    return raw
      .filter(validStoredJob)
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .filter((job) => {
        if (seen.has(job.id)) return false;
        seen.add(job.id);
        return true;
      })
      .slice(0, MAX_SAVED_JOBS)
      .map(storageRecord);
  } catch {
    return [];
  }
}

/** Returns the next list even if private browsing/storage quota prevents persistence. */
export function saveJob(
  job: StoredJob,
  storage: StorageLike | undefined = browserStorage(),
): StoredJob[] {
  if (!validStoredJob(job)) throw new TypeError('Invalid saved analysis');
  const next = [storageRecord(job), ...loadJobs(storage).filter((item) => item.id !== job.id)]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, MAX_SAVED_JOBS);
  try {
    storage?.setItem(JOBS_STORAGE_KEY, JSON.stringify(next));
  } catch {
    /* Keep the in-memory analysis usable. */
  }
  return next;
}

/** Removes browser history only. Does not claim to delete server-side media. */
export function removeJob(
  id: string,
  storage: StorageLike | undefined = browserStorage(),
): StoredJob[] {
  const next = loadJobs(storage).filter((job) => job.id !== id);
  try {
    storage?.setItem(JOBS_STORAGE_KEY, JSON.stringify(next));
  } catch {
    /* In-memory history still updates. */
  }
  return next;
}

/** Clears only RallyMirror's history key, never unrelated browser data. */
export function clearJobs(storage: StorageLike | undefined = browserStorage()): void {
  try {
    storage?.removeItem(JOBS_STORAGE_KEY);
  } catch {
    /* Storage may be disabled. */
  }
}
