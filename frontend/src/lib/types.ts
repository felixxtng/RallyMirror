/** Public contracts of RallyMirror's existing FastAPI video-job service. */
export const ANALYSIS_MODES = [
  'combined',
  'court',
  'tracking',
  'pose',
  'racket',
  'shuttle',
  'phases',
] as const;
export type AnalysisMode = (typeof ANALYSIS_MODES)[number];

export function isAnalysisMode(value: unknown): value is AnalysisMode {
  return typeof value === 'string' && ANALYSIS_MODES.includes(value as AnalysisMode);
}

export type JobState =
  | 'uploading'
  | 'receiving'
  | 'uploaded'
  | 'dispatching'
  | 'dispatch_unknown'
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed';

export interface JobCredential {
  id: string;
  /** Private access capability. Never include it in a URL or visible label. */
  token: string;
}

export interface CreatedJob extends JobCredential {
  analysis_mode: AnalysisMode;
  max_bytes: number;
  upload_url: string;
  upload_headers: Record<string, string>;
}

export interface JobStatus {
  analysis_mode: AnalysisMode;
  id: string;
  state: JobState;
  stage: string;
  error: string | null;
  /** Unix timestamps in seconds, as returned by the API. */
  created: number;
  updated: number;
}

export interface StoredJob extends JobCredential {
  /** Missing on legacy browser records, which always used the combined pipeline. */
  analysis_mode?: AnalysisMode;
  name: string;
  size: number;
  /** Browser timestamps in milliseconds. */
  createdAt: number;
  updatedAt: number;
  state: JobState;
  stage: string;
  error?: string | null;
}

export interface ResultLinks {
  videoUrl: string;
  reportUrl: string;
  /** Refresh signed links before this browser timestamp. */
  expiresAt: number;
}

export interface RunReport {
  analysis_mode?: AnalysisMode;
  frames?: number;
  modules?: Partial<
    Record<
      Exclude<AnalysisMode, 'combined'>,
      {
        status: 'available' | 'unavailable' | 'not_requested';
        records: number;
      }
    >
  >;
  phase_counts?: Record<string, number>;
  inference_mode?: AnalysisMode;
  provenance?: string;
  pose_records?: number;
  input?: {
    duration_seconds?: number;
    fps?: number;
    source_width?: number;
    source_height?: number;
    width?: number;
    height?: number;
    resized?: boolean;
  };
  device?: string;
  models?: Array<{ path: string; sha256: string }>;
  stage_seconds?: Record<string, number>;
  total_seconds?: number;
  warnings?: string[];
  output?: string;
  feature_contract?: string;
  [key: string]: unknown;
}

export const JOB_STATES: readonly JobState[] = [
  'uploading',
  'receiving',
  'uploaded',
  'dispatching',
  'dispatch_unknown',
  'queued',
  'running',
  'succeeded',
  'failed',
];

export const ACTIVE_JOB_STATES: readonly JobState[] = ['dispatching', 'queued', 'running'];

export function isJobState(value: unknown): value is JobState {
  return typeof value === 'string' && JOB_STATES.includes(value as JobState);
}

export function isActiveJob(state: JobState): boolean {
  return ACTIVE_JOB_STATES.includes(state);
}
