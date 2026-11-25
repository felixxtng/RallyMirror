import {
  isActiveJob,
  isAnalysisMode,
  isJobState,
  type AnalysisMode,
  type CreatedJob,
  type JobCredential,
  type JobStatus,
  type ResultLinks,
  type RunReport,
} from './types';

export class ApiError extends Error {
  readonly status: number;
  readonly retryable: boolean;
  constructor(message: string, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.retryable = status === 0 || status === 408 || status === 429 || status >= 500;
  }
}

const abortError = () => new DOMException('Request cancelled', 'AbortError');

export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.';
}

function responseDetail(body: unknown, fallback: string): string {
  if (body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string')
    return body.detail;
  return fallback;
}

function responseError(status: number, body: unknown): ApiError {
  const defaults: Record<number, string> = {
    401: 'The team access key was not accepted. Check it and try again.',
    403: 'This result link has expired. Refresh the result to continue.',
    404: 'This analysis is unavailable or its private access token is no longer valid.',
    409: 'This analysis is not ready for that action yet.',
    413: 'The video exceeds the server upload limit. Choose a smaller clip.',
    422: 'The server could not accept this request.',
    429: 'The service is busy. Wait a moment before trying again.',
    502: 'The analysis API is unavailable. Start the RallyMirror API and worker, then retry.',
    503: 'The analysis service is unavailable. Try again after checking its status.',
    504: 'The service took too long to respond. Check the analysis status before submitting again.',
  };
  return new ApiError(
    responseDetail(body, defaults[status] || `The request failed (${status}).`),
    status,
  );
}

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const done = () => {
      signal?.removeEventListener('abort', cancel);
      resolve();
    };
    const timer = setTimeout(done, ms);
    const cancel = () => {
      clearTimeout(timer);
      reject(abortError());
    };
    signal?.addEventListener('abort', cancel, { once: true });
  });
}

export interface UploadOptions {
  signal?: AbortSignal;
  onProgress?: (percent: number) => void;
}

export interface WatchOptions {
  signal?: AbortSignal;
  onUpdate: (status: JobStatus) => void;
  intervalMs?: number;
}

/** Client for the existing API only. It never simulates training or inference. */
export class RallyMirrorApi {
  readonly baseUrl: string;
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl.trim().replace(/\/+$/, '');
    if (this.baseUrl && !/^https?:\/\//i.test(this.baseUrl) && !/^\/[^/]/.test(this.baseUrl)) {
      throw new TypeError('API URL must use http(s) or a same-origin path.');
    }
    if (this.baseUrl) {
      const parsed = new URL(this.baseUrl, 'http://localhost');
      if (parsed.username || parsed.password || parsed.search || parsed.hash) {
        throw new TypeError('API URL must not contain credentials, a query or a fragment.');
      }
    }
  }

  private endpoint(path: string): string {
    return this.baseUrl + path;
  }

  /** Resolve only http(s) capabilities; reject script/data URLs from a bad server. */
  private mediaUrl(value: unknown): string {
    if (typeof value !== 'string' || !value.trim())
      throw new ApiError('The server returned an invalid result link.');
    const origin = typeof window !== 'undefined' ? window.location.origin : 'http://localhost';
    const apiBase = new URL((this.baseUrl || '') + '/', origin);
    const prefix = apiBase.pathname.replace(/\/+$/, '');
    // The backend returns root-relative capabilities. Retain any reverse-proxy
    // mount prefix configured for the API, without changing signed cloud URLs.
    const resource =
      prefix && value.startsWith('/') && !value.startsWith('//') ? prefix + value : value;
    let url: URL;
    try {
      url = new URL(resource, apiBase);
    } catch {
      throw new ApiError('The server returned an invalid result link.');
    }
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) {
      throw new ApiError('The server returned an unsafe result link.');
    }
    return url.href;
  }

  private async request<T>(
    url: string,
    init: RequestInit,
    networkMessage: string,
    read: (response: Response) => Promise<T>,
  ): Promise<T> {
    if (init.signal?.aborted) throw abortError();
    const controller = new AbortController();
    const relayAbort = () => controller.abort();
    init.signal?.addEventListener('abort', relayAbort, { once: true });
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, 20000);
    try {
      const response = await fetch(url, {
        ...init,
        signal: controller.signal,
        credentials: 'omit',
        cache: 'no-store',
      });
      return await read(response);
    } catch (error) {
      if (timedOut)
        throw new ApiError(
          'The service took too long to respond. Check the analysis status before submitting again.',
          408,
        );
      if (isAbortError(error) || init.signal?.aborted) throw abortError();
      if (error instanceof ApiError) throw error;
      throw new ApiError(networkMessage);
    } finally {
      clearTimeout(timeout);
      init.signal?.removeEventListener('abort', relayAbort);
    }
  }

  private async json(path: string, init: RequestInit = {}): Promise<unknown> {
    return this.request(
      this.endpoint(path),
      init,
      'Cannot reach the RallyMirror API. Start the API service or check your connection.',
      async (response) => {
        let body: unknown;
        try {
          body = await response.json();
        } catch (error) {
          if (isAbortError(error)) throw error;
          body = undefined;
        }
        if (!response.ok) throw responseError(response.status, body);
        if (body === undefined)
          throw new ApiError(
            'The server returned an unexpected response. Check the configured API address.',
          );
        return body;
      },
    );
  }

  private auth(job: JobCredential): HeadersInit {
    return { Authorization: `Bearer ${job.token}` };
  }
  private jobPath(job: JobCredential): string {
    return '/api/jobs/' + encodeURIComponent(job.id);
  }

  async health(
    signal?: AbortSignal,
  ): Promise<{ status: string; backend?: string; supported_modes?: AnalysisMode[] }> {
    const body = (await this.json('/healthz', { signal })) as {
      status?: unknown;
      backend?: unknown;
      supported_modes?: unknown;
    };
    if (body?.status !== 'ok') throw new ApiError('The analysis API is not ready.');
    return {
      status: 'ok',
      ...(typeof body.backend === 'string' ? { backend: body.backend } : {}),
      ...(Array.isArray(body.supported_modes)
        ? { supported_modes: body.supported_modes.filter(isAnalysisMode) }
        : {}),
    };
  }

  async createJob(
    accessKey = '',
    signal?: AbortSignal,
    analysisMode: AnalysisMode = 'combined',
  ): Promise<CreatedJob> {
    if (!isAnalysisMode(analysisMode)) throw new ApiError('Choose a supported analysis mode.', 422);
    const body = (await this.json('/api/jobs', {
      method: 'POST',
      headers: { Authorization: `Bearer ${accessKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ analysis_mode: analysisMode }),
      signal,
    })) as Partial<CreatedJob>;
    if (
      !body ||
      typeof body.id !== 'string' ||
      !body.id ||
      typeof body.token !== 'string' ||
      !body.token ||
      typeof body.max_bytes !== 'number' ||
      !Number.isFinite(body.max_bytes) ||
      body.max_bytes <= 0 ||
      typeof body.upload_url !== 'string' ||
      !body.upload_headers ||
      typeof body.upload_headers !== 'object' ||
      Array.isArray(body.upload_headers) ||
      !Object.values(body.upload_headers).every((value) => typeof value === 'string') ||
      (body.analysis_mode != null && !isAnalysisMode(body.analysis_mode))
    ) {
      throw new ApiError('The server returned an invalid upload session.');
    }
    // Validate the URL now, before handing a file or supplied headers to it.
    this.mediaUrl(body.upload_url);
    if (analysisMode !== 'combined' && body.analysis_mode == null) {
      throw new ApiError(
        'This API does not support individual analysis modes yet. Update the backend before uploading.',
      );
    }
    if ((body.analysis_mode ?? 'combined') !== analysisMode) {
      throw new ApiError(
        'The server selected a different analysis mode. No video was uploaded; please try again.',
      );
    }
    return { ...body, analysis_mode: body.analysis_mode ?? 'combined' } as CreatedJob;
  }

  upload(job: CreatedJob, file: File, options: UploadOptions = {}): Promise<void> {
    return new Promise((resolve, reject) => {
      if (options.signal?.aborted) {
        reject(abortError());
        return;
      }
      if (!file.size) {
        reject(new ApiError('The selected video is empty.'));
        return;
      }
      if (file.size > job.max_bytes) {
        reject(new ApiError('The video exceeds the server upload limit.', 413));
        return;
      }
      const xhr = new XMLHttpRequest();
      const onAbort = () => xhr.abort();
      const cleanup = () => options.signal?.removeEventListener('abort', onAbort);
      try {
        xhr.open('PUT', this.mediaUrl(job.upload_url));
        xhr.timeout = 5 * 60 * 1000;
        for (const [name, value] of Object.entries(job.upload_headers))
          xhr.setRequestHeader(name, value);
      } catch {
        reject(new ApiError('The upload session could not be opened. Choose the video again.'));
        return;
      }
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && event.total > 0)
          options.onProgress?.(
            Math.min(100, Math.max(0, Math.round((event.loaded * 100) / event.total))),
          );
      };
      xhr.onload = () => {
        cleanup();
        if (xhr.status >= 200 && xhr.status < 300) {
          options.onProgress?.(100);
          resolve();
          return;
        }
        let detail: unknown;
        try {
          detail = JSON.parse(xhr.responseText);
        } catch {
          /* Cloud uploads may return XML. */
        }
        reject(responseError(xhr.status, detail));
      };
      xhr.onerror = () => {
        cleanup();
        reject(
          new ApiError('Upload interrupted. Check your connection and the server upload settings.'),
        );
      };
      xhr.ontimeout = () => {
        cleanup();
        reject(
          new ApiError(
            'The upload timed out. Check your connection before submitting a new analysis.',
          ),
        );
      };
      xhr.onabort = () => {
        cleanup();
        reject(abortError());
      };
      options.signal?.addEventListener('abort', onAbort, { once: true });
      try {
        xhr.send(file);
      } catch {
        cleanup();
        reject(new ApiError('The browser could not upload this video. Select it again and retry.'));
      }
    });
  }

  async startJob(
    job: JobCredential,
    signal?: AbortSignal,
  ): Promise<{ id: string; state: import('./types').JobState; analysis_mode: AnalysisMode }> {
    const body = (await this.json(this.jobPath(job) + '/start', {
      method: 'POST',
      headers: this.auth(job),
      signal,
    })) as { id?: unknown; state?: unknown; analysis_mode?: unknown };
    if (
      !body ||
      body.id !== job.id ||
      !isJobState(body.state) ||
      (body.analysis_mode != null && !isAnalysisMode(body.analysis_mode))
    )
      throw new ApiError('The server returned an invalid analysis status.');
    return { id: job.id, state: body.state, analysis_mode: body.analysis_mode ?? 'combined' };
  }

  async getJob(job: JobCredential, signal?: AbortSignal): Promise<JobStatus> {
    const body = (await this.json(this.jobPath(job), {
      headers: this.auth(job),
      signal,
    })) as Partial<JobStatus>;
    if (
      !body ||
      body.id !== job.id ||
      !isJobState(body.state) ||
      typeof body.stage !== 'string' ||
      typeof body.created !== 'number' ||
      typeof body.updated !== 'number' ||
      (body.analysis_mode != null && !isAnalysisMode(body.analysis_mode))
    ) {
      throw new ApiError('The server returned an invalid analysis status.');
    }
    return {
      id: body.id,
      analysis_mode: body.analysis_mode ?? 'combined',
      state: body.state,
      stage: body.stage,
      created: body.created,
      updated: body.updated,
      error: typeof body.error === 'string' ? body.error : null,
    };
  }

  /** Only reads status. It never retries a submission or starts a job implicitly. */
  async watchJob(job: JobCredential, options: WatchOptions): Promise<JobStatus> {
    for (;;) {
      const status = await this.getJob(job, options.signal);
      options.onUpdate(status);
      if (!isActiveJob(status.state)) return status;
      const delay =
        typeof document !== 'undefined' && document.hidden
          ? Math.max(10000, options.intervalMs ?? 3000)
          : Math.max(100, options.intervalMs ?? 3000);
      await wait(delay, options.signal);
    }
  }

  async getResultLinks(job: JobCredential, signal?: AbortSignal): Promise<ResultLinks> {
    const get = async (artifact: 'video' | 'report') => {
      const body = (await this.json(`${this.jobPath(job)}/result?artifact=${artifact}`, {
        headers: this.auth(job),
        signal,
      })) as { url?: unknown };
      return this.mediaUrl(body?.url);
    };
    const [videoUrl, reportUrl] = await Promise.all([get('video'), get('report')]);
    // Cap at 14 minutes even for opaque cloud signatures (server links last 15).
    let expiresAt = Date.now() + 14 * 60 * 1000;
    for (const value of [videoUrl, reportUrl]) {
      const expires = Number(new URL(value).searchParams.get('expires'));
      if (Number.isFinite(expires) && expires > 0)
        expiresAt = Math.min(expiresAt, expires * 1000 - 30000);
    }
    return { videoUrl, reportUrl, expiresAt };
  }

  async getReport(reportUrl: string, signal?: AbortSignal): Promise<RunReport> {
    return this.request(
      this.mediaUrl(reportUrl),
      { signal },
      'The run report could not be downloaded. Refresh the result links and try again.',
      async (response) => {
        let body: unknown;
        try {
          body = await response.json();
        } catch (error) {
          if (isAbortError(error)) throw error;
          body = undefined;
        }
        if (!response.ok) throw responseError(response.status, body);
        if (!body || typeof body !== 'object' || Array.isArray(body))
          throw new ApiError('The server returned an invalid run report.');
        return body as RunReport;
      },
    );
  }
}
