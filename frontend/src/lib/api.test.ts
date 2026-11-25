import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, RallyMirrorApi, isAbortError } from './api';
import type { CreatedJob, JobCredential, JobStatus } from './types';
import { ANALYSIS_MODES } from './types';

const credential: JobCredential = { id: 'job-1', token: 'private-job-token' };
const status = (state: JobStatus['state']): JobStatus => ({
  analysis_mode: 'combined',
  ...credential,
  state,
  stage: state,
  error: null,
  created: 10,
  updated: 20,
});
const json = (body: unknown, statusCode = 200) =>
  new Response(JSON.stringify(body), {
    status: statusCode,
    headers: { 'Content-Type': 'application/json' },
  });
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('existing video-job API contract', () => {
  it.each(ANALYSIS_MODES)(
    'submits and reads the persisted %s analysis mode',
    async (analysisMode) => {
      const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(
          json({
            ...credential,
            analysis_mode: analysisMode,
            max_bytes: 100,
            upload_url: '/api/jobs/job-1/input',
            upload_headers: { 'Content-Type': 'video/mp4' },
          }),
        )
        .mockResolvedValueOnce(json({ ...status('running'), analysis_mode: analysisMode }));
      vi.stubGlobal('fetch', fetchMock);
      const api = new RallyMirrorApi();
      expect((await api.createJob('team', undefined, analysisMode)).analysis_mode).toBe(
        analysisMode,
      );
      expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ analysis_mode: analysisMode });
      expect((await api.getJob(credential)).analysis_mode).toBe(analysisMode);
    },
  );

  it('normalises legacy status responses to combined and rejects unknown modes', async () => {
    const legacy = { ...status('running') } as Partial<JobStatus>;
    delete legacy.analysis_mode;
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(json(legacy))
        .mockResolvedValueOnce(json({ ...legacy, analysis_mode: 'arbitrary' })),
    );
    const api = new RallyMirrorApi();
    expect((await api.getJob(credential)).analysis_mode).toBe('combined');
    await expect(api.getJob(credential)).rejects.toThrow('invalid analysis status');
  });

  it('does not silently run combined when an old server ignores a requested individual mode', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        json({
          ...credential,
          max_bytes: 100,
          upload_url: '/api/jobs/job-1/input',
          upload_headers: {},
        }),
      ),
    );
    await expect(new RallyMirrorApi().createJob('', undefined, 'court')).rejects.toThrow(
      'does not support individual analysis modes',
    );
  });

  it('keeps the team key on creation and the private job token in headers only', async () => {
    const requests: Array<{ url: string; init: RequestInit }> = [];
    const replies = [
      {
        ...credential,
        max_bytes: 100,
        upload_url: '/api/jobs/job-1/input',
        upload_headers: { Authorization: 'Bearer private-job-token', 'Content-Type': 'video/mp4' },
      },
      { id: credential.id, state: 'queued' },
      status('succeeded'),
      { url: '/api/jobs/job-1/download/result.mp4?expires=9999999999&signature=signed-video' },
      { url: 'https://storage.example/report.json?signed=report' },
      { warnings: ['Court calibration rejected'], total_seconds: 10 },
    ];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url, init = {}) => {
        requests.push({ url: String(url), init });
        return json(replies.shift());
      }),
    );
    const api = new RallyMirrorApi('https://api.example');
    const created = await api.createJob('team-secret');
    await api.startJob(created);
    expect((await api.getJob(created)).state).toBe('succeeded');
    const links = await api.getResultLinks(created);
    expect(links.videoUrl).toContain('https://api.example/api/jobs/job-1/download/result.mp4');
    expect((await api.getReport(links.reportUrl)).warnings).toEqual(['Court calibration rejected']);
    expect(requests[0].init.headers).toEqual({
      Authorization: 'Bearer team-secret',
      'Content-Type': 'application/json',
    });
    expect(JSON.parse(requests[0].init.body as string)).toEqual({ analysis_mode: 'combined' });
    expect(requests[1].init.headers).toEqual({ Authorization: 'Bearer private-job-token' });
    expect(requests[5].init.headers).toBeUndefined();
    expect(
      requests.every(
        (item) => !item.url.includes('private-job-token') && !item.url.includes('team-secret'),
      ),
    ).toBe(true);
    expect(requests.every((item) => item.init.credentials === 'omit')).toBe(true);
    expect(
      requests
        .filter((item) => item.url.includes('/result?'))
        .map((item) => item.url.split('artifact=')[1]),
    ).toEqual(['video', 'report']);
  });

  it('never resubmits an uncertain dispatch', async () => {
    const fetchMock = vi.fn(async () => json(status('dispatch_unknown')));
    vi.stubGlobal('fetch', fetchMock);
    const onUpdate = vi.fn();
    const result = await new RallyMirrorApi().watchJob(credential, { onUpdate });
    expect(result.state).toBe('dispatch_unknown');
    expect(onUpdate).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0]?.length).toBe(2);
  });

  it('polls active jobs sequentially and stops on completion', async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(json(status('running')))
      .mockResolvedValueOnce(json(status('succeeded')));
    vi.stubGlobal('fetch', fetchMock);
    const onUpdate = vi.fn();
    const watch = new RallyMirrorApi().watchJob(credential, { onUpdate, intervalMs: 500 });
    await vi.advanceTimersByTimeAsync(500);
    expect((await watch).state).toBe('succeeded');
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(onUpdate.mock.calls.map((call) => call[0].state)).toEqual(['running', 'succeeded']);
  });

  it('cancels the wait without performing another status request', async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => json(status('running'))),
    );
    const controller = new AbortController();
    const watch = new RallyMirrorApi().watchJob(credential, {
      signal: controller.signal,
      onUpdate: () => {},
    });
    const rejection = expect(watch).rejects.toMatchObject({ name: 'AbortError' });
    await vi.advanceTimersByTimeAsync(1);
    controller.abort();
    await rejection;
    await vi.advanceTimersByTimeAsync(10000);
    expect(fetch).toHaveBeenCalledOnce();
  });

  it('preserves HTTP errors and handles an offline API clearly', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(json({ detail: 'Enter the team access key' }, 401))
        .mockRejectedValueOnce(new TypeError('Network failed')),
    );
    const api = new RallyMirrorApi();
    await expect(api.createJob()).rejects.toMatchObject({
      status: 401,
      retryable: false,
      message: 'Enter the team access key',
    });
    await expect(api.health()).rejects.toMatchObject({
      status: 0,
      retryable: true,
      message: expect.stringContaining('Cannot reach'),
    });
  });

  it('recognises an HTML fallback instead of treating a frontend server as the API', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('<!doctype html><title>Frontend</title>')),
    );
    await expect(new RallyMirrorApi().health()).rejects.toThrow('unexpected response');
  });

  it('refuses a result from the wrong job', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => json({ ...status('succeeded'), id: 'another-job' })),
    );
    await expect(new RallyMirrorApi().getJob(credential)).rejects.toThrow(
      'invalid analysis status',
    );
  });

  it('rejects unsafe media links and refreshes ahead of local expiry', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => json({ url: 'javascript:alert(1)' })),
    );
    await expect(new RallyMirrorApi().getResultLinks(credential)).rejects.toThrow(
      'unsafe result link',
    );
    const expiry = Math.floor(Date.now() / 1000) + 60;
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => json({ url: `/download?expires=${expiry}&signature=safe` })),
    );
    expect((await new RallyMirrorApi().getResultLinks(credential)).expiresAt).toBe(
      expiry * 1000 - 30000,
    );
  });

  it('retains an API mount prefix for backend capabilities without altering cloud URLs', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(json({ url: '/api/jobs/job-1/download/result.mp4?signature=safe' }))
      .mockResolvedValueOnce(json({ url: 'https://bucket.example/report.json?signed=cloud' }));
    vi.stubGlobal('fetch', fetchMock);
    const links = await new RallyMirrorApi('/backend').getResultLinks(credential);
    expect(links.videoUrl).toBe(
      'http://localhost/backend/api/jobs/job-1/download/result.mp4?signature=safe',
    );
    expect(links.reportUrl).toBe('https://bucket.example/report.json?signed=cloud');
    expect(fetchMock.mock.calls[0][0]).toBe('/backend/api/jobs/job-1/result?artifact=video');
  });

  it('does not turn cancellation into an offline error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new DOMException('Cancelled', 'AbortError');
      }),
    );
    await expect(new RallyMirrorApi().getJob(credential)).rejects.toMatchObject({
      name: 'AbortError',
    });
    expect(isAbortError(new DOMException('Cancelled', 'AbortError'))).toBe(true);
  });

  it('times out an unresponsive request without retrying the mutation', async () => {
    vi.useFakeTimers();
    const request = vi.fn(
      (_url: string, init: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener(
            'abort',
            () => reject(new DOMException('Aborted', 'AbortError')),
            { once: true },
          );
        }),
    );
    vi.stubGlobal('fetch', request);
    const pending = new RallyMirrorApi().startJob(credential);
    const rejection = expect(pending).rejects.toMatchObject({ status: 408, retryable: true });
    await vi.advanceTimersByTimeAsync(20000);
    await rejection;
    expect(request).toHaveBeenCalledOnce();
  });
});

describe('direct upload', () => {
  const job: CreatedJob = {
    analysis_mode: 'combined',
    ...credential,
    max_bytes: 10,
    upload_url: 'https://bucket.example/signed-upload',
    upload_headers: { 'Content-Type': 'video/mp4' },
  };
  class FakeXHR {
    static latest: FakeXHR;
    headers: Record<string, string> = {};
    upload: {
      onprogress?: (event: { lengthComputable: boolean; loaded: number; total: number }) => void;
    } = {};
    timeout = 0;
    status = 200;
    responseText = '';
    onload?: () => void;
    onerror?: () => void;
    ontimeout?: () => void;
    onabort?: () => void;
    opened: unknown[] = [];
    file?: File;
    constructor() {
      FakeXHR.latest = this;
    }
    open(...args: unknown[]) {
      this.opened = args;
    }
    setRequestHeader(name: string, value: string) {
      this.headers[name] = value;
    }
    send(file: File) {
      this.file = file;
    }
    abort() {
      this.onabort?.();
    }
  }
  const file = () => new File(['video'], 'rally.mp4', { type: 'video/mp4' });

  it('uses exactly the supplied cloud headers and reports upload progress', async () => {
    vi.stubGlobal('XMLHttpRequest', FakeXHR);
    const progress = vi.fn();
    const upload = new RallyMirrorApi().upload(job, file(), { onProgress: progress });
    const xhr = FakeXHR.latest;
    expect(xhr.opened).toEqual(['PUT', 'https://bucket.example/signed-upload']);
    expect(xhr.headers).toEqual({ 'Content-Type': 'video/mp4' });
    xhr.upload.onprogress?.({ lengthComputable: true, loaded: 3, total: 5 });
    xhr.onload?.();
    await upload;
    expect(progress.mock.calls.map((call) => call[0])).toEqual([60, 100]);
  });

  it('blocks an oversized file before opening a request', async () => {
    const constructor = vi.fn();
    vi.stubGlobal('XMLHttpRequest', constructor);
    await expect(
      new RallyMirrorApi().upload({ ...job, max_bytes: 2 }, file()),
    ).rejects.toMatchObject({ status: 413 });
    expect(constructor).not.toHaveBeenCalled();
  });

  it('aborts an in-flight upload', async () => {
    vi.stubGlobal('XMLHttpRequest', FakeXHR);
    const controller = new AbortController();
    const upload = new RallyMirrorApi().upload(job, file(), { signal: controller.signal });
    controller.abort();
    await expect(upload).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('reports a signed-upload failure without starting an analysis', async () => {
    vi.stubGlobal('XMLHttpRequest', FakeXHR);
    const upload = new RallyMirrorApi().upload(job, file());
    FakeXHR.latest.status = 403;
    FakeXHR.latest.onload?.();
    await expect(upload).rejects.toBeInstanceOf(ApiError);
  });
});
