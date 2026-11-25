import { test, expect, type Page, type Route } from '@playwright/test';
import { mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const samplePath = fileURLToPath(new URL('../tests/fixtures/court-example.mp4', import.meta.url));
const sampleVideo = readFileSync(samplePath);
const jobId = '628c566a-f52a-42ae-bf76-4ad6be973cd8';
const jobToken = 'qa-private-job-token';

/** Serve the same byte-range contract as FastAPI FileResponse, not a fake body. */
async function fulfillSampleVideo(route: Route) {
  const length = sampleVideo.byteLength;
  const headers: Record<string, string> = { 'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-store' };
  const range = route.request().headers().range;
  if (!range) return route.fulfill({ status: 200, headers: { ...headers, 'Content-Length': String(length) }, body: sampleVideo });
  const match = /^bytes=(\d*)-(\d*)$/.exec(range);
  if (!match || (!match[1] && !match[2])) return route.fulfill({ status: 416, headers: { ...headers, 'Content-Range': `bytes */${length}` }, body: '' });
  const start = match[1] ? Number(match[1]) : Math.max(0, length - Number(match[2]));
  const end = match[1] && match[2] ? Math.min(length - 1, Number(match[2])) : length - 1;
  if (start >= length || end < start) return route.fulfill({ status: 416, headers: { ...headers, 'Content-Range': `bytes */${length}` }, body: '' });
  const body = sampleVideo.subarray(start, end + 1);
  return route.fulfill({ status: 206, headers: { ...headers, 'Content-Range': `bytes ${start}-${end}/${length}`, 'Content-Length': String(body.byteLength) }, body });
}

async function noHorizontalOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
}

async function uploadSample(page: Page) {
  await page.locator('input[type="file"]').first().setInputFiles(samplePath);
  const submit = page.locator('form button[type="submit"]');
  await expect(submit).toBeEnabled();
  await submit.click();
}

test.beforeEach(async ({ page }) => {
  await page.route('**/healthz', route => route.fulfill({ json: { status: 'ok', backend: 'local' } }));
});

test('all primary screens render without overflow or uncaught errors', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const route of ['/', '/analyse', '/rallies', '/compare', '/project']) {
    await page.goto('/#' + route);
    await expect(page.locator('main')).toBeVisible();
    await expect(page.locator('h1').first()).toBeVisible();
    await page.evaluate(async () => { await document.fonts.ready; });
    expect(await page.locator('h1').first().evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
    if (route === '/') {
      const court = page.locator('.court-scene');
      await expect(court).toHaveAttribute('data-renderer', /webgl|static/);
      // Give the lazy WebGL renderer a chance to load, but retain valid coverage
      // on machines that correctly use the static fallback.
      await expect(court).toHaveAttribute('data-renderer', 'webgl', { timeout: 3000 }).catch(() => {});
    }
    await noHorizontalOverflow(page);
    const directory = join('test-results', 'screenshots', testInfo.project.name);
    mkdirSync(directory, { recursive: true });
    await page.screenshot({ path: join(directory, `${route === '/' ? 'overview' : route.slice(1)}.png`), fullPage: true, animations: 'disabled' });
  }
  if (testInfo.project.name === 'mobile') {
    await page.setViewportSize({ width: 320, height: 780 });
    for (const route of ['/', '/analyse', '/compare', '/project']) {
      await page.goto('/#' + route);
      await expect(page.locator('h1').first()).toBeVisible();
      await noHorizontalOverflow(page);
      expect(await page.locator('h1').first().evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
      await page.screenshot({ path: join('test-results', 'screenshots', 'mobile', `${route === '/' ? 'overview' : route.slice(1)}-320.png`), fullPage: true, animations: 'disabled' });
    }
  }
  expect(errors).toEqual([]);
});

test('the default recorded example is identified as combined analysis', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /Explore example/i }).or(page.getByRole('link', { name: /Explore example/i })).first().click();
  await expect(page.locator('video').first()).toHaveAttribute('src', /\.mp4(?:\?|$)/);
  await expect(page.locator('.example-notice')).toContainText(/combined/i);
  await expect(page.locator('.example-notice')).not.toContainText(/court-only|court prototype|component example/i);
  await noHorizontalOverflow(page);
});

test('reduced motion and keyboard navigation remain usable', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true);
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('main')).toBeFocused();
  const courtTab = page.getByRole('tab', { name: /Court/ });
  await courtTab.focus();
  await courtTab.press('ArrowRight');
  const playersTab = page.getByRole('tab', { name: /Players/ });
  await expect(playersTab).toBeFocused();
  await expect(playersTab).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('tabpanel')).toContainText(/Player tracking/);
  await page.goto('/#/analyse');
  await expect(page.locator('input[type="file"]').first()).toBeAttached();
  const dropzone = page.getByRole('button', { name: /Drop your rally here/ });
  await dropzone.focus();
  const chooserEvent = page.waitForEvent('filechooser');
  await dropzone.press('Enter');
  const chooser = await chooserEvent;
  await chooser.setFiles(samplePath);
  await expect(page.getByRole('button', { name: 'Analyse video', exact: true })).toBeEnabled();
  await expect(page.locator('html')).toHaveAttribute('data-motion', 'off');
  await noHorizontalOverflow(page);
});

test('a missing WebGL context leaves the static court visible', async ({ page }) => {
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, name: string, ...options: unknown[]) {
      if (name === 'webgl' || name === 'webgl2' || name === 'experimental-webgl') return null;
      return Reflect.apply(original, this, [name, ...options]);
    } as typeof original;
  });
  await page.goto('/');
  const court = page.locator('.court-scene');
  await expect(court).toHaveAttribute('data-renderer', 'static');
  await expect(court.locator('svg')).toBeVisible();
  await expect(court).toHaveAccessibleName(/Illustrative.*badminton court/i);
});

test('WebGL context loss shows the fallback without breaking navigation', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  const court = page.locator('.court-scene');
  await expect(court).toHaveAttribute('data-renderer', /webgl|static/);
  await page.waitForTimeout(1500);
  test.skip(await court.getAttribute('data-renderer') !== 'webgl', 'WebGL2 is unavailable in this browser environment.');
  const lost = await court.locator('canvas').evaluate((canvas: HTMLCanvasElement) => {
    const extension = canvas.getContext('webgl2')?.getExtension('WEBGL_lose_context');
    extension?.loseContext();
    return Boolean(extension);
  });
  test.skip(!lost, 'The browser does not expose the context-loss test extension.');
  await expect(court).toHaveAttribute('data-renderer', 'static');
  await page.goto('/#/project');
  await expect(page.locator('main')).toBeVisible();
  expect(errors).toEqual([]);
});

test('invalid input does not create an inference job', async ({ page }) => {
  let creates = 0;
  await page.route('**/api/jobs', async route => {
    creates += 1;
    await route.fulfill({ status: 500, json: { detail: 'Unexpected request' } });
  });
  await page.goto('/#/analyse');
  await page.locator('input[type="file"]').first().setInputFiles({ name: 'wrong.txt', mimeType: 'text/plain', buffer: Buffer.from('not a video') });
  await expect(page.getByRole('alert').first()).toContainText(/MP4|video|supported/i);
  expect(creates).toBe(0);
});

test('a real file follows the job API contract and displays result warnings', async ({ page }, testInfo) => {
  const requests: Array<{ method: string; path: string; authorization: string | undefined }> = [];
  let reads = 0;
  await page.addInitScript(() => {
    type UploadEvidence = { size: number; type: string; name: string };
    const state = window as unknown as { __qaFileUploads: UploadEvidence[] };
    state.__qaFileUploads = [];
    const original = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send = function (body?: Document | XMLHttpRequestBodyInit | null) {
      if (body instanceof File) state.__qaFileUploads.push({ size: body.size, type: body.type, name: body.name });
      return original.call(this, body);
    };
  });
  await page.route(/\/api\/jobs(?:\/[^?]*)?(?:\?.*)?$/, async route => {
    const request = route.request();
    const url = new URL(request.url());
    requests.push({ method: request.method(), path: url.pathname, authorization: request.headers().authorization });
    if (url.pathname === '/api/jobs' && request.method() === 'POST') {
      return route.fulfill({ status: 201, json: { id: jobId, token: jobToken, max_bytes: 209715200, upload_url: `/api/jobs/${jobId}/input`, upload_headers: { 'Content-Type': 'video/mp4', Authorization: `Bearer ${jobToken}` } } });
    }
    expect(request.headers().authorization).toBe(`Bearer ${jobToken}`);
    if (url.pathname.endsWith('/input') && request.method() === 'PUT') {
      // Edge does not expose File-backed XHR bodies through postDataBuffer.
      // Check the actual send argument below and the wire content type here.
      expect(request.headers()['content-type']).toContain('video/mp4');
      return route.fulfill({ json: { bytes: sampleVideo.byteLength } });
    }
    if (url.pathname.endsWith('/start')) return route.fulfill({ json: { id: jobId, state: 'queued' } });
    if (url.pathname.endsWith('/result')) return route.fulfill({ json: { url: url.searchParams.get('artifact') === 'report' ? '/qa/report.json' : '/qa/result.mp4' } });
    reads += 1;
    return route.fulfill({ json: { id: jobId, state: reads === 1 ? 'running' : 'succeeded', stage: reads === 1 ? 'court' : 'complete', error: null, created: 1795590000, updated: 1795590010 } });
  });
  await page.route('**/qa/result.mp4', fulfillSampleVideo);
  await page.route('**/qa/report.json', route => route.fulfill({ json: {
    pose_records: 600, input: { duration_seconds: 10, fps: 30, source_width: 1280, source_height: 720, width: 1280, height: 720, resized: false },
    device: 'cpu', total_seconds: 12.3, stage_seconds: { court: 1.2, pose_and_racket: 5.4 }, models: [],
    warnings: ['Court calibration rejected: court mapping unavailable.'], output: 'H.264 MP4, silent',
  } }));
  await page.goto('/#/analyse');
  const access = page.locator('input[type="password"]').first();
  if (await access.count()) {
    await page.locator('summary').filter({ hasText: 'Team access' }).click();
    await access.fill('qa-team-secret');
  }
  await uploadSample(page);
  await expect(page.locator('video[src*="/qa/result.mp4"]')).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText(/Court calibration rejected/).first()).toBeVisible();
  expect(requests.filter(request => request.method === 'POST' && request.path === '/api/jobs')).toHaveLength(1);
  expect(requests.filter(request => request.path.endsWith('/start'))).toHaveLength(1);
  expect(await page.evaluate(() => (window as unknown as { __qaFileUploads: unknown[] }).__qaFileUploads)).toEqual([
    { size: sampleVideo.byteLength, type: 'video/mp4', name: 'court-example.mp4' },
  ]);
  expect(page.url()).not.toContain(jobToken);
  const storage = await page.evaluate(() => JSON.stringify({ ...localStorage }));
  expect(storage).toContain(jobId);
  expect(storage).not.toContain('qa-team-secret');
  await page.screenshot({ path: join('test-results', 'screenshots', testInfo.project.name, 'review-warning.png'), fullPage: true, animations: 'disabled' });
  await page.goto('/#/rallies');
  await expect(page.getByText('court-example.mp4').first()).toBeVisible();
});

test('an unavailable backend presents a recoverable error without duplicate submissions', async ({ page }) => {
  let creates = 0;
  await page.route('**/api/jobs', async route => {
    creates += 1;
    await route.fulfill({ status: 503, json: { detail: 'The analysis service is unavailable.' } });
  });
  await page.goto('/#/analyse');
  await uploadSample(page);
  await expect(page.getByRole('alert').first()).toContainText(/unavailable/i);
  expect(creates).toBe(1);
  await noHorizontalOverflow(page);
});

test('an uncertain remote submission is not automatically submitted again', async ({ page }) => {
  let creates = 0;
  let starts = 0;
  await page.route(/\/api\/jobs(?:\/[^?]*)?(?:\?.*)?$/, async route => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === '/api/jobs' && request.method() === 'POST') {
      creates += 1;
      return route.fulfill({ status: 201, json: { id: jobId, token: jobToken, max_bytes: 209715200, upload_url: `/api/jobs/${jobId}/input`, upload_headers: { 'Content-Type': 'video/mp4', Authorization: `Bearer ${jobToken}` } } });
    }
    if (url.pathname.endsWith('/input')) return route.fulfill({ json: { bytes: sampleVideo.byteLength } });
    if (url.pathname.endsWith('/start')) {
      starts += 1;
      return route.fulfill({ status: 503, json: { detail: 'Submission uncertain; check job status before submitting again.' } });
    }
    return route.fulfill({ json: { id: jobId, state: 'dispatch_unknown', stage: 'dispatch_unknown', error: 'Submission uncertain; check job status before submitting again.', created: 1795590000, updated: 1795590010 } });
  });
  await page.goto('/#/analyse');
  await uploadSample(page);
  await expect(page.getByText(/Submission uncertain/).first()).toBeVisible();
  await page.waitForTimeout(1200);
  expect(creates).toBe(1);
  expect(starts).toBe(1);
});

test('signed link refresh preserves playback time and changing jobs clears the previous report', async ({ page }) => {
  const secondId = 'a98b6f5e-54c2-444e-8ef5-74b885386a54';
  const generations = new Map<string, number>();
  await page.addInitScript(({ first, second, token }) => {
    const stamp = Date.now();
    localStorage.setItem('rallymirror.jobs.v1', JSON.stringify([
      { id: first, token, name: 'First rally.mp4', size: 2000, createdAt: stamp, updatedAt: stamp, state: 'succeeded', stage: 'complete', error: null },
      { id: second, token: 'second-private-token', name: 'Second rally.mp4', size: 2000, createdAt: stamp - 1, updatedAt: stamp - 1, state: 'succeeded', stage: 'complete', error: null },
    ]));
  }, { first: jobId, second: secondId, token: jobToken });
  await page.route(/\/api\/jobs\/[^/]+(?:\/result)?(?:\?.*)?$/, route => {
    const request = route.request();
    const url = new URL(request.url());
    const id = url.pathname.split('/')[3];
    expect(request.headers().authorization).toBe('Bearer ' + (id === jobId ? jobToken : 'second-private-token'));
    if (url.pathname.endsWith('/result')) {
      if (url.searchParams.get('artifact') === 'report') return route.fulfill({ json: { url: `/qa/${id}/report.json` } });
      const next = (generations.get(id) || 0) + 1;
      generations.set(id, next);
      return route.fulfill({ json: { url: `/qa/${id}/video-${next}.mp4` } });
    }
    return route.fulfill({ json: { id, state: 'succeeded', stage: 'complete', error: null, created: 1795590000, updated: 1795590010 } });
  });
  await page.route(/\/qa\/[^/]+\/video-\d+\.mp4$/, fulfillSampleVideo);
  await page.route(/\/qa\/[^/]+\/report\.json$/, route => {
    const isFirst = new URL(route.request().url()).pathname.includes(jobId);
    return route.fulfill({ json: { input: { fps: 30 }, device: isFirst ? 'cpu' : 'cuda', total_seconds: isFirst ? 1.1 : 2.2, warnings: [isFirst ? 'First source quality note' : 'Second source quality note'] } });
  });
  await page.goto('/#/analyse?job=' + jobId);
  const video = page.getByLabel('Annotated rally video', { exact: true });
  await expect(video).toBeVisible();
  await expect(page.getByText('First source quality note').first()).toBeVisible();
  await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.readyState)).toBeGreaterThanOrEqual(2);
  await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.seekable.length ? element.seekable.end(element.seekable.length - 1) : 0)).toBeGreaterThan(2);
  await video.evaluate((element: HTMLVideoElement) => { element.currentTime = 2; });
  await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.currentTime)).toBeCloseTo(2, 1);
  const before = await video.getAttribute('src');
  await video.evaluate(element => element.dispatchEvent(new Event('error')));
  await expect(video).not.toHaveAttribute('src', before || '');
  await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.currentTime)).toBeCloseTo(2, 1);
  await page.getByRole('tab', { name: 'Run details', exact: true }).click();
  await expect(page.getByRole('tabpanel')).toContainText('cpu');
  await page.evaluate(id => { window.location.hash = '/analyse?job=' + id; }, secondId);
  await expect(page.getByText('Second rally.mp4', { exact: true })).toBeVisible();
  await expect(page.getByText('Second source quality note').first()).toBeVisible();
  await expect(page.getByText('First source quality note')).toHaveCount(0);
  await expect(page.getByRole('tab', { name: 'Overview', exact: true })).toHaveAttribute('aria-selected', 'true');
  await page.getByRole('tab', { name: 'Run details', exact: true }).click();
  await expect(page.getByRole('tabpanel')).toContainText('cuda');
  await expect(page.getByRole('tabpanel')).not.toContainText('cpu');
});

test('local comparison links playback, seeking and timing alignment without uploads', async ({ page }) => {
  let uploads = 0;
  page.on('request', request => {
    if (request.method() !== 'GET' && new URL(request.url()).pathname.startsWith('/api/')) uploads += 1;
  });
  await page.goto('/#/compare');
  await expect(page.getByRole('button', { name: 'Play both', exact: true })).toBeDisabled();
  await page.getByLabel('Choose attempt A', { exact: true }).setInputFiles(samplePath);
  await page.getByLabel('Choose attempt B', { exact: true }).setInputFiles(samplePath);
  await page.waitForFunction(() => {
    const videos = [...document.querySelectorAll('video')];
    return videos.length === 2 && videos.every(video => video.readyState >= 1 && video.duration > 0);
  });
  await page.getByRole('button', { name: 'Play both', exact: true }).click();
  await expect.poll(() => page.locator('video').evaluateAll(videos => videos.every(video => !(video as HTMLVideoElement).paused))).toBe(true);
  await expect.poll(() => page.getByLabel('Attempt A video', { exact: true }).evaluate((video: HTMLVideoElement) => video.currentTime)).toBeGreaterThan(0.1);
  await page.getByRole('button', { name: 'Pause both', exact: true }).click();
  await expect.poll(() => page.locator('video').evaluateAll(videos => videos.every(video => (video as HTMLVideoElement).paused))).toBe(true);
  const alignment = page.getByRole('slider', { name: /Attempt B offset/ });
  await alignment.focus();
  for (let i = 0; i < 5; i += 1) await alignment.press('ArrowRight');
  await expect(alignment).toHaveValue('0.5');
  await page.getByLabel('Attempt A video', { exact: true }).evaluate((video: HTMLVideoElement) => { video.currentTime = 2; });
  await expect.poll(() => page.getByLabel('Attempt B video', { exact: true }).evaluate((video: HTMLVideoElement) => video.currentTime)).toBeCloseTo(2.5, 1);
  await page.getByRole('button', { name: 'Reset alignment', exact: true }).click();
  await expect(alignment).toHaveValue('0');
  await expect.poll(() => page.getByLabel('Attempt B video', { exact: true }).evaluate((video: HTMLVideoElement) => video.currentTime)).toBeCloseTo(2, 1);
  await page.getByLabel('Attempt A video', { exact: true }).evaluate((video: HTMLVideoElement) => { video.currentTime = 0; });
  await alignment.focus();
  for (let i = 0; i < 10; i += 1) await alignment.press('ArrowLeft');
  await expect(alignment).toHaveValue('-1');
  await expect.poll(() => page.getByLabel('Attempt B video', { exact: true }).evaluate((video: HTMLVideoElement) => video.currentTime)).toBeCloseTo(0, 1);
  await page.getByRole('button', { name: 'Play both', exact: true }).click();
  await expect.poll(() => page.locator('video').evaluateAll(videos => {
    const [a, b] = videos as HTMLVideoElement[];
    return !a.paused && a.currentTime < 1 && b.paused && b.currentTime < 0.1;
  })).toBe(true);
  await expect.poll(() => page.getByLabel('Attempt A video', { exact: true }).evaluate((video: HTMLVideoElement) => video.currentTime)).toBeGreaterThan(1.1);
  await expect.poll(() => page.getByLabel('Attempt B video', { exact: true }).evaluate((video: HTMLVideoElement) => video.paused)).toBe(false);
  await page.getByLabel('Attempt A video', { exact: true }).evaluate((video: HTMLVideoElement) => { video.playbackRate = 1.5; });
  await expect(page.getByLabel('Attempt B video', { exact: true })).toHaveJSProperty('playbackRate', 1.5);
  await page.getByRole('button', { name: 'Pause both', exact: true }).click();
  await expect.poll(() => page.locator('video').evaluateAll(videos => videos.every(video => (video as HTMLVideoElement).paused))).toBe(true);
  // With B ahead, reaching B's end must stop A as well instead of letting it run.
  await alignment.focus();
  for (let i = 0; i < 20; i += 1) await alignment.press('ArrowRight');
  await expect(alignment).toHaveValue('1');
  await page.locator('video').evaluateAll(videos => {
    const [a, b] = videos as HTMLVideoElement[];
    a.currentTime = Math.max(0, b.duration - 1.2);
  });
  await page.getByRole('button', { name: 'Play both', exact: true }).click();
  await expect.poll(() => page.locator('video').evaluateAll(videos => videos.every(video => (video as HTMLVideoElement).paused))).toBe(true);
  await expect(page.getByRole('button', { name: 'Play both', exact: true })).toBeVisible();
  expect(uploads).toBe(0);
  await noHorizontalOverflow(page);
});
