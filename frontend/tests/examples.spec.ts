import { test, expect } from '@playwright/test';
import { createHash } from 'node:crypto';

const modes = ['combined', 'court', 'tracking', 'pose', 'racket', 'shuttle', 'phases'];
type Provenance = {
  runtime_commit: string;
  inference: string;
  caveat: string;
  views: Array<{ mode: string; video: string; bytes: number; sha256: string }>;
};
type ExampleReport = {
  analysis_mode: string;
  frames: number;
  modules: Record<string, { status: string; records: number }>;
  coverage_note: string;
};

test('all seven recorded example views play and match their reports and provenance', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const provenanceResponse = await page.request.get('/media/demo-provenance.json');
  expect(provenanceResponse.ok()).toBe(true);
  const provenance = await provenanceResponse.json() as Provenance;
  expect(provenance.runtime_commit).toMatch(/^[a-f0-9]{40}$/);
  expect(provenance.inference).toMatch(/automatic inference/i);
  expect(provenance.caveat).toMatch(/not.*accuracy benchmark/i);
  expect(provenance.views.map(view => view.mode)).toEqual(modes);
  await page.goto('/#/analyse?example=1');
  const navigation = page.getByRole('navigation', { name: 'Example analysis mode', exact: true });
  await expect(navigation.getByRole('link')).toHaveCount(7);
  for (const mode of modes) {
    const active = navigation.locator(`a[href="#/analyse?example=1&mode=${mode}"]`);
    await active.click();
    await expect(active).toHaveAttribute('aria-current', 'page');
    const video = page.locator('.review-layout video');
    await expect(video).toHaveAttribute('src', `/media/${mode}-analysis.mp4`);
    await expect(video).toHaveAttribute('poster', `/media/${mode}-analysis.jpg`);
    await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.readyState)).toBeGreaterThanOrEqual(2);
    const metadata = await video.evaluate((element: HTMLVideoElement) => ({ duration: element.duration, width: element.videoWidth, height: element.videoHeight }));
    expect(metadata.duration).toBeGreaterThan(0);
    expect(metadata.duration).toBeLessThanOrEqual(60);
    expect(metadata.width).toBe(1280);
    expect(metadata.height).toBe(720);
    await video.evaluate(async (element: HTMLVideoElement) => { element.muted = true; await element.play(); });
    await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.currentTime)).toBeGreaterThan(0.1);
    await video.evaluate((element: HTMLVideoElement) => element.pause());
    await expect(page.getByRole('alert')).toHaveCount(0);
    const reportLink = page.getByRole('link', { name: 'Open run report', exact: true });
    await expect(reportLink).toHaveAttribute('href', `/media/${mode}-report.json`);
    const response = await page.request.get((await reportLink.getAttribute('href'))!);
    expect(response.ok()).toBe(true);
    const report = await response.json() as ExampleReport;
    expect(report.analysis_mode).toBe(mode);
    expect(report.frames).toBeGreaterThan(0);
    expect(report.coverage_note).toMatch(/not.*accuracy/i);
    for (const name of mode === 'combined' ? modes.slice(1) : [mode]) {
      expect(report.modules[name].status).toBe('available');
      expect(report.modules[name].records).toBeGreaterThan(0);
    }
    const record = provenance.views.find(view => view.mode === mode)!;
    expect(record.video).toBe(`${mode}-analysis.mp4`);
    const mediaResponse = await page.request.get('/media/' + record.video);
    expect(mediaResponse.ok()).toBe(true);
    const bytes = await mediaResponse.body();
    expect(bytes.byteLength).toBe(record.bytes);
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(record.sha256);
  }
  expect(errors).toEqual([]);
});

test('the built-in comparison loads and plays the source rally with its combined analysis', async ({ page }) => {
  let postedJobs = 0;
  page.on('request', request => {
    if (request.method() === 'POST' && new URL(request.url()).pathname.startsWith('/api/jobs')) postedJobs += 1;
  });
  await page.goto('/#/compare');
  await page.getByRole('button', { name: 'Load project example', exact: true }).click();
  const first = page.getByLabel('Attempt A video', { exact: true });
  const second = page.getByLabel('Attempt B video', { exact: true });
  await expect(first).toHaveAttribute('src', '/media/source-analysis.mp4');
  await expect(second).toHaveAttribute('src', '/media/combined-analysis.mp4');
  await expect.poll(() => page.locator('video').evaluateAll(videos => videos.every(video => (video as HTMLVideoElement).readyState >= 2))).toBe(true);
  const durations = await page.locator('video').evaluateAll(videos => videos.map(video => (video as HTMLVideoElement).duration));
  expect(durations[0]).toBeGreaterThan(0);
  expect(durations[0]).toBeCloseTo(durations[1], 1);
  await page.getByRole('button', { name: 'Play both', exact: true }).click();
  await expect.poll(() => first.evaluate((element: HTMLVideoElement) => element.currentTime)).toBeGreaterThan(0.15);
  await expect.poll(() => second.evaluate((element: HTMLVideoElement) => element.paused)).toBe(false);
  expect(await page.locator('video').evaluateAll(videos => Math.abs((videos[0] as HTMLVideoElement).currentTime - (videos[1] as HTMLVideoElement).currentTime))).toBeLessThan(0.35);
  await page.getByRole('button', { name: 'Pause both', exact: true }).click();
  await expect.poll(() => page.locator('video').evaluateAll(videos => videos.every(video => (video as HTMLVideoElement).paused))).toBe(true);
  expect(postedJobs).toBe(0);
});
