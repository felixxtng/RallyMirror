import { test, expect } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import { statSync } from 'node:fs';

const modes = ['combined', 'court', 'tracking', 'pose', 'racket', 'shuttle', 'phases'];
const inputPath = fileURLToPath(new URL('../tests/fixtures/court-example.mp4', import.meta.url));
const inputBytes = statSync(inputPath).size;
const jobId = '4fb04792-cead-49eb-9f86-19bb87457bc9';
const token = 'qa-shuttle-private-token';

test.beforeEach(async ({ page }) => {
  await page.route('**/healthz', route => route.fulfill({ json: { status: 'ok', backend: 'local', supported_modes: modes } }));
});

test('the analysis picker defaults to combined and offers every individual module', async ({ page }) => {
  let created = 0;
  await page.route('**/api/jobs', route => {
    created += 1;
    return route.fulfill({ status: 500, json: { detail: 'Changing a selector must not create a job.' } });
  });
  await page.goto('/#/analyse');
  const picker = page.getByRole('combobox', { name: 'Analysis mode', exact: true });
  await expect(picker).toHaveValue('combined');
  expect(await picker.locator('option').evaluateAll(options => options.map(option => (option as HTMLOptionElement).value))).toEqual(modes);
  for (const mode of modes) {
    await picker.selectOption(mode);
    await expect(picker).toHaveValue(mode);
  }
  expect(created).toBe(0);
});

test('the selected module is sent in the job request and survives browser history reload', async ({ page }) => {
  const posted: unknown[] = [];
  let statusReads = 0;
  await page.route(/\/api\/jobs(?:\/[^?]*)?(?:\?.*)?$/, async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === '/api/jobs' && request.method() === 'POST') {
      posted.push(request.postDataJSON());
      expect(request.headers()['content-type']).toContain('application/json');
      return route.fulfill({ status: 201, json: {
        id: jobId, token, analysis_mode: 'shuttle', max_bytes: 209715200,
        upload_url: `/api/jobs/${jobId}/input`, upload_headers: { 'Content-Type': 'video/mp4', Authorization: `Bearer ${token}` },
      } });
    }
    expect(request.headers().authorization).toBe(`Bearer ${token}`);
    if (path.endsWith('/input') && request.method() === 'PUT') return route.fulfill({ json: { bytes: inputBytes } });
    if (path.endsWith('/start')) return route.fulfill({ json: { id: jobId, state: 'queued', analysis_mode: 'shuttle' } });
    statusReads += 1;
    return route.fulfill({ json: { id: jobId, state: 'running', stage: 'shuttle', analysis_mode: 'shuttle', error: null, created: 1795590000, updated: 1795590010 } });
  });
  await page.goto('/#/analyse');
  await page.getByRole('combobox', { name: 'Analysis mode', exact: true }).selectOption('shuttle');
  await page.getByLabel('Choose rally video', { exact: true }).setInputFiles(inputPath);
  const analyse = page.getByRole('button', { name: 'Analyse video', exact: true });
  await expect(analyse).toBeEnabled();
  await analyse.click();
  await expect.poll(() => statusReads).toBeGreaterThan(0);
  expect(posted).toEqual([{ analysis_mode: 'shuttle' }]);
  await expect.poll(() => page.evaluate(id => {
    const saved = JSON.parse(localStorage.getItem('rallymirror.jobs.v1') || '[]') as Array<{ id: string; analysis_mode?: string }>;
    return saved.find(job => job.id === id)?.analysis_mode;
  }, jobId)).toBe('shuttle');
  await page.evaluate(() => { location.hash = '/rallies'; });
  const row = page.locator('.rally-row').filter({ hasText: 'court-example.mp4' });
  await expect(row).toBeVisible();
  await expect(row).toContainText(/Shuttle/i);
  await page.reload();
  await expect(page.locator('.rally-row').filter({ hasText: 'court-example.mp4' })).toContainText(/Shuttle/i);
  const saved = await page.evaluate(id => {
    const records = JSON.parse(localStorage.getItem('rallymirror.jobs.v1') || '[]') as Array<{ id: string; analysis_mode?: string; token?: string }>;
    return records.find(job => job.id === id);
  }, jobId);
  expect(saved?.analysis_mode).toBe('shuttle');
  expect(saved?.token).toBe(token);
  expect(page.url()).not.toContain(token);
  expect(posted).toHaveLength(1);
});
