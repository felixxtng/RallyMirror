import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';

const baseURL = process.env.PERF_BASE_URL || 'http://127.0.0.1:4173';
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const records = {};
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  await page.addInitScript(() => {
    window.__rmRenderTimes = [];
    const original = WebGL2RenderingContext.prototype.clear;
    WebGL2RenderingContext.prototype.clear = function (...args) {
      window.__rmRenderTimes.push(performance.now());
      return Reflect.apply(original, this, args);
    };
  });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(baseURL);
  await page.locator('.court-scene[data-renderer="webgl"]').waitFor({ timeout: 10000 });
  await page.waitForTimeout(1000);
  const before = await page.evaluate(() => window.__rmRenderTimes.length);
  await page.waitForTimeout(3000);
  records.visible = await page.evaluate((start) => {
    const times = window.__rmRenderTimes.slice(start);
    const gaps = times
      .slice(1)
      .map((t, i) => t - times[i])
      .sort((a, b) => a - b);
    const canvas = document.querySelector('.court-scene canvas');
    const bounds = canvas.getBoundingClientRect();
    return {
      rendersOverThreeSeconds: times.length,
      medianRenderIntervalMs: gaps[Math.floor(gaps.length / 2)],
      canvasPixels: [canvas.width, canvas.height],
      canvasCssPixels: [bounds.width, bounds.height],
      effectivePixelRatio: canvas.width / bounds.width,
    };
  }, before);
  // A shorter window can scroll the entire court out of view. In a 1000px
  // window the page's maximum scroll still leaves part of the hero visible.
  await page.setViewportSize({ width: 1440, height: 600 });
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
  await page.waitForTimeout(400);
  const offscreenBefore = await page.evaluate(() => window.__rmRenderTimes.length);
  await page.waitForTimeout(1500);
  records.offscreen = {
    rendersOverOnePointFiveSeconds: await page.evaluate(
      (start) => window.__rmRenderTimes.length - start,
      offscreenBefore,
    ),
  };
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(500);
  const resumeBefore = await page.evaluate(() => window.__rmRenderTimes.length);
  await page.waitForTimeout(1000);
  records.resumed = {
    rendersOverOneSecond: await page.evaluate(
      (start) => window.__rmRenderTimes.length - start,
      resumeBefore,
    ),
  };
  await page.getByRole('button', { name: 'Motion on', exact: true }).click();
  await page.waitForTimeout(400);
  const reducedBefore = await page.evaluate(() => window.__rmRenderTimes.length);
  await page.waitForTimeout(1500);
  records.motionOff = {
    rendersOverOnePointFiveSeconds: await page.evaluate(
      (start) => window.__rmRenderTimes.length - start,
      reducedBefore,
    ),
  };
  const upload = await context.newPage();
  await upload.route('**/healthz', (r) => r.fulfill({ json: { status: 'ok', backend: 'local' } }));
  await upload.goto(baseURL + '/#/analyse');
  await upload.getByRole('heading', { name: 'Start with your footage' }).waitFor();
  records.uploadRoute = await upload.evaluate(() => ({
    threeRequested: performance
      .getEntriesByType('resource')
      .some((r) => /three\.module|CourtScene/.test(r.name)),
  }));
  records.pageErrors = errors;
  const valid =
    records.visible.rendersOverThreeSeconds > 0 &&
    records.visible.rendersOverThreeSeconds <= 95 &&
    records.visible.effectivePixelRatio <= 1.51 &&
    records.offscreen.rendersOverOnePointFiveSeconds === 0 &&
    records.resumed.rendersOverOneSecond > 0 &&
    records.motionOff.rendersOverOnePointFiveSeconds === 0 &&
    !records.uploadRoute.threeRequested &&
    errors.length === 0;
  records.passed = valid;
  records.scope =
    'One installed Edge browser on this machine. Rendering observations are not a universal FPS or inference-throughput guarantee.';
  await mkdir('test-results', { recursive: true });
  await writeFile('test-results/performance.json', JSON.stringify(records, null, 2));
  console.log(JSON.stringify(records, null, 2));
  if (!valid) process.exitCode = 1;
} finally {
  await browser.close();
}
