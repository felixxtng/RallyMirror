import { test, expect, type Page } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

const errors = new WeakMap<Page, string[]>();

test.beforeEach(async ({ page }) => {
  const messages: string[] = [];
  errors.set(page, messages);
  page.on('pageerror', error => messages.push(error.message));
});

test.afterEach(async ({ page }) => {
  expect(errors.get(page) || []).toEqual([]);
});

async function readableWords(page: Page, selector: string) {
  await expect.poll(() => page.locator(selector).evaluateAll(words => words.length > 0 && words.every(word => {
    const style = getComputedStyle(word);
    return Number(style.opacity) > 0.99 && style.transform === 'none' && style.filter === 'none' && style.willChange === 'auto';
  }))).toBe(true);
}

test('the native scroll story changes camera acts on desktop and stays compact on mobile', async ({ page }, testInfo) => {
  await page.goto('/');
  await page.evaluate(async () => { await document.fonts.ready; });
  const story = page.locator('.court-story');
  const scene = page.locator('.court-scene');
  await expect(scene).toHaveAttribute('data-renderer', 'webgl');
  const directory = join('test-results', 'template-motion', testInfo.project.name);
  mkdirSync(directory, { recursive: true });
  if ((page.viewportSize()?.width || 0) < 1100) {
    await expect(story).not.toHaveClass(/is-scroll-enabled/);
    await expect(page.getByRole('group', { name: 'Court camera views' })).toHaveCount(0);
    await expect(scene).toHaveAttribute('data-scroll-mode', 'static');
    await expect(story).toBeVisible();
    return;
  }
  await expect(story).toHaveClass(/is-scroll-enabled/);
  await expect(scene).toHaveAttribute('data-scroll-mode', 'cinematic');
  const startDistance = Number(await scene.getAttribute('data-camera-distance'));
  await page.screenshot({ path: join(directory, 'court-start.png'), animations: 'disabled' });
  const travel = await story.evaluate(element => {
    const bounds = element.getBoundingClientRect();
    return { top: bounds.top + scrollY, distance: Math.max(1, bounds.height - innerHeight) };
  });
  await page.evaluate(({ top, distance }) => { window.scrollTo({ top: top + distance * 0.52, behavior: 'instant' }); }, travel);
  await expect.poll(async () => Number(await scene.getAttribute('data-scroll-progress'))).toBeGreaterThan(0.42);
  await expect(page.getByRole('button', { name: 'Orbit', exact: true })).toHaveAttribute('aria-pressed', 'true');
  expect(await page.evaluate(() => scrollY)).toBeCloseTo(travel.top + travel.distance * 0.52, -1);
  await page.screenshot({ path: join(directory, 'court-orbit.png'), animations: 'disabled' });
  await page.evaluate(({ top, distance }) => { window.scrollTo({ top: top + distance * 0.96, behavior: 'instant' }); }, travel);
  await expect.poll(async () => Number(await scene.getAttribute('data-scroll-progress'))).toBeGreaterThan(0.88);
  await expect(page.getByRole('button', { name: 'Shuttle', exact: true })).toHaveAttribute('aria-pressed', 'true');
  expect(Number(await scene.getAttribute('data-camera-distance'))).toBeLessThan(startDistance - 5);
  await page.screenshot({ path: join(directory, 'court-shuttle.png'), animations: 'disabled' });
});

test('system reduced motion and the motion control disable the scroll story', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await expect(page.locator('html')).toHaveAttribute('data-motion', 'off');
  await expect(page.locator('.court-story')).not.toHaveClass(/is-scroll-enabled/);
  await expect(page.getByRole('group', { name: 'Court camera views' })).toHaveCount(0);
  await readableWords(page, '#hero-title [data-editorial-word]');
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await expect(page.getByRole('button', { name: 'Motion on', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Motion on', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-motion', 'off');
  await expect(page.locator('.court-story')).not.toHaveClass(/is-scroll-enabled/);
  await expect(page.getByRole('group', { name: 'Court camera views' })).toHaveCount(0);
  await expect(page.locator('.court-scene')).toHaveAttribute('data-scroll-mode', 'static');
  await readableWords(page, '#hero-title [data-editorial-word]');
});

test('word reveals play once and finish as visible text with released animation hints', async ({ page }) => {
  await page.addInitScript(() => {
    const state = window as unknown as { __qaWordEntrances: number; __qaHeroEntrances: number };
    state.__qaWordEntrances = 0;
    state.__qaHeroEntrances = 0;
    const original = Element.prototype.animate;
    Element.prototype.animate = function (keyframes, options) {
      if (this.matches('[data-editorial-word]')) {
        state.__qaWordEntrances += 1;
        if (this.closest('#hero-title')) state.__qaHeroEntrances += 1;
      }
      return original.call(this, keyframes, options);
    };
  });
  await page.goto('/');
  await expect.poll(() => page.evaluate(() => (window as unknown as { __qaWordEntrances: number }).__qaWordEntrances)).toBeGreaterThan(0);
  await readableWords(page, '#hero-title [data-editorial-word]');
  const count = await page.evaluate(() => (window as unknown as { __qaHeroEntrances: number }).__qaHeroEntrances);
  await page.getByRole('heading', { name: 'Vision modules', exact: true }).scrollIntoViewIfNeeded();
  await readableWords(page, '.module-section [data-editorial-word]');
  await page.getByRole('heading', { name: /Every rally/ }).scrollIntoViewIfNeeded();
  await readableWords(page, '#hero-title [data-editorial-word]');
  expect(await page.evaluate(() => (window as unknown as { __qaHeroEntrances: number }).__qaHeroEntrances)).toBe(count);
  expect(await page.evaluate(() => document.getAnimations().filter(animation => {
    const target = (animation.effect as KeyframeEffect | null)?.target;
    return target instanceof Element && target.closest('#hero-title') && animation.playState === 'running';
  }).length)).toBe(0);
});

test('the gallery and list keep the same real example links and support keyboard activation', async ({ page }, testInfo) => {
  await page.goto('/');
  const gallery = page.locator('.rally-gallery');
  await gallery.scrollIntoViewIfNeeded();
  await expect(gallery).toHaveAttribute('data-view', 'gallery');
  const links = gallery.locator('.rally-gallery__link');
  await expect(links).toHaveCount(4);
  const original = await links.evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
  expect(original.every(href => href?.startsWith('#/'))).toBe(true);
  for (const image of await gallery.locator('img').all()) {
    await image.scrollIntoViewIfNeeded();
    await expect.poll(() => image.evaluate((node: HTMLImageElement) => node.complete && node.naturalWidth > 0)).toBe(true);
  }
  const listButton = gallery.getByRole('button', { name: 'List view', exact: true });
  await listButton.focus();
  await listButton.press('Enter');
  await expect(listButton).toBeFocused();
  await expect(listButton).toHaveAttribute('aria-pressed', 'true');
  await expect(gallery).toHaveAttribute('data-view', 'list');
  await expect(links).toHaveCount(4);
  expect(await links.evaluateAll(nodes => nodes.map(node => node.getAttribute('href')))).toEqual(original);
  await expect.poll(() => gallery.evaluate(node => node.getAnimations({ subtree: true }).filter(animation => animation.playState === 'running').length)).toBe(0);
  const directory = join('test-results', 'template-motion', testInfo.project.name);
  mkdirSync(directory, { recursive: true });
  await gallery.screenshot({ path: join(directory, 'gallery-list.png'), animations: 'disabled' });
  const galleryButton = gallery.getByRole('button', { name: 'Gallery view', exact: true });
  await galleryButton.focus();
  await galleryButton.press('Enter');
  await expect(gallery).toHaveAttribute('data-view', 'gallery');
  expect(await links.evaluateAll(nodes => nodes.map(node => node.getAttribute('href')))).toEqual(original);
  await gallery.screenshot({ path: join(directory, 'gallery-space.png'), animations: 'disabled' });
  await listButton.focus();
  await listButton.press('Enter');
  await page.keyboard.press('Tab');
  await expect(links.first()).toBeFocused();
  const href = await links.first().getAttribute('href');
  await page.keyboard.press('Enter');
  await expect.poll(() => new URL(page.url()).hash).toBe(href);
});

test('the full overview and both gallery layouts fit narrow screens', async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto('/');
    await page.evaluate(async () => { await document.fonts.ready; });
    await readableWords(page, '#hero-title [data-editorial-word]');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
    expect(await page.locator('#hero-title').evaluate(node => node.scrollWidth <= node.clientWidth + 1)).toBe(true);
    const gallery = page.locator('.rally-gallery');
    await gallery.scrollIntoViewIfNeeded();
    await expect(gallery.locator('.rally-gallery__link')).toHaveCount(4);
    for (const mode of ['List view', 'Gallery view']) {
      await gallery.getByRole('button', { name: mode, exact: true }).click();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
      expect(await gallery.evaluate(node => node.scrollWidth <= node.clientWidth + 1)).toBe(true);
    }
    const directory = join('test-results', 'template-motion', testInfo.project.name);
    mkdirSync(directory, { recursive: true });
    await gallery.screenshot({ path: join(directory, `gallery-${width}.png`), animations: 'disabled' });
  }
});
