// The queue and history panel in the real page, with a graph that needs no GPU (Load Image -> Save Image on "This browser"): every run adds an
// entry with its picture, a thumbnail opens a full-size view, "Load workflow" opens the run's workflow in a new tab, Clear forgets the runs.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu', '--enable-unsafe-webgpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const queueAndWait = async () => {
    await page.locator('button:has-text("Queue")').click();
    for (let i = 0; i < 40; i++) { await page.waitForTimeout(500); if (i > 2 && await page.locator('button:has-text("Queue")').isVisible()) break; }
    await page.waitForTimeout(500);
  };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  await page.setInputFiles('input[accept=".png,.jpg,.jpeg"]', path.join(__dirname, 'apple.png'));
  await page.waitForTimeout(800);
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'load-save-workflow.json'));
  await page.waitForTimeout(1200);

  check('the history is closed at first', (await page.locator('.history').count()) === 0);
  await page.click('button:has-text("History")');
  await page.waitForSelector('.history');
  check('it says nothing has run yet', /Nothing has run/.test(await page.textContent('.history')));

  await queueAndWait();
  await queueAndWait();
  const entries = await page.locator('.hist-entry').count();
  const titles = await page.locator('.hist-title').allTextContents();
  check('every run adds an entry, newest first', entries === 2 && /#2/.test(titles[0]) && /#1/.test(titles[1]), titles.join(' | '));
  check('both runs finished', (await page.locator('.hist-entry.done').count()) === 2);
  const thumbs = await page.locator('.hist-thumbs img').count();
  check('the first run shows its saved picture; the second, with nothing changed, says its outputs were cached', thumbs === 1 && /cached/.test(await page.textContent('.hist-note')), `${thumbs} thumbnails`);
  const loaded = await page.evaluate(() => [...document.querySelectorAll('.hist-thumbs img')].every(i => i.complete && i.naturalWidth > 0));
  check('the pictures load', loaded);
  await page.screenshot({ path: path.join(OUT, 'history.png') });

  await page.locator('.hist-thumbs img').first().click();
  await page.waitForSelector('.lightbox');
  check('a thumbnail opens the full-size view', true);
  await page.locator('.lightbox').click({ position: { x: 20, y: 20 } });
  await page.waitForTimeout(300);
  check('a click on it closes it', (await page.locator('.lightbox').count()) === 0);

  const tabsBefore = await page.locator('.tabs .tab').count();
  await page.locator('.hist-actions button').first().click();
  await page.waitForTimeout(1000);
  check('Load workflow opens the run in a new tab', (await page.locator('.tabs .tab').count()) === tabsBefore + 1, await page.locator('.tabs .tab.active').textContent());

  await page.click('.hist-head button');
  await page.waitForTimeout(300);
  check('Clear forgets the runs', (await page.locator('.hist-entry').count()) === 0);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
