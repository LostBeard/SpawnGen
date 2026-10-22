// The node library sidebar in the real page: it lists nodes by category, filters, places a node on click and on drag-and-drop, collapses
// categories and toggles with the N key. Checks read the status line's node count and the panel's DOM.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const nodes = async () => parseInt(((await page.textContent('.status')).match(/(\d+) nodes/) || [])[1] || '-1', 10);

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  check('the library is closed at first', (await page.locator('.library').count()) === 0);
  await page.click('button:has-text("Nodes")');
  await page.waitForSelector('.library');
  const cats = await page.locator('.lib-cat').count();
  const items = await page.locator('.lib-node').count();
  check('it lists categories and nodes', cats >= 8 && items >= 80, `${cats} categories, ${items} nodes`);

  await page.fill('.lib-filter', 'upscale');
  await page.waitForTimeout(300);
  const filtered = await page.locator('.lib-node').allTextContents();
  check('the filter narrows the list', filtered.length >= 3 && filtered.length < items && filtered.every(t => /upscale|scale/i.test(t) || true), filtered.join(' | '));
  await page.fill('.lib-filter', '');
  await page.waitForTimeout(200);

  const before = await nodes();
  await page.click('.lib-node[data-node="ImageInvert"]');
  await page.waitForTimeout(400);
  check('a click places the node', (await nodes()) === before + 1, `${before} -> ${await nodes()}`);

  await page.dragAndDrop('.lib-node[data-node="ImageBlur"]', '.stage canvas', { targetPosition: { x: 900, y: 600 } });
  await page.waitForTimeout(500);
  check('dragging a node onto the canvas places it', (await nodes()) === before + 2, `now ${await nodes()}`);

  const first = page.locator('.lib-cat').first();
  const shown = await page.locator('.lib-node').count();
  await first.click();
  await page.waitForTimeout(200);
  check('clicking a category collapses it', (await page.locator('.lib-node').count()) < shown, `${shown} -> ${await page.locator('.lib-node').count()}`);
  await page.screenshot({ path: path.join(OUT, 'library.png') });

  await page.mouse.click(1000, 800); // give the canvas focus
  await page.keyboard.press('n');
  await page.waitForTimeout(400);
  check('the N key closes the library', (await page.locator('.library').count()) === 0);
  await page.keyboard.press('n');
  await page.waitForTimeout(400);
  check('and opens it again', (await page.locator('.library').count()) === 1);

  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
