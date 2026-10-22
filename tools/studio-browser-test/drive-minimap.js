// The minimap in the real page: an overview in the bottom right corner; clicking it moves the view (the canvas picture changes), the Map button
// hides it. Look at out/minimap-1.png (overview with the view rectangle) and out/minimap-2.png (after a click near its top left).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
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
  const hash = async () => crypto.createHash('md5').update(await page.locator('canvas').screenshot()).digest('hex');
  const canvasBox = async () => await page.locator('canvas').boundingBox();

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  // Zoom in so there is more graph than view: the view rectangle is then smaller than the map.
  const box = await canvasBox();
  await page.mouse.move(box.x + 400, box.y + 300);
  for (let i = 0; i < 4; i++) await page.mouse.wheel(0, -300);
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, 'minimap-1.png') });
  const before = await hash();

  // A click near the top left of the minimap (200x140 at the bottom right with a 12 px margin).
  const mapLeft = box.x + box.width - 12 - 200, mapTop = box.y + box.height - 12 - 140;
  await page.mouse.click(mapLeft + 25, mapTop + 20);
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, 'minimap-2.png') });
  const after = await hash();
  check('a click on the minimap moves the view', before !== after);

  await page.click('button:has-text("Map")');
  await page.waitForTimeout(300);
  const hidden = await hash();
  await page.click('button:has-text("Map")');
  await page.waitForTimeout(300);
  check('the Map button hides and shows it', hidden !== after && (await hash()) === after);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
