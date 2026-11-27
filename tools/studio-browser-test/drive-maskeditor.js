// The mask editor in the real page, on the in-page engine: upload a picture, open a Load Image -> Mask To Image -> Save Image workflow, right-click the Load Image node,
// "Open in Mask Editor", paint a stroke, Save. The node then points at the masked picture; running the graph shows the mask: white along the stroke, black elsewhere.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  // the real GPU (TJ 2026-11-20: never SwiftShader / --disable-gpu, it froze NV1080)
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  // every result file the page turns into a blob URL is kept, so the run's picture can be read (the History panel no longer has thumbnails)
  await page.addInitScript(() => {
    window.__resultBlobs = [];
    const make = URL.createObjectURL;
    URL.createObjectURL = function (b) { const u = make.call(URL, b); if (b instanceof Blob) window.__resultBlobs.push(b); return u; };
  });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  await page.setInputFiles('input[accept*=".wav"]', path.join(__dirname, 'apple.png'));
  await page.waitForTimeout(800);
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'mask-workflow.json'));
  await page.waitForTimeout(1200);

  // Find the Load Image node by right-clicking until the menu offers the mask editor.
  let opened = false;
  for (let y = 380; y < 640 && !opened; y += 16)
    for (let x = 300; x < 700 && !opened; x += 40) {
      await page.mouse.click(x, y, { button: 'right' });
      await page.waitForTimeout(140);
      if (await page.locator('.menu li:has-text("Open in Mask Editor")').count()) { await page.click('.menu li:has-text("Open in Mask Editor")'); opened = true; }
      else if (await page.locator('.menu').count()) { await page.keyboard.press('Escape'); await page.mouse.click(5, 850); }
    }
  check('the node menu offers the mask editor', opened);
  await page.waitForSelector('.maskeditor');
  const box = await page.locator('.me-mask').boundingBox();
  // A stroke across the picture's left half, from 20% to 80% of the height.
  const x = box.x + box.width * 0.3, y1 = box.y + box.height * 0.2, y2 = box.y + box.height * 0.8;
  await page.mouse.move(x, y1); await page.mouse.down();
  for (let i = 1; i <= 12; i++) await page.mouse.move(x, y1 + (y2 - y1) * i / 12);
  await page.mouse.up();
  await page.screenshot({ path: path.join(OUT, 'maskeditor.png') });
  await page.click('.me-save');
  await page.waitForSelector('.maskeditor', { state: 'detached' });
  await page.waitForTimeout(500);
  check('saving closes the editor and tells where the picture went', /Saved as mask-apple/.test(await page.locator('.toast').first().textContent().catch(() => '')), await page.locator('.toast').first().textContent().catch(() => ''));

  const from = await page.evaluate(() => window.__resultBlobs.length);
  await page.locator('button:has-text("Queue")').click();
  const status = () => page.evaluate(() => (document.querySelector('.runbar .status') || {}).textContent || '');
  const t0 = Date.now();
  await page.waitForTimeout(3000);
  while (Date.now() - t0 < 120000 && /Running|Queued|Loading|Starting/i.test(await status())) await page.waitForTimeout(500);
  await page.waitForTimeout(2000);
  const stats = await page.evaluate(async from => {
    const png = window.__resultBlobs.slice(from).filter(b => b.type === 'image/png').pop();
    if (!png) return null;
    const bmp = await createImageBitmap(png);
    const c = new OffscreenCanvas(bmp.width, bmp.height);
    const g = c.getContext('2d'); g.drawImage(bmp, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let white = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] > 200) white++;
    const at = (fx, fy) => d[((Math.floor(fy * c.height)) * c.width + Math.floor(fx * c.width)) * 4];
    return { w: c.width, h: c.height, fraction: white / (c.width * c.height), onStroke: at(0.3, 0.5), farAway: at(0.9, 0.5) };
  }, from);
  console.log(JSON.stringify(stats));
  check('the mask picture is white on the stroke and black away from it', stats && stats.onStroke > 200 && stats.farAway < 50, JSON.stringify(stats));
  check('the painted area is a part of the picture, not all of it', stats && stats.fraction > 0.02 && stats.fraction < 0.5, stats && stats.fraction.toFixed(3));

  // Open the editor again on the masked picture: its transparent pixels come back as paint; Invert swaps paint and no paint; Cancel keeps the node.
  const paintAt = () => page.evaluate(() => {
    const c = document.querySelector('.me-mask'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    const a = (fx, fy) => d[((Math.floor(fy * c.height)) * c.width + Math.floor(fx * c.width)) * 4 + 3];
    return { onStroke: a(0.3, 0.5), farAway: a(0.9, 0.5) };
  });
  let reopened = false;
  for (let y = 380; y < 640 && !reopened; y += 16)
    for (let x = 300; x < 700 && !reopened; x += 40) {
      await page.mouse.click(x, y, { button: 'right' });
      await page.waitForTimeout(140);
      if (await page.locator('.menu li:has-text("Open in Mask Editor")').count()) { await page.click('.menu li:has-text("Open in Mask Editor")'); reopened = true; }
      else if (await page.locator('.menu').count()) { await page.keyboard.press('Escape'); await page.mouse.click(5, 850); }
    }
  await page.waitForSelector('.maskeditor');
  await page.waitForTimeout(800);
  const again = await paintAt();
  check('reopened: the saved mask comes back as paint, nothing elsewhere', again.onStroke === 255 && again.farAway === 0, JSON.stringify(again));
  await page.click('.me-invert');
  await page.waitForTimeout(200);
  const inverted = await paintAt();
  check('Invert: the stroke unpainted, the rest painted', inverted.onStroke === 0 && inverted.farAway === 255, JSON.stringify(inverted));
  await page.keyboard.press('Escape');
  await page.waitForSelector('.maskeditor', { state: 'detached' });
  check('Escape closes the editor without saving', true);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
