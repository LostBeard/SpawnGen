// An animated result in the BROWSER engine (TJ 2026-11-19: the LTX example's Save Animated WEBP showed a still apple and "Save picture" wrote a .png).
// Load Image (example.png) -> Invert -> Batch Images (2 frames, picture and negative) -> Save Animated WEBP at 2 fps, on "This browser". Checks:
// the node's picture changes between frames on the canvas; the History thumbnail is a blob: URL (no base64 data URL built in .NET);
// right-click offers "Save animation" and the download keeps the file's name and is an animated WEBP (RIFF / WEBP / ANIM).
// STUDIO_URL (default local :5280). Writes out/animated-preview-a.png and -b.png.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = process.env.PROFILE_DIR || path.join(__dirname, 'out', 'webgpu-profile');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const context = await chromium.launchPersistentContext(PROFILE, {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 }, acceptDownloads: true,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
  });
  const page = context.pages()[0] || await context.newPage();
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  const prompt = {
    '1': { class_type: 'LoadImage', inputs: { image: 'example.png' } },
    '2': { class_type: 'ImageInvert', inputs: { image: ['1', 0] } },
    '3': { class_type: 'ImageBatch', inputs: { image1: ['1', 0], image2: ['2', 0] } },
    '4': { class_type: 'SaveAnimatedWEBP', inputs: { images: ['3', 0], filename_prefix: 'animcheck', fps: 2, lossless: process.env.LOSSLESS === '1', quality: 80, method: 'default' } },
  };
  const promptPath = path.join(OUT, 'animated-prompt.json');
  fs.writeFileSync(promptPath, JSON.stringify(prompt));
  await page.setInputFiles('input[accept=".json,.png"]', promptPath);
  await page.waitForTimeout(1500);
  await page.locator('button:has-text("Queue")').click();
  for (let i = 0; i < 600; i++) {
    await page.waitForTimeout(500);
    const s = (await page.textContent('.status')).trim();
    if (i > 4 && !/Running|Queued/.test(s)) break;
  }
  await page.waitForTimeout(2000);
  check('the run finished', !/fail|error/i.test(await page.textContent('.status')), (await page.textContent('.status')).trim().slice(0, 80));
  // A node's error is drawn on the canvas, not logged (2026-11-20: Save Animated WEBP failed and only the node said so): the History
  // entry's state says whether the run succeeded.
  if (!(await page.locator('.hist-entry').count())) { await page.click('button:has-text("History")'); await page.waitForTimeout(800); }
  const state = ((await page.locator('.hist-entry .hist-title').first().textContent().catch(() => '')) ?? '').trim();
  check('the run succeeded (History says done)', state.endsWith('done'), state.slice(0, 80));
  await page.click('button:has-text("History")'); await page.waitForTimeout(500); // closed again: the canvas checks need the node in view

  // The canvas repaints the playing frames: two samples half a frame apart differ (nothing else on the page moves).
  const sample = () => page.evaluate(() => {
    const c = document.querySelector('canvas'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let h = 0; for (let i = 0; i < d.length; i += 4) h = (h * 31 + d[i] + d[i + 1] * 7 + d[i + 2] * 13) | 0; return h;
  });
  const seen = new Set();
  for (let i = 0; i < 8; i++) { seen.add(await sample()); await page.waitForTimeout(260); }
  await page.screenshot({ path: path.join(OUT, 'animated-preview-a.png') });
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'animated-preview-b.png') });
  check('the node\'s picture plays (the canvas shows more than one frame)', seen.size >= 2, `${seen.size} distinct canvas states in 2 s`);

  // Right-click the Save Animated WEBP node's picture: "Save animation" downloads the .webp itself.
  let menu = false;
  for (let y = 200; y < 760 && !menu; y += 24)
    for (let x = 300; x < 1500 && !menu; x += 48) {
      await page.mouse.click(x, y, { button: 'right' });
      await page.waitForTimeout(80);
      if (await page.locator('.menu li:has-text("Save animation")').count()) menu = true;
      else if (await page.locator('.menu').count()) { await page.keyboard.press('Escape'); await page.mouse.click(5, 880); }
    }
  check('the node menu offers "Save animation"', menu);
  if (menu) {
    const [download] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }), page.click('.menu li:has-text("Save animation")')]);
    const file = path.join(OUT, 'animated-download.webp');
    await download.saveAs(file);
    const b = fs.readFileSync(file);
    check('the download keeps the file\'s name', /^animcheck_\d{5}_\.webp$/.test(download.suggestedFilename()), download.suggestedFilename());
    check('it is an animated WEBP', b.subarray(0, 4).toString() === 'RIFF' && b.subarray(8, 12).toString() === 'WEBP' && b.includes(Buffer.from('ANIM')), `${b.length} bytes`);
  }
  if (!(await page.locator('.hist-thumbs img').count())) { await page.click('button:has-text("History")'); await page.waitForTimeout(800); }
  const thumb = await page.locator('.hist-thumbs img').first().getAttribute('src', { timeout: 3000 }).catch(() => null);
  check('the History thumbnail is a blob URL (no base64 data URL)', (thumb || '').startsWith('blob:'), (thumb || '').slice(0, 40));

  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await context.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
