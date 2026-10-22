// Image to 3D through the whole Studio (server engine, needs the TRELLIS.2 weights cached or ~10 GB of download): open the example, queue it, and when the run is done the
// History panel offers "View 3D model" and the viewer opens it. Needs example.png in the server's input folder (a picture of one object).
// Look at out/3d-viewer.png and out/3d-graph.png.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
  await page.click('.menu li:has-text("Image to 3D")'); await page.waitForTimeout(1200);
  check('the example opens with 3 nodes', /3 nodes/.test(await page.textContent('.status')));
  await page.click('button.primary');
  const t0 = Date.now();
  await page.waitForSelector('button.danger');
  while (Date.now() - t0 < 25 * 60 * 1000) {
    await page.waitForTimeout(5000);
    if ((await page.locator('button.danger').count()) === 0) break;
  }
  console.log(`run took ${((Date.now() - t0) / 1000).toFixed(0)} s; status: ${(await page.textContent('.status')).trim()}`);
  await page.screenshot({ path: path.join(OUT, '3d-graph.png') });
  await page.click('button:has-text("History")'); await page.waitForSelector('.hist-entry');
  check('the history entry offers the model', (await page.locator('.hist-entry button:has-text("View 3D model")').count()) === 1);
  await page.click('.hist-entry button:has-text("View 3D model")');
  await page.waitForSelector('.glbviewer[data-ready="1"]', { timeout: 60000 });
  await page.waitForTimeout(1500);
  const tris = await page.evaluate(() => document.querySelector('.glbviewer').dataset.triangles);
  check('the viewer shows the model', Number(tris) > 1000, tris + ' triangles');
  await page.screenshot({ path: path.join(OUT, '3d-viewer.png') });
  await page.keyboard.press('Escape');
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
