// One of ComfyUI's templates run to the end in the browser, as a visitor would: Templates, search, open, Queue. Its model files come from where
// the template says ComfyUI gets them (model-catalog.json: Hugging Face), cached or streamed by the browser engine. Prints the engine's [models]
// and [gpu] lines and the outcome; writes out/template-run.png.
// STUDIO_URL (default http://localhost:5280/), TEMPLATE (search text, default "SDXL Turbo"), MAX_MINUTES (40).
const { chromium } = require('playwright-core');
const path = require('path');
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const SEARCH = process.env.TEMPLATE || 'SDXL Turbo';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const MAX = parseInt(process.env.MAX_MINUTES || '40', 10) * 60000;

(async () => {
  const ctx = await chromium.launchPersistentContext(path.join(__dirname, 'out', 'template-run-profile'), { executablePath: CHROME, headless: true,
    viewport: { width: 1600, height: 900 }, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
  const page = ctx.pages()[0] || await ctx.newPage();
  let t0 = Date.now(); const t = () => ((Date.now() - t0) / 1000).toFixed(1).padStart(7);
  page.on('console', m => { const s = m.text(); if (/\[models\]|\[gpu\]|error/i.test(s) && !/powerPreference/.test(s)) console.log(t() + ' ' + s.slice(0, 220)); });
  page.on('pageerror', e => console.log(t() + ' PAGEERROR ' + e.message.slice(0, 220)));
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  await page.click('button:has-text("Templates")'); await page.waitForSelector('.templates .tpl-card', { timeout: 60000 });
  await page.fill('.templates .tpl-head input', SEARCH); await page.waitForTimeout(800);
  const card = page.locator('.templates .tpl-card').first();
  const title = (await card.locator('.tpl-title').textContent()).trim();
  const badge = (await card.locator('.tpl-badge, .tpl-api').first().textContent().catch(() => '')).trim();
  console.log(`template: ${title} [${badge}]`);
  await card.click(); await page.waitForTimeout(2500);
  t0 = Date.now(); await page.click('button:has-text("Queue")');
  let last = '', outcome = 'timed out';
  while (Date.now() - t0 < MAX) {
    await page.waitForTimeout(2000);
    const st = (await page.locator('.status').first().textContent({ timeout: 60000 }).catch(() => '')) || '';
    if (st !== last && !/Downloading/.test(st) || /Downloading.*(00|50)%/.test(st)) { console.log(t() + ' STATUS ' + st.slice(0, 170)); }
    last = st;
    const toast = (await page.locator('.toast').last().textContent({ timeout: 200 }).catch(() => '')) || '';
    if (/Run failed/.test(toast)) { outcome = 'FAILED: ' + toast.slice(0, 300); break; }
    if (t0 + 6000 < Date.now() && !/Running|Queued/.test(st)) { outcome = 'ran'; break; }
  }
  console.log(`${outcome === 'ran' ? 'PASS' : 'FAIL'} ${title}: ${outcome} in ${t().trim()} s`);
  await page.screenshot({ path: path.join(__dirname, 'out', 'template-run.png') });
  await ctx.close();
  process.exit(outcome === 'ran' ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
