// Runs several workflows one after another in ONE page of the in-browser engine and prints, per step, its time and the page's measuring lines
// ([warmup], [heap], [models] with ?nodetimes=1): what a model switch costs in .NET heap and reload time. For SpawnDev.ILGPU.ML's
// CompilerCachePolicy (compiler caches released after a model's warm-up): run it once as it is and once with STUDIO_URL ending in ?warmup=off.
// STEPS: entries separated by | , each "file:<API prompt JSON path>" or "example:<substring of an Examples-menu label>".
// STUDIO_URL (default local :5280). Models come from the persistent profile out/site-profile. Exit 1 if a step fails.
const { chromium } = require('playwright-core');
const path = require('path');
const BASE = process.env.STUDIO_URL || 'http://localhost:5280/';
const URL = BASE + (BASE.includes('?') ? '&' : '?') + 'nodetimes=1';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = process.env.PROFILE_DIR || path.join(__dirname, 'out', 'site-profile');
const STEPS = (process.env.STEPS || '').split('|').map(s => s.trim()).filter(Boolean);
if (!STEPS.length) { console.error('STEPS is empty'); process.exit(2); }

(async () => {
  const context = await chromium.launchPersistentContext(OUT, {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'],
  });
  const page = context.pages()[0] || await context.newPage();
  const problems = [];
  let lines = [];
  page.on('console', m => {
    const t = m.text();
    if (m.type() === 'error' && !/powerPreference/.test(t)) problems.push(t.slice(0, 300));
    if (/^\[(warmup|heap|models)\]/.test(t) || /^\[nodetimes\] run:/.test(t)) lines.push(t);
  });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message.slice(0, 300)));
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(4000);
  console.log('ADAPTER ' + await page.evaluate(async () => { const a = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' }); return a ? `${a.info.vendor} ${a.info.description} ${a.info.backend || ''}` : 'none'; }) + ' | ' + URL);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  await page.click('button:has-text("History")'); await page.waitForTimeout(500);
  const status = async () => (await page.locator('.status').first().textContent({ timeout: 60000 }).catch(() => '')) ?? '';
  let failed = 0;
  for (let i = 0; i < STEPS.length; i++) {
    const step = STEPS[i], kind = step.slice(0, step.indexOf(':')), what = step.slice(step.indexOf(':') + 1);
    if (kind === 'file') { await page.setInputFiles('input[accept=".json,.png"]', what); }
    else { await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu'); await page.locator('.menu li', { hasText: what }).first().click(); }
    await page.waitForTimeout(2000);
    lines = [];
    const t0 = Date.now();
    let toast = '';
    await page.click('button:has-text("Queue")');
    await page.waitForTimeout(300);
    while (Date.now() - t0 < 40 * 60000) {
      toast = ((await page.locator('.toast').first().textContent({ timeout: 50 }).catch(() => '')) || toast).slice(0, 400);
      if (!/Running|Queued/.test(await status())) break;
      await page.waitForTimeout(100);
    }
    const wall = (Date.now() - t0) / 1000;
    await page.waitForTimeout(1500); // the heap line is printed a moment after the run ends
    const title = ((await page.locator('.hist-entry .hist-title').first().textContent().catch(() => '')) ?? '').trim();
    const bad = !/done/.test(title);
    if (bad) failed++;
    console.log(`STEP ${i + 1} ${path.basename(what)}: wall ${wall.toFixed(1)} s; ${title}${bad && toast ? ' | message: ' + toast.trim() : ''}`);
    for (const l of lines) console.log('  ' + l);
  }
  console.log(`SUMMARY ${STEPS.length} steps, ${failed} failed; browser errors: ${problems.length ? problems.join(' | ') : 'none'}`);
  await context.close();
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
