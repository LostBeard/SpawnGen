// Times one Examples-menu entry run several times in one page (the models stay loaded after the first), for before / after measurements of a
// change: each run's own time as the History panel records it (Started to finished, to 0.1 s) and the wall time seen here (polled every 100 ms).
// EXAMPLE (a substring of the menu label, default "Text to image (SD-Turbo"), RUNS (default 5), STUDIO_URL (default local :5280).
// Models come from the persistent profile out/site-profile (shared with drive-site-examples.js). Prints one line per run and a summary.
const { chromium } = require('playwright-core');
const path = require('path');
const BASE = process.env.STUDIO_URL || 'http://localhost:5280/';
// NODETIMES=1 / PEROP=1 open the page with ?nodetimes=1 / ?perop=1 and print the last run's [nodetimes] / [perop] lines (perop waits for every GPU op: slower).
const FLAGS = [process.env.NODETIMES === '1' && 'nodetimes=1', process.env.PEROP === '1' && 'perop=1'].filter(Boolean).join('&');
const URL = FLAGS ? BASE + (BASE.includes('?') ? '&' : '?') + FLAGS : BASE;
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = path.join(__dirname, 'out');
const EXAMPLE = process.env.EXAMPLE || 'Text to image (SD-Turbo';
const RUNS = parseInt(process.env.RUNS || '5', 10);

(async () => {
  const context = await chromium.launchPersistentContext(path.join(OUT, 'site-profile'), {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'],
  });
  const page = context.pages()[0] || await context.newPage();
  const problems = [];
  page.on('console', m => { if (m.type() === 'error' && !/powerPreference/.test(m.text())) problems.push(m.text().slice(0, 200)); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message.slice(0, 200)));
  let timing = [];
  page.on('console', m => { const t = m.text(); if (/^\[(nodetimes|perop)\]/.test(t)) timing.push(t); });
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(4000);
  await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
  await page.locator('.menu li', { hasText: EXAMPLE }).first().click(); await page.waitForTimeout(2000);
  await page.click('button:has-text("History")'); await page.waitForTimeout(500);
  const status = async () => (await page.locator('.status').first().textContent({ timeout: 60000 }).catch(() => '')) ?? '';
  const pageSeconds = [], wall = [];
  for (let r = 0; r < RUNS; r++) {
    const t0 = Date.now();
    timing = [];
    await page.click('button:has-text("Queue")');
    await page.waitForTimeout(300);
    while (Date.now() - t0 < 30 * 60000) {
      if (!/Running|Queued/.test(await status())) break;
      await page.waitForTimeout(100);
    }
    wall.push((Date.now() - t0) / 1000);
    await page.waitForTimeout(500);
    const title = (await page.locator('.hist-entry .hist-title').first().textContent().catch(() => '')) ?? '';
    const m = title.match(/· ([\d.]+) s ·/);
    pageSeconds.push(m ? parseFloat(m[1]) : NaN);
    console.log(`run ${r + 1}: ${m ? m[1] + ' s (page)' : '?'}; wall ${wall[r].toFixed(1)} s; ${title.trim()}`);
  }
  if (timing.length) console.log('last run:\n  ' + timing.join('\n  '));
  const warm = pageSeconds.slice(1).filter(x => !isNaN(x)).sort((a, b) => a - b);
  const median = warm.length ? warm[Math.floor(warm.length / 2)] : NaN;
  console.log(`SUMMARY ${EXAMPLE}: first ${pageSeconds[0]} s; warm median ${median} s of ${warm.length} (${warm.join(', ')}); errors: ${problems.length ? problems.join(' | ') : 'none'}`);
  await context.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
