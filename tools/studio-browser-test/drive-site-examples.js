// Every Examples-menu entry run to the end on a Studio (default: the live site) in the installed Chrome on the real GPU, the way a first-time
// visitor would: open the example, Queue, wait. Per example: the outcome (ran / error with its text / timed out), the time, browser errors, and a
// screenshot out/site-ex-<n>.png to look at. Models download into a persistent profile (out/site-profile) so a second run is fast.
// STUDIO_URL (default https://lostbeard.github.io/SpawnGen/), ONLY (substrings separated by |: run only matching examples), MAX_MINUTES per example (30).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'https://lostbeard.github.io/SpawnGen/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const MAX = parseInt(process.env.MAX_MINUTES || '30', 10) * 60000;

(async () => {
  const context = await chromium.launchPersistentContext(path.join(OUT, 'site-profile'), {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'],
  });
  const page = context.pages()[0] || await context.newPage();
  let problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text().slice(0, 300)); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message.slice(0, 300)));
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(4000);
  const adapter = await page.evaluate(async () => { const a = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' }); return a ? `${a.info.vendor} ${a.info.description} ${a.info.backend || ''}` : 'none'; });
  console.log('ADAPTER ' + adapter + ' | ' + URL);
  await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
  const labels = (await page.locator('.menu li').allTextContents()).map(s => s.trim()).filter(s => !/Starter/.test(s));
  await page.keyboard.press('Escape'); await page.mouse.click(800, 860);
  const read = async (sel, t = 120000) => (await page.locator(sel).first().textContent({ timeout: t }).catch(() => null)) ?? '';
  const runs = async () => parseInt(((await read('.toolbar')).match(/History \((\d+)\)/) || [])[1] || '0', 10);
  const results = [];
  for (const [i, label] of labels.entries()) {
    if (process.env.ONLY && !process.env.ONLY.split('|').some(o => label.includes(o))) continue;
    problems = [];
    await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
    await page.locator('.menu li', { hasText: label }).first().click(); await page.waitForTimeout(2000);
    const before = await runs(), t0 = Date.now();
    await page.click('button:has-text("Queue")');
    let outcome = 'timed out', detail = '', toasts = new Set();
    while (Date.now() - t0 < MAX) {
      await page.waitForTimeout(2000);
      const toast = await read('.toast', 300);
      if (toast) toasts.add(toast.slice(0, 200));
      const status = await read('.status');
      if (/fail|error|could not|not supported|needs|cannot/i.test(toast) && !/Running/.test(status)) { outcome = 'ERROR'; detail = toast; break; }
      if ((await runs()) > before && !/Running|Queued/.test(status)) { outcome = 'ran'; break; }
    }
    const secs = Math.round((Date.now() - t0) / 1000);
    const shot = path.join(OUT, `site-ex-${i + 1}.png`);
    await page.screenshot({ path: shot }).catch(() => {});
    const errs = problems.filter(p => !/powerPreference/.test(p));
    results.push({ label, outcome, secs, detail, toasts: [...toasts].slice(0, 4), errors: errs.slice(0, 4), shot: path.basename(shot) });
    console.log(`${outcome === 'ran' && errs.length === 0 ? 'PASS' : 'FAIL'} ${label}: ${outcome} in ${secs} s${detail ? ' | ' + detail : ''}${errs.length ? ' | browser errors: ' + errs.join(' || ') : ''} (${path.basename(shot)})`);
  }
  fs.writeFileSync(path.join(OUT, 'site-examples.json'), JSON.stringify({ url: URL, adapter, results }, null, 1));
  await context.close();
  process.exit(results.every(r => r.outcome === 'ran' && r.errors.length === 0) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
