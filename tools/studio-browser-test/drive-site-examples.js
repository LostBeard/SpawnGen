// Every Examples-menu entry run to the end on a Studio (default: the live site) in the installed Chrome on the real GPU, the way a first-time
// visitor would: open the example, Queue, wait. Per example: the outcome (ran / error with its text / timed out), the time, browser errors, and a
// screenshot out/site-ex-<n>.png to look at. Models download into a persistent profile (out/site-profile) so a second run is fast.
// MODEL_MEMORY_GB: sets Settings' "GPU memory for models" before the page starts (0 = one workflow at a time; unset = whatever the profile has)
// and prints the engine's [models] lines about kept, dropped and given-way models after each example. PASSES (default 1): run the list again in
// the same page (the page opens with ?testhooks=1 and each later run first forgets the cached node results, keeping the models), which shows
// what a second visit to each workflow still finds loaded.
// STUDIO_URL (default https://lostbeard.github.io/SpawnGen/), ONLY (substrings separated by |: run only matching examples), MAX_MINUTES per example (30).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const BASE = process.env.STUDIO_URL || 'https://lostbeard.github.io/SpawnGen/';
const QUERY = [process.env.NODETIMES === '1' ? 'nodetimes=1' : '', parseInt(process.env.PASSES || '1', 10) > 1 ? 'testhooks=1' : ''].filter(Boolean).join('&');
const URL = QUERY ? BASE + (BASE.includes('?') ? '&' : '?') + QUERY : BASE;
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const MAX = parseInt(process.env.MAX_MINUTES || '30', 10) * 60000;

(async () => {
  const context = await chromium.launchPersistentContext(path.join(OUT, 'site-profile'), {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'],
  });
  const page = context.pages()[0] || await context.newPage();
  // Every result file the page turns into a blob URL is kept, so each run's pictures can be decoded: a run that "ends" with a file nobody can
  // open is a failure (2026-11-20: the live site's Save Image PNGs had a broken deflate stream and the drive still said PASS).
  await page.addInitScript(() => {
    window.__resultBlobs = [];
    const make = URL.createObjectURL;
    URL.createObjectURL = function (b) { const u = make.call(URL, b); if (b instanceof Blob) window.__resultBlobs.push(b); return u; };
  });
  if (process.env.MODEL_MEMORY_GB !== undefined) await page.addInitScript(gb => {
    try { const s = JSON.parse(localStorage.getItem('spawngen.settings') || '{}'); s.ModelMemoryGb = gb; localStorage.setItem('spawngen.settings', JSON.stringify(s)); } catch (e) { }
  }, parseFloat(process.env.MODEL_MEMORY_GB));
  const undecodable = async from => page.evaluate(async from => {
    const bad = [];
    for (const b of window.__resultBlobs.slice(from)) {
      if (!/^image\/(png|webp|jpeg|gif)$/.test(b.type)) continue;
      try { (await createImageBitmap(b)).close(); } catch (e) { bad.push(`${b.type} ${b.size} B: ${e.message}`); }
    }
    return bad;
  }, from);
  let problems = [];
  // With NODETIMES=1 the page opens with ?nodetimes=1 and prints each stage a node reports: the last ones are shown after a failed example
  // (which stage a lost device or a missing file hit), with the engine's [gpu] and [models] lines.
  let trace = [];
  page.on('console', m => { const t = m.text(); if (m.type() === 'error') problems.push(t.slice(0, 300)); if (/^\[(stage|gpu|models)\]/.test(t)) trace.push(t.slice(0, 220)); });
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
  const passes = parseInt(process.env.PASSES || '1', 10);
  for (let pass = 1; pass <= passes; pass++)
  for (const [i, label] of labels.entries()) {
    if (process.env.ONLY && !process.env.ONLY.split('|').some(o => label.includes(o))) continue;
    problems = []; trace = [];
    await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
    await page.locator('.menu li', { hasText: label }).first().click(); await page.waitForTimeout(2000);
    // a later pass: the cached node results go (the loaded models stay), so the workflow really runs again and shows what was still loaded
    if (pass > 1) await page.evaluate(() => window.spawngenForgetResults && window.spawngenForgetResults());
    const before = await runs(), t0 = Date.now(), blobsBefore = await page.evaluate(() => window.__resultBlobs.length);
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
    if (outcome === 'ran') { await page.waitForTimeout(1500); for (const bad of await undecodable(blobsBefore)) problems.push('UNDECODABLE RESULT ' + bad); }
    const shot = path.join(OUT, `site-ex-${i + 1}.png`);
    await page.screenshot({ path: shot }).catch(() => {});
    const errs = problems.filter(p => !/powerPreference/.test(p));
    results.push({ label, outcome, secs, detail, toasts: [...toasts].slice(0, 4), errors: errs.slice(0, 4), shot: path.basename(shot) });
    // the engine's own word on the backend it opened (printed when it opens one: the first example, and after a device loss)
    for (const l of trace.filter(t => t.startsWith('[gpu] backend'))) console.log('    ' + l);
    if (process.env.MODEL_MEMORY_GB !== undefined) for (const l of trace.filter(t => /^\[models\] (dropped|room for|the loaded transformer|GPU memory|no GPU memory|freed|.* over the budget)/.test(t))) console.log('    ' + l);
    console.log(`${outcome === 'ran' && errs.length === 0 ? 'PASS' : 'FAIL'} ${passes > 1 ? 'pass ' + pass + ' ' : ''}${label}: ${outcome} in ${secs} s${detail ? ' | ' + detail : ''}${errs.length ? ' | browser errors: ' + errs.join(' || ') : ''} (${path.basename(shot)})`);
    if (outcome !== 'ran' || errs.length) for (const l of trace.slice(-14)) console.log('    ' + l);
  }
  fs.writeFileSync(path.join(OUT, 'site-examples.json'), JSON.stringify({ url: URL, adapter, results }, null, 1));
  await context.close();
  process.exit(results.every(r => r.outcome === 'ran' && r.errors.length === 0) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
