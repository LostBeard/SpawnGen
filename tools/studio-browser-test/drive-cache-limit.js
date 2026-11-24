// The Studio's "Model cache limit" setting on the in-browser engine: the browser keeps at most that many GB of models, the least recently
// used go when a new one needs room, and a model larger than the limit streams instead of being stored. Found 2026-11-24: with no limit a
// row of examples stored 42 GB (the browser's own estimate said 3.6 of 14.3) and Chrome then cleared the whole site's storage mid-session.
// Phase A, limit 0.25 GB: three small models in a row (Depth map, Remove the background, Pose and object detection); after each run the
// bytes stored must be within the limit. Phase B, limit 0.02 GB (smaller than any of them): Depth map still runs, streamed, and nothing
// above the limit is stored. Phase C (between them): a store already over a lowered limit is trimmed at the next run. STUDIO_URL (default http://localhost:5280/). Own profile (out/cache-limit-profile), emptied at the start.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const BASE = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = path.join(__dirname, 'out', 'cache-limit-profile');
let failed = 0;
const check = (ok, what) => { console.log((ok ? 'PASS ' : 'FAIL ') + what); if (!ok) failed++; };

async function phase(limitGb, labels, fresh) {
  if (fresh) fs.rmSync(PROFILE, { recursive: true, force: true });
  const ctx = await chromium.launchPersistentContext(PROFILE, { executablePath: CHROME, headless: true,
    viewport: { width: 1600, height: 900 }, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
  const page = ctx.pages()[0] || await ctx.newPage();
  const logs = [];
  page.on('console', m => logs.push(m.text()));
  // the setting as the Settings panel stores it, in place before the Studio starts
  await page.addInitScript(gb => { try { localStorage.setItem('spawngen.settings', JSON.stringify({ ModelCacheGb: gb })); } catch (e) { } }, limitGb);
  await page.goto(BASE + '?nodetimes=1'); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  const read = async sel => (await page.locator(sel).first().textContent({ timeout: 60000 }).catch(() => '')) || '';
  const runs = async () => parseInt(((await read('.toolbar')).match(/History \((\d+)\)/) || [])[1] || '0', 10);
  const runExample = async label => {
    for (let i = 0; i < 30 && /Run failed/.test((await page.locator('.toast').first().textContent({ timeout: 200 }).catch(() => '')) || ''); i++) await page.waitForTimeout(1000);
    await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
    await page.locator('.menu li', { hasText: label }).first().click(); await page.waitForTimeout(1500);
    const before = await runs();
    await page.click('button:has-text("Queue")');
    for (let i = 0; i < 900; i++) {
      await page.waitForTimeout(1000);
      const toast = (await page.locator('.toast').first().textContent({ timeout: 200 }).catch(() => '')) || '';
      if (/Run failed/.test(toast)) return 'failed: ' + toast.trim().split('\n')[0].slice(0, 200);
      if ((await runs()) > before && !/Running|Queued/.test(await read('.status'))) return 'ran';
    }
    return 'timed out';
  };
  // bytes of every file in the origin-private file system (what the limit counts), and the largest file
  const stored = () => page.evaluate(async () => {
    const walk = async dir => {
      let total = 0, files = 0, largest = 0;
      for await (const [, h] of dir.entries()) {
        if (h.kind === 'file') { const size = (await h.getFile()).size; total += size; files++; largest = Math.max(largest, size); }
        else { const sub = await walk(h); total += sub.total; files += sub.files; largest = Math.max(largest, sub.largest); }
      }
      return { total, files, largest };
    };
    return walk(await navigator.storage.getDirectory());
  });
  const limit = limitGb * 1e9;
  for (const label of labels) {
    const outcome = await runExample(label);
    const s = await stored();
    console.log(`limit ${limitGb} GB, ${label}: ${outcome}; stored ${(s.total / 1e6).toFixed(1)} MB in ${s.files} files (largest ${(s.largest / 1e6).toFixed(1)} MB)`);
    check(outcome === 'ran', `${label} runs under a ${limitGb} GB limit`);
    check(s.total <= limit, `and the stored models stay within the limit (${(s.total / 1e6).toFixed(1)} of ${(limit / 1e6).toFixed(0)} MB)`);
  }
  const said = logs.filter(l => /^\[models\] (cache limit|streaming|the browser refused)/.test(l));
  console.log('console: ' + said.slice(0, 8).join(' || ').slice(0, 700));
  await ctx.close();
  return { logs, said };
}

(async () => {
  const a = await phase(0.25, ['Depth map', 'Remove the background', 'Pose and object detection'], true);
  check(a.said.some(l => /cache limit 0\.[23]/.test(l)), 'the engine took the limit from the settings');
  // Phase C, the profile phase A left (about 0.2 GB stored under its 0.25 GB limit) opened with a limit of 0.05 GB: every model this run needs
  // is already stored (a cache hit downloads nothing), so only the engine's own trim at the start of the run can bring the store down.
  const c = await phase(0.05, ['Pose and object detection'], false);
  check(c.logs.some(l => /^\[models\] removed .* of stored models/.test(l)), 'a store already over a lowered limit is trimmed when the next run starts');
  const b = await phase(0.02, ['Depth map'], true);
  check(b.said.some(l => /^\[models\] streaming/.test(l)), 'a model larger than the limit streams instead of being stored');
  console.log(failed ? `${failed} FAILED` : 'ALL PASS');
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
