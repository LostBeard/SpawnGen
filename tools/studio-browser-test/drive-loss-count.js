// Counts how often an example loses the WebGPU device: runs it ATTEMPTS times in the in-browser engine (installed Chrome, real GPU), in one
// page while runs succeed. After a failed run it asks the browser itself, from plain JS, for a new adapter and device at 0 / 15 / 45 s
// (does Chrome hand out a working device after the driver reset, apart from anything the engine does?), then closes the browser and starts
// a new one for the remaining attempts, so one loss does not eat the rest of the count.
// EXAMPLE (substring of an Examples-menu label), ATTEMPTS (default 10), FRESH=1 (a new browser per attempt: every run is a page's first), STUDIO_URL (default local :5280). Models: out/site-profile.
// Exit 1 if any attempt failed.
const { chromium } = require('playwright-core');
const path = require('path');
const BASE = process.env.STUDIO_URL || 'http://localhost:5280/';
const URL = BASE + (BASE.includes('?') ? '&' : '?') + 'nodetimes=1';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const EXAMPLE = process.env.EXAMPLE || 'FLUX.2 Klein text to image';
const ATTEMPTS = parseInt(process.env.ATTEMPTS || '10', 10);
const FRESH = process.env.FRESH === '1'; // a new browser for every attempt: counts FIRST runs (model load and kernel compiles included)

const probe = page => page.evaluate(async () => {
  try {
    const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!a) return 'no adapter';
    const d = await a.requestDevice();
    const buffer = d.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    await d.queue.onSubmittedWorkDone();
    buffer.destroy(); d.destroy();
    return `device ok (${a.info.description} ${a.info.backend || ''})`;
  } catch (e) { return 'refused: ' + String(e.message || e).split('\n')[0].slice(0, 160); }
});

(async () => {
  let done = 0, failed = 0, launches = 0;
  const times = [];
  while (done < ATTEMPTS) {
    launches++;
    const context = await chromium.launchPersistentContext(path.join(__dirname, 'out', 'site-profile'), {
      executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
      args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'],
    });
    const page = context.pages()[0] || await context.newPage();
    let gpuLines = [], browserLines = [];
    page.on('console', m => {
      const t = m.text();
      if (/^\[(gpu|stage)\]/.test(t)) gpuLines.push(t.slice(0, 200));
      // Browser errors and warnings (a WebGPU validation error shows here, and a dropped submit leaves the device alive): kept per run.
      if ((m.type() === 'error' || m.type() === 'warning') && !/powerPreference/.test(t)) browserLines.push(`${m.type()}: ${t.slice(0, 400)}`);
    });
    page.on('pageerror', e => browserLines.push('pageerror: ' + String(e.message).slice(0, 400)));
    await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(4000);
    if (launches === 1) console.log('ADAPTER ' + await page.evaluate(async () => { const a = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' }); return a ? `${a.info.vendor} ${a.info.description} ${a.info.backend || ''}` : 'none'; }) + ' | ' + URL);
    await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
    await page.locator('.menu li', { hasText: EXAMPLE }).first().click(); await page.waitForTimeout(2000);
    await page.click('button:has-text("History")'); await page.waitForTimeout(500);
    const status = async () => (await page.locator('.status').first().textContent({ timeout: 60000 }).catch(() => '')) ?? '';
    let inPage = 0;
    while (done < ATTEMPTS) {
      const t0 = Date.now();
      let toast = '';
      gpuLines = []; browserLines = [];
      await page.click('button:has-text("Queue")');
      await page.waitForTimeout(300);
      while (Date.now() - t0 < 30 * 60000) {
        toast = ((await page.locator('.toast').first().textContent({ timeout: 50 }).catch(() => '')) || toast).slice(0, 300);
        if (!/Running|Queued/.test(await status())) break;
        await page.waitForTimeout(100);
      }
      const secs = (Date.now() - t0) / 1000;
      await page.waitForTimeout(500);
      const title = ((await page.locator('.hist-entry .hist-title').first().textContent().catch(() => '')) ?? '').trim();
      done++; inPage++;
      const ok = /done/.test(title);
      console.log(`attempt ${done} (browser ${launches}, run ${inPage} in its page): ${ok ? 'ran' : 'FAILED'} in ${secs.toFixed(1)} s${ok ? '' : ' | ' + toast.trim().split('\n')[0]}`);
      for (const l of browserLines.slice(0, 12)) console.log('  browser ' + l);
      if (ok) { times.push(secs); if (FRESH) break; continue; }
      failed++;
      for (const wait of [0, 15, 30]) {
        if (wait) await page.waitForTimeout(wait * 1000);
        console.log(`  plain-JS probe ${((Date.now() - t0) / 1000 - secs).toFixed(0)} s after the failure: ${await probe(page)}`);
      }
      for (const l of gpuLines.slice(-10)) console.log('  ' + l); // the last stages the run reported, then the loss
      // Does reloading the page get a device back (what a visitor would try first), or only a new browser?
      await page.reload(); await page.waitForSelector('canvas'); await page.waitForTimeout(3000);
      console.log('  plain-JS probe after a page reload: ' + await probe(page));
      break;
    }
    await context.close();
  }
  times.sort((a, b) => a - b);
  console.log(`SUMMARY ${EXAMPLE}: ${ATTEMPTS} attempts, ${failed} failed, ${launches} browser launches; successful runs ${times.length ? times[0].toFixed(1) + ' - ' + times[times.length - 1].toFixed(1) + ' s' : 'none'}`);
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
