// After the WebGPU device is lost, the next Queue works without reloading the page (found 2026-11-19 on the live site: one loss failed every
// later example). Runs the SD-Turbo example (its seed randomizes, so each Queue really runs), destroys the engine's WebGPU device through
// the ?testhooks=1 hook (a real device loss), queues again.
// STUDIO_URL (default http://localhost:5280/).
const { chromium } = require('playwright-core');
const path = require('path');
const URL = (process.env.STUDIO_URL || 'http://localhost:5280/') + '?testhooks=1';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
let failed = 0;
const check = (ok, what) => { console.log((ok ? 'PASS ' : 'FAIL ') + what); if (!ok) failed++; };

(async () => {
  const ctx = await chromium.launchPersistentContext(path.join(__dirname, 'out', 'device-lost-profile'), { executablePath: CHROME, headless: true,
    viewport: { width: 1600, height: 900 }, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
  const page = ctx.pages()[0] || await ctx.newPage();
  const logs = [];
  page.on('console', m => logs.push(m.text()));
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  const read = async sel => (await page.locator(sel).first().textContent({ timeout: 60000 }).catch(() => '')) || '';
  const runs = async () => parseInt(((await read('.toolbar')).match(/History \((\d+)\)/) || [])[1] || '0', 10);
  const queue = async () => {
    const before = await runs();
    await page.click('button:has-text("Queue")');
    for (let i = 0; i < 300; i++) {
      await page.waitForTimeout(1000);
      const toast = (await page.locator('.toast').first().textContent({ timeout: 200 }).catch(() => '')) || '';
      if (/Run failed/.test(toast)) return 'failed: ' + toast;
      if ((await runs()) > before && !/Running|Queued/.test(await read('.status'))) return 'ran';
    }
    return 'timed out';
  };

  await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
  await page.locator('.menu li', { hasText: 'Text to image (SD-Turbo' }).first().click(); await page.waitForTimeout(1500);
  check((await queue()) === 'ran', 'SD-Turbo runs');

  // A real device loss: the engine's own GPUDevice destroyed (?testhooks=1), as a GPU reset or a driver timeout loses it.
  // ...and, as after a driver reset (2026-11-23, RX 590: requestDevice refused with DXGI_ERROR_DEVICE_REMOVED seconds after the loss), the
  // browser refuses the next two attempts to open a device: the engine waits and tries again inside the same run.
  const hooked = await page.evaluate(() => { if (!window.spawngenLoseGpu) return false; window.spawngenLoseGpu(); window.spawngenRefuseDevices(2); return true; });
  check(hooked, 'the test hook destroyed the WebGPU device');
  await page.waitForTimeout(1500);

  // The SD-Turbo example randomizes its seed after each run, so every Queue really runs on the GPU (no cached result).
  console.log('lost-related console lines: ' + logs.filter(l => /lost/i.test(l)).slice(0, 6).join(' || '));
  const second = await queue();
  console.log('second run: ' + second);
  check(second === 'ran', 'Queue after the device loss runs (a new device, no reload)');
  check(logs.some(l => /\[gpu\] the WebGPU device was lost/.test(l)), 'the engine noticed the lost device and opened a new one');
  check(logs.filter(l => /\[gpu\] no new device yet/.test(l)).length === 2, 'the two refused opens were retried in the same run');
  const third = await queue();
  check(third === 'ran', 'and keeps running');

  // A loss the browser does not recover from in time (more refusals than the engine's tries): that run fails with the reason, and the
  // NEXT Queue opens a device again (a failed open must not answer every later run).
  await page.evaluate(() => { window.spawngenLoseGpu(); window.spawngenRefuseDevices(8); });
  await page.waitForTimeout(1500);
  const fourth = await queue();
  console.log('run with 8 refused opens: ' + fourth.slice(0, 200));
  check(/^failed/.test(fourth), 'a run fails when no device opens after every try');
  const fifth = await queue();
  check(fifth === 'ran', 'and the Queue after that failed open runs (the failure is not kept)');
  await page.screenshot({ path: path.join(__dirname, 'out', 'device-lost.png') });
  await ctx.close();
  console.log(failed ? `${failed} FAILED` : 'ALL PASS');
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
