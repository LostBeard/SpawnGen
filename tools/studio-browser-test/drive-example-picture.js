// The shipped example.png through the picture examples in this browser: each opens, runs as it ships, and shows a result. Look at out/expic-*.png.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const EXAMPLES = (process.env.EXAMPLES || 'Remove the background,Pose and object detection,Upscale a picture with a model').split(',');

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const runs = async () => parseInt(((await page.textContent('.toolbar')).match(/History \((\d+)\)/) || [])[1] || '0', 10);
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  for (const [i, label] of EXAMPLES.entries()) {
    await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
    await page.click(`.menu li:has-text("${label}")`); await page.waitForTimeout(1500);
    const before = await runs(), t0 = Date.now();
    await page.click('button:has-text("Queue")');
    let done = false, failed = '';
    for (let s = 0; s < 600 && !done; s++) {
      await page.waitForTimeout(1000);
      // A toast can go between finding and reading it: read with a short timeout, or the read waits for the whole default timeout.
      const toast = await page.locator('.toast').first().textContent({ timeout: 300 }).catch(() => '') ?? '';
      if (/fail|error|not one of/i.test(toast)) { failed = toast; break; }
      done = (await runs()) > before && !/Running|Queued/.test(await page.textContent('.status'));
    }
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(OUT, `expic-${i}.png`) });
    check(`"${label}" runs on the shipped example.png (out/expic-${i}.png)`, done && !failed, failed || `${Math.round((Date.now() - t0) / 1000)} s`);
  }
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
