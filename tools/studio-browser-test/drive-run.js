// End to end through the server: opens the Studio the server hosts, presses Queue, follows the run on the toolbar, and checks the server history
// and the screenshots. Needs the SpawnGen server running (it serves the Studio), a GPU, and the SD-Turbo model installed. Takes about 40 seconds.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');

const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  // CDP_URL attaches to an already running window (the desktop shell with WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9333).
  const attach = process.env.CDP_URL;
  const browser = attach ? await chromium.connectOverCDP(attach) : await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = attach ? browser.contexts()[0].pages()[0] : await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));

  if (!attach) await page.goto(URL); else await page.reload();
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  const status = async () => (await page.textContent('.status')).trim();
  const historyOf = () => page.evaluate(async () => Object.values(await (await fetch('/history')).json()).map(h => h.status.status_str));
  const before = (await historyOf()).length;
  const queue = page.locator('button:has-text("Queue")');
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  check('Queue is enabled when a server hosts the page', await queue.isEnabled());
  await queue.click();

  const seen = [];
  const started = Date.now();
  while (Date.now() - started < 150000) {
    const s = await status();
    const head = s.split(' \u00b7 ')[0];
    if (/Running|Queued/.test(head) && !seen.includes(head.split(':')[0])) {
      seen.push(head.split(':')[0]);
      console.log(`  +${((Date.now() - started) / 1000).toFixed(0)}s ${head}`);
      if (seen.length === 3) await page.screenshot({ path: path.join(OUT, 'run-2-running.png') });
    }
    if (!/Running|Queued/.test(s) && Date.now() - started > 3000) break;
    await page.waitForTimeout(500);
  }
  await page.waitForTimeout(2500); // the result picture loads after the last event
  await page.screenshot({ path: path.join(OUT, 'run-3-done.png') });

  const history = (await historyOf()).slice(before);
  check('the server ran it to success', history.length === 1 && history[0] === 'success', JSON.stringify(history));
  // A server that already ran this graph serves the loader and the text encoders from its cache; the sampler and decode always run (new random seed).
  check('the toolbar followed the sampler and the decode', ['Running KSampler', 'Running VAE Decode'].every(s => seen.includes(s)), seen.join(' > '));
  check('no browser errors', problems.length === 0, problems.join(' | '));
  console.log('Look at out/run-3-done.png: the Save Image node should show the generated picture.');

  if (!attach) await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
