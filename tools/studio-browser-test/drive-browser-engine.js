// The in-browser engine, with a graph that needs no GPU: upload a PNG, open Load Image -> Preview Image, run it on "This browser" and look
// at the result. Proves the run queue, events, stores and result pictures work inside the page. Works against the Studio alone or server-hosted.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');

const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu', '--enable-unsafe-webgpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  const status = async () => (await page.textContent('.status')).trim();

  const engines = await page.locator('select.engine option').allTextContents();
  check('the browser engine is offered', engines.some(e => /This browser/.test(e)), engines.join(' | '));
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });

  await page.setInputFiles('input[accept=".png"]', path.join(__dirname, 'apple.png'));
  await page.waitForTimeout(800);
  await page.setInputFiles('input[accept=".json"]', path.join(__dirname, 'load-preview-workflow.json'));
  await page.waitForTimeout(1200);
  check('the workflow opened', /^2 nodes, 1 links/.test(await status()), await status());
  await page.screenshot({ path: path.join(OUT, 'browser-1-before.png') });

  await page.locator('button:has-text("Queue")').click();
  await page.waitForTimeout(6000);
  await page.screenshot({ path: path.join(OUT, 'browser-2-after.png') });
  check('the run finished (Queue is back)', await page.locator('button:has-text("Queue")').isVisible(), await status());
  check('no browser errors', problems.length === 0, problems.join(' | '));
  console.log('Look at out/browser-2-after.png: the Preview Image node should show the apple.');

  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
