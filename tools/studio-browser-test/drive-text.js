// A text result in a node: Preview Any on the in-page engine (no GPU, no model) shows the text it was given in a box under its widget.
// Look at out/text-result.png after the run: three lines of text in a dark box inside the node.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

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
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'preview-any-workflow.json'));
  await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(OUT, 'text-before.png') });
  await page.locator('button:has-text("Queue")').click();
  await page.waitForTimeout(4000);
  await page.screenshot({ path: path.join(OUT, 'text-result.png') });
  check('the run finished', await page.locator('button:has-text("Queue")').isVisible());
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
