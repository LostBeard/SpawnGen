// Autogrow inputs in the real page: a ComfyUI workflow whose Batch Images node has two pictures connected opens with both links and one free
// socket, and running it in the browser makes a batch of two. Look at out/autogrow.png.
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
  const status = async () => (await page.locator('.status').textContent()).trim();

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  await page.setInputFiles('input[accept*=".wav"]', path.join(__dirname, 'apple.png'));
  await page.waitForTimeout(800);
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'autogrow-workflow.json'));
  await page.waitForTimeout(1500);
  check('the workflow opens with every link', /^4 nodes, 3 links/.test(await status()), await status());
  await page.screenshot({ path: path.join(OUT, 'autogrow.png') });

  await page.click('button:has-text("History")');
  await page.locator('button:has-text("Queue")').click();
  for (let i = 0; i < 40; i++) { await page.waitForTimeout(500); if (i > 2 && await page.locator('button:has-text("Queue")').isVisible()) break; }
  await page.waitForTimeout(800);
  const thumbs = await page.locator('.hist-thumbs img').count();
  check('the batch of two reaches Preview Image (two pictures)', thumbs === 2, `${thumbs} pictures`);
  await page.screenshot({ path: path.join(OUT, 'autogrow-run.png') });
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
