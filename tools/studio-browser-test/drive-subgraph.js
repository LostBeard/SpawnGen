// Subgraphs in the real page, on the in-page engine (no GPU): pack two nodes (click one, shift-click the other, Ctrl+Shift+E), the graph shrinks to two nodes and still runs,
// the saved workflow carries the definition, and Ctrl+Shift+E on the subgraph node unpacks it again. Look at out/subgraph.png.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu', '--enable-unsafe-webgpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const nodes = async () => parseInt(((await page.textContent('.status')).match(/(\d+) nodes/) || [])[1] || '-1', 10);
  const runToEnd = async () => { await page.locator('button:has-text("Queue")').click(); for (let i = 0; i < 40; i++) { await page.waitForTimeout(400); if (i > 2 && await page.locator('button:has-text("Queue")').isVisible()) break; } await page.waitForTimeout(500); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'subgraph-workflow.json'));
  await page.waitForTimeout(1200);
  check('three nodes to start with', (await nodes()) === 3);

  await page.mouse.click(480, 440);
  await page.keyboard.down('Shift');
  await page.mouse.click(800, 450);
  await page.keyboard.up('Shift');
  await page.keyboard.press('Control+Shift+E');
  await page.waitForTimeout(500);
  check('Ctrl+Shift+E packs the two nodes into one', (await nodes()) === 2, `${await nodes()} nodes`);
  await page.screenshot({ path: path.join(OUT, 'subgraph.png') });

  await runToEnd();
  const errorToast = await page.locator('.toast').first().textContent().catch(() => '');
  check('the graph with the subgraph runs', await page.locator('button:has-text("Queue")').isVisible() && !/failed/i.test(errorToast || ''), errorToast);

  const [download] = await Promise.all([page.waitForEvent('download'), page.click('button:has-text("Save")')]);
  await download.saveAs(path.join(OUT, 'subgraph-saved.json'));
  const saved = fs.readFileSync(path.join(OUT, 'subgraph-saved.json'), 'utf8');
  check('the saved workflow carries the definition', /"definitions"/.test(saved) && /"inputNode"/.test(saved) && /"target_id": -20/.test(saved) && /StringConcatenate/.test(saved));

  await page.locator('canvas').focus(); // Queue and Save took the keyboard focus; the subgraph node is still selected
  await page.keyboard.press('Control+Shift+E');
  await page.waitForTimeout(500);
  check('Ctrl+Shift+E on the subgraph node unpacks it', (await nodes()) === 3, `${await nodes()} nodes`);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
