// Editing a subgraph's inside in the real page: pack two nodes, open the subgraph from its menu (breadcrumbs, the inner nodes between the input and
// output boundary), add a node inside, go back out by the breadcrumb, and find the edit kept when the subgraph is opened again; the workflow still runs.
// Look at out/subgraph-inside.png.
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
  const nodes = async () => parseInt(((await page.textContent('.status')).match(/(\d+) nodes/) || [])[1] || '-1', 10);
  const runToEnd = async () => { await page.locator('button:has-text("Queue")').click(); for (let i = 0; i < 40; i++) { await page.waitForTimeout(400); if (i > 2 && await page.locator('button:has-text("Queue")').isVisible()) break; } await page.waitForTimeout(500); };
  // The packed node lands where the first packed node was: right-click it there.
  const openFromMenu = async () => {
    await page.mouse.click(440, 440, { button: 'right' });
    await page.waitForTimeout(300);
    if (!(await page.locator('.menu li:has-text("Open Subgraph")').count())) return false;
    await page.click('.menu li:has-text("Open Subgraph")');
    return true;
  };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'subgraph-workflow.json'));
  await page.waitForTimeout(1200);
  await page.mouse.click(480, 440);
  await page.keyboard.down('Shift');
  await page.mouse.click(800, 450);
  await page.keyboard.up('Shift');
  await page.keyboard.press('Control+Shift+E');
  await page.waitForTimeout(500);
  check('packed into a subgraph', (await nodes()) === 2, `${await nodes()} nodes`);

  check('the subgraph node offers Open Subgraph', await openFromMenu());
  await page.waitForTimeout(500);
  check('inside: breadcrumbs back to the workflow', await page.locator('.crumbs').count() === 1, await page.locator('.crumbs').textContent().catch(() => ''));
  check('inside: the two inner nodes and the two boundary nodes', (await nodes()) === 4, `${await nodes()} nodes`);
  await page.screenshot({ path: path.join(OUT, 'subgraph-inside.png') });

  // Add a node inside.
  await page.mouse.dblclick(200, 800);
  await page.waitForSelector('.search input');
  await page.keyboard.type('Trim Text');
  await page.waitForTimeout(300);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(400);
  check('a node added inside', (await nodes()) === 5, `${await nodes()} nodes`);

  await page.click('.crumbs button');
  await page.waitForTimeout(500);
  check('the breadcrumb goes back out', await page.locator('.crumbs').count() === 0 && (await nodes()) === 2, `${await nodes()} nodes`);
  await runToEnd();
  const toast = await page.locator('.toast').first().textContent().catch(() => '');
  check('the workflow still runs', !/failed|error/i.test(toast || ''), toast);

  check('opening it again', await openFromMenu());
  await page.waitForTimeout(500);
  check('the node added inside is still there', (await nodes()) === 5, `${await nodes()} nodes`);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
