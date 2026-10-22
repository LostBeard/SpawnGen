// Custom nodes through the server: the server loaded SpawnGen.SampleNodes from its custom_nodes folder, the Studio learns its nodes from /object_info,
// lists them in the library, opens a workflow that uses them and runs it on the server. Start the server with SpawnGen__CustomNodesPath pointing at a folder that
// holds samples/SpawnGen.SampleNodes/bin/Release/net10.0/SpawnGen.SampleNodes.dll. Look at out/custom.png: "negneG" ... reversed text in Preview Any.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.click('button:has-text("Nodes")');
  await page.waitForSelector('.library');
  await page.fill('.lib-filter', 'sample');
  await page.waitForTimeout(300);
  const names = await page.locator('.lib-node').allTextContents();
  check("the server's custom nodes are in the library", names.includes('Reverse Text') && names.includes('Tint Image'), names.join(' | '));
  await page.fill('.lib-filter', '');
  await page.click('button:has-text("Nodes")');

  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'custom-nodes-workflow.json'));
  await page.waitForTimeout(1200);
  check('a workflow with a custom node opens without a missing-node warning', /^3 nodes, 2 links/.test((await page.textContent('.status')).trim()), (await page.textContent('.status')).trim());
  await page.locator('button:has-text("Queue")').click();
  for (let i = 0; i < 40; i++) { await page.waitForTimeout(500); if (i > 2 && await page.locator('button:has-text("Queue")').isVisible()) break; }
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, 'custom.png') });
  const history = await page.evaluate(async () => Object.values(await (await fetch('/history')).json()).pop());
  check('the run succeeded on the server', history.status.status_str === 'success', history.status.status_str);
  check('the custom node reversed the text', history.outputs['3'].text[0] === 'neGnwapS', JSON.stringify(history.outputs['3']));
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
