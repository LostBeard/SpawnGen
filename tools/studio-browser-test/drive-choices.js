// Combo lists follow the engine: a server with an extra checkpoint (checkpoints.json in its models folder) offers it in Load Checkpoint while the server engine
// is selected, and the browser engine's list does not have it. Needs the server (it serves the Studio) started with SpawnGen__ModelsPath on a folder whose
// checkpoints.json names "catalog_demo".
const { chromium } = require('playwright-core');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu', '--enable-unsafe-webgpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const choices = async () => {
    await page.mouse.click(200, 477); // the ckpt_name widget of Load Checkpoint in the fitted starter graph
    await page.waitForSelector('.widget-edit select');
    const options = await page.locator('.widget-edit select option').allTextContents();
    await page.keyboard.press('Escape');
    await page.mouse.click(800, 800);
    await page.waitForTimeout(300);
    return options;
  };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  const engines = await page.locator('select.engine option').allTextContents();
  check('both engines are offered', engines.length === 2, engines.join(' | '));
  const onServer = await choices();
  check("the server's list has its extra checkpoint", onServer.includes('catalog_demo') && onServer.includes('sd_turbo'), onServer.join(', '));
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  await page.waitForTimeout(500);
  const inBrowser = await choices();
  check("the browser engine's list does not", !inBrowser.includes('catalog_demo') && inBrowser.includes('sd_turbo'), inBrowser.join(', '));
  await page.selectOption('select.engine', { index: 0 });
  await page.waitForTimeout(500);
  check('switching back to the server brings it back', (await choices()).includes('catalog_demo'));
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
