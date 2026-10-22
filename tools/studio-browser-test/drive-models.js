// The model manager in the real page, through the server: it lists the files the models folder's hub cache holds (biggest first), deletes one, and clears the rest.
// Start the server with SpawnGen__ModelsPath on a folder whose hub/ has two files named hf_owner_repo_main_big.bin and hf_owner_repo_main_small.bin.
const { chromium } = require('playwright-core');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';

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
  await page.click('button:has-text("Models")');
  await page.waitForSelector('.models .model-file');
  const names = await page.locator('.model-file .name').allTextContents();
  check('the files are listed, biggest first, without the plumbing', names.length === 2 && names[0] === 'owner_repo / big.bin', names.join(' | '));
  check('with their sizes and the total', /3 KB|2 KB/.test(await page.locator('.model-file .size').first().textContent()) && /2 files/.test(await page.locator('.model-total').textContent()), await page.locator('.model-total').textContent());
  await page.locator('.model-file button').first().click();
  await page.waitForTimeout(500);
  check('a file can be deleted', (await page.locator('.model-file').count()) === 1);
  await page.click('.model-total button');
  await page.waitForTimeout(500);
  check('Delete all empties the list', (await page.locator('.model-file').count()) === 0 && /Nothing downloaded/.test(await page.locator('.models').textContent()));
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
