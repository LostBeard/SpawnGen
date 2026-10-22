// Node id badges in the real page: Settings > Show node ids draws #id in each title bar (the picture changes), and the choice survives a reload.
// Look at out/nodeids.png: "#3", "#4"... at the right of every title bar.
const { chromium } = require('playwright-core');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
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
  const hash = async () => crypto.createHash('md5').update(await page.locator('canvas').screenshot()).digest('hex');
  const open = async () => { await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500); };
  await open();
  const plain = await hash();
  await page.click('button:has-text("Settings")');
  await page.waitForSelector('.settings');
  await page.click('.settings label.row:has-text("node ids") input');
  await page.waitForTimeout(400);
  await page.click('.settings .set-head button');
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(OUT, 'nodeids.png') });
  check('ids are drawn', (await hash()) !== plain);
  await page.waitForTimeout(1500);
  await open();
  await page.click('button:has-text("Settings")');
  await page.waitForSelector('.settings');
  check('the setting survives a reload', await page.locator('.settings label.row:has-text("node ids") input').isChecked());
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
