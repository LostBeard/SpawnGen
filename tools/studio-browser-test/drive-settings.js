// Settings in the real page: the link style changes how links are drawn (the canvas picture changes), settings survive a reload, and hidden links
// draw nothing. Look at out/settings-straight.png: right-angled links.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
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
  const spline = await hash();
  await page.click('button:has-text("Settings")');
  await page.waitForSelector('.settings');
  await page.selectOption('.settings label:has-text("Link style") select', { label: 'Straight' });
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, 'settings-straight.png') });
  const straight = await hash();
  check('Straight links look different from splines', straight !== spline);

  await page.selectOption('.settings label:has-text("Link style") select', { label: 'Hidden' });
  await page.waitForTimeout(400);
  const hidden = await hash();
  check('hidden links draw nothing (another picture again)', hidden !== straight && hidden !== spline);

  await page.selectOption('.settings label:has-text("Link style") select', { label: 'Linear' });
  await page.waitForTimeout(300);
  await page.waitForTimeout(1500);
  await open();
  await page.click('button:has-text("Settings")');
  await page.waitForSelector('.settings');
  const chosen = await page.locator('.settings label:has-text("Link style") select').evaluate(s => s.options[s.selectedIndex].text);
  check('the choice survives a reload', chosen === 'Linear', chosen);

  await page.click('.settings label.row:has-text("minimap") input');
  await page.waitForTimeout(300);
  check('the minimap can be switched off in the settings', !(await page.locator('button:has-text("Map")').evaluate(b => b.classList.contains('on'))));
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
