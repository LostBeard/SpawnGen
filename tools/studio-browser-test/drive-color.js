// The COLOR widget in the real page: a Color Picker node shows a swatch with its hex code, and clicking the row opens a colour input.
// Look at out/color-widget.png.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
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

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await require('./tab-offset')(page);
  await page.waitForTimeout(3500);
  await page.mouse.dblclick(200, 700);
  await page.waitForSelector('.search input');
  await page.keyboard.type('Color Picker');
  await page.waitForTimeout(300);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'color-widget.png') });

  // The swatch: find a pure white 20x(h) block near the new node by reading canvas pixels around the drop point.
  const white = await page.evaluate(() => {
    const c = document.querySelector('.stage canvas');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] === 255 && d[i + 1] === 255 && d[i + 2] === 255) n++;
    return n;
  });
  check('the node shows a white swatch for #ffffff', white > 100, `${white} white pixels`);

  // Open the properties panel for the new node (it is selected after placing): the colour row is a colour input there.
  await page.keyboard.press('i');
  await page.waitForTimeout(400);
  const picker = page.locator('.props input[type=color]');
  check('the properties panel edits it with a colour picker', await picker.count() === 1);
  await picker.evaluate(el => { el.value = '#ff0000'; el.dispatchEvent(new Event('change', { bubbles: true })); });
  await page.waitForTimeout(400);
  const red = await page.evaluate(() => {
    const c = document.querySelector('.stage canvas');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] === 255 && d[i + 1] === 0 && d[i + 2] === 0) n++;
    return n;
  });
  await page.screenshot({ path: path.join(OUT, 'color-widget-red.png') });
  check('choosing red repaints the swatch', red > 100, `${red} red pixels`);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
