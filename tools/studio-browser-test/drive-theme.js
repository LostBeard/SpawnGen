// Colour palettes in the real page: Light repaints the canvas (light grid, light nodes) and the page (toolbar, panels) at once, survives a reload
// without a dark flash, and Dark brings the original picture back. Look at out/theme-light.png.
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
  const open = async () => { await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500); };
  // The canvas pixel at a CSS point (top-left corner is empty canvas background), and the mean luminance of the whole canvas.
  const canvasStats = () => page.evaluate(() => {
    const c = document.querySelector('.stage canvas');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let sum = 0, n = 0;
    for (let i = 0; i < d.length; i += 4 * 97) { sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; n++; }
    const corner = [d[(5 * c.width + 5) * 4], d[(5 * c.width + 5) * 4 + 1], d[(5 * c.width + 5) * 4 + 2]];
    return { mean: sum / n, corner };
  });
  const toolbarBg = () => page.locator('.toolbar').evaluate(t => getComputedStyle(t).backgroundColor);
  const choose = async label => {
    if (!(await page.locator('.settings').count())) await page.click('button:has-text("Settings")');
    await page.selectOption('.settings label:has-text("Colour palette") select', { label });
    await page.waitForTimeout(500);
  };

  await page.goto(URL);
  await page.evaluate(() => localStorage.removeItem('spawngen.settings'));
  await open();
  const dark = await canvasStats();
  const darkBar = await toolbarBg();
  check('dark is the default (dark canvas)', dark.mean < 80, dark.mean.toFixed(1));

  await choose('Light');
  const light = await canvasStats();
  await page.screenshot({ path: path.join(OUT, 'theme-light.png') });
  check('Light repaints the canvas light', light.mean > 150, light.mean.toFixed(1));
  check('the empty canvas is the light background', light.corner.every(v => v > 180), light.corner.join(','));
  const lightBar = await toolbarBg();
  check('the toolbar turns light too', lightBar !== darkBar && /rgb\((2[0-9]{2}|1[89][0-9]),/.test(lightBar), lightBar);
  check('html carries data-theme=light', (await page.evaluate(() => document.documentElement.dataset.theme)) === 'light');

  await page.waitForTimeout(800);
  await page.goto(URL);
  // Before the app has booted the page is already light (no dark flash).
  check('the page is light before the app boots', (await page.evaluate(() => document.documentElement.dataset.theme)) === 'light');
  await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  check('Light survives a reload', (await canvasStats()).mean > 150);

  await choose('Dark');
  const back = await canvasStats();
  check('Dark brings the dark picture back', Math.abs(back.mean - dark.mean) < 2, back.mean.toFixed(1) + ' vs ' + dark.mean.toFixed(1));
  check('and the dark toolbar', (await toolbarBg()) === darkBar);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
