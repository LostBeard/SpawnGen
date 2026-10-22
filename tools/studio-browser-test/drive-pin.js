// Pinning in the real page: P pins the selected node (a pin shows in its title, no resize grip), a pinned node cannot be dragged, the workflow carries flags.pinned,
// and the context menu offers Unpin. Look at out/pin.png: the KSampler with a pin in its title bar and no grip.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const hash = async () => crypto.createHash('md5').update(await page.locator('canvas').screenshot()).digest('hex');
  const saved = async (name) => {
    const [d] = await Promise.all([page.waitForEvent('download'), page.click('button:has-text("Save")')]);
    await d.saveAs(path.join(OUT, name));
    return JSON.parse(fs.readFileSync(path.join(OUT, name), 'utf8')).nodes.find(n => n.type === 'KSampler');
  };

  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500); await require('./tab-offset')(page);
  const plain = await hash();
  await page.mouse.click(1040, 234);              // select the KSampler by its title
  await page.keyboard.press('p');
  await page.waitForTimeout(300);
  const pinned = await hash();
  check('P changes the picture (pin drawn, grip gone)', pinned !== plain);
  await page.screenshot({ path: path.join(OUT, 'pin.png') });

  await page.mouse.move(1060, 234); await page.mouse.down(); await page.mouse.move(1160, 330, { steps: 8 }); await page.mouse.up();
  await page.waitForTimeout(300);
  check('a pinned node does not move when dragged (only the selection box / pan is unaffected)', (await hash()) === pinned);

  const flags = (await saved('pin-saved.json')).flags;
  check('the workflow carries flags.pinned', flags && flags.pinned === true, JSON.stringify(flags));

  await page.mouse.click(1040, 234, { button: 'right' });
  await page.waitForSelector('.menu');
  check('the menu offers Unpin and no Collapse', (await page.locator('.menu li:has-text("Unpin")').count()) === 1 && (await page.locator('.menu li:has-text("Collapse")').count()) === 0);
  await page.click('.menu li:has-text("Unpin")');
  await page.waitForTimeout(300);
  const after = (await saved('pin-saved2.json')).flags;
  check('Unpin clears the flag', !after || !after.pinned, JSON.stringify(after));
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
