// Node colours in the real page: right-click a node, pick a colour (context menu), the node is drawn in it and the colour survives a reload.
// Look at out/colors.png: the KSampler in purple, the VAE Decode in yellow.
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
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const hash = async () => crypto.createHash('md5').update(await page.locator('canvas').screenshot()).digest('hex');
  const open = async () => { await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500); await require('./tab-offset')(page); };
  const colour = async (x, y, label) => {
    await page.mouse.click(x, y, { button: 'right' });
    await page.waitForSelector('.menu');
    await page.click(`.menu li:has-text("${label}")`);
    await page.waitForTimeout(300);
  };

  await open();
  const before = await hash();
  await colour(1040, 234, 'Colour: purple'); // KSampler title
  const after = await hash();
  check('colouring a node changes the picture', before !== after);
  await colour(1400, 234, 'Colour: yellow'); // VAE Decode title
  await page.screenshot({ path: path.join(OUT, 'colors.png') });

  const [download] = await Promise.all([page.waitForEvent('download'), page.click('button:has-text("Save")')]);
  await download.saveAs(path.join(OUT, 'colors-saved.json'));
  const saved = JSON.parse(fs.readFileSync(path.join(OUT, 'colors-saved.json'), 'utf8'));
  const sampler = saved.nodes.find(n => n.type === 'KSampler');
  check('the workflow carries color and bgcolor', sampler.color === '#323' && sampler.bgcolor === '#535', `${sampler.color} ${sampler.bgcolor}`);

  const coloured = await hash();
  await page.waitForTimeout(1600);
  await page.reload(); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  const [again] = await Promise.all([page.waitForEvent('download'), page.click('button:has-text("Save")')]);
  await again.saveAs(path.join(OUT, 'colors-saved2.json'));
  const reloaded = JSON.parse(fs.readFileSync(path.join(OUT, 'colors-saved2.json'), 'utf8')).nodes.find(n => n.type === 'KSampler');
  check('the colours survive a reload', reloaded.color === '#323' && reloaded.bgcolor === '#535', `${reloaded.color} ${reloaded.bgcolor}`);

  await require('./tab-offset')(page);
  await colour(1040, 234, 'Colour: default');
  check('Colour: default goes back to the theme', (await hash()) !== coloured);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
