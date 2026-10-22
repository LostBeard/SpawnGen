// Drives the Studio in headless Chrome: real pointer/keyboard events on the canvas, checks via the toolbar status text and screenshots.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const logs = [];
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') logs.push(m.type() + ': ' + m.text()); });
  page.on('pageerror', e => logs.push('PAGEERROR ' + e.message));

  const status = async () => (await page.textContent('.status')).trim();
  const shot = async n => page.screenshot({ path: `${OUT}/${n}.png` });
  const undoEnabled = async () => !(await page.getAttribute('button[title^="Undo"]', 'disabled') !== null);
  const results = [];
  const check = (name, ok, detail) => { results.push({ name, ok }); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await require('./tab-offset')(page);
  await page.waitForTimeout(3500);
  const s0 = await status();
  check('example graph loads', /^7 nodes, 9 links/.test(s0), s0);

  // Drag the KSampler by its title bar.
  await page.mouse.move(1040, 234);
  await page.mouse.down();
  for (let i = 1; i <= 8; i++) await page.mouse.move(1040 + i * 15, 234 + i * 15);
  await page.mouse.up();
  await page.waitForTimeout(400);
  check('dragging a node enables undo', await undoEnabled());
  await shot('drive-2-dragged');
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(400);
  check('ctrl+z undoes the drag', !(await undoEnabled()));

  // Wheel zoom at the cursor.
  const zoom = async () => parseInt((await status()).match(/(\d+)%/)[1]);
  const z0 = await zoom(); // 100 after the initial fit
  await page.mouse.move(800, 450);
  await page.mouse.wheel(0, -400);
  await page.waitForTimeout(400);
  const z1 = await zoom();
  check('wheel zooms in', z1 > z0, `${z0}% -> ${z1}%`);
  await page.mouse.click(200, 700); // deselect first: F fits the selection when there is one
  await page.keyboard.press('f');
  await page.waitForTimeout(400);
  check('F fits the view again', (await zoom()) === z0, `${await zoom()}%`);

  // Double-click empty canvas: node search, add a node.
  await page.mouse.dblclick(200, 700);
  await page.waitForSelector('.search input');
  await shot('drive-3-search');
  await page.keyboard.type('preview');
  await page.waitForTimeout(300);
  const listed = await page.locator('.search li').allTextContents();
  check('search filters the node list', listed.length >= 1 && /Preview Image/i.test(listed[0]), listed.join(' | '));
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  check('Enter in search places the node', /^8 nodes/.test(await status()), await status());
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(300);
  check('undo removes the placed node', /^7 nodes/.test(await status()), await status());

  // The seed's "control after generate" combo sits right under the seed: pick a mode.
  await page.mouse.click(1030, 378);
  await page.waitForSelector('.widget-edit select');
  await page.selectOption('.widget-edit select', 'increment');
  await page.waitForTimeout(400);
  check('control after generate combo closes after a pick', (await page.locator('.widget-edit').count()) === 0);

  // Edit a widget in place: KSampler steps row (below the control row).
  await page.mouse.click(1030, 404);
  await page.waitForSelector('.widget-edit input');
  await shot('drive-4-widget-edit');
  await page.keyboard.press('Control+a');
  await page.keyboard.type('4');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(400);
  check('widget editor closes on Enter', (await page.locator('.widget-edit').count()) === 0);
  await shot('drive-5-after-edit');

  // Save the workflow and read what was downloaded.
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('button:has-text("Save")')]);
  const savePath = `${OUT}/saved-workflow.json`;
  await download.saveAs(savePath);
  const saved = JSON.parse(fs.readFileSync(savePath, 'utf8'));
  const sampler = saved.nodes.find(n => n.type === 'KSampler');
  check('saved workflow has the edited steps value', sampler && sampler.widgets_values[2] === 4, JSON.stringify(sampler && sampler.widgets_values));
  check('saved workflow has the chosen control mode', sampler && sampler.widgets_values[1] === 'increment', JSON.stringify(sampler && sampler.widgets_values));
  check('saved workflow has all nodes and links', saved.nodes.length === 7 && saved.links.length === 9, `${saved.nodes.length} nodes ${saved.links.length} links`);

  // Open ComfyUI's stock workflow through the file input.
  await page.setInputFiles('input[type=file]', path.join(__dirname, 'comfy-default-workflow.json'));
  await page.waitForTimeout(1200);
  check('opens a ComfyUI workflow', /^7 nodes, 9 links/.test(await status()), await status());
  await shot('drive-6-comfy-default');

  // Right-click the canvas: menu.
  await page.mouse.click(200, 760, { button: "right" });
  await page.waitForSelector('.menu');
  const items = await page.locator('.menu li').allTextContents();
  check('context menu on the canvas', items.some(i => /Add node/.test(i)), items.join(' | '));
  await shot('drive-7-menu');

  console.log('\nconsole warnings/errors:', logs.length ? '\n' + logs.slice(0, 10).join('\n') : 'none');
  const failed = results.filter(r => !r.ok).length;
  console.log(`\n${results.length - failed}/${results.length} passed`);
  await browser.close();
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
