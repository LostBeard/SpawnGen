// Group boxes in the real page: Ctrl+G around the selection, drag the title (the nodes inside come along), rename by double click, colour from the
// menu, delete, undo. Checks go through the saved workflow JSON. Look at out/groups-1.png and out/groups-2.png too.
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
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const saved = async () => {
    const [download] = await Promise.all([page.waitForEvent('download'), page.click('button:has-text("Save")')]);
    const p = path.join(OUT, 'groups-saved.json');
    await download.saveAs(p);
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  };
  const sampler = wf => wf.nodes.find(n => n.type === 'KSampler');

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.mouse.click(200, 700);          // focus the canvas, deselect
  await page.keyboard.press('Control+a');
  await page.keyboard.press('Control+g');
  await page.waitForTimeout(300);
  let wf = await saved();
  check('Ctrl+G adds a group around the selection', wf.groups.length === 1, JSON.stringify(wf.groups[0] && wf.groups[0].bounding));
  const g0 = wf.groups[0], s0 = sampler(wf).pos.slice();
  await page.screenshot({ path: path.join(OUT, 'groups-1.png') });

  // After Ctrl+G on the fitted example the group's title bar spans about x 93..1507, y 110..144 on screen (measured: see out/groups-1.png).
  const title = { x: 300, y: 125 };
  await page.mouse.move(title.x, title.y);
  await page.mouse.down();
  await page.mouse.move(title.x + 30, title.y + 20, { steps: 4 });
  await page.mouse.move(title.x + 60, title.y + 40, { steps: 4 });
  await page.mouse.up();
  await page.waitForTimeout(300);
  wf = await saved();
  const dx = (wf.groups[0].bounding[0] - g0.bounding[0]), dy = (wf.groups[0].bounding[1] - g0.bounding[1]);
  const sdx = sampler(wf).pos[0] - s0[0], sdy = sampler(wf).pos[1] - s0[1];
  check('dragging the group title moves the group', dx > 20 && dy > 10, `${dx.toFixed(0)}, ${dy.toFixed(0)}`);
  check('the nodes inside came along by the same amount', Math.abs(sdx - dx) < 0.5 && Math.abs(sdy - dy) < 0.5, `${sdx.toFixed(0)}, ${sdy.toFixed(0)}`);

  // Rename by double click on the title.
  const title2 = { x: title.x + 60, y: title.y + 40 }; // the title bar after the drag
  await page.waitForTimeout(500);
  await page.mouse.dblclick(title2.x, title2.y);
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(OUT, 'groups-dbl.png') });
  await page.waitForSelector('.widget-edit input');
  await page.waitForTimeout(300);
  await page.keyboard.press('Control+a');
  await page.keyboard.type('Pipeline');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(300);
  wf = await saved();
  check('double click renames the group', wf.groups[0].title === 'Pipeline', wf.groups[0].title);

  // Colour from the menu.
  await page.mouse.click(title2.x, title2.y, { button: 'right' });
  await page.waitForSelector('.menu');
  await page.click('.menu li:has-text("Colour: green")');
  await page.waitForTimeout(300);
  wf = await saved();
  check('the context menu changes the colour', wf.groups[0].color === '#8A8', wf.groups[0].color);
  await page.screenshot({ path: path.join(OUT, 'groups-2.png') });

  // Delete and undo.
  await page.mouse.click(title2.x, title2.y);
  await page.keyboard.press('Delete');
  await page.waitForTimeout(300);
  wf = await saved();
  check('Delete removes the selected group and keeps the nodes', wf.groups.length === 0 && wf.nodes.length === 7, `${wf.groups.length} groups ${wf.nodes.length} nodes`);
  await page.mouse.click(200, 800);          // the Save button had the focus: give it back to the canvas
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(300);
  wf = await saved();
  check('undo brings the group back', wf.groups.length === 1 && wf.groups[0].title === 'Pipeline', wf.groups.length + ' groups');

  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
