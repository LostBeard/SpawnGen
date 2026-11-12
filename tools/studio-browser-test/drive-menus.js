// Right-click menus and partial runs through the server (no GPU): Queue Selected Output Node runs only that output (partial_execution_targets);
// a link's Add Reroute here puts a reroute in it (undo takes it out); an output slot's and an input slot's Disconnect take their links out; an empty
// input slot offers Add node; Clone copies a node; a Queue with one output left unconnected runs the other and says which node it left out.
// Coordinates assume 1600x900 and menus-workflow.json as the Studio fits it.
const { chromium } = require('playwright-core');
const path = require('path');
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
  const counts = async () => { const m = (await page.textContent('.status')).match(/(\d+) nodes, (\d+) links/); return m ? [+m[1], +m[2]] : [-1, -1]; };
  const menu = async (x, y) => { await page.mouse.click(x, y, { button: 'right' }); await page.waitForTimeout(300); return page.locator('.menu li').allTextContents(); };
  const pick = async label => { await page.click(`.menu li:has-text("${label}")`); await page.waitForTimeout(400); };
  // The newest run's output nodes, once it has finished.
  const newestRun = async after => {
    for (let i = 0; i < 100; i++) {
      const runs = Object.values(await page.evaluate(async () => (await fetch('api/history')).json()));
      const newest = runs.sort((a, b) => b.prompt[0] - a.prompt[0])[0];
      if (newest && newest.prompt[0] > after && newest.status && newest.status.completed) return { number: newest.prompt[0], outputs: Object.keys(newest.outputs).sort() };
      await page.waitForTimeout(200);
    }
    return { number: after, outputs: [] };
  };
  const lastNumber = async () => Math.max(0, ...Object.values(await page.evaluate(async () => (await fetch('api/history')).json())).map(r => r.prompt[0]));

  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'menus-workflow.json')); await page.waitForTimeout(1200);
  check('the workflow opens: 4 nodes, 2 links', JSON.stringify(await counts()) === '[4,2]');

  // Queue Selected Output Node on the top Preview as Text.
  let items = await menu(1000, 313);
  check('an output node offers Queue Selected Output Node and Clone', items.some(t => /Queue Selected Output Node/.test(t)) && items.some(t => /^Clone/.test(t)), items.slice(0, 4).join(' | '));
  let before = await lastNumber();
  await pick('Queue Selected Output Node');
  let run = await newestRun(before);
  check('it runs only that output', JSON.stringify(run.outputs) === '["2"]', JSON.stringify(run.outputs));
  items = await menu(590, 313);
  check('a node that is not an output does not offer it', !items.some(t => /Queue Selected/.test(t)), items.slice(0, 3).join(' | '));
  await page.mouse.click(300, 820); await page.waitForTimeout(200);

  // A link: Add Reroute here, then undo.
  items = await menu(800, 600);
  check('a link offers Add Reroute here', items.some(t => /Add Reroute here/.test(t)), items.join(' | '));
  await pick('Add Reroute here');
  check('the reroute goes into the link', JSON.stringify(await counts()) === '[5,3]', JSON.stringify(await counts()));
  await page.mouse.click(300, 820); await page.keyboard.press('Control+z'); await page.waitForTimeout(300);
  check('one undo takes it out', JSON.stringify(await counts()) === '[4,2]', JSON.stringify(await counts()));

  // An output slot: Disconnect; the input it fed now offers Add node.
  items = await menu(720, 339);
  check('a linked output slot offers Disconnect', items.includes('Disconnect'), items.join(' | '));
  await pick('Disconnect');
  check('its link is gone', JSON.stringify(await counts()) === '[4,1]', JSON.stringify(await counts()));
  items = await menu(880, 339);
  check('an empty input slot offers Add node', items.some(t => /Add node/.test(t)), items.join(' | '));
  await page.mouse.click(300, 820); await page.waitForTimeout(200);

  // Queue with the top output unconnected: the other runs and the toast says what was left out.
  before = await lastNumber();
  await page.click('.toolbar button.primary:has-text("Queue")');
  await page.waitForTimeout(600);
  const toast = await page.locator('.toast').first().textContent().catch(() => '');
  check('Queue says which output it left out', /Queued without the outputs that need node 2/.test(toast || ''), toast);
  run = await newestRun(before);
  check('and runs the other', JSON.stringify(run.outputs) === '["4"]', JSON.stringify(run.outputs));

  // Clone, then an input slot's Disconnect.
  await page.mouse.click(300, 820);
  items = await menu(1000, 573);
  await pick('Clone');
  check('Clone adds a copy', JSON.stringify(await counts()) === '[5,1]', JSON.stringify(await counts()));
  items = await menu(880, 599);
  check('a linked input slot offers Disconnect', items.includes('Disconnect'), items.join(' | '));
  await pick('Disconnect');
  check('its link is gone', JSON.stringify(await counts()) === '[5,0]', JSON.stringify(await counts()));
  // Shapes: card on the bottom Preview as Text; it changes the drawing and is saved as ComfyUI's shape 4.
  const canvasHash = async () => require('crypto').createHash('md5').update(await page.locator('canvas').screenshot()).digest('hex');
  await page.mouse.click(300, 820); await page.waitForTimeout(200);
  const plain = await canvasHash();
  items = await menu(1000, 573);
  check('the node menu has Colours and Shapes submenus', items.some(t => /^Colours/.test(t)) && items.some(t => /^Shapes/.test(t)), items.join(' | ').slice(0, 200));
  await page.click('.menu li.has-sub:has-text("Shapes")');
  const shapes = await page.locator('.menu li.has-sub.open .sub li').allTextContents();
  check('Shapes offers default, box, round and card', JSON.stringify(shapes) === '["default","box","round","card"]', JSON.stringify(shapes));
  await page.click('.menu .sub li:text-is("card")'); await page.waitForTimeout(300);
  await page.mouse.click(300, 820); await page.waitForTimeout(300);
  check('the card outline is drawn', (await canvasHash()) !== plain);
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('.toolbar button:has-text("Export")')]);
  const wf = JSON.parse(require('fs').readFileSync(await download.path(), 'utf8'));
  check('it is saved as shape 4', wf.nodes.find(n => n.id === 4).shape === 4 && wf.nodes.find(n => n.id === 2).shape === undefined,
    JSON.stringify(wf.nodes.map(n => [n.id, n.shape])));
  await page.screenshot({ path: path.join(__dirname, 'out', 'menus-1.png') });

  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
