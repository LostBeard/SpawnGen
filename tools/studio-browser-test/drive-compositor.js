// The compositor editor (Create Layered Image) in the real page: Load Image -> Add Layer x2 -> Create Layered Image runs once, the node's
// "Open Compositor" button opens the editor, and the actions measured on ComfyUI's own editor (probe-comfy-compositor-steps*.js) are
// repeated; after each the fields, the layer list and the node's saved composition (through Export) are checked against what ComfyUI
// stored. Then the workflow runs again and the result must show the edit. Real GPU (the browser engine runs the nodes).
//   STUDIO_URL (default http://localhost:5280/). LOOK=1 stops after the first run and only writes out/compositor-0-ran.png.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

const prompt = {
  '1': { class_type: 'LoadImage', inputs: { image: 'example.png' } },
  '2': { class_type: 'AddLayer', inputs: { image: ['1', 0], name: 'coat' } },
  '3': { class_type: 'AddLayer', inputs: { image: ['1', 0], layers: ['2', 0], name: 'apple', x: 120, y: 60, opacity: 0.8 } },
  '4': { class_type: 'ImageCompositor', inputs: { layers: ['3', 0], compositor: {} } },
};

(async () => {
  const wfPath = path.join(OUT, 'compositor-prompt.json');
  fs.writeFileSync(wfPath, JSON.stringify(prompt));
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const status = () => page.evaluate(() => (document.querySelector('.runbar .status') || {}).textContent || '');
  const run = async () => {
    await page.click('button:has-text("Queue")');
    const t0 = Date.now();
    await page.waitForTimeout(3000);
    while (Date.now() - t0 < 240000 && /Running|Queued|Loading|Starting/i.test(await status())) await page.waitForTimeout(500);
    await page.waitForTimeout(2500);
    return status();
  };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(2000);
  console.log('status after the first run: ' + (await run()).trim().slice(0, 160));
  await page.screenshot({ path: path.join(OUT, 'compositor-0-ran.png') });
  if (process.env.LOOK) { console.log('problems: ' + problems.slice(0, 5).join(' || ')); await browser.close(); return; }
  const OPEN = [1445, 429]; // the node's Open Compositor row (out/compositor-0-ran.png)
  const PREVIEW = { x: 1360, y: 447, width: 170, height: 153 };
  const before = (await page.screenshot({ clip: PREVIEW })).toString('base64');

  const saved = async () => {
    const [download] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }),
      page.evaluate(() => [...document.querySelectorAll('.toolbar button')].find(b => b.textContent.trim() === 'Export').click())]);
    const wf = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    const node = wf.nodes.find(n => n.type === 'ImageCompositor');
    return node.widgets_values.find(v => v && typeof v === 'object') || {};
  };
  const ui = () => page.evaluate(() => {
    const val = id => { const e = document.querySelector(`[data-testid="${id}"]`); return e ? e.value : null; };
    return {
      open: !!document.querySelector('[data-testid="compositor"]'),
      fields: ['compositor-x', 'compositor-y', 'compositor-w', 'compositor-h', 'compositor-rotation', 'compositor-opacity'].map(val),
      canvas: ['compositor-canvas-w', 'compositor-canvas-h'].map(val),
      rows: [...document.querySelectorAll('[data-testid="compositor-row"] .cmp-name')].map(e => e.children[0].textContent.trim() + ' ' + e.children[1].textContent.trim()),
      background: (document.querySelector('[data-testid="compositor-background"] .cmp-line') || {}).textContent,
      zoom: (document.querySelector('[data-testid="compositor-zoom"] option') || {}).textContent,
    };
  });
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const fieldsAre = async (name, want) => { const u = await ui(); check(name, same(u.fields, want.map(String)), 'fields ' + u.fields.join(', ')); };
  const close = (a, b, t = 0.02) => Math.abs(a - b) <= t;

  await page.mouse.click(...OPEN);
  await page.waitForSelector('[data-testid="compositor"]', { timeout: 10000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, 'compositor-1-open.png') });
  let u = await ui();
  check('the editor opens on the run: top layer selected, the list top-most first, the canvas size', same(u.fields, ['120', '60', '512', '512', '0', '80'])
    && same(u.rows, ['apple N 80%', 'coat N 100%']) && u.background === '632×572' && u.zoom === '90%', JSON.stringify(u));
  check('nothing is saved before the first edit', same(await saved(), {}));

  const view = await page.locator('[data-testid="compositor-canvas"]').boundingBox();
  const Z = 0.9, OX = view.x + (view.width - 632 * Z) / 2, OY = view.y + (view.height - 572 * Z) / 2;
  const S = (x, y) => [OX + x * Z, OY + y * Z];
  const drag = async (from, by, shift) => {
    await page.mouse.move(from[0], from[1]); await page.mouse.down();
    if (shift) await page.keyboard.down('Shift');
    await page.mouse.move(from[0] + by[0] / 2, from[1] + by[1] / 2, { steps: 5 });
    await page.mouse.move(from[0] + by[0], from[1] + by[1], { steps: 5 });
    await page.mouse.up();
    if (shift) await page.keyboard.up('Shift');
    await page.waitForTimeout(250);
  };
  const press = async title => { await page.click(`[data-testid="compositor"] button[title="${title}"]`); await page.waitForTimeout(200); };
  const type = async (id, text) => { await page.fill(`[data-testid="${id}"]`, text); await page.press(`[data-testid="${id}"]`, 'Enter'); await page.waitForTimeout(200); };

  // moving: +45, +27 screen pixels at 90% is +50, +30 on the canvas (ComfyUI stored x 170.00027, y 90.00018)
  await drag(S(376, 316), [45, 27]);
  await fieldsAre('a body drag moves the layer by the drag over the zoom', [170, 90, 512, 512, 0, 80]);
  let v = await saved();
  const apple = () => v.layers[1], coat = () => v.layers[0];
  check('the first edit writes the whole composition, in the shape ComfyUI stores', v.version === 1 && same(v.canvas, { w: 632, h: 572 }) && v.inputs.length === 2
    && same(v.background, { color: '#ffffff', opacity: 1, visible: false }) && !('order' in v)
    && same(coat(), { name: 'coat', visible: true, opacity: 1, blend: 'normal', transform: { x: 0, y: 0, w: 512, h: 512, rotation: 0 }, flipH: false, flipV: false })
    && apple().name === 'apple' && apple().opacity === 0.8 && close(apple().transform.x, 170) && close(apple().transform.y, 90)
    && same(Object.keys(v), ['version', 'canvas', 'inputs', 'background', 'layers']), JSON.stringify(v).slice(0, 420));
  // resizing by the bottom-right corner (at 682, 602 now): free; then undo / redo
  await drag(S(682, 602), [45, 45]);
  await fieldsAre('a corner drag resizes with the top-left staying put', [170, 90, 562, 562, 0, 80]);
  await drag(S(732, 652), [54, 9]);
  await fieldsAre('a corner drag off the diagonal does not keep the aspect', [170, 90, 622, 572, 0, 80]);
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(200);
  await fieldsAre('Ctrl+Z undoes one action', [170, 90, 562, 562, 0, 80]);
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(200);
  await fieldsAre('Ctrl+Z again undoes the one before', [170, 90, 512, 512, 0, 80]);
  await press('Redo');
  await fieldsAre('the Redo button', [170, 90, 562, 562, 0, 80]);
  await press('Undo');
  await fieldsAre('the Undo button', [170, 90, 512, 512, 0, 80]);
  await drag(S(682, 346), [45, 0]);
  await fieldsAre('an edge handle changes one side', [170, 90, 562, 512, 0, 80]);
  await press('Undo');
  await drag(S(682, 602), [45, 9], true);
  await fieldsAre('with Shift a corner keeps the aspect (the drag projected on the diagonal)', [170, 90, 542, 542, 0, 80]);
  await press('Undo');
  // the rotation handle: 22 screen px above the top edge's middle; dragged level with the centre on the right is a quarter turn
  const top = S(426, 90), centre = S(426, 346);
  await drag([top[0], top[1] - 22], [centre[0] + 250 - top[0], centre[1] - (top[1] - 22)]);
  await fieldsAre('the rotation handle turns the layer about its centre', [170, 90, 512, 512, 90, 80]);
  v = await saved();
  check('rotation is stored in radians', close(apple().transform.rotation, Math.PI / 2, 0.01), String(apple().transform.rotation));
  await press('Undo');
  await press('Undo');
  await fieldsAre('undone back to where the run placed the layer', [120, 60, 512, 512, 0, 80]);
  v = await saved();
  check('the saved value follows undo', close(apple().transform.x, 120) && apple().transform.rotation === 0);

  const aligned = [];
  for (const a of ['Align left', 'Align horizontal center', 'Align right', 'Align top', 'Align vertical center', 'Align bottom']) {
    await press(a); const f = (await ui()).fields; aligned.push(f[0] + ',' + f[1]);
  }
  check('alignment is to the canvas', same(aligned, ['0,60', '60,60', '120,60', '120,0', '120,30', '120,60']), aligned.join(' | '));
  await press('Flip horizontal'); await press('Flip vertical');
  await type('compositor-rotation', '30'); await type('compositor-opacity', '55'); await type('compositor-x', '10');
  await type('compositor-w', '300'); await type('compositor-h', '100');
  u = await ui(); v = await saved();
  check('flips and typed fields are stored as ComfyUI stores them', same(u.fields, ['10', '60', '300', '100', '30', '55']) && same(u.rows, ['apple N 55%', 'coat N 100%'])
    && same(apple(), { name: 'apple', visible: true, opacity: 0.55, blend: 'normal', transform: { x: 10, y: 60, w: 300, h: 100, rotation: 0.5235987755982988 }, flipH: true, flipV: true }),
    JSON.stringify(apple()));
  await page.screenshot({ path: path.join(OUT, 'compositor-2-edited.png') });
  const blends = await page.evaluate(() => [...document.querySelectorAll('[data-testid="compositor-blend"] option')].map(o => o.textContent.trim()));
  check('the blend list has the 26 modes of ComfyUI in its order', blends.length === 26 && blends[0] === 'Normal' && blends[1] === 'Multiply' && blends[25] === 'Luminosity', blends.length + ': ' + blends.slice(0, 4).join(', '));
  await page.selectOption('[data-testid="compositor-blend"]', 'multiply');
  await page.waitForTimeout(200);
  u = await ui(); v = await saved();
  check('a blend mode is stored by its id and shown by its letter', apple().blend === 'multiply' && u.rows[0] === 'apple M 55%', u.rows[0]);

  // the list: selecting, reordering, visibility, the background
  await page.locator('[data-testid="compositor-row"]').nth(1).click();
  await page.waitForTimeout(200);
  await fieldsAre('a click on a row selects that layer', [0, 0, 512, 512, 0, 100]);
  await press('Move layer up');
  u = await ui(); v = await saved();
  check('Move layer up: the list changes and an order key appears (layers stay in input order)', same(u.rows, ['coat N 100%', 'apple M 55%']) && same(v.order, [1, 0])
    && v.layers[0].name === 'coat' && same(Object.keys(v), ['version', 'canvas', 'inputs', 'background', 'order', 'layers']), JSON.stringify(v.order) + ' ' + u.rows.join(' | '));
  await press('Move layer down');
  u = await ui(); v = await saved();
  check('Move layer down: back to the input order, the order key is gone', same(u.rows, ['apple M 55%', 'coat N 100%']) && !('order' in v));
  const eyes = page.locator('[data-testid="compositor-row"] button[title="Toggle visibility"]');
  await eyes.nth(0).click(); await page.waitForTimeout(200);
  v = await saved();
  const hidden = apple().visible === false;
  await eyes.nth(0).click(); await page.waitForTimeout(200);
  v = await saved();
  check('the toggle of a row hides and shows its layer', hidden && apple().visible === true);
  await page.click('[data-testid="compositor-background"] button[title="Toggle visibility"]'); await page.waitForTimeout(200);
  v = await saved();
  check('the Background toggle', v.background.visible === true);
  await page.mouse.click(view.x + 40, view.y + view.height / 2);
  await page.waitForTimeout(200);
  u = await ui();
  check('a click on empty space selects the Background: the canvas size fields', same(u.canvas, ['632', '572']) && u.fields[0] === null, JSON.stringify(u.canvas));
  await page.screenshot({ path: path.join(OUT, 'compositor-3-background.png') });

  // closing keeps everything; reopening shows the saved composition; Restore goes back to what the editor opened with
  const keep = JSON.stringify(v);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(400);
  u = await ui();
  check('Escape closes the editor and keeps the composition', !u.open && JSON.stringify(await saved()) === keep);
  await page.mouse.click(...OPEN);
  await page.waitForSelector('[data-testid="compositor"]', { timeout: 10000 });
  await page.waitForTimeout(600);
  u = await ui();
  check('reopened, the editor shows the saved composition', same(u.fields, ['10', '60', '300', '100', '30', '55']) && same(u.rows, ['apple M 55%', 'coat N 100%']), JSON.stringify(u.fields));
  await page.click('[data-testid="compositor-restore"]'); await page.waitForTimeout(200);
  check('Restore right after opening changes nothing', JSON.stringify(await saved()) === keep);
  await type('compositor-opacity', '70');
  v = await saved();
  const edited = apple().opacity === 0.7;
  await page.click('[data-testid="compositor-restore"]'); await page.waitForTimeout(200);
  check('Restore returns to the composition the editor opened with', edited && JSON.stringify(await saved()) === keep);
  await page.click('[data-testid="compositor-close"]');
  await page.waitForTimeout(400);
  check('Close closes the editor', !(await ui()).open);

  // the run replays the composition
  const text = (await run()).trim();
  await page.screenshot({ path: path.join(OUT, 'compositor-4-rerun.png') });
  const after = (await page.screenshot({ clip: PREVIEW })).toString('base64');
  check('the workflow runs again and the result shows the edit', !/fail|error/i.test(text) && after !== before, text.slice(0, 120));
  check('no browser errors', problems.length === 0, problems.slice(0, 3).join(' || ').slice(0, 300));
  console.log(`${results.filter(Boolean).length}/${results.length} passed`);
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
