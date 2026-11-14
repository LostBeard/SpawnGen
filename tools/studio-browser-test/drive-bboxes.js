// Create Bounding Boxes' canvas in the real page: the same pointer actions the probe made on ComfyUI's own editor (probe-comfy-bboxes-steps2.js),
// checked against the editor_state ComfyUI stored, through Export. The canvas is found by its pixels: the Grid button's accent (Grid is on) and
// the dark field below it. STUDIO_URL (default http://localhost:5280/).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

const workflow = {
  last_node_id: 1, last_link_id: 0,
  nodes: [{ id: 1, type: 'CreateBoundingBoxes', pos: [300, 40], size: [380, 700], flags: {}, order: 0, mode: 0,
    inputs: [{ name: 'background', type: 'IMAGE', link: null }, { name: 'bboxes', type: 'BOUNDING_BOX,ARRAY,STRING', link: null }],
    outputs: [{ name: 'preview', type: 'IMAGE', links: [] }, { name: 'bboxes', type: 'BOUNDING_BOX', links: [] }, { name: 'elements', type: 'ARRAY', links: [] }],
    properties: {}, widgets_values: [1024, 1024, [], []] }],
  links: [], groups: [], config: {}, extra: {}, version: 0.4,
};
const META = { type: 'obj', text: '', desc: '', palette: [] };

(async () => {
  const wfPath = path.join(OUT, 'bboxes-workflow.json');
  fs.writeFileSync(wfPath, JSON.stringify(workflow));
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(OUT, 'bboxes-0-loaded.png') });

  const canvasBox = await page.locator('canvas').first().boundingBox();
  const grab = async () => page.evaluate(() => {
    const c = document.querySelector('canvas'); const r = c.getBoundingClientRect(); const k = c.width / r.width;
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height);
    return { w: c.width, h: c.height, k, left: r.left, top: r.top, data: Array.from(d.data) };
  });
  const img = await grab();
  const px = (x, y) => { const i = (y * img.w + x) * 4; return [img.data[i], img.data[i + 1], img.data[i + 2]]; };
  const near = (p, q, t = 6) => Math.abs(p[0] - q[0]) <= t && Math.abs(p[1] - q[1]) <= t && Math.abs(p[2] - q[2]) <= t;
  // The Grid button: the first accent (#4f8fe8) pixel, scanning rows.
  let grid = null;
  for (let y = 0; y < img.h && !grid; y++) for (let x = 0; x < img.w && !grid; x++) if (near(px(x, y), [79, 143, 232], 4)) grid = { x, y };
  // The canvas: straight below the button, the first run of the dark field (#202020); then its edges.
  let field = null;
  if (grid) for (let y = grid.y + 8; y < img.h && !field; y++) if (near(px(grid.x + 10, y), [32, 32, 32], 3) && near(px(grid.x + 10, y + 3), [32, 32, 32], 3)) field = { x: grid.x + 10, y };
  let rect = null;
  if (field) {
    // Walk the field's colour along its top row and its left column (the grid's first dots are a step in), stopping at the #4a4a4a border.
    let l = field.x, r = field.x, b = field.y;
    const inField = (x, y) => near(px(x, y), [32, 32, 32], 3);
    while (l > 0 && inField(l - 1, field.y + 1)) l--;
    while (r < img.w - 1 && inField(r + 1, field.y + 1)) r++;
    while (b < img.h - 1 && inField(l + 1, b + 1)) b++;
    rect = { x: img.left + l / img.k, y: img.top + field.y / img.k, w: (r - l + 1) / img.k, h: (b - field.y + 1) / img.k };
  }
  check('the canvas is drawn (toolbar and field)', !!rect, rect ? JSON.stringify(rect) : `grid ${JSON.stringify(grid)} field ${JSON.stringify(field)}`);
  if (!rect) { await browser.close(); process.exit(1); }
  check('the canvas is square for 1024 x 1024', Math.abs(rect.w - rect.h) <= 2, `${rect.w.toFixed(1)} x ${rect.h.toFixed(1)}`);

  const at = (fx, fy) => ({ x: rect.x + rect.w * fx, y: rect.y + rect.h * fy });
  async function drag(fx0, fy0, fx1, fy1) {
    const a = at(fx0, fy0), b = at(fx1, fy1);
    await page.mouse.move(a.x, a.y); await page.mouse.down();
    for (let i = 1; i <= 8; i++) await page.mouse.move(a.x + (b.x - a.x) * i / 8, a.y + (b.y - a.y) * i / 8);
    await page.mouse.up(); await page.waitForTimeout(200);
  }
  async function exported() {
    const [download] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }), page.click('.toolbar button:has-text("Export")')]);
    const saved = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    return saved.nodes.find(n => n.type === 'CreateBoundingBoxes').widgets_values[2];
  }
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const box = (x, y, width, height) => ({ x, y, width, height, metadata: META });
  // Within a pixel of the probe's fractions: the canvas here is about as many screen pixels wide as ComfyUI's, so a rounding step can land one
  // grid cell or pixel apart; the expected values are ComfyUI's, the check allows that one step.
  const close = (got, want, tol) => got.length === want.length && got.every((g, i) => ['x', 'y', 'width', 'height'].every(k => Math.abs(g[k] - want[i][k]) <= tol));

  await drag(0.2, 0.2, 0.7, 0.6);
  let s = await exported();
  check('draw A with Grid on: ComfyUI {192, 192, 512, 416}', close(s, [box(192, 192, 512, 416)], 0) && same(s[0].metadata, META), JSON.stringify(s));
  await drag(0.13, 0.71, 0.37, 0.93);
  await drag(0.25, 0.8, 0.3, 0.85);
  s = await exported();
  check('draw B and move it: ComfyUI {192, 800, 256, 224}', close(s, [box(192, 192, 512, 416), box(192, 800, 256, 224)], 0), JSON.stringify(s));
  // Grid off: the button is the accent pixel found above.
  await page.mouse.click(img.left + (grid.x + 12) / img.k, img.top + (grid.y + 6) / img.k);
  await page.waitForTimeout(200);
  await drag(0.81, 0.11, 0.93, 0.27);
  s = await exported();
  check('draw C with Grid off: ComfyUI {829, 113, 123, 164} (to a pixel)', s.length === 3 && close([s[2]], [box(829, 113, 123, 164)], 2), JSON.stringify(s[2]));
  await page.screenshot({ path: path.join(OUT, 'bboxes-1-drawn.png') });
  // Select C with a click (as a user would; Export took the keyboard focus), then Delete.
  { const c = at(0.87, 0.19); await page.mouse.click(c.x, c.y); await page.waitForTimeout(150); }
  await page.keyboard.press('Delete');
  await page.waitForTimeout(200);
  s = await exported();
  check('Delete removes the selected box (C), the node stays', s.length === 2, JSON.stringify(s));
  // The panel (100% zoom: world units are screen pixels): select A, then its description field (below obj / text and the text field).
  { const a = at(0.45, 0.4); await page.mouse.click(a.x, a.y); await page.waitForTimeout(150); }
  const descY = rect.y + rect.h + 4 + 4 + 20 + 6 + 36 + 6 + 27;
  await page.mouse.click(rect.x + rect.w / 2, descY);
  await page.waitForTimeout(300);
  const editor = page.locator('.widget-edit textarea');
  check('the description opens in a text box', await editor.count() === 1);
  if (await editor.count() === 1) {
    await editor.fill('a red apple');
    await editor.evaluate(el => el.blur());
    await page.waitForTimeout(300);
    s = await exported();
    check('the description is saved in the box', s[0].metadata.desc === 'a red apple' && s[0].metadata.type === 'obj', JSON.stringify(s[0].metadata));
  }
  await page.screenshot({ path: path.join(OUT, 'bboxes-2-described.png') });
  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await browser.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})();
