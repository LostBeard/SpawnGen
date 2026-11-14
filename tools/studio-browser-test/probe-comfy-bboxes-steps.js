// Black-box look at ComfyUI's Create Bounding Boxes canvas editor (Node 2.0 view): a sequence of user actions, and after each the stored
// editor_state and the controls on screen, plus a screenshot. Nothing of ComfyUI's code is read.  COMFY_URL (default http://127.0.0.1:8199/)
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out', 'bboxes');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.COMFY_URL || 'http://127.0.0.1:8199/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.evaluate(async () => { await window.app.ui.settings.setSettingValueAsync('Comfy.VueNodes.Enabled', true); });
  await page.reload({ waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.evaluate(() => {
    const app = window.app; app.graph.clear();
    const LG = window.LiteGraph || window.comfyAPI.litegraph.LiteGraph;
    const node = LG.createNode('CreateBoundingBoxes'); node.pos = [100, 50]; app.graph.add(node); app.canvas.ds.scale = 0.75; app.canvas.ds.offset = [0, 0]; app.canvas.setDirty(true, true);
  });
  await page.waitForTimeout(1500);

  const log = [];
  const controls = () => page.evaluate(() => [...document.querySelectorAll('[data-node-id] *, [role="dialog"] *, [role="menu"] *, .p-popover *, .p-colorpicker *')]
    .filter(el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && (['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'CANVAS'].includes(el.tagName) || el.getAttribute('role') === 'menuitem'); })
    .map(el => { const r = el.getBoundingClientRect(); return `${el.tagName}${el.type ? ':' + el.type : ''} "${(el.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 30)}" aria="${el.getAttribute('aria-label') || ''}" ph="${el.placeholder || ''}" val="${(el.value ?? '').toString().slice(0, 30)}" @${Math.round(r.x)},${Math.round(r.y)} ${Math.round(r.width)}x${Math.round(r.height)}`; }));
  const state = () => page.evaluate(() => { const w = window.app.graph._nodes[0].widgets.find(w => w.name === 'editor_state'); return JSON.stringify(w.value); });
  const canvasRect = async () => (await page.locator('[data-node-id] canvas').first().boundingBox());
  async function record(name) {
    console.log('step', name);
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(OUT, `${String(log.length).padStart(2, '0')}-${name}.png`) });
    log.push({ step: name, state: await state(), controls: await controls() });
  }
  async function drag(r, fx0, fy0, fx1, fy1, mods = []) {
    for (const m of mods) await page.keyboard.down(m);
    await page.mouse.move(r.x + r.width * fx0, r.y + r.height * fy0);
    await page.mouse.down();
    await page.mouse.move(r.x + r.width * (fx0 + fx1) / 2, r.y + r.height * (fy0 + fy1) / 2, { steps: 6 });
    await page.mouse.move(r.x + r.width * fx1, r.y + r.height * fy1, { steps: 6 });
    await page.mouse.up();
    for (const m of mods) await page.keyboard.up(m);
  }
  const clickText = async (text) => { const b = page.locator('[data-node-id] button', { hasText: text }).first(); const bb = await b.boundingBox(); console.log('click', text, JSON.stringify(bb)); await page.mouse.click(bb.x + bb.width / 2, bb.y + bb.height / 2); };

  let r = await canvasRect();
  log.push({ step: 'canvas', rect: r });
  await drag(r, 0.2, 0.2, 0.7, 0.6); await record('draw-box-1-grid-on');
  await clickText('text'); await record('type-text');
  await clickText('obj'); await record('type-obj-again');
  await page.locator('[data-node-id] textarea').first().fill('a red apple'); await page.mouse.click(1500, 1000); await record('description');
  await page.locator('[data-node-id] button', { hasText: '+' }).first().click(); await record('palette-plus');
  await page.keyboard.press('Escape'); await record('escape');
  await clickText('Grid'); await record('grid-toggled');
  r = await canvasRect();
  await drag(r, 0.1, 0.65, 0.33, 0.93); await record('draw-box-2-grid-off');
  await drag(r, 0.45, 0.4, 0.55, 0.5); await record('drag-inside-box-1');
  await drag(r, 0.7, 0.6, 0.8, 0.7); await record('drag-corner-box-1');
  await page.mouse.click(r.x + r.width * 0.2, r.y + r.height * 0.8); await record('click-box-2');
  await page.keyboard.press('Delete'); await record('delete-key');
  await page.mouse.click(r.x + r.width * 0.5, r.y + r.height * 0.5, { button: 'right' }); await record('right-click-box-1');
  await page.keyboard.press('Escape');
  await page.mouse.dblclick(r.x + r.width * 0.5, r.y + r.height * 0.5); await record('dblclick-box-1');
  await clickText('Clear all'); await record('clear-all');

  fs.writeFileSync(path.join(OUT, 'steps.json'), JSON.stringify(log, null, 1));
  for (const s of log) { console.log(`== ${s.step}: ${s.state ?? JSON.stringify(s.rect)}`); }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
