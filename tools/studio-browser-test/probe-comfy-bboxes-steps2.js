// Black-box, second pass on ComfyUI's Create Bounding Boxes editor: Grid off, moving a box, the palette swatch, Clear all, and the snap step
// at another canvas size. Every position is read again before each action (the graph view pans when a text field is edited).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out', 'bboxes2');
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
  const width = Number(process.env.BOX_WIDTH || 1024), height = Number(process.env.BOX_HEIGHT || 1024);
  await page.evaluate(([w, h]) => {
    const app = window.app; app.graph.clear();
    const LG = window.LiteGraph || window.comfyAPI.litegraph.LiteGraph;
    const node = LG.createNode('CreateBoundingBoxes'); node.pos = [100, 50]; app.graph.add(node);
    node.widgets.find(x => x.name === 'width').value = w; node.widgets.find(x => x.name === 'height').value = h;
    app.canvas.ds.scale = 0.75; app.canvas.ds.offset = [0, 0]; app.canvas.setDirty(true, true);
  }, [width, height]);
  await page.waitForTimeout(1500);

  const log = [];
  const state = () => page.evaluate(() => JSON.stringify(window.app.graph._nodes[0].widgets.find(w => w.name === 'editor_state').value));
  const rect = async () => page.locator('[data-node-id] canvas').first().boundingBox();
  const button = async (text) => page.locator('[data-node-id] button', { hasText: text }).first().boundingBox();
  async function record(name, extra = {}) {
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(OUT, `${String(log.length).padStart(2, '0')}-${name}.png`) });
    const s = await state();
    log.push({ step: name, state: s, ...extra });
    console.log(`== ${name}: ${s}${Object.keys(extra).length ? ' ' + JSON.stringify(extra) : ''}`);
  }
  async function drag(fx0, fy0, fx1, fy1) {
    const r = await rect();
    await page.mouse.move(r.x + r.width * fx0, r.y + r.height * fy0);
    await page.mouse.down();
    await page.mouse.move(r.x + r.width * (fx0 + fx1) / 2, r.y + r.height * (fy0 + fy1) / 2, { steps: 6 });
    await page.mouse.move(r.x + r.width * fx1, r.y + r.height * fy1, { steps: 6 });
    await page.mouse.up();
  }
  async function clickAt(fx, fy) { const r = await rect(); await page.mouse.click(r.x + r.width * fx, r.y + r.height * fy); }
  async function clickButton(text) { const b = await button(text); await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2); }

  console.log('canvas', JSON.stringify(await rect()), 'size', width, height);
  await drag(0.2, 0.2, 0.7, 0.6); await record('draw-A-grid-on');
  await drag(0.13, 0.71, 0.37, 0.93); await record('draw-B-grid-on');
  await drag(0.25, 0.8, 0.3, 0.85); await record('drag-inside-B-selected-move');
  await clickAt(0.45, 0.3); await record('click-A');
  await drag(0.45, 0.3, 0.52, 0.36); await record('drag-inside-A-move');
  await clickButton('Grid'); await record('grid-off');
  await drag(0.81, 0.11, 0.93, 0.27); await record('draw-C-grid-off');
  await drag(0.85, 0.15, 0.88, 0.17); await record('drag-inside-C-grid-off');
  await clickButton('+'); await record('palette-plus');
  const sw = await page.locator('[data-node-id] button:not(:has-text("+"))').evaluateAll(els => els.map(el => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height, t: (el.innerText || '').trim() }; }).filter(e => e.w > 10 && e.w < 20 && e.h < 20 && !e.t));
  if (sw.length) {
    const s = sw[sw.length - 1];
    await page.mouse.click(s.x + s.w / 2, s.y + s.h / 2);
    const popup = await page.evaluate(() => [...document.querySelectorAll('input, button, [role="dialog"], .p-colorpicker, .p-popover')].filter(el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; })
      .map(el => `${el.tagName}:${el.type || ''} cls=${(el.className || '').toString().slice(0, 40)} val=${(el.value ?? '').toString().slice(0, 20)} aria=${el.getAttribute('aria-label') || ''}`).slice(-25));
    await record('palette-swatch-click', { popup });
    await page.keyboard.press('Escape');
    await page.mouse.click(s.x + s.w / 2, s.y + s.h / 2, { button: 'right' }); await record('palette-swatch-right-click');
  }
  await clickButton('Clear all'); await record('clear-all');
  fs.writeFileSync(path.join(OUT, `steps-${width}x${height}.json`), JSON.stringify(log, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
