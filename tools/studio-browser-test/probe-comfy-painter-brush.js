// Black-box measurements of how ComfyUI's Painter brush paints (Node 2.0 view; see probe-comfy-painter.js for the widget's layout): for each
// case the canvas is cleared, the settings are changed through the widget's own controls, one action is made, and the painting the page
// uploads when the prompt is built is fetched and saved to out/painter-brush/<case>.png (read them with painter-brush-read.py). Nothing
// of ComfyUI's code is read. The canvas is 426 px on screen for 512 bitmap pixels, at (312, 126).
//   COMFY_URL (default http://127.0.0.1:8199/). ONLY=<substring> runs matching cases.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out', 'painter-brush');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.COMFY_URL || 'http://127.0.0.1:8199/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const X = 312, Y = 126, W = 426, K = W / 512;   // screen = X + bitmap * K

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.evaluate(async () => { await window.app.ui.settings.setSettingValueAsync('Comfy.VueNodes.Enabled', true); });
  await page.reload({ waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  const id = await page.evaluate(() => {
    const app = window.app;
    app.graph.clear();
    const LG = window.LiteGraph || (window.comfyAPI && window.comfyAPI.litegraph && window.comfyAPI.litegraph.LiteGraph);
    const n = LG.createNode('Painter'); n.pos = [300, 80]; app.graph.add(n);
    app.canvas.ds.scale = 1; app.canvas.ds.offset = [0, 0]; app.canvas.setDirty(true, true);
    return n.id;
  });
  await page.waitForTimeout(1500);

  const at = (bx, by) => [X + bx * K, Y + by * K];
  const dot = async (bx, by) => { const [x, y] = at(bx, by); await page.mouse.move(x, y); await page.mouse.down(); await page.mouse.up(); await page.waitForTimeout(250); };
  const line = async (ax, ay, bx, by, steps = 16) => {
    const [x0, y0] = at(ax, ay), [x1, y1] = at(bx, by);
    await page.mouse.move(x0, y0); await page.mouse.down(); await page.mouse.move(x1, y1, { steps }); await page.mouse.up(); await page.waitForTimeout(300);
  };
  // the sliders are role=slider thumbs: focused, then moved with the keyboard (Home = minimum, End = maximum, arrows = one step)
  const sliders = page.locator('[data-node-id] [role="slider"]');
  const slider = async (index, keys) => { await sliders.nth(index).focus(); for (const k of keys) await page.keyboard.press(k); await page.waitForTimeout(150); };
  const texts = () => page.evaluate(() => [...document.querySelectorAll('[data-node-id] span')].filter(e => e.children.length === 0).map(e => e.textContent.trim()).filter(Boolean).join(' | '));
  const sliderInfo = () => page.evaluate(() => [...document.querySelectorAll('[data-node-id] [role="slider"]')].map(e => `${e.getAttribute('aria-valuemin')}..${e.getAttribute('aria-valuemax')} now ${e.getAttribute('aria-valuenow')}`).join(' | '));
  const opacity = async v => { const f = page.locator('[data-node-id] input[type=number]').first(); await f.click(); await f.fill(String(v)); await f.press('Enter'); await page.waitForTimeout(150); };
  const colour = async hex => { await page.evaluate(h => { const c = document.querySelector('[data-node-id] input[type=color]'); c.value = h; c.dispatchEvent(new Event('input', { bubbles: true })); c.dispatchEvent(new Event('change', { bubbles: true })); }, hex); await page.waitForTimeout(150); };
  const tool = async name => { await page.locator(`[data-node-id] button:has-text("${name}")`).first().click(); await page.waitForTimeout(150); };
  const clear = async () => { await page.locator('[data-node-id] button:has-text("Clear")').first().click(); await page.waitForTimeout(300); };
  const save = async name => {
    const out = JSON.parse(await page.evaluate(async () => JSON.stringify((await window.app.graphToPrompt()).output)));
    const node = Object.values(out)[0];
    const file = node.inputs.mask;
    if (!file) { console.log(`${name}: nothing uploaded (mask "${file}") | inputs ${JSON.stringify(node.inputs)}`); return; }
    const b64 = await page.evaluate(async url => { const r = await fetch(url); const b = new Uint8Array(await r.arrayBuffer()); let s = ''; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000)); return btoa(s); },
      `/view?filename=${encodeURIComponent(file.replace(' [input]', ''))}&type=input`);
    fs.writeFileSync(path.join(OUT, name + '.png'), Buffer.from(b64, 'base64'));
    console.log(`${name}: saved | width ${node.inputs.width} height ${node.inputs.height} bg ${node.inputs.bg_color} | ${await texts()}`);
  };
  console.log('sliders (cursor size, hardness, width, height): ' + await sliderInfo());
  console.log('node properties: ' + await page.evaluate(id => JSON.stringify(window.app.graph.getNodeById(id).properties).slice(0, 400), id));

  // slider order as probe-comfy-painter.js listed them: 0 cursor size, 1 hardness, 2 width, 3 height
  const cases = {
    'dot-size20-hard100': async () => { await dot(256, 256); },
    'dot-size1': async () => { await slider(0, ['Home']); await dot(256, 256); },
    'dot-sizemax': async () => { await slider(0, ['End']); await dot(256, 256); },
    'dot-size50-hard100': async () => { await slider(0, ['Home']); for (let i = 0; i < 49; i++) await page.keyboard.press('ArrowRight'); await dot(256, 256); },
    'dot-size50-hard50': async () => { await slider(1, ['Home']); for (let i = 0; i < 50; i++) await page.keyboard.press('ArrowRight'); await dot(256, 256); },
    'dot-size50-hard0': async () => { await slider(1, ['Home']); await dot(256, 256); },
    'dot-size50-hard100-opacity50': async () => { await slider(1, ['End']); await opacity(50); await dot(256, 256); },
    'twodots-opacity50': async () => { await dot(246, 256); await dot(266, 256); },
    'line-opacity50': async () => { await line(100, 256, 400, 256); },
    'cross-opacity50': async () => { await line(100, 256, 400, 256); await line(256, 100, 256, 400); },
    'line-self-overlap-opacity50': async () => { await line(100, 256, 400, 256, 12); const [x, y] = at(100, 256); await page.mouse.move(...at(100, 256)); await page.mouse.down(); await page.mouse.move(...at(400, 256), { steps: 10 }); await page.mouse.move(...at(100, 256), { steps: 10 }); await page.mouse.up(); await page.waitForTimeout(300); },
    'line-fast-hard100': async () => { await opacity(100); await line(60, 200, 460, 300, 2); },
    'line-red': async () => { await colour('#ff0000'); await line(100, 150, 400, 150); },
    'erase-through-line': async () => { await line(100, 256, 400, 256); await tool('Eraser'); await dot(256, 256); await tool('Brush'); },
    'erase-soft': async () => { await colour('#ffffff'); await line(100, 256, 400, 256); await tool('Eraser'); await slider(1, ['Home']); await dot(256, 256); await tool('Brush'); await slider(1, ['End']); },
    'edge-dot': async () => { await dot(2, 256); },
  };
  for (const [name, action] of Object.entries(cases)) {
    if (process.env.ONLY && !name.includes(process.env.ONLY)) continue;
    await clear();
    await action();
    await save(name);
  }
  console.log('sliders at the end: ' + await sliderInfo());
  await page.screenshot({ path: path.join(OUT, 'last.png') });
  await browser.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
