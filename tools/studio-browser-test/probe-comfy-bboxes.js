// Black-box look at ComfyUI's frontend: adds a Create Bounding Boxes node to an empty graph on a running ComfyUI and records what its canvas
// widget shows (screenshot, the DOM controls inside the node's widget area, the widget values). Nothing of ComfyUI's code is read.
//   COMFY_URL (default http://127.0.0.1:8199/)
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.COMFY_URL || 'http://127.0.0.1:8199/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const NODE = process.env.NODE_TYPE || 'CreateBoundingBoxes';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const logs = [];
  page.on('console', m => logs.push(`${m.type()}: ${m.text()}`));
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  // The canvas editor needs ComfyUI's "Node 2.0" view (the classic view says "Bounding Boxes: Node 2.0 only").
  await page.evaluate(async () => { await window.app.ui.settings.setSettingValueAsync('Comfy.VueNodes.Enabled', true); });
  await page.reload({ waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  const info = await page.evaluate((type) => {
    const app = window.app;
    app.graph.clear();
    const LG = window.LiteGraph || (window.comfyAPI && window.comfyAPI.litegraph && window.comfyAPI.litegraph.LiteGraph);
    const node = LG.createNode(type);
    node.pos = [100, 100];
    app.graph.add(node);
    app.canvas.setDirty(true, true);
    return {
      size: node.size,
      widgets: (node.widgets || []).map(w => ({ name: w.name, type: w.type, value: JSON.stringify(w.value)?.slice(0, 400), options: JSON.stringify(w.options)?.slice(0, 400) })),
    };
  }, NODE);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(OUT, `comfy-${NODE}.png`) });
  // DOM controls the frontend placed over the canvas for this node (DOM widgets): tag, type, text, title, aria-label.
  const dom = await page.evaluate(() => {
    const out = [];
    for (const el of document.querySelectorAll('[data-node-id] *')) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const t = (el.innerText || '').trim().slice(0, 60);
      if (['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'OPTION', 'CANVAS', 'LABEL'].includes(el.tagName) || el.getAttribute('role') || el.title)
        out.push({ tag: el.tagName, type: el.type || '', cls: (el.className && el.className.baseVal === undefined ? el.className : '').toString().slice(0, 80),
          text: t, title: el.title || '', aria: el.getAttribute('aria-label') || '', placeholder: el.placeholder || '', value: (el.value || '').toString().slice(0, 60),
          rect: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)] });
    }
    return out;
  });
  // Draw one box by dragging on the widget's canvas, then record what the frontend shows and stores.
  const steps = [];
  const canvasBox = dom.find(e => e.tag === 'CANVAS');
  if (canvasBox) {
    const [cx, cy, cw, chh] = canvasBox.rect;
    await page.mouse.move(cx + cw * 0.2, cy + chh * 0.2);
    await page.mouse.down();
    await page.mouse.move(cx + cw * 0.5, cy + chh * 0.45, { steps: 8 });
    await page.mouse.move(cx + cw * 0.7, cy + chh * 0.6, { steps: 8 });
    await page.mouse.up();
    await page.waitForTimeout(800);
    await page.screenshot({ path: path.join(OUT, `comfy-${NODE}-drawn.png`) });
    const state = await page.evaluate(() => {
      const n = window.app.graph._nodes[0];
      return (n.widgets || []).map(w => ({ name: w.name, value: JSON.stringify(w.value)?.slice(0, 800) }));
    });
    const dom2 = await page.evaluate(() => [...document.querySelectorAll('body *')].filter(el => {
      const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'LABEL'].includes(el.tagName);
    }).map(el => { const r = el.getBoundingClientRect(); return { tag: el.tagName, type: el.type || '', text: (el.innerText || '').trim().slice(0, 50),
      aria: el.getAttribute('aria-label') || '', placeholder: el.placeholder || '', value: (el.value || '').toString().slice(0, 50),
      rect: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)] }; }).filter(e => e.rect[0] > 400 && e.rect[0] < 1300));
    steps.push({ after: 'drag', state, dom: dom2 });
  }
  fs.writeFileSync(path.join(OUT, `comfy-${NODE}.json`), JSON.stringify({ info, dom, steps, logs: logs.slice(-30) }, null, 1));
  console.log(JSON.stringify(info, null, 1).slice(0, 3000));
  console.log(`${dom.length} DOM controls; see out/comfy-${NODE}.json and .png`);
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
