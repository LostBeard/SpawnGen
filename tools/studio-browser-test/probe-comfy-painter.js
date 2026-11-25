// Black-box look at ComfyUI's Painter node (its PAINTER widget) in the Node 2.0 view: adds the node, records every control it shows (tag,
// text, title, aria-label, value, place) and the widgets' values, paints one stroke on its canvas and records the values again, with
// screenshots and the requests the page made meanwhile (uploads). Nothing of ComfyUI's code is read.
//   COMFY_URL (default http://127.0.0.1:8199/). Output: out/comfy-painter-1-node.png, -2-stroke.png, comfy-painter.json.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.COMFY_URL || 'http://127.0.0.1:8199/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const requests = [];
  page.on('request', r => { if (r.method() !== 'GET') requests.push(`${r.method()} ${r.url().replace(URL, '/')} ${(r.postData() || '').slice(0, 300).replace(/[^\x20-\x7e]/g, '.')}`); });
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
  const values = () => page.evaluate(id => { const n = window.app.graph.getNodeById(id); return { size: n.size, widgets: (n.widgets || []).map(w => ({ name: w.name, type: w.type, value: JSON.stringify(w.value)?.slice(0, 300) })) }; }, id);
  const controls = () => page.evaluate(() => {
    const out = [];
    const root = document.querySelector('[data-node-id]') || document.body;
    for (const el of root.querySelectorAll('*')) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent.trim()).join(' ').trim();
      const interesting = ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'CANVAS', 'LABEL', 'IMG'].includes(el.tagName) || el.getAttribute('role') || el.title || own;
      if (!interesting) continue;
      out.push({ tag: el.tagName, type: el.type || '', role: el.getAttribute('role') || '', text: (own || (el.tagName === 'BUTTON' ? (el.innerText || '').trim() : '')).slice(0, 60),
        title: el.title || '', aria: el.getAttribute('aria-label') || '', value: (el.value === undefined ? '' : String(el.value)).slice(0, 40),
        size: el.tagName === 'CANVAS' ? `${el.width}x${el.height}` : '', rect: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)] });
    }
    return out;
  });
  const print = items => { for (const it of items) console.log(`  ${it.tag}${it.type ? ':' + it.type : ''}${it.role ? '[' + it.role + ']' : ''} "${it.text}"${it.title ? ' title=' + it.title : ''}${it.aria ? ' aria=' + it.aria : ''}${it.value ? ' value=' + it.value : ''}${it.size ? ' bitmap=' + it.size : ''} @${it.rect.join(',')}`); };

  await page.screenshot({ path: path.join(OUT, 'comfy-painter-1-node.png') });
  const before = await values();
  const dom = await controls();
  console.log('widgets: ' + JSON.stringify(before));
  console.log(`controls (${dom.length}):`); print(dom);

  // one stroke across the largest canvas inside the node
  const canvases = dom.filter(d => d.tag === 'CANVAS').sort((a, b) => b.rect[2] * b.rect[3] - a.rect[2] * a.rect[3]);
  let after = null;
  if (canvases.length) {
    const [x, y, w, h] = canvases[0].rect;
    console.log(`stroke on the canvas at ${x},${y} ${w}x${h}`);
    await page.mouse.move(x + w * 0.25, y + h * 0.25); await page.mouse.down();
    await page.mouse.move(x + w * 0.5, y + h * 0.5, { steps: 8 }); await page.mouse.move(x + w * 0.75, y + h * 0.3, { steps: 8 });
    await page.mouse.up();
    await page.waitForTimeout(2500);
    await page.screenshot({ path: path.join(OUT, 'comfy-painter-2-stroke.png') });
    after = await values();
    console.log('widgets after the stroke: ' + JSON.stringify(after));
  } else console.log('no canvas found inside the node');
  console.log('requests: ' + (requests.length ? '\n  ' + requests.join('\n  ') : 'none'));
  // what a queued prompt carries for the node
  const prompt = await page.evaluate(async () => { try { const p = await window.app.graphToPrompt(); return JSON.stringify(p.output); } catch (e) { return 'error ' + e.message; } });
  console.log('prompt: ' + prompt.slice(0, 800));
  console.log('requests after graphToPrompt: ' + requests.slice(-4).join(' || ').slice(0, 900));
  fs.writeFileSync(path.join(OUT, 'comfy-painter.json'), JSON.stringify({ before, dom, after, requests, prompt }, null, 1));
  await browser.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
