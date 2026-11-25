// Black-box look at ComfyUI's compositor editor (Create Layered Image, the COMPOSITOR widget): builds Load Image -> Add Layer (x2) -> Create
// Layered Image on a running ComfyUI, runs it once (the editor needs the input pictures), opens the editor with the node's "Open Compositor"
// button and records what it shows: a screenshot, every control in the dialog (tag, text, title, aria-label, value, place), and the widget's
// value before and after. Nothing of ComfyUI's code is read. STEP (optional) names an extra action to try after opening (see below).
//   COMFY_URL (default http://127.0.0.1:8199/); the two pictures are uploaded to its input folder through /upload/image.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.COMFY_URL || 'http://127.0.0.1:8199/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PICTURES = [
  ['compositor-a.png', path.join(__dirname, '..', '..', 'src', 'SpawnGen.Studio', 'wwwroot', 'examples', 'example.png')],
  ['compositor-b.png', path.join(__dirname, 'apple.png')],
];

const controls = page => page.evaluate(() => {
  const out = [];
  const roots = [...document.querySelectorAll('[role="dialog"], .p-dialog, dialog')];
  const scope = roots.length ? roots : [document.body];
  for (const root of scope) for (const el of root.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent.trim()).join(' ').trim();
    const interesting = ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'CANVAS', 'LABEL', 'IMG'].includes(el.tagName) || el.getAttribute('role') || el.title || own;
    if (!interesting) continue;
    out.push({ tag: el.tagName, type: el.type || '', role: el.getAttribute('role') || '', text: (own || (el.tagName === 'BUTTON' ? (el.innerText || '').trim() : '')).slice(0, 70),
      title: el.title || '', aria: el.getAttribute('aria-label') || '', value: (el.value === undefined ? '' : String(el.value)).slice(0, 40),
      rect: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)] });
  }
  return { dialogs: roots.length, items: out };
});

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const logs = [];
  page.on('console', m => { if (m.type() === 'error') logs.push(m.text().slice(0, 200)); });
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  for (const [name, file] of PICTURES) {
    const b64 = fs.readFileSync(file).toString('base64');
    const status = await page.evaluate(async ({ name, b64 }) => {
      const bin = atob(b64); const data = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) data[i] = bin.charCodeAt(i);
      const form = new FormData(); form.append('image', new File([data], name, { type: 'image/png' })); form.append('overwrite', 'true');
      return (await fetch('/upload/image', { method: 'POST', body: form })).status;
    }, { name, b64 });
    console.log(`upload ${name}: ${status}`);
  }
  await page.evaluate(async () => { await window.app.ui.settings.setSettingValueAsync('Comfy.VueNodes.Enabled', true); });
  await page.reload({ waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });

  const built = await page.evaluate(() => {
    const app = window.app;
    app.graph.clear();
    const LG = window.LiteGraph || (window.comfyAPI && window.comfyAPI.litegraph && window.comfyAPI.litegraph.LiteGraph);
    const add = (type, x, y) => { const n = LG.createNode(type); n.pos = [x, y]; app.graph.add(n); return n; };
    const set = (n, name, value) => { const w = (n.widgets || []).find(w => w.name === name); if (w) w.value = value; return !!w; };
    const slot = (n, name, out) => (out ? n.outputs : n.inputs).findIndex(s => s.name === name);
    const la = add('LoadImage', 40, 80), lb = add('LoadImage', 40, 480);
    set(la, 'image', 'compositor-a.png'); set(lb, 'image', 'compositor-b.png');
    const a = add('AddLayer', 420, 80), b = add('AddLayer', 420, 480);
    set(a, 'name', 'coat'); set(b, 'name', 'apple'); set(b, 'x', 120); set(b, 'y', 60); set(b, 'opacity', 0.8);
    const c = add('ImageCompositor', 820, 120);
    la.connect(0, a, slot(a, 'image'));
    lb.connect(0, b, slot(b, 'image'));
    a.connect(0, b, slot(b, 'layers'));
    b.connect(0, c, slot(c, 'layers'));
    app.canvas.setDirty(true, true);
    return { addLayerWidgets: (a.widgets || []).map(w => `${w.name}=${JSON.stringify(w.value)}`), addLayerInputs: a.inputs.map(i => i.name), compositorId: c.id };
  });
  console.log('graph: ' + JSON.stringify(built));
  await page.waitForTimeout(1000);
  // run once
  const queued = await page.evaluate(async () => { try { await window.app.queuePrompt(0); return 'queued'; } catch (e) { return 'error ' + e.message; } });
  console.log('queue: ' + queued);
  for (let i = 0; i < 120; i++) {
    await page.waitForTimeout(1000);
    const q = await page.evaluate(async () => { const r = await (await fetch('/queue')).json(); return r.queue_running.length + r.queue_pending.length; });
    if (q === 0 && i > 2) break;
  }
  await page.waitForTimeout(2000);
  await page.screenshot({ path: path.join(OUT, 'comfy-compositor-1-after-run.png') });
  const valueBefore = await page.evaluate(id => { const n = window.app.graph.getNodeById(id); return (n.widgets || []).map(w => ({ name: w.name, value: JSON.stringify(w.value)?.slice(0, 1500) })); }, built.compositorId);
  console.log('value after the run: ' + JSON.stringify(valueBefore));
  // open the editor
  const button = page.locator('button:has-text("Open Compositor")').first();
  console.log('Open Compositor: enabled=' + await button.isEnabled().catch(() => '?') + ' title=' + await button.getAttribute('title').catch(() => '?'));
  await button.click({ timeout: 10000 }).catch(e => console.log('click failed: ' + e.message.split('\n')[0]));
  await page.waitForTimeout(2500);
  await page.screenshot({ path: path.join(OUT, 'comfy-compositor-2-editor.png') });
  const dom = await controls(page);
  fs.writeFileSync(path.join(OUT, 'comfy-compositor-editor.json'), JSON.stringify({ built, valueBefore, dom, logs }, null, 1));
  console.log(`editor: ${dom.dialogs} dialog(s), ${dom.items.length} controls; see out/comfy-compositor-2-editor.png and comfy-compositor-editor.json`);
  for (const it of dom.items.filter(i => i.text || i.title || i.aria).slice(0, 80))
    console.log(`  ${it.tag}${it.role ? '[' + it.role + ']' : ''} "${it.text}"${it.title ? ' title=' + it.title : ''}${it.aria ? ' aria=' + it.aria : ''}${it.value ? ' value=' + it.value : ''} @${it.rect.join(',')}`);
  if (logs.length) console.log('browser errors: ' + logs.slice(0, 5).join(' || '));
  await browser.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
