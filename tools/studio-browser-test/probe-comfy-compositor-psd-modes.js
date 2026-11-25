// Black-box: one PSD from ComfyUI's compositor editor that answers the questions a PSD writer has: a 64 x 64 layer per blend mode (the 26 of
// Add Layer's blend_mode list, named by the mode), one at a negative place, one with a Unicode name, the top one hidden in the editor, and
// the Background made visible. Saved to out/comfy-compositor-export-modes.psd; read it with psd-dump.py. Nothing of ComfyUI's code is read.
//   COMFY_URL (default http://127.0.0.1:8199/).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
const URL = process.env.COMFY_URL || 'http://127.0.0.1:8199/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.evaluate(async () => { await window.app.ui.settings.setSettingValueAsync('Comfy.VueNodes.Enabled', true); });
  await page.reload({ waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  const built = await page.evaluate(async () => {
    const info = await (await fetch('/object_info/AddLayer')).json();
    const spec = info.AddLayer.input.optional.blend_mode;
    const modes = Array.isArray(spec[0]) ? spec[0] : (spec[1] && spec[1].options) || [];
    const app = window.app;
    app.graph.clear();
    const LG = window.LiteGraph || (window.comfyAPI && window.comfyAPI.litegraph && window.comfyAPI.litegraph.LiteGraph);
    const add = (type, x, y) => { const n = LG.createNode(type); n.pos = [x, y]; app.graph.add(n); return n; };
    const set = (n, name, value) => { const w = (n.widgets || []).find(w => w.name === name); if (w) w.value = value; return !!w; };
    const slot = (n, name) => n.inputs.findIndex(s => s.name === name);
    const load = add('LoadImage', 40, 80); set(load, 'image', 'compositor-b.png');
    const scale = add('ImageScale', 40, 480);
    set(scale, 'upscale_method', 'bilinear'); set(scale, 'width', 64); set(scale, 'height', 64); set(scale, 'crop', 'disabled');
    load.connect(0, scale, slot(scale, 'image'));
    const layers = modes.map((m, i) => ({ name: m, x: 8 + (i % 6) * 40, y: 8 + Math.floor(i / 6) * 40, blend: m }));
    layers.push({ name: 'negative', x: -20, y: -10, blend: 'normal' });
    layers.push({ name: 'Ünïcødé 層 ✓', x: 200, y: 180, blend: 'normal' });
    layers.push({ name: 'hidden-top', x: 120, y: 200, blend: 'normal' });
    let prev = null;
    layers.forEach((l, i) => {
      const n = add('AddLayer', 420 + (i % 8) * 60, 80 + Math.floor(i / 8) * 60);
      set(n, 'name', l.name); set(n, 'x', l.x); set(n, 'y', l.y); set(n, 'blend_mode', l.blend); set(n, 'opacity', 0.5 + (i % 5) * 0.1);
      scale.connect(0, n, slot(n, 'image'));
      if (prev) prev.connect(0, n, slot(n, 'layers'));
      prev = n;
    });
    const c = add('ImageCompositor', 1100, 120);
    prev.connect(0, c, slot(c, 'layers'));
    app.canvas.ds.scale = 1; app.canvas.ds.offset = [0, 0]; app.canvas.setDirty(true, true);
    return { modes, id: c.id, count: layers.length };
  });
  console.log(`modes (${built.modes.length}): ${built.modes.join(', ')}; layers ${built.count}`);
  await page.waitForTimeout(1000);
  await page.evaluate(() => window.app.queuePrompt(0));
  for (let i = 0; i < 240; i++) {
    await page.waitForTimeout(1000);
    const q = await page.evaluate(async () => { const r = await (await fetch('/queue')).json(); return r.queue_running.length + r.queue_pending.length; });
    if (q === 0 && i > 2) break;
  }
  await page.waitForTimeout(2500);
  const box = await page.locator('button:has-text("Open Compositor")').first().boundingBox();
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await page.waitForTimeout(2500);
  // hide the top row (its visibility toggle), show the Background
  const eyes = page.locator('[role="dialog"] button[aria-label="Toggle visibility"]');
  const first = await eyes.first().boundingBox();
  if (first) await page.mouse.click(first.x + first.width / 2, first.y + first.height / 2);
  await page.waitForTimeout(300);
  const last = await eyes.last().boundingBox();
  if (last) await page.mouse.click(last.x + last.width / 2, last.y + last.height / 2);
  await page.waitForTimeout(500);
  const value = await page.evaluate(id => { const n = window.app.graph.getNodeById(id); const w = (n.widgets || []).find(w => w.name === 'compositor'); return w && w.value; }, built.id);
  console.log('canvas ' + JSON.stringify(value && value.canvas) + ' background ' + JSON.stringify(value && value.background)
    + ' hidden: ' + ((value && value.layers) || []).filter(l => l.visible === false).map(l => l.name).join(', '));
  await page.screenshot({ path: path.join(OUT, 'comfy-compositor-psd-modes.png') });
  const [download] = await Promise.all([page.waitForEvent('download', { timeout: 30000 }), page.mouse.click(898, 95)]);
  const file = path.join(OUT, 'comfy-compositor-export-modes.psd');
  await download.saveAs(file);
  fs.writeFileSync(path.join(OUT, 'comfy-compositor-export-modes.json'), JSON.stringify({ built, value }, null, 1));
  console.log(`saved ${file} (${fs.statSync(file).size} bytes)`);
  await browser.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
