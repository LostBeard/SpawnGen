// Black-box: what ComfyUI's Painter widget SHOWS when an image is linked to its optional `image` input (display only: the outputs with a
// linked image are settled by probe-comfy-painter-run.js). Load Image (compositor-a.png, 512 x 512; then a 300 x 200 picture) -> Painter,
// a stroke painted, screenshots of the node before the run, after a run through the page's own queue, and the canvas bitmap's pixels
// read back (does the bitmap hold the picture, or is it drawn under the canvas?). Nothing of ComfyUI's code is read.
//   COMFY_URL (default http://127.0.0.1:8199/). Writes out/painter-linked/*.png and prints what it found.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out', 'painter-linked');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.COMFY_URL || 'http://127.0.0.1:8199/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.evaluate(async () => { await window.app.ui.settings.setSettingValueAsync('Comfy.VueNodes.Enabled', true); });
  await page.reload({ waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  const ids = await page.evaluate(() => {
    const app = window.app;
    app.graph.clear();
    const LG = window.LiteGraph || (window.comfyAPI && window.comfyAPI.litegraph && window.comfyAPI.litegraph.LiteGraph);
    const p = LG.createNode('Painter'); p.pos = [300, 80]; app.graph.add(p);
    const l = LG.createNode('LoadImage'); l.pos = [900, 80]; app.graph.add(l);
    const w = l.widgets.find(x => x.name === 'image'); w.value = 'compositor-a.png';
    l.connect(0, p, p.inputs.findIndex(i => i.name === 'image'));
    const s = LG.createNode('PreviewImage'); s.pos = [900, 600]; app.graph.add(s); p.connect(0, s, 0);
    app.canvas.ds.scale = 1; app.canvas.ds.offset = [0, 0]; app.canvas.setDirty(true, true);
    return { painter: p.id, load: l.id };
  });
  await page.waitForTimeout(2000);
  const canvasInfo = () => page.evaluate(id => {
    const node = document.querySelector(`[data-node-id="${id}"]`);
    const cs = node ? [...node.querySelectorAll('canvas')] : [];
    const imgs = node ? [...node.querySelectorAll('img')].map(i => `${i.naturalWidth}x${i.naturalHeight} ${i.src.slice(0, 120)}`) : [];
    const out = cs.map(c => {
      const b = c.getBoundingClientRect();
      let px = 'unreadable';
      try { const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let painted = 0, opaque = 0; for (let i = 3; i < d.length; i += 4) { if (d[i]) painted++; if (d[i] === 255) opaque++; } px = `${painted} px alpha>0, ${opaque} opaque of ${c.width * c.height}; pixel(10,10) ${[...d.slice((10 * c.width + 10) * 4, (10 * c.width + 10) * 4 + 4)]}`; } catch (e) { px = 'getImageData: ' + e.message; }
      const st = getComputedStyle(c);
      return `canvas ${c.width}x${c.height} shown ${b.width.toFixed(0)}x${b.height.toFixed(0)} at (${b.left.toFixed(0)},${b.top.toFixed(0)}) bg '${st.backgroundImage.slice(0, 80)}' bgcolor ${st.backgroundColor} | ${px}`;
    });
    // anything drawn under the canvas: its ancestors' / siblings' backgrounds
    const under = cs[0] ? [...cs[0].parentElement.children].filter(e => e !== cs[0]).map(e => `${e.tagName} ${getComputedStyle(e).backgroundImage.slice(0, 80)}`) : [];
    return { canvases: out, imgs, under, parentBg: cs[0] ? getComputedStyle(cs[0].parentElement).backgroundImage.slice(0, 80) + ' ' + getComputedStyle(cs[0].parentElement).backgroundColor : '' };
  }, ids.painter);
  const shot = async name => { const el = page.locator(`[data-node-id="${ids.painter}"]`); await el.screenshot({ path: path.join(OUT, name + '.png') }); console.log(`${name}: ${JSON.stringify(await canvasInfo())}`); };

  await shot('1-linked-before-stroke');
  // a stroke across the canvas (located live)
  const r = await page.evaluate(id => { const c = document.querySelector(`[data-node-id="${id}"] canvas`); const b = c.getBoundingClientRect(); return { x: b.left, y: b.top, k: b.width / c.width }; }, ids.painter);
  await page.mouse.move(r.x + 100 * r.k, r.y + 100 * r.k); await page.mouse.down(); await page.mouse.move(r.x + 400 * r.k, r.y + 300 * r.k, { steps: 12 }); await page.mouse.up(); await page.waitForTimeout(400);
  await shot('2-linked-after-stroke');
  // run through the page's own queue, wait for it to finish
  await page.evaluate(async () => { await window.app.queuePrompt(0, 1); });
  for (let i = 0; i < 120; i++) { const q = await page.evaluate(async () => (await (await fetch('/queue')).json())); if (!q.queue_running.length && !q.queue_pending.length) break; await page.waitForTimeout(500); }
  await page.waitForTimeout(1500);
  await shot('3-linked-after-run');
  // the linked picture changed to another size (300 x 200 if present) without a run, then with a run
  const other = await page.evaluate(async () => { const r = await fetch('/object_info/LoadImage'); const j = await r.json(); return j.LoadImage.input.required.image[0].filter(n => /\.(png|jpg|jpeg|webp)$/i.test(n)); });
  console.log('input pictures: ' + other.slice(0, 30).join(', '));
  // a picture that is not 512 x 512 (sizes read in the page)
  let pick = null;
  for (const n of other) {
    const s = await page.evaluate(async n => { const i = new Image(); i.src = `/api/view?type=input&filename=${encodeURIComponent(n)}`; await i.decode(); return [i.naturalWidth, i.naturalHeight]; }, n).catch(() => null);
    if (s && (s[0] !== 512 || s[1] !== 512) && s[0] >= 64 && s[1] >= 64 && s[0] !== s[1]) { pick = n; console.log(`other picture: ${n} ${s[0]}x${s[1]}`); break; }
  }
  if (pick) {
    // the widget's callback is what the combo calls on a user pick (a bare value change did not update the preview on the first run)
    await page.evaluate(([id, n]) => { const l = window.app.graph.getNodeById(id); const w = l.widgets.find(x => x.name === 'image'); w.value = n; if (w.callback) w.callback(n, window.app.canvas, l); window.app.graph.setDirtyCanvas(true, true); }, [ids.load, pick]);
    await page.waitForTimeout(800); await shot('4-other-size-before-run');
    await page.evaluate(async () => { await window.app.queuePrompt(0, 1); });
    for (let i = 0; i < 120; i++) { const q = await page.evaluate(async () => (await (await fetch('/queue')).json())); if (!q.queue_running.length && !q.queue_pending.length) break; await page.waitForTimeout(500); }
    await page.waitForTimeout(1500); await shot('5-other-size-after-run');
  } else console.log('no 300 x 200 picture in input/: other-size display not checked');
  await page.screenshot({ path: path.join(OUT, 'page.png') });
  await browser.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
