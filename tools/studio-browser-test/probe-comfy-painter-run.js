// Black-box: what ComfyUI's Painter node outputs. Paints in the widget (as probe-comfy-painter.js), lets the page build the prompt (which
// uploads the painting), then queues variants of that prompt through /prompt with Save Image on IMAGE and on MASK (through Convert Mask to
// Image) and downloads everything to out/painter-fixtures/: the uploaded painting and each variant's two outputs. Nothing of ComfyUI's
// code is read.  Variants: the painting as painted; bg_color #336699; width 256 height 384 (not the painting's size); no painting ("");
// with a base image linked (compositor-a.png, 512 x 512), same and other size.
//   COMFY_URL (default http://127.0.0.1:8199/). STROKES: "white" (default) or "mixed" (a second, half-transparent red soft stroke and an erased gap).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out', 'painter-fixtures');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.COMFY_URL || 'http://127.0.0.1:8199/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const MIXED = process.env.STROKES === 'mixed';
const TAG = MIXED ? 'mixed' : 'white';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.evaluate(async () => { await window.app.ui.settings.setSettingValueAsync('Comfy.VueNodes.Enabled', true); });
  await page.reload({ waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.evaluate(() => {
    const app = window.app;
    app.graph.clear();
    const LG = window.LiteGraph || (window.comfyAPI && window.comfyAPI.litegraph && window.comfyAPI.litegraph.LiteGraph);
    const n = LG.createNode('Painter'); n.pos = [300, 80]; app.graph.add(n);
    app.canvas.ds.scale = 1; app.canvas.ds.offset = [0, 0]; app.canvas.setDirty(true, true);
  });
  await page.waitForTimeout(1500);
  // the canvas: 426 px shown for 512 (probe-comfy-painter.js): at 312,126
  const X = 312, Y = 126, W = 426;
  const stroke = async pts => {
    await page.mouse.move(X + pts[0][0] * W, Y + pts[0][1] * W); await page.mouse.down();
    for (const p of pts.slice(1)) await page.mouse.move(X + p[0] * W, Y + p[1] * W, { steps: 8 });
    await page.mouse.up(); await page.waitForTimeout(300);
  };
  await stroke([[0.25, 0.25], [0.5, 0.5], [0.75, 0.3]]);
  if (MIXED) {
    // colour #ff0000 at 50 %, hardness lowered by dragging its slider left, a second stroke; then the eraser across the first stroke
    await page.evaluate(() => { const c = document.querySelector('[data-node-id] input[type=color]'); c.value = '#ff0000'; c.dispatchEvent(new Event('input', { bubbles: true })); c.dispatchEvent(new Event('change', { bubbles: true })); });
    const pct = page.locator('[data-node-id] input[type=number]').first();
    await pct.click(); await pct.fill('50'); await pct.press('Enter'); await page.waitForTimeout(200);
    await page.mouse.move(683, 680); await page.mouse.down(); await page.mouse.move(560, 680, { steps: 6 }); await page.mouse.up(); await page.waitForTimeout(200);
    await stroke([[0.2, 0.8], [0.8, 0.8]]);
    await page.mouse.click(659, 572); await page.waitForTimeout(200); // Eraser
    await stroke([[0.375, 0.2], [0.375, 0.6]]);
  }
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, `${TAG}-widget.png`) });
  const labels = await page.evaluate(() => [...document.querySelectorAll('[data-node-id] span')].filter(e => e.children.length === 0).map(e => e.textContent.trim()).filter(Boolean).join(' | '));
  console.log('widget texts: ' + labels);
  const base = JSON.parse(await page.evaluate(async () => JSON.stringify((await window.app.graphToPrompt()).output)));
  const painted = base['1'].inputs.mask;
  console.log('painted file: ' + painted);

  const fetchTo = async (url, file) => {
    const b64 = await page.evaluate(async url => { const r = await fetch(url); if (!r.ok) return null; const b = new Uint8Array(await r.arrayBuffer()); let s = ''; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000)); return btoa(s); }, url);
    if (b64 === null) { console.log('  could not fetch ' + url); return; }
    fs.writeFileSync(path.join(OUT, file), Buffer.from(b64, 'base64'));
  };
  await fetchTo(`/view?filename=${encodeURIComponent(painted.replace(' [input]', ''))}&type=input`, `${TAG}-uploaded.png`);

  const variants = {
    asis: {},
    bg: { bg_color: '#336699' },
    size: { width: 256, height: 384 },
    ...(MIXED ? {} : { empty: { mask: '' }, emptybg: { mask: '', bg_color: '#336699', width: 128, height: 64 } }),
    image: { image: ['9', 0] },
    imagesize: { image: ['9', 0], width: 256, height: 384 },
  };
  for (const [name, change] of Object.entries(variants)) {
    const prompt = {
      '1': { class_type: 'Painter', inputs: { ...base['1'].inputs, ...change } },
      '2': { class_type: 'SaveImage', inputs: { images: ['1', 0], filename_prefix: `painter-${TAG}-${name}-image` } },
      '3': { class_type: 'MaskToImage', inputs: { mask: ['1', 1] } },
      '4': { class_type: 'SaveImage', inputs: { images: ['3', 0], filename_prefix: `painter-${TAG}-${name}-mask` } },
    };
    if (change.image) prompt['9'] = { class_type: 'LoadImage', inputs: { image: 'compositor-a.png' } };
    const result = await page.evaluate(async prompt => {
      const r = await fetch('/prompt', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt }) });
      const j = await r.json();
      if (!j.prompt_id) return { error: JSON.stringify(j).slice(0, 600) };
      for (let i = 0; i < 240; i++) {
        await new Promise(r => setTimeout(r, 500));
        const h = await (await fetch('/history/' + j.prompt_id)).json();
        if (h[j.prompt_id]) return { status: h[j.prompt_id].status, outputs: h[j.prompt_id].outputs };
      }
      return { error: 'timeout' };
    }, prompt);
    if (result.error) { console.log(`${name}: ERROR ${result.error}`); continue; }
    const msg = (result.status.messages || []).filter(m => m[0] === 'execution_error').map(m => m[1].exception_message).join(' ');
    console.log(`${name}: ${result.status.status_str} ${msg.slice(0, 300)}`);
    for (const [node, kind] of [['2', 'image'], ['4', 'mask']]) {
      const f = result.outputs && result.outputs[node] && result.outputs[node].images && result.outputs[node].images[0];
      if (f) await fetchTo(`/view?filename=${encodeURIComponent(f.filename)}&subfolder=${encodeURIComponent(f.subfolder)}&type=${f.type}`, `${TAG}-${name}-${kind}.png`);
    }
  }
  fs.writeFileSync(path.join(OUT, `${TAG}-prompt.json`), JSON.stringify(base, null, 1));
  await browser.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
