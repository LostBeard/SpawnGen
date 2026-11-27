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

  // 2026-11-27: the canvas is located on every stroke (focusing a slider can scroll the page, which moved the canvas off X, Y and made the
  // erase-soft case time out on 2026-11-26); a canvas that is not where the old constants say is reported, so earlier fixtures stay checkable
  let R = { x: X, y: Y, k: K };
  const locate = async () => {
    const r = await page.evaluate(() => { const c = document.querySelector('[data-node-id] canvas'); if (!c) return null; const b = c.getBoundingClientRect(); return { x: b.left, y: b.top, w: b.width, bw: c.width }; });
    if (!r) throw new Error('no painter canvas');
    R = { x: r.x, y: r.y, k: r.w / r.bw };
    if (Math.abs(r.x - X) > 0.5 || Math.abs(r.y - Y) > 0.5 || Math.abs(R.k - K) > 1e-3) console.log(`  canvas at (${r.x.toFixed(1)}, ${r.y.toFixed(1)}) scale ${R.k.toFixed(4)}, not (${X}, ${Y}) ${K.toFixed(4)}`);
  };
  const at = (bx, by) => [R.x + bx * R.k, R.y + by * R.k];
  const dot = async (bx, by) => { await locate(); const [x, y] = at(bx, by); await page.mouse.move(x, y); await page.mouse.down(); await page.mouse.up(); await page.waitForTimeout(250); };
  const line = async (ax, ay, bx, by, steps = 16) => {
    await locate();
    const [x0, y0] = at(ax, ay), [x1, y1] = at(bx, by);
    await page.mouse.move(x0, y0); await page.mouse.down(); await page.mouse.move(x1, y1, { steps }); await page.mouse.up(); await page.waitForTimeout(300);
  };
  // the sliders are role=slider thumbs: focused, then moved with the keyboard (Home = minimum, End = maximum, arrows = one step)
  const sliders = page.locator('[data-node-id] [role="slider"]');
  const slider = async (index, keys) => { await sliders.nth(index).focus(); for (const k of keys) await page.keyboard.press(k); await page.waitForTimeout(150); };
  // set slider `index` (minimum `min`, step `step`) to `value`: Home, then one ArrowRight per step
  const set = async (index, min, value, step = 1) => { await sliders.nth(index).focus(); await page.keyboard.press('Home'); for (let i = 0; i < (value - min) / step; i++) await page.keyboard.press('ArrowRight'); await page.waitForTimeout(150); };
  const texts = () => page.evaluate(() => [...document.querySelectorAll('[data-node-id] span')].filter(e => e.children.length === 0).map(e => e.textContent.trim()).filter(Boolean).join(' | '));
  const sliderInfo = () => page.evaluate(() => [...document.querySelectorAll('[data-node-id] [role="slider"]')].map(e => `${e.getAttribute('aria-valuemin')}..${e.getAttribute('aria-valuemax')} now ${e.getAttribute('aria-valuenow')}`).join(' | '));
  const opacity = async v => { const f = page.locator('[data-node-id] input[type=number]').first(); await f.click(); await f.fill(String(v)); await f.press('Enter'); await page.waitForTimeout(150); };
  const colour = async hex => { await page.evaluate(h => { const c = document.querySelector('[data-node-id] input[type=color]'); c.value = h; c.dispatchEvent(new Event('input', { bubbles: true })); c.dispatchEvent(new Event('change', { bubbles: true })); }, hex); await page.waitForTimeout(150); };
  // buttons are clicked through the DOM: a scrolled page leaves them "outside of the viewport" for Playwright's pointer click
  const button = async name => { const ok = await page.evaluate(n => { const b = [...document.querySelectorAll('[data-node-id] button')].find(e => e.textContent.trim().includes(n)); if (b) b.click(); return !!b; }, name); if (!ok) throw new Error(`no ${name} button`); };
  const tool = async name => { await button(name); await page.waitForTimeout(150); };
  const clear = async () => { await button('Clear'); await page.waitForTimeout(300); };
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
    // (the old 'erase-soft' case focused slider 1 in Eraser mode, which is Width there, not Hardness: replaced by eraser-controls / erase-*-b)
    'edge-dot': async () => { await dot(2, 256); },
    'eraser-controls': async () => {
      console.log('  brush mode : sliders ' + await sliderInfo() + ' | ' + await texts());
      await tool('Eraser');
      console.log('  eraser mode: sliders ' + await sliderInfo() + ' | ' + await texts() + ' | number fields ' + await page.evaluate(() => document.querySelectorAll('[data-node-id] input[type=number]').length) + ' | colour inputs ' + await page.evaluate(() => document.querySelectorAll('[data-node-id] input[type=color]').length));
      await tool('Brush'); },
    // 2026-11-27: the cases below set every setting they rely on (set = Home then n steps right), so they do not depend on the cases above
    ...Object.fromEntries([0, 10, 25, 50, 75, 90, 100].map(h => [`dot-size50-hard${String(h).padStart(3, '0')}-b`, async () => {
      await tool('Brush'); await opacity(100); await set(0, 1, 50); await set(1, 0, h); await dot(256, 256); }])),
    ...Object.fromEntries([0, 50].map(h => [`dot-size100-hard${String(h).padStart(3, '0')}-b`, async () => {
      await tool('Brush'); await opacity(100); await set(0, 1, 100); await set(1, 0, h); await dot(256, 256); }])),
    // every setting is changed in Brush mode (where slider 1 is Hardness), then the Eraser is used: does it take the brush's opacity / hardness?
    'erase-opacity50': async () => {
      await tool('Brush'); await opacity(100); await set(0, 1, 50); await set(1, 0, 100); await line(100, 256, 400, 256);
      await opacity(50); await tool('Eraser'); await dot(256, 256); await tool('Brush'); await opacity(100); },
    'erase-soft-b': async () => {
      await tool('Brush'); await opacity(100); await set(0, 1, 50); await set(1, 0, 100); await line(100, 256, 400, 256);
      await set(1, 0, 0); await tool('Eraser'); await dot(256, 256); await tool('Brush'); await set(1, 0, 100); },
    'edge-dot-b': async () => { await tool('Brush'); await opacity(100); await set(0, 1, 50); await set(1, 0, 100); await dot(2, 256); },
    // Width / Height changed AFTER painting: does the painting stay, scale, crop, or clear? (64 per step: 512 -> 256 is 4 steps left)
    'size-change-after-paint': async () => {
      await tool('Brush'); await opacity(100); await set(0, 1, 50); await set(1, 0, 100); await dot(128, 128); await dot(384, 384);
      await slider(2, ['ArrowLeft', 'ArrowLeft', 'ArrowLeft', 'ArrowLeft']); await slider(3, ['ArrowRight', 'ArrowRight']); await page.waitForTimeout(500);
      console.log('  after the size change: ' + await sliderInfo() + ' | canvas ' + await page.evaluate(() => { const c = document.querySelector('[data-node-id] canvas'); return c ? `${c.width}x${c.height} shown ${c.getBoundingClientRect().width.toFixed(0)}x${c.getBoundingClientRect().height.toFixed(0)}` : 'none'; })); },
    'paint-after-size-change': async () => { await dot(128, 128); },
    // soft LINES: does a soft stroke build up along its path (stamps overlapping), and does the pointer speed (stamp spacing) change it?
    ...Object.fromEntries([['line-soft-hard000', 0, 100, 16], ['line-soft-hard050', 50, 100, 16], ['line-soft-hard000-opacity50', 0, 50, 16],
      ['line-soft-hard000-slow', 0, 100, 96], ['line-soft-hard000-fast', 0, 100, 2]].map(([name, h, o, steps]) => [name, async () => {
      await tool('Brush'); await opacity(o); await set(0, 1, 50); await set(1, 0, h); await line(100, 256, 400, 256, steps); await opacity(100); await set(1, 0, 100); }])),
    // exact pointer positions (synthetic PointerEvents at fractional client coordinates on the canvas): a soft 50 brush along y 256 from x 100
    // to 400 in N equal segments, to separate the stamp spacing from the per-event stamps
    ...Object.fromEntries([1, 2, 30, 300].map(n => [`synth-soft-seg${String(n).padStart(3, '0')}`, async () => {
      await tool('Brush'); await opacity(100); await set(0, 1, 50); await set(1, 0, 0); await locate();
      const info = await page.evaluate(([R, n]) => {
        const c = document.querySelector('[data-node-id] canvas'); const seen = [];
        for (const t of ['pointerdown', 'pointermove', 'pointerup']) c.addEventListener(t, () => seen.push(t[7]), { once: false });
        const ev = (type, bx, by, buttons) => new PointerEvent(type, { bubbles: true, cancelable: true, composed: true, pointerId: 1, pointerType: 'mouse', isPrimary: true,
          button: 0, buttons, clientX: R.x + bx * R.k, clientY: R.y + by * R.k });
        c.dispatchEvent(ev('pointerdown', 100, 256, 1));
        for (let i = 1; i <= n; i++) c.dispatchEvent(ev('pointermove', 100 + 300 * i / n, 256, 1));
        c.dispatchEvent(ev('pointerup', 400, 256, 0)); document.dispatchEvent(ev('pointerup', 400, 256, 0)); window.dispatchEvent(ev('pointerup', 400, 256, 0));
        return seen.length;
      }, [R, n]);
      await page.waitForTimeout(300); await set(1, 0, 100); }])),
    // shrink then grow back without clearing: is the cropped part lost (the canvas keeps only what it shows) or kept?
    'shrink-then-grow': async () => {
      await tool('Brush'); await opacity(100); await set(0, 1, 50); await set(1, 0, 100); await set(2, 64, 512, 64); await set(3, 64, 512, 64);
      // arrows from the current value (set() presses Home first, which passes through 64 and crops everything)
      await dot(128, 128); await dot(384, 384); await slider(2, ['ArrowLeft', 'ArrowLeft', 'ArrowLeft', 'ArrowLeft']); await page.waitForTimeout(400);
      await slider(2, ['ArrowRight', 'ArrowRight', 'ArrowRight', 'ArrowRight']); await page.waitForTimeout(400); },
    'size-restore': async () => { await set(2, 64, 512, 64); await set(3, 64, 512, 64); console.log('  restored: ' + await sliderInfo()); },
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
