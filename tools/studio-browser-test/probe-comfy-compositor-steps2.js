// Second round of black-box measurements of ComfyUI's compositor editor (see probe-comfy-compositor-steps.js): what the first round's clicks
// missed. Controls are clicked by their place on screen (1600x1000, as probe-comfy-compositor.js recorded them), since the editor's
// viewport takes pointer events meant for the dialog's own buttons when asked through selectors.
//   Blend list, Close / reopen, Restore, the zoom menu, the Hand tool, the background swatch, the rotation handle, arrow keys, Delete,
//   a click on empty canvas, and resize rounding (several corner drags of known size).
//   COMFY_URL (default http://127.0.0.1:8199/). Output: out/comfy-compositor-steps2.json, out/comfy-compositor-s2-<n>.png.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
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
  const id = await page.evaluate(() => {
    const app = window.app;
    app.graph.clear();
    const LG = window.LiteGraph || (window.comfyAPI && window.comfyAPI.litegraph && window.comfyAPI.litegraph.LiteGraph);
    const add = (type, x, y) => { const n = LG.createNode(type); n.pos = [x, y]; app.graph.add(n); return n; };
    const set = (n, name, value) => { const w = (n.widgets || []).find(w => w.name === name); if (w) w.value = value; };
    const slot = (n, name) => n.inputs.findIndex(s => s.name === name);
    const la = add('LoadImage', 40, 80), lb = add('LoadImage', 40, 480);
    set(la, 'image', 'compositor-a.png'); set(lb, 'image', 'compositor-b.png');
    const a = add('AddLayer', 420, 80), b = add('AddLayer', 420, 480);
    set(a, 'name', 'coat'); set(b, 'name', 'apple'); set(b, 'x', 120); set(b, 'y', 60); set(b, 'opacity', 0.8);
    const c = add('ImageCompositor', 820, 120);
    la.connect(0, a, slot(a, 'image')); lb.connect(0, b, slot(b, 'image')); a.connect(0, b, slot(b, 'layers')); b.connect(0, c, slot(c, 'layers'));
    return c.id;
  });
  await page.waitForTimeout(800);
  await page.evaluate(() => window.app.queuePrompt(0));
  for (let i = 0; i < 120; i++) {
    await page.waitForTimeout(1000);
    const q = await page.evaluate(async () => { const r = await (await fetch('/queue')).json(); return r.queue_running.length + r.queue_pending.length; });
    if (q === 0 && i > 2) break;
  }
  await page.waitForTimeout(2000);
  const openBox = await page.locator('button:has-text("Open Compositor")').first().boundingBox();
  const openAt = [openBox.x + openBox.width / 2, openBox.y + openBox.height / 2];
  console.log('Open Compositor button at ' + openAt.map(Math.round));

  const state = () => page.evaluate(id => {
    const dialogs = [...document.querySelectorAll('[role="dialog"]')].filter(d => d.getBoundingClientRect().width > 0);
    const dialog = dialogs[0];
    const inputs = dialog ? [...dialog.querySelectorAll('input')].filter(e => e.getBoundingClientRect().width > 0).map(e => e.value) : [];
    const rows = dialog ? [...dialog.querySelectorAll('span')].filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.x < 220 && e.children.length === 0; }).map(e => e.textContent.trim()).filter(Boolean) : [];
    const combo = dialog && dialog.querySelector('[role="combobox"]');
    const zoom = dialog && dialog.querySelector('button[aria-label="Zoom"]');
    const header = dialog ? [...dialog.querySelectorAll('span')].filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.x > 1380 && r.y < 90 && e.children.length === 0; }).map(e => e.textContent.trim()).join(' ') : '';
    const n = window.app.graph.getNodeById(id);
    const w = (n.widgets || []).find(w => w.name === 'compositor');
    return { open: dialogs.length, inputs, rows, blend: combo ? combo.textContent.trim() : null, zoom: zoom ? zoom.textContent.trim() : null, selected: header, value: JSON.stringify(w && w.value) };
  }, id);
  const steps = [];
  let n = 0;
  const record = async what => {
    await page.waitForTimeout(700);
    const s = await state();
    n++;
    await page.screenshot({ path: path.join(OUT, `comfy-compositor-s2-${n}.png`) });
    steps.push({ n, what, ...s });
    fs.writeFileSync(path.join(OUT, 'comfy-compositor-steps2.json'), JSON.stringify({ steps }, null, 1));
    console.log(`${n}. ${what}: open ${s.open} sel "${s.selected}" fields [${s.inputs.join(', ')}] blend ${s.blend} zoom ${s.zoom} | ${s.rows.join(' / ')} | value ${s.value.length} chars`);
    return s;
  };
  const at = async (x, y) => { await page.mouse.click(x, y); };
  const drag = async (from, by) => {
    await page.mouse.move(from[0], from[1]); await page.mouse.down();
    await page.mouse.move(from[0] + by[0] / 2, from[1] + by[1] / 2, { steps: 6 });
    await page.mouse.move(from[0] + by[0], from[1] + by[1], { steps: 6 });
    await page.mouse.up();
  };
  const options = () => page.evaluate(() => [...document.querySelectorAll('[role="option"], [role="menuitem"], [role="menuitemradio"], [role="menuitemcheckbox"]')]
    .filter(e => e.getBoundingClientRect().width > 0).map(e => e.textContent.trim()));

  await at(...openAt); await record('opened');
  // blend list: the combo at (1532, 445)
  await at(1532, 445); await page.waitForTimeout(700);
  const blends = await options();
  console.log(`   blend list (${blends.length}): ${blends.join(' | ')}`);
  await page.screenshot({ path: path.join(OUT, 'comfy-compositor-s2-blendlist.png') });
  const multiply = page.locator('[role="option"]', { hasText: /^Multiply$/ }).first();
  const mb = await multiply.boundingBox().catch(() => null);
  if (mb) await at(mb.x + mb.width / 2, mb.y + mb.height / 2); else await page.keyboard.press('Escape');
  await record('chose Multiply from the blend list');
  // zoom menu at (774, 95)
  await at(774, 95); await page.waitForTimeout(700);
  const zooms = await options();
  console.log(`   zoom menu (${zooms.length}): ${zooms.join(' | ')}`);
  await page.screenshot({ path: path.join(OUT, 'comfy-compositor-s2-zoommenu.png') });
  await page.keyboard.press('Escape'); await page.waitForTimeout(400);
  await record('zoom menu closed with Escape (is the editor still open?)');
  // resize rounding: bottom-right corner of the apple layer is at canvas (632, 572) -> screen (508 + 632 z, 267 + 572 z)
  const z = 0.9;
  const corner = () => [508 + 632 * z, 267 + 572 * z];
  let c = corner();
  await drag(c, [-90, -90]); let s = await record('corner dragged by (-90, -90) screen px');
  await page.keyboard.press('Control+z'); await record('undo');
  await drag(c, [-45, -45]); await record('corner dragged by (-45, -45)');
  await page.keyboard.press('Control+z'); await record('undo');
  await drag(c, [-9, -9]); await record('corner dragged by (-9, -9)');
  await page.keyboard.press('Control+z'); await record('undo');
  // rotation handle above the top edge: (846, 299); drag it to the right and down around the layer centre (846, 551)
  await drag([846, 299], [252, 252]); await record('rotation handle dragged from the top to the right side (a quarter turn?)');
  await page.keyboard.press('Control+z'); await record('undo');
  // arrow keys and Delete with the layer selected (click the body first)
  await at(900, 600); await record('clicked the layer body');
  await page.keyboard.press('ArrowRight'); await record('ArrowRight');
  await page.keyboard.press('Shift+ArrowDown'); await record('Shift+ArrowDown');
  await page.keyboard.press('Delete'); await record('Delete');
  // empty canvas (left of the composition) and the Hand tool
  await at(350, 500); await record('clicked empty space left of the canvas');
  await at(722, 95); await record('Hand tool chosen');
  await drag([800, 500], [60, 40]); await record('dragged with the Hand tool by (+60, +40)');
  await at(686, 95); await record('Select tool chosen');
  // background swatch at (65, 932) and its row
  await at(120, 925); await record('clicked the Background row');
  await at(65, 932); await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(OUT, 'comfy-compositor-s2-swatch.png') });
  const afterSwatch = await page.evaluate(() => [...document.querySelectorAll('input')].filter(e => e.getBoundingClientRect().width > 0).map(e => `${e.type}:${e.value}`));
  console.log('   inputs visible after clicking the swatch: ' + afterSwatch.join(', '));
  await page.keyboard.press('Escape'); await record('after the swatch click and Escape');
  // Close at (1545, 33); reopen; Restore at (1458, 33)
  if ((await state()).open === 0) { await at(...openAt); await record('reopened (Escape had closed the editor)'); }
  await at(1545, 33); s = await record('Close'); console.log('   value after Close: ' + s.value.slice(0, 500));
  await at(...openAt); await record('opened again');
  await at(1458, 33); s = await record('Restore'); console.log('   value after Restore: ' + s.value.slice(0, 500));
  await at(1545, 33); s = await record('Close after Restore'); console.log('   value after Close: ' + s.value.slice(0, 300));
  await browser.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
