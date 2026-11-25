// Black-box measurements of ComfyUI's compositor editor, action by action (see probe-comfy-compositor.js for the setup and the first look):
// after each action it records the property fields of the selected layer, the layer list and the node's widget value. Nothing of ComfyUI's
// code is read. Geometry assumes the editor as probe-comfy-compositor.js found it at 1600x1000: canvas origin (508, 267) at 90% zoom.
//   COMFY_URL (default http://127.0.0.1:8199/). Output: out/comfy-compositor-steps.json and out/comfy-compositor-step-<n>.png.
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
  const open = async () => { await page.locator('button:has-text("Open Compositor")').first().click({ timeout: 8000 }).catch(e => console.log('   (open: ' + e.message.split(String.fromCharCode(10))[0] + ')')); await page.waitForTimeout(2000); };
  const dialogs = () => page.evaluate(() => document.querySelectorAll('[role="dialog"][data-state="open"]').length);
  await open();

  const state = () => page.evaluate(id => {
    const dialog = document.querySelector('[role="dialog"], .p-dialog, dialog') || document.body;
    const inputs = [...dialog.querySelectorAll('input')].filter(e => e.getBoundingClientRect().width > 0).map(e => e.value);
    const spans = [...dialog.querySelectorAll('span')].filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.x < 220 && e.children.length === 0; }).map(e => e.textContent.trim()).filter(Boolean);
    const combo = dialog.querySelector('[role="combobox"]');
    const n = window.app.graph.getNodeById(id);
    const w = (n.widgets || []).find(w => w.name === 'compositor');
    return { inputs, layers: spans, blend: combo ? combo.textContent.trim() : null, value: JSON.stringify(w && w.value) };
  }, id);
  const steps = [];
  let n = 0;
  const record = async what => {
    await page.waitForTimeout(700);
    const s = await state();
    n++;
    await page.screenshot({ path: path.join(OUT, `comfy-compositor-step-${n}.png`) });
    steps.push({ n, what, ...s });
    fs.writeFileSync(path.join(OUT, 'comfy-compositor-steps.json'), JSON.stringify({ steps }, null, 1));
    console.log(`${n}. ${what}: fields [${s.inputs.join(', ')}] blend ${s.blend} | layers ${s.layers.join(' / ')} | value ${s.value.slice(0, 160)}`);
  };
  const drag = async (from, by, opts = {}) => {
    await page.mouse.move(from[0], from[1]);
    await page.mouse.down();
    if (opts.shift) await page.keyboard.down('Shift');
    await page.mouse.move(from[0] + by[0] / 2, from[1] + by[1] / 2, { steps: 6 });
    await page.mouse.move(from[0] + by[0], from[1] + by[1], { steps: 6 });
    if (opts.shift) await page.keyboard.up('Shift');
    await page.mouse.up();
  };
  const click = async sel => { await page.locator(sel).first().click({ timeout: 5000, force: true }).catch(e => console.log(`   (click ${sel} failed: ${e.message.split('\n')[0]})`)); };

  await record('editor opened (apple selected: x 120, y 60, 512x512, opacity 80)');
  // the selected layer's frame at 90%: corners (616,321) .. (1077,782)
  await drag([850, 550], [45, 27]); await record('dragged the layer body by (+45, +27) screen px');
  await drag([1122, 809], [45, 45]); await record('dragged the bottom-right corner by (+45, +45)');
  await drag([1167, 854], [54, 9]); await record('dragged the bottom-right corner by (+54, +9) (off the diagonal)');
  await page.keyboard.press('Control+z'); await record('Ctrl+Z');
  await page.keyboard.press('Control+z'); await record('Ctrl+Z again');
  await click('button[aria-label="Redo"]'); await record('Redo button');
  await click('button[aria-label="Undo"]'); await record('Undo button');
  // after the undos the frame should be back at (661,348)..(1122,809): right edge middle at (1122, 578)
  await drag([1122, 578], [45, 0]); await record('dragged the right edge handle by (+45, 0)');
  await click('button[aria-label="Undo"]'); await record('Undo');
  await drag([1122, 809], [45, 9], { shift: true }); await record('dragged the bottom-right corner by (+45, +9) holding Shift');
  await click('button[aria-label="Undo"]'); await record('Undo');
  for (const a of ['Align left', 'Align horizontal center', 'Align right', 'Align top', 'Align vertical center', 'Align bottom']) {
    await click(`button[aria-label="${a}"]`); await record(a);
  }
  await click('button[aria-label="Flip horizontal"]'); await record('Flip horizontal');
  await click('button[aria-label="Flip vertical"]'); await record('Flip vertical');
  // the fields: type a rotation and an opacity (order of the inputs is read from the first record)
  const inputs = page.locator('[role="dialog"] input, .p-dialog input, dialog input');
  const count = await inputs.count();
  console.log('   inputs in the dialog: ' + count);
  const type = async (index, text, what) => { const el = inputs.nth(index); await el.click(); await el.fill(text); await el.press('Enter'); await record(what); };
  if (count >= 6) {
    await type(4, '30', 'typed 30 in the rotation field');
    await type(5, '55', 'typed 55 in the opacity field');
    await type(0, '10', 'typed 10 in X');
    await type(2, '300', 'typed 300 in W');
    await type(3, '100', 'typed 100 in H');
  }
  await page.locator('[role="combobox"]').first().focus(); await page.keyboard.press('Enter'); await page.waitForTimeout(800);
  const blends = await page.evaluate(() => [...document.querySelectorAll('[role="option"]')].map(e => e.textContent.trim()));
  console.log('   blend modes offered (' + blends.length + '): ' + blends.join(', '));
  await page.locator('[role="option"]', { hasText: 'Multiply' }).first().click({ force: true, timeout: 3000 }).catch(e => console.log('   (Multiply: ' + e.message.split(String.fromCharCode(10))[0] + ')'));
  await record('chose Multiply');
  await click('span[title="coat"]'); await record('clicked the coat layer in the list');
  await click('button[aria-label="Move layer up"]'); await record('Move layer up (coat selected)');
  await click('button[aria-label="Move layer down"]'); await record('Move layer down');
  const toggles = page.locator('button[aria-label="Toggle visibility"]');
  await toggles.nth(0).click(); await record('toggled the first layer row visibility');
  await toggles.nth(0).click(); await record('toggled it back');
  await toggles.nth((await toggles.count()) - 1).click(); await record('toggled the Background visibility');
  await click('button:has-text("Close")'); await record('Close'); console.log('   open dialogs after Close: ' + await dialogs());
  await open(); await record('opened again');
  await click('button:has-text("Restore")'); await record('Restore');
  await click('button:has-text("Close")'); await record('Close after Restore');
  fs.writeFileSync(path.join(OUT, 'comfy-compositor-steps.json'), JSON.stringify({ steps, blends }, null, 1));
  await browser.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
