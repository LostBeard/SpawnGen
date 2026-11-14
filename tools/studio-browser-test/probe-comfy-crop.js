// Black-box look at ComfyUI's crop editor (Crop Image / ImageCropV2, Node 2.0 view) with a picture connected (Load Image -> Crop Image): the
// controls, the Ratio choices, and the stored crop_region after dragging the rectangle, a corner, and choosing a ratio. Nothing of ComfyUI's
// code is read.  COMFY_URL (default http://127.0.0.1:8199/), CROP_IMAGE (a file in ComfyUI's input folder, default example.png).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out', 'crop');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.COMFY_URL || 'http://127.0.0.1:8199/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const IMAGE = process.env.CROP_IMAGE || 'example.png';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.evaluate(async () => { await window.app.ui.settings.setSettingValueAsync('Comfy.VueNodes.Enabled', true); });
  await page.reload({ waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  const size = await page.evaluate((image) => {
    const app = window.app; app.graph.clear();
    const LG = window.LiteGraph || window.comfyAPI.litegraph.LiteGraph;
    const load = LG.createNode('LoadImage'); load.pos = [20, 40]; app.graph.add(load);
    load.widgets.find(w => w.name === 'image').value = image;
    load.widgets.find(w => w.name === 'image').callback?.(image);
    const crop = LG.createNode('ImageCropV2'); crop.pos = [420, 40]; app.graph.add(crop);
    load.connect(0, crop, 0);
    app.canvas.ds.scale = 0.8; app.canvas.ds.offset = [0, 0]; app.canvas.setDirty(true, true);
    return crop.size;
  }, IMAGE);
  await page.waitForTimeout(3000);

  const log = [];
  const crop = () => page.evaluate(() => { const n = window.app.graph._nodes.find(n => n.type === 'ImageCropV2'); return JSON.stringify(n.widgets.find(w => w.name === 'crop_region').value); });
  const controls = () => page.evaluate(() => {
    const node = [...document.querySelectorAll('[data-node-id]')].find(e => (e.innerText || '').includes('Crop Image'));
    if (!node) return [];
    return [...node.querySelectorAll('*')].filter(el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && ['BUTTON', 'INPUT', 'SELECT', 'IMG', 'CANVAS', 'LABEL'].includes(el.tagName); })
      .map(el => { const r = el.getBoundingClientRect(); return `${el.tagName} "${(el.innerText || el.alt || '').trim().slice(0, 30)}" aria="${el.getAttribute('aria-label') || ''}" val="${(el.value ?? '').toString().slice(0, 20)}" @${Math.round(r.x)},${Math.round(r.y)} ${Math.round(r.width)}x${Math.round(r.height)}`; });
  });
  async function record(name, extra = {}) {
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(OUT, `${String(log.length).padStart(2, '0')}-${name}.png`) });
    const s = await crop();
    log.push({ step: name, crop: s, ...extra });
    console.log(`== ${name}: ${s}${Object.keys(extra).length ? ' ' + JSON.stringify(extra).slice(0, 1500) : ''}`);
  }
  await record('loaded', { size, controls: await controls() });
  // The picture the editor shows: the largest IMG or CANVAS inside the Crop Image node.
  const picture = await page.evaluate(() => {
    const node = [...document.querySelectorAll('[data-node-id]')].find(e => (e.innerText || '').includes('Crop Image'));
    const els = node ? [...node.querySelectorAll('img, canvas')].map(el => el.getBoundingClientRect()).filter(r => r.width > 40) : [];
    els.sort((a, b) => b.width * b.height - a.width * a.height);
    return els[0] ? { x: els[0].x, y: els[0].y, w: els[0].width, h: els[0].height } : null;
  });
  console.log('picture', JSON.stringify(picture));
  if (picture) {
    const at = (fx, fy) => [picture.x + picture.w * fx, picture.y + picture.h * fy];
    async function drag(a, b) { await page.mouse.move(...a); await page.mouse.down(); await page.mouse.move((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, { steps: 5 }); await page.mouse.move(...b, { steps: 5 }); await page.mouse.up(); }
    await drag(at(0.2, 0.2), at(0.35, 0.3)); await record('drag-inside-rectangle');
    await drag(at(0.05, 0.05), at(0.6, 0.7)); await record('drag-from-outside-or-corner');
  }
  // The Ratio choices.
  const ratio = page.locator('[data-node-id] button', { hasText: 'Custom' }).first();
  if (await ratio.count()) {
    await ratio.click();
    await page.waitForTimeout(500);
    const options = await page.evaluate(() => [...document.querySelectorAll('[role="option"], [role="menuitem"], li')].filter(el => el.getBoundingClientRect().width > 0).map(el => (el.innerText || '').trim()).filter(t => t && t.length < 30));
    await record('ratio-open', { options });
    const pick = page.locator('[role="option"], [role="menuitem"], li', { hasText: '16:9' }).first();
    if (await pick.count()) { await pick.click(); await record('ratio-16:9'); }
    else await page.keyboard.press('Escape');
  }
  // The lock (Custom ratio), typing in the fields, and the picture's edge: start from a known region each time by typing it.
  if (picture) {
    const imgW = 768, imgH = 768;
    const toScreen = (x, y) => [picture.x + x / imgW * picture.w, picture.y + y / imgH * picture.h];
    async function drag(a, b) { await page.mouse.move(...a); await page.mouse.down(); await page.mouse.move((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, { steps: 5 }); await page.mouse.move(...b, { steps: 5 }); await page.mouse.up(); }
    const cropNode = page.locator('[data-node-id]', { hasText: 'Crop Image' }).last();
    const field = (i) => cropNode.locator('input').nth(i);
    async function type(i, v) { await field(i).fill(String(v), { force: true }); await field(i).press('Enter'); await field(i).evaluate(el => el.blur()); await page.waitForTimeout(250); }
    // Back to Custom.
    const ratioButton = page.locator('[data-node-id] button', { hasText: /^(1:1|3:4|4:3|16:9|9:16|Custom)$/ }).first();
    await ratioButton.click(); await page.waitForTimeout(300);
    await page.locator('[role="option"], li', { hasText: 'Custom' }).first().click(); await record('custom');
    await type(0, 100); await type(1, 100); await type(2, 400); await type(3, 200); await record('typed-100-100-400x200');
    await type(2, 900); await record('typed-width-900-past-the-edge');
    await type(0, 700); await record('typed-x-700');
    await type(0, 100); await type(2, 400); await type(3, 200);
    await drag(toScreen(300, 200), toScreen(300 + 600, 200 + 600)); await record('drag-inside-past-the-edge');
    await type(0, 100); await type(1, 100); await type(2, 400); await type(3, 200);
    await drag(toScreen(500, 300), toScreen(500 + 600, 300 + 50)); await record('drag-corner-past-the-edge');
    await type(0, 100); await type(1, 100); await type(2, 400); await type(3, 200);
    const lock = cropNode.locator('button[aria-label*="aspect ratio"]').first();
    await lock.click(); await record('lock', { aria: await lock.getAttribute('aria-label') });
    await drag(toScreen(500, 300), toScreen(500 + 100, 300 + 10)); await record('drag-corner-locked');
    await type(2, 600); await record('typed-width-600-locked');
  }
  fs.writeFileSync(path.join(OUT, 'steps.json'), JSON.stringify(log, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
