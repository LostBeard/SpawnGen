// Black-box: what ComfyUI's compositor editor writes with Export PSD. Same setup as probe-comfy-compositor-steps2.js; the apple layer gets
// Multiply, a rotation and a flip first (typed / chosen), then the Export PSD button (toolbar, at 898, 95) is pressed and the download is
// saved to out/comfy-compositor-export.psd with its suggested name printed. Nothing of ComfyUI's code is read.
//   COMFY_URL (default http://127.0.0.1:8199/). PLAIN=1 exports without any edit (out/comfy-compositor-export-plain.psd).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
const URL = process.env.COMFY_URL || 'http://127.0.0.1:8199/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PLAIN = process.env.PLAIN === '1';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.evaluate(async () => { await window.app.ui.settings.setSettingValueAsync('Comfy.VueNodes.Enabled', true); });
  await page.reload({ waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  await page.evaluate(() => {
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
  });
  await page.waitForTimeout(800);
  await page.evaluate(() => window.app.queuePrompt(0));
  for (let i = 0; i < 120; i++) {
    await page.waitForTimeout(1000);
    const q = await page.evaluate(async () => { const r = await (await fetch('/queue')).json(); return r.queue_running.length + r.queue_pending.length; });
    if (q === 0 && i > 2) break;
  }
  await page.waitForTimeout(2000);
  const box = await page.locator('button:has-text("Open Compositor")').first().boundingBox();
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await page.waitForTimeout(2000);
  if (!PLAIN) {
    await page.mouse.click(1532, 445); await page.waitForTimeout(700);
    const mb = await page.locator('[role="option"]', { hasText: /^Multiply$/ }).first().boundingBox().catch(() => null);
    if (mb) await page.mouse.click(mb.x + mb.width / 2, mb.y + mb.height / 2);
    await page.waitForTimeout(400);
    const inputs = page.locator('[role="dialog"] input');
    const type = async (index, text) => { const el = inputs.nth(index); await el.click(); await el.fill(text); await el.press('Enter'); await page.waitForTimeout(300); };
    await type(4, '30');   // rotation
    await type(2, '300');  // W
    await page.mouse.click(1533, 368); await page.waitForTimeout(300); // Flip horizontal
    await page.mouse.click(29, 932); await page.waitForTimeout(300);   // Background visible
  }
  await page.screenshot({ path: path.join(OUT, 'comfy-compositor-psd-before.png') });
  const name = PLAIN ? 'comfy-compositor-export-plain.psd' : 'comfy-compositor-export.psd';
  const [download] = await Promise.all([page.waitForEvent('download', { timeout: 20000 }), page.mouse.click(898, 95)]);
  await download.saveAs(path.join(OUT, name));
  console.log(`suggested name: ${download.suggestedFilename()}; saved out/${name} (${fs.statSync(path.join(OUT, name)).size} bytes)`);
  await browser.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
