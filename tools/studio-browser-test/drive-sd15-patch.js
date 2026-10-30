// SD 1.5 with a UNet patch (FreeU, FreeU_V2, PerturbedAttentionGuidance) on one engine, to compare the browser's WebGPU picture with the server's.
// Opens ComfyUI's stock workflow patched to v1-5-pruned-emaonly-fp16.safetensors (downloaded from Hugging Face on first use), 512x512, euler, cfg 7.5,
// with the patch node between Load Checkpoint and the KSampler (PATCH=none leaves it out), runs it on ENGINE ("browser" or "server") and saves the
// Save Image picture to out/sd15-<patch>-<engine>.png.
//   STUDIO_URL (default http://localhost:8188/, the server-hosted Studio, which has both engines), PATCH, ENGINE, STEPS (default 10), SEED (default 7),
//   PROFILE_DIR (a persistent profile keeps the browser's downloaded model), MAX_MINUTES (default 40).
// Compare two pictures with: node drive-sd15-patch.js --compare a.png b.png
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

if (process.argv[2] === '--compare') {
  // Mean and largest difference per channel value, in 1/255 steps, of two same-size PNGs (decoded by Chrome, so no PNG library is needed).
  (async () => {
    const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
    const page = await browser.newPage();
    const pixels = async f => page.evaluate(async b64 => {
      const img = new Image();
      img.src = 'data:image/png;base64,' + b64;
      await img.decode();
      const c = new OffscreenCanvas(img.width, img.height);
      const g = c.getContext('2d');
      g.drawImage(img, 0, 0);
      return { w: img.width, h: img.height, data: Array.from(g.getImageData(0, 0, img.width, img.height).data) };
    }, fs.readFileSync(f).toString('base64'));
    const [a, b] = [await pixels(process.argv[3]), await pixels(process.argv[4])];
    if (a.w !== b.w || a.h !== b.h) { console.log(`sizes differ: ${a.w}x${a.h} vs ${b.w}x${b.h}`); process.exit(1); }
    let sum = 0, max = 0, n = 0;
    for (let i = 0; i < a.data.length; i++) {
      if (i % 4 === 3) continue;
      const d = Math.abs(a.data[i] - b.data[i]);
      sum += d; n++; if (d > max) max = d;
    }
    console.log(`mean difference ${(sum / n).toFixed(2)}/255, largest ${max}/255`);
    await browser.close();
  })();
  return;
}

const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = process.env.PROFILE_DIR || path.join(OUT, 'webgpu-profile');
const PATCH = process.env.PATCH || 'FreeU';
const ENGINE = process.env.ENGINE || 'browser';
const MAX_MINUTES = parseInt(process.env.MAX_MINUTES || '40', 10);
const PATCHES = {
  FreeU: [1.1, 1.2, 0.9, 0.2],
  FreeU_V2: [1.3, 1.4, 0.9, 0.2],
  PerturbedAttentionGuidance: [3.0],
};

(async () => {
  const context = await chromium.launchPersistentContext(PROFILE, {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 }, acceptDownloads: true,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'],
  });
  const page = context.pages()[0] || await context.newPage();
  const problems = [];
  const started = Date.now();
  const t = () => `+${((Date.now() - started) / 1000).toFixed(0)}s`;
  page.on('console', m => {
    if (m.type() === 'error') { problems.push(m.text()); console.log(t(), 'console error:', m.text().slice(0, 300)); }
    else if (/\[SpawnGen\]|device|lost|Unhandled/i.test(m.text())) console.log(t(), 'console:', m.text().slice(0, 400));
  });
  page.on('pageerror', e => { problems.push('PAGEERROR ' + e.message); console.log(t(), 'PAGEERROR', e.message.slice(0, 300)); });

  // The stock workflow, patched: SD 1.5 fp16, 512x512, fixed seed, and the patch node on the model link (link 1: Load Checkpoint -> KSampler).
  const wf = JSON.parse(fs.readFileSync(path.join(__dirname, 'comfy-default-workflow.json'), 'utf8'));
  for (const n of wf.nodes) {
    if (n.type === 'CheckpointLoaderSimple') n.widgets_values = ['v1-5-pruned-emaonly-fp16.safetensors'];
    if (n.type === 'EmptyLatentImage') n.widgets_values = [512, 512, 1];
    if (n.type === 'KSampler') n.widgets_values = [parseInt(process.env.SEED || '7', 10), 'fixed', parseInt(process.env.STEPS || '10', 10), 7.5, 'euler', 'normal', 1];
    if (n.type === 'CLIPTextEncode' && n.id === 6) n.widgets_values = ['a photo of a red apple on a wooden table'];
    if (n.type === 'CLIPTextEncode' && n.id === 7) n.widgets_values = [''];
  }
  if (PATCH !== 'none') {
    const ckpt = wf.nodes.find(n => n.id === 4), sampler = wf.nodes.find(n => n.id === 3);
    wf.nodes.push({
      id: 10, type: PATCH, pos: [420, 60], size: { 0: 315, 1: 130 }, flags: {}, order: 1, mode: 0,
      inputs: [{ name: 'model', type: 'MODEL', link: 1 }], outputs: [{ name: 'MODEL', type: 'MODEL', links: [10], slot_index: 0 }],
      properties: {}, widgets_values: PATCHES[PATCH],
    });
    wf.links[0] = [1, 4, 0, 10, 0, 'MODEL'];
    wf.links.push([10, 10, 0, 3, 0, 'MODEL']);
    sampler.inputs[0].link = 10;
    ckpt.outputs[0].links = [1];
    wf.last_node_id = 10;
    wf.last_link_id = 10;
  }
  const wfPath = path.join(OUT, `sd15-${PATCH}.json`);
  fs.writeFileSync(wfPath, JSON.stringify(wf));

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await require('./tab-offset')(page);
  await page.waitForTimeout(3500);
  const status = async () => (await page.textContent('.status')).trim();
  await page.selectOption('select.engine', { label: ENGINE === 'browser' ? 'This browser (WebGPU)' : 'Server (CUDA / OpenCL)' });
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(1200);
  console.log(t(), 'opened:', await status());
  await page.locator('button:has-text("Queue")').click();
  const queued = Date.now();

  let last = '', lastToast = '', finished = false;
  while (Date.now() - started < MAX_MINUTES * 60 * 1000) {
    const toast = await page.locator('.toast').first().textContent({ timeout: 300 }).catch(() => null);
    if (toast && toast !== lastToast) { lastToast = toast; console.log(t(), 'TOAST:', toast); }
    const s = (await status()).split(' \u00b7 ')[0];
    if (s !== last && /Running|Queued/.test(s)) { console.log(t(), s); last = s; }
    if (!/Running|Queued/.test(await status()) && Date.now() - queued > 4000) { finished = true; break; }
    await page.waitForTimeout(1500);
  }
  console.log(t(), `run took ${((Date.now() - queued) / 1000).toFixed(0)} s`);
  await page.waitForTimeout(3000);
  await page.screenshot({ path: path.join(OUT, `sd15-${PATCH}-${ENGINE}-screen.png`) });
  const saved = path.join(OUT, `sd15-${PATCH}-${ENGINE}.png`);
  try {
    const [download] = await Promise.all([
      page.waitForEvent('download', { timeout: 15000 }),
      (async () => { await page.mouse.click(1462, 378, { button: 'right' }); await page.click('.menu li:has-text("Save picture")'); })(),
    ]);
    await download.saveAs(saved);
    console.log(t(), 'saved the picture to', saved);
  } catch (e) { console.log(t(), 'could not save the picture:', e.message.split('\n')[0]); }
  console.log(t(), finished ? 'run ended' : 'TIMED OUT', '| status:', await status());
  console.log('errors:', problems.length ? problems.join('\n') : 'none');
  await context.close();
  process.exit(finished && problems.length === 0 ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
