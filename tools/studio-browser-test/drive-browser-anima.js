// Anima in the BROWSER on WebGPU (2026-11-28, Lore): the image_anima_preview graph as AnimaTests.The_template_graph_matches_ComfyUI runs it
// (anima-preview3-base + Qwen3-0.6B + Qwen-Image VAE, 512 x 512, 4 steps er_sde / simple, cfg 4, seed 42) on "This browser"; the picture is
// saved to out/browser-anima.png and compared with ComfyUI's (tests/.../Fixtures/anima/comfyui_picture_512.png) by tools/.../png-psnr.py.
// The weights (about 6 GB from circlestone-labs/Anima) download on first use; PROFILE_DIR keeps them. STUDIO_URL (default local :5300).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const URL = process.env.STUDIO_URL || 'http://localhost:5300/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = process.env.PROFILE_DIR || path.join(__dirname, 'out', 'webgpu-profile');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const POS = 'masterpiece, best quality, score_7, safe, anime, a girl with a red umbrella standing in the rain';
const NEG = 'worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts';

(async () => {
  const context = await chromium.launchPersistentContext(PROFILE, {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
  });
  const page = context.pages()[0] || await context.newPage();
  await page.addInitScript(() => {
    window.__resultBlobs = [];
    const make = URL.createObjectURL;
    URL.createObjectURL = function (b) { const u = make.call(URL, b); if (b instanceof Blob && /png/.test(b.type)) window.__resultBlobs.push(b); return u; };
  });
  const problems = [];
  const started = Date.now();
  const t = () => `+${((Date.now() - started) / 1000).toFixed(0)}s`;
  page.on('console', m => {
    const s = m.text();
    if (m.type() === 'error') { problems.push(s); console.log(t(), 'console error:', s.slice(0, 300)); }
    else if (/\[models\]|\[gpu\]|anima|lost/i.test(s) && !/Downloading/.test(s)) console.log(t(), 'log:', s.slice(0, 300));
    if (/device was lost|a run failed/i.test(s)) problems.push(s.slice(0, 300));
  });
  page.on('pageerror', e => { problems.push('PAGEERROR ' + e.message); console.log(t(), 'PAGEERROR', e.message.slice(0, 300)); });
  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  const prompt = {
    unet: { class_type: 'UNETLoader', inputs: { unet_name: 'anima-preview3-base.safetensors', weight_dtype: 'default' } },
    clip: { class_type: 'CLIPLoader', inputs: { clip_name: 'qwen_3_06b_base.safetensors', type: 'stable_diffusion', device: 'default' } },
    vae: { class_type: 'VAELoader', inputs: { vae_name: 'qwen_image_vae.safetensors' } },
    pos: { class_type: 'CLIPTextEncode', inputs: { clip: ['clip', 0], text: POS } },
    neg: { class_type: 'CLIPTextEncode', inputs: { clip: ['clip', 0], text: NEG } },
    latent: { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
    ks: { class_type: 'KSampler', inputs: { model: ['unet', 0], positive: ['pos', 0], negative: ['neg', 0], latent_image: ['latent', 0],
      seed: 42, steps: 4, cfg: 4, sampler_name: 'er_sde', scheduler: 'simple', denoise: 1 } },
    dec: { class_type: 'VAEDecode', inputs: { samples: ['ks', 0], vae: ['vae', 0] } },
    save: { class_type: 'PreviewImage', inputs: { images: ['dec', 0] } },
  };
  const promptPath = path.join(OUT, 'anima-prompt.json');
  fs.writeFileSync(promptPath, JSON.stringify(prompt));
  await page.setInputFiles('input[accept=".json,.png"]', promptPath);
  await page.waitForTimeout(1500);
  await page.evaluate(() => { window.__resultBlobs = []; });
  await page.locator('button:has-text("Queue")').click();
  let last = '';
  for (let i = 0; i < 7200; i++) {
    await page.waitForTimeout(1000);
    const s = (await page.textContent('.status')).trim().split(' \u00b7 ')[0];
    if (s !== last && /Running|Queued/.test(s)) { console.log(t(), s); last = s; }
    if (i > 5 && !/Running|Queued/.test(s)) break;
  }
  await page.waitForTimeout(2000);
  await page.screenshot({ path: path.join(OUT, 'browser-anima-screen.png') });
  console.log(t(), 'status:', (await page.textContent('.status')).trim());
  const b64 = await page.evaluate(async () => {
    const b = window.__resultBlobs.at(-1);
    if (!b) return null;
    const bytes = new Uint8Array(await b.arrayBuffer());
    let s = ''; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  });
  if (b64) { fs.writeFileSync(path.join(OUT, 'browser-anima.png'), Buffer.from(b64, 'base64')); console.log('picture: out/browser-anima.png'); }
  else problems.push('no picture came back');
  console.log('errors:', problems.length ? problems.join('\n') : 'none');
  await context.close();
  process.exit(problems.length === 0 ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
