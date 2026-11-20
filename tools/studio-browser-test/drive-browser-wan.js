// Wan 2.2 5B Turbo in the BROWSER on WebGPU: Wan 2.2 Video (TI2V 5B) -> Preview Image on "This browser" (out/browser-wan.png, frames in the
// Preview node). The weights (DiT turbo q8, VAE f16, umT5 q8, about 12 GB) download on first use; keep PROFILE_DIR to reuse them.
// WAN_W / WAN_H / WAN_FRAMES / WAN_TILED override the size (default 1280x704 x 9 frames, untiled decode). STUDIO_URL (default local :5280).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = process.env.PROFILE_DIR || path.join(__dirname, 'out', 'webgpu-profile');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const context = await chromium.launchPersistentContext(PROFILE, {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
  });
  const page = context.pages()[0] || await context.newPage();
  const problems = [];
  const started = Date.now();
  const t = () => `+${((Date.now() - started) / 1000).toFixed(0)}s`;
  page.on('console', m => {
    const s = m.text();
    if (m.type() === 'error') { problems.push(s); console.log(t(), 'console error:', s.slice(0, 300)); }
    else if (/\[models\]|\[gpu\]|Wan|lost/i.test(s) && !/Downloading/.test(s)) console.log(t(), 'log:', s.slice(0, 300));
  });
  page.on('pageerror', e => { problems.push('PAGEERROR ' + e.message); console.log(t(), 'PAGEERROR', e.message.slice(0, 300)); });
  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  const prompt = {
    '1': { class_type: 'Wan22Video', inputs: {
      prompt: 'A young woman in a yellow raincoat walks along a beach at sunset and waves at the camera with her right hand, waves rolling in behind her',
      negative_prompt: '', width: +(process.env.WAN_W || 1280), height: +(process.env.WAN_H || 704), length: +(process.env.WAN_FRAMES || 9),
      steps: 4, cfg: 1, shift: 8, seed: 42, sampler: 'few_step', model: '5B Turbo', tiled_decode: process.env.WAN_TILED === '1' } },
    '2': { class_type: 'PreviewImage', inputs: { images: ['1', 0] } },
  };
  const promptPath = path.join(OUT, 'wan-prompt.json');
  fs.writeFileSync(promptPath, JSON.stringify(prompt));
  await page.setInputFiles('input[accept=".json,.png"]', promptPath);
  await page.waitForTimeout(1500);
  await page.locator('button:has-text("Queue")').click();
  let last = '';
  for (let i = 0; i < 7200; i++) {
    await page.waitForTimeout(1000);
    const s = (await page.textContent('.status')).trim().split(' \u00b7 ')[0];
    if (s !== last && /Running|Queued/.test(s)) { console.log(t(), s); last = s; }
    if (i > 5 && !/Running|Queued/.test(s)) break;
  }
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(OUT, 'browser-wan.png') });
  console.log(t(), 'status:', (await page.textContent('.status')).trim());
  console.log('errors:', problems.length ? problems.join('\n') : 'none');
  await context.close();
  process.exit(problems.length === 0 ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
