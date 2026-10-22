// Library vision models in the BROWSER on WebGPU: uploads a picture, runs LoadImage -> <node> -> SaveImage on "This browser", follows the status and
// leaves a screenshot (out/browser-vision-<node>.png: look at the picture in the Save Image node). The models download from Hugging Face into the
// profile's storage on first use; keep PROFILE_DIR to reuse them. NODE: DepthAnythingV2 (default), RemoveBackground, StyleTransfer, ImageUpscaleWithModel.
// INPUT_PICTURE: a PNG to process (default: a generated gradient with a disc).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');

const URL = process.env.STUDIO_URL || 'http://localhost:5300/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = process.env.PROFILE_DIR || path.join(__dirname, 'out', 'webgpu-profile');
const NODE = process.env.NODE || 'DepthAnythingV2';
const MAX_MINUTES = parseInt(process.env.MAX_MINUTES || '15', 10);
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const context = await chromium.launchPersistentContext(PROFILE, {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 }, acceptDownloads: true,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'],
  });
  const page = context.pages()[0] || await context.newPage();
  const problems = [];
  const started = Date.now();
  const t = () => `+${((Date.now() - started) / 1000).toFixed(0)}s`;
  page.on('console', m => { if (m.type() === 'error') { problems.push(m.text()); console.log(t(), 'console error:', m.text().slice(0, 300)); } });
  page.on('pageerror', e => { problems.push('PAGEERROR ' + e.message); console.log(t(), 'PAGEERROR', e.message.slice(0, 300)); });

  // The picture to process, as a PNG file the Upload button takes.
  let picture = process.env.INPUT_PICTURE;
  if (!picture) {
    // A tiny PNG writer would be overkill here: draw it in the page and save it from there.
    picture = path.join(OUT, 'vision-input.png');
    const tmp = await context.newPage();
    await tmp.goto('about:blank');
    const b64 = await tmp.evaluate(() => {
      const c = document.createElement('canvas'); c.width = c.height = 256; const x = c.getContext('2d');
      const g = x.createLinearGradient(0, 0, 256, 0); g.addColorStop(0, '#2020e0'); g.addColorStop(1, '#a0a040'); x.fillStyle = g; x.fillRect(0, 0, 256, 256);
      x.fillStyle = '#e03030'; x.beginPath(); x.arc(128, 128, 70, 0, 7); x.fill();
      return c.toDataURL('image/png').split(',')[1];
    });
    fs.writeFileSync(picture, Buffer.from(b64, 'base64'));
    await tmp.close();
  }

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  await page.setInputFiles('label.button:has-text("Upload") input', picture);
  await page.waitForTimeout(1500);
  const name = path.basename(picture);

  const inputs = { DepthAnythingV2: {}, RemoveBackground: { threshold: 0.5 }, StyleTransfer: { style: 'mosaic' }, ImageUpscaleWithModel: {} };
  const prompt = { '1': { class_type: 'LoadImage', inputs: { image: name } } };
  if (NODE === 'ImageUpscaleWithModel') {
    prompt['2'] = { class_type: 'UpscaleModelLoader', inputs: { model_name: 'espcn_x3.onnx' } };
    prompt['3'] = { class_type: NODE, inputs: { upscale_model: ['2', 0], image: ['1', 0] } };
  } else prompt['3'] = { class_type: NODE, inputs: { image: ['1', 0], ...inputs[NODE] } };
  const outputLink = NODE === 'RemoveBackground' ? ['3', 1] : ['3', 0];
  if (NODE === 'RemoveBackground') { prompt['4'] = { class_type: 'MaskToImage', inputs: { mask: ['3', 1] } }; prompt['5'] = { class_type: 'SaveImage', inputs: { images: ['4', 0], filename_prefix: 'vision' } }; }
  else prompt['5'] = { class_type: 'SaveImage', inputs: { images: outputLink, filename_prefix: 'vision' } };
  const promptPath = path.join(OUT, 'vision-prompt.json');
  fs.writeFileSync(promptPath, JSON.stringify(prompt));
  await page.setInputFiles('input[accept=".json,.png"]', promptPath);
  await page.waitForTimeout(1500);
  const status = async () => (await page.textContent('.status')).trim();
  console.log(t(), 'opened:', await status());
  await page.locator('button:has-text("Queue")').click();

  let last = '', lastToast = '', finished = false;
  while (Date.now() - started < MAX_MINUTES * 60 * 1000) {
    const toast = await page.locator('.toast').first().textContent({ timeout: 300 }).catch(() => null);
    if (toast && toast !== lastToast) { lastToast = toast; console.log(t(), 'TOAST:', toast); }
    const s = (await status()).split(' \u00b7 ')[0];
    if (s !== last && /Running|Queued/.test(s)) { console.log(t(), s); last = s; }
    if (!/Running|Queued/.test(await status()) && Date.now() - started > 6000) { finished = true; break; }
    await page.waitForTimeout(1000);
  }
  await page.waitForTimeout(3000);
  await page.screenshot({ path: path.join(OUT, `browser-vision-${NODE}.png`) });
  console.log(t(), finished ? 'run ended' : 'TIMED OUT', '| status:', await status(), '| last toast:', lastToast);
  console.log('errors:', problems.length ? problems.join('\n') : 'none');
  console.log('DONE');
  await context.close();
  process.exit(finished && problems.length === 0 ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); console.log('DONE'); process.exit(2); });
