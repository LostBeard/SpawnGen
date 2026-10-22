// SD-Turbo in the BROWSER on WebGPU: opens ComfyUI's stock workflow (patched to sd_turbo, 1 step, small picture), runs it on "This browser",
// follows the status line and reports timings. The first run downloads the model (about 2.5 GB) into the profile's origin-private storage; keep
// PROFILE_DIR to reuse it. Needs a WebGPU browser (the system Chrome here) and network access to Hugging Face on the first run.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');

const URL = process.env.STUDIO_URL || 'http://localhost:5300/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = process.env.PROFILE_DIR || path.join(__dirname, 'out', 'webgpu-profile');
const SIZE = parseInt(process.env.PICTURE_SIZE || '64', 10);
const MAX_MINUTES = parseInt(process.env.MAX_MINUTES || '40', 10);
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  // A persistent profile keeps the downloaded model between runs.
  const context = await chromium.launchPersistentContext(PROFILE, {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'],
  });
  const page = context.pages()[0] || await context.newPage();
  const problems = [];
  const started = Date.now();
  const t = () => `+${((Date.now() - started) / 1000).toFixed(0)}s`;
  page.on('console', m => {
    if (m.type() === 'error') { problems.push(m.text()); console.log(t(), 'console error:', m.text().slice(0, 300)); }
    else if (/\[SpawnGen\]|GenLoad|\[perop\]|device|lost|Unhandled/i.test(m.text())) console.log(t(), 'console:', m.text().slice(0, 400));
  });
  page.on('pageerror', e => { problems.push('PAGEERROR ' + e.message); console.log(t(), 'PAGEERROR', e.message.slice(0, 300)); });

  // The stock workflow, patched: our checkpoint name, 1 step, cfg 1, a small picture.
  const wf = JSON.parse(fs.readFileSync(path.join(__dirname, 'comfy-default-workflow.json'), 'utf8'));
  for (const n of wf.nodes) {
    if (n.type === 'CheckpointLoaderSimple') n.widgets_values = ['sd_turbo'];
    if (n.type === 'EmptyLatentImage') n.widgets_values = [SIZE, SIZE, 1];
    if (n.type === 'KSampler') n.widgets_values = [7, 'fixed', 1, 1, 'euler', 'normal', 1];
    if (n.type === 'CLIPTextEncode' && n.id === 6) n.widgets_values = ['a photo of a red apple on a wooden table'];
    if (n.type === 'CLIPTextEncode' && n.id === 7) n.widgets_values = [''];
  }
  const wfPath = path.join(OUT, 'sd-turbo-small.json');
  fs.writeFileSync(wfPath, JSON.stringify(wf));

  await page.goto(URL + (process.env.PEROP ? '?perop=1' : ''));
  await page.waitForSelector('canvas');
  await require('./tab-offset')(page);
  await page.waitForTimeout(3500);
  const status = async () => (await page.textContent('.status')).trim();
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(1200);
  console.log(t(), 'opened:', await status());
  await page.locator('button:has-text("Queue")').click();

  let last = '', lastToast = '';
  let finished = false;
  while (Date.now() - started < MAX_MINUTES * 60 * 1000) {
    // A failed run shows its reason in a toast for a few seconds: keep the full text.
    const toast = await page.locator('.toast').first().textContent({ timeout: 300 }).catch(() => null);
    if (toast && toast !== lastToast) { lastToast = toast; console.log(t(), 'TOAST:', toast); }
    const s = (await status()).split(' \u00b7 ')[0];
    if (s !== last && /Running|Queued/.test(s)) { console.log(t(), s); last = s; }
    if (!/Running|Queued/.test(await status()) && Date.now() - started > 4000) { finished = true; break; }
    await page.waitForTimeout(1500);
  }
  await page.waitForTimeout(3000);
  await page.screenshot({ path: path.join(OUT, 'gpu-result.png') });
  // Save the picture the Save Image node shows (right-click on it) so it can be compared pixel by pixel with another engine's.
  try {
    const [download] = await Promise.all([
      page.waitForEvent('download', { timeout: 15000 }),
      (async () => { await page.mouse.click(1462, 378, { button: 'right' }); await page.click('.menu li:has-text("Save picture")'); })(),
    ]);
    await download.saveAs(path.join(OUT, 'browser-result.png'));
    console.log(t(), 'saved the picture to out/browser-result.png');
  } catch (e) { console.log(t(), 'could not save the picture:', e.message.split('\n')[0]); }
  console.log(t(), finished ? 'run ended' : 'TIMED OUT', '| status:', await status());
  console.log('errors:', problems.length ? problems.join('\n') : 'none');
  console.log('DONE');
  await context.close();
  process.exit(finished && problems.length === 0 ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); console.log('DONE'); process.exit(2); });
