// SDXL from a single-file checkpoint through the Studio and the server: opens sdxl-workflow.json (ComfyUI's default workflow with Load Checkpoint set to
// sd_xl_base_1.0.safetensors, 1024x1024, 20 steps), checks Load Checkpoint offers the .safetensors files, presses Queue, follows the run and checks the history.
// Needs the server running with <models>/checkpoints/sd_xl_base_1.0.safetensors and a GPU with 8 GB. Look at out/sdxl-done.png: the picture in Save Image.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const status = async () => (await page.textContent('.status')).trim();

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  const info = await page.evaluate(async () => (await (await fetch('/object_info/CheckpointLoaderSimple')).json()));
  const names = info.CheckpointLoaderSimple.input.required.ckpt_name[0];
  check('Load Checkpoint offers the .safetensors files', names.includes('sd_xl_base_1.0.safetensors') && names.includes('v1-5-pruned-emaonly.safetensors'), names.join(', '));

  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'sdxl-workflow.json'));
  await page.waitForTimeout(1200);
  check('the workflow opens', /^7 nodes, 9 links/.test(await status()), await status());
  const before = await page.evaluate(async () => Object.keys(await (await fetch('/history')).json()).length);
  await page.locator('button:has-text("Queue")').click();

  const seen = [];
  const started = Date.now();
  while (Date.now() - started < 900000) {
    const s = await status();
    const head = s.split(' \u00b7 ')[0];
    if (/Running|Queued/.test(head) && !seen.includes(head.split(':')[0])) {
      seen.push(head.split(':')[0]);
      console.log(`  +${((Date.now() - started) / 1000).toFixed(0)}s ${head}`);
    }
    if (!/Running|Queued|Loading/.test(s) && Date.now() - started > 5000) break;
    await page.waitForTimeout(1000);
  }
  console.log(`  done after ${((Date.now() - started) / 1000).toFixed(0)}s: ${await status()}`);
  await page.waitForTimeout(3000); // the result picture loads after the last event
  await page.screenshot({ path: path.join(OUT, 'sdxl-done.png') });

  const history = await page.evaluate(async () => Object.values(await (await fetch('/history')).json()));
  const last = history[history.length - 1];
  check('the server ran it to success', history.length === before + 1 && last.status.status_str === 'success', last && last.status.status_str);
  const images = Object.values(last.outputs || {}).flatMap(o => o.images || []);
  check('Save Image wrote a picture', images.length === 1 && /^studio_sdxl_\d+_\.png$/.test(images[0].filename), JSON.stringify(images));
  check('the toolbar followed the sampler and the decode', ['Running KSampler', 'Running VAE Decode'].every(s => seen.includes(s)), seen.join(' > '));
  check('no browser errors', problems.length === 0, problems.join(' | '));

  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
