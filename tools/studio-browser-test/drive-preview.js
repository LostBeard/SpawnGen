// Latent preview while a sampler works, through the server (GPU + SD-Turbo): opens the stock workflow patched to 8 steps at 384x384, queues it,
// counts the "preview" events on the websocket and takes screenshots during the run (out/preview-*.png: the KSampler node should show a blurry
// colour guess that sharpens, then the picture appears in Save Image).
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
  let previews = 0;
  page.on('websocket', ws => ws.on('framereceived', f => { if (typeof f.payload === 'string' && f.payload.includes('"type":"preview"')) previews++; }));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  const wf = JSON.parse(fs.readFileSync(path.join(__dirname, 'comfy-default-workflow.json'), 'utf8'));
  for (const n of wf.nodes) {
    if (n.type === 'CheckpointLoaderSimple') n.widgets_values = ['sd_turbo'];
    if (n.type === 'EmptyLatentImage') n.widgets_values = [384, 384, 1];
    if (n.type === 'KSampler') n.widgets_values = [11, 'fixed', 8, 1, 'euler', 'normal', 1];
    if (n.type === 'CLIPTextEncode' && n.id === 6) n.widgets_values = ['a photo of a red apple on a wooden table'];
    if (n.type === 'CLIPTextEncode' && n.id === 7) n.widgets_values = [''];
  }
  const wfPath = path.join(OUT, 'preview-workflow.json');
  fs.writeFileSync(wfPath, JSON.stringify(wf));

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(1200);
  await page.locator('button:has-text("Queue")').click();
  let shots = 0;
  for (let i = 0; i < 240; i++) {
    await page.waitForTimeout(500);
    const running = !(await page.locator('button:has-text("Queue")').isVisible());
    if (running && previews >= 2 && shots < 2 && i % 4 === 0) await page.screenshot({ path: path.join(OUT, `preview-${++shots}.png`) });
    if (i > 6 && !running) break;
  }
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(OUT, 'preview-final.png') });
  check('the sampler sent a preview for its steps', previews >= 6, `${previews} previews`);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
