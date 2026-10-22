// A generated picture carries its workflow: generate on the server's engine, open a different workflow, then DROP the picture on the canvas and
// check that the original workflow (same seed) is back. Needs the SpawnGen server (it serves the Studio), a GPU and the SD-Turbo model. ~45 s.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');

const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

/** The text chunks of a PNG (tEXt only: that is what SpawnGen writes for ASCII workflows). */
function pngText(buf) {
  const out = {};
  for (let pos = 8; pos + 12 <= buf.length;) {
    const len = buf.readUInt32BE(pos), type = buf.toString('ascii', pos + 4, pos + 8);
    if (type === 'tEXt') { const data = buf.subarray(pos + 8, pos + 8 + len); const z = data.indexOf(0); out[data.toString('latin1', 0, z)] = data.toString('latin1', z + 1); }
    if (type === 'IEND') break;
    pos += 12 + len;
  }
  return out;
}

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const status = async () => (await page.textContent('.status')).trim();

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);

  // 1. Generate (the seed is used as it is, then control after generate moves it on).
  await page.locator('button:has-text("Queue")').click();
  for (let i = 0; i < 160; i++) { await page.waitForTimeout(1000); if (i > 3 && !/Running|Queued/.test(await status())) break; }
  await page.waitForTimeout(2000);
  const files = await page.evaluate(async () => {
    const h = Object.values(await (await fetch('/history')).json()).pop();
    return Object.values(h.outputs).flatMap(o => o.images);
  });
  check('a picture was saved', files.length === 1, JSON.stringify(files));
  const pngB64 = await page.evaluate(async f => {
    const bytes = new Uint8Array(await (await fetch(`/view?filename=${f.filename}&subfolder=&type=${f.type}`)).arrayBuffer());
    let s = ''; for (const b of bytes) s += String.fromCharCode(b); return btoa(s);
  }, files[0]);
  const png = Buffer.from(pngB64, 'base64');
  fs.writeFileSync(path.join(OUT, 'generated.png'), png);
  const text = pngText(png);
  check('the PNG carries prompt and workflow', 'prompt' in text && 'workflow' in text, Object.keys(text).join(', '));
  const sampler = JSON.parse(text.workflow).nodes.find(n => n.type === 'KSampler');
  const seed = sampler.widgets_values[0];
  check('the embedded workflow has the seed that was used', Number.isInteger(seed) && seed >= 0, 'seed ' + seed);

  // 2. Replace the graph with something else.
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'load-preview-workflow.json'));
  await page.waitForTimeout(1200);
  check('a different workflow is open', /^2 nodes/.test(await status()), await status());

  // 3. Drop the generated picture on the canvas.
  await page.evaluate(async b64 => {
    const bin = atob(b64); const bytes = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const dt = new DataTransfer(); dt.items.add(new File([bytes], 'generated.png', { type: 'image/png' }));
    const stage = document.querySelector('.stage');
    stage.dispatchEvent(new DragEvent('dragover', { dataTransfer: dt, bubbles: true, cancelable: true }));
    stage.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
  }, pngB64);
  await page.waitForTimeout(2000);
  check('dropping the picture restores the workflow', /^7 nodes, 9 links/.test(await status()), await status());
  await page.screenshot({ path: path.join(OUT, 'roundtrip-after-drop.png') });

  // 4. The restored graph is the one that made the picture: same seed.
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('button:has-text("Save")')]);
  await download.saveAs(path.join(OUT, 'restored.json'));
  const restored = JSON.parse(fs.readFileSync(path.join(OUT, 'restored.json'), 'utf8')).nodes.find(n => n.type === 'KSampler');
  check('the restored KSampler has the same seed', restored.widgets_values[0] === seed, `${restored.widgets_values[0]} vs ${seed}`);
  check('no browser errors', problems.length === 0, problems.join(' | '));

  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
