// Files dropped on the canvas, on the in-page engine (the drop is handled in C# on the stage, and the file is read in JS: a picture's workflow
// is found chunk by chunk, a plain picture goes to the inputs from JS). Drops, as the browser delivers them (a DataTransfer of Files):
//   1. a PNG whose workflow tEXt chunk comes AFTER the picture data: it opens as a workflow in its own tab;
//   2. a PNG with no text: it is added as an input picture and Load Image lists it;
//   3. a workflow .json: it opens in its own tab.
// STUDIO_URL (default local :5280).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? '  [' + detail + ']' : ''}`); };

function crc32(bytes) {
  let crc = 0xFFFFFFFF;
  for (const b of bytes) { crc ^= b; for (let k = 0; k < 8; k++) crc = (crc & 1) ? (0xEDB88320 ^ (crc >>> 1)) : (crc >>> 1); }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

/** The PNG with a tEXt chunk (keyword, text) inserted just before IEND (after every IDAT). */
function withLateText(png, keyword, text) {
  const body = Buffer.concat([Buffer.from('tEXt', 'ascii'), Buffer.from(keyword + '\0', 'latin1'), Buffer.from(text, 'latin1')]);
  const len = Buffer.alloc(4); len.writeUInt32BE(body.length - 4);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body));
  return Buffer.concat([png.subarray(0, png.length - 12), len, body, crc, png.subarray(png.length - 12)]);
}

(async () => {
  const apple = fs.readFileSync(path.join(__dirname, 'apple.png'));
  const workflow = fs.readFileSync(path.join(__dirname, 'comfy-default-workflow.json'), 'utf8');
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  const tabNames = async () => (await page.locator('.tabs .tab span').allTextContents()).map(s => s.trim());
  const toast = async () => (await page.locator('.toast').count()) ? (await page.textContent('.toast')).trim() : '';
  const drop = async (name, bytes, type) => {
    const transfer = await page.evaluateHandle(({ name, b64, type }) => {
      const bin = atob(b64); const data = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) data[i] = bin.charCodeAt(i);
      const dt = new DataTransfer(); dt.items.add(new File([data], name, { type })); return dt;
    }, { name, b64: bytes.toString('base64'), type });
    await page.dispatchEvent('.stage', 'drop', { dataTransfer: transfer });
    await page.waitForTimeout(1500);
  };

  await drop('dropped-picture-workflow.png', withLateText(apple, 'workflow', workflow), 'image/png');
  check('a dropped PNG with its workflow after the picture data opens it in its own tab', (await tabNames()).includes('dropped-picture-workflow'), (await tabNames()).join(', '));

  await drop('dropped-plain.png', apple, 'image/png');
  check('a dropped PNG with no workflow is added as an input picture', /dropped-plain\.png/.test(await toast()) && /input picture/.test(await toast()), await toast());
  // a taken name, as ComfyUI's /upload/image does it (measured 2026-11-20): the same bytes reuse it, other bytes are numbered
  await drop('dropped-plain.png', apple, 'image/png');
  check('the same picture dropped again keeps its name (nothing stored twice)', /input picture \(dropped-plain\.png\)/.test(await toast()), await toast());
  await drop('dropped-plain.png', withLateText(apple, 'note', 'other bytes'), 'image/png');
  check('other bytes under that name become "dropped-plain (1).png"', /input picture \(dropped-plain \(1\)\.png\)/.test(await toast()), await toast());
  // Load Image offers the dropped picture, and the Depth map example runs on it: Load Image decodes the file the page kept in JS.
  await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
  await page.click('.menu li:has-text("Depth map")'); await page.waitForTimeout(1500);
  await page.mouse.click(400, 507); // the example fits the view the same way every time: Load Image's image combo (drive-upload.js)
  await page.waitForSelector('.widget-edit select');
  const options = await page.$$eval('.widget-edit select option', os => os.map(o => o.value));
  check('Load Image lists the dropped pictures, each once', [...options].filter(o => o.startsWith('dropped-plain')).sort().join('|') === 'dropped-plain (1).png|dropped-plain.png', options.join(', '));
  await page.selectOption('.widget-edit select', 'dropped-plain.png'); await page.waitForTimeout(500);
  await page.mouse.click(800, 250); await page.waitForTimeout(300);
  const runs = async () => parseInt(((await page.textContent('.toolbar', { timeout: 120000 })).match(/History \((\d+)\)/) || [])[1] || '0', 10);
  const before = await runs();
  await page.click('button:has-text("Queue")');
  let ran = false;
  for (let i = 0; i < 300 && !ran; i++) { await page.waitForTimeout(1000); ran = (await runs()) > before && !/Running|Queued/.test(await page.textContent('.status', { timeout: 120000 })); }
  await page.screenshot({ path: path.join(OUT, 'drop-depth.png') });
  check('the Depth map example runs on the dropped picture (see out/drop-depth.png)', ran && !/fail|error/i.test(await toast()), await toast());

  // a JPEG (no workflow chunk to look for) is a picture too: it goes to the inputs from JS
  await drop('dropped-photo.jpg', fs.readFileSync(path.join(__dirname, '..', '..', 'tests', 'SpawnGen.Nodes.Tests', 'Fixtures', 'decode_corpus', 'jpeg_420.jpg')), 'image/jpeg');
  check('a dropped JPEG is added as an input picture', /dropped-photo\.jpg was added as an input picture/.test(await toast()), await toast());
  await drop('dropped-json.json', Buffer.from(workflow, 'utf8'), 'application/json');
  check('a dropped workflow .json opens in its own tab', (await tabNames()).includes('dropped-json'), (await tabNames()).join(', '));

  await page.screenshot({ path: path.join(OUT, 'drop.png') });
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
