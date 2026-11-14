// Crop Image's crop region in the real page on the server: Load Image (a 768 x 768 magenta picture put into the server's input folder) -> Crop
// Image; the picture shows without a run, and the probe's actions on ComfyUI's own editor (probe-comfy-crop.js) give ComfyUI's regions, read
// through Export. The picture is found by its colour (magenta, dimmed outside the region).
//   STUDIO_URL (default http://localhost:8188/); INPUT_DIR: the server's input folder (default %LOCALAPPDATA%/SpawnGen/input).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const INPUT_DIR = process.env.INPUT_DIR || path.join(process.env.LOCALAPPDATA, 'SpawnGen', 'input');

function solidPng(w, h, [r, g, b]) {
  const crc = (buf) => { let c, t = []; for (let n = 0; n < 256; n++) { c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
    let x = 0xffffffff; for (const v of buf) x = t[(x ^ v) & 0xff] ^ (x >>> 8); return (x ^ 0xffffffff) >>> 0; };
  const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]);
    const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([len, td, c]); };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  const raw = Buffer.alloc((w * 3 + 1) * h); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) raw.set([r, g, b], y * (w * 3 + 1) + 1 + x * 3);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

const workflow = {
  last_node_id: 2, last_link_id: 1,
  nodes: [
    { id: 1, type: 'LoadImage', pos: [20, 60], size: [260, 300], flags: {}, order: 0, mode: 0, inputs: [],
      outputs: [{ name: 'IMAGE', type: 'IMAGE', links: [1] }, { name: 'MASK', type: 'MASK', links: [] }], properties: {}, widgets_values: ['crop_test_magenta.png', 'image'] },
    { id: 2, type: 'ImageCropV2', pos: [340, 40], size: [320, 520], flags: {}, order: 1, mode: 0, inputs: [{ name: 'image', type: 'IMAGE', link: 1 }],
      outputs: [{ name: 'IMAGE', type: 'IMAGE', links: [] }], properties: {}, widgets_values: [{ x: 0, y: 0, width: 512, height: 512 }] },
  ],
  links: [[1, 1, 0, 2, 0, 'IMAGE']], groups: [], config: {}, extra: {}, version: 0.4,
};

(async () => {
  fs.mkdirSync(INPUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(INPUT_DIR, 'crop_test_magenta.png'), solidPng(768, 768, [255, 0, 255]));
  const wfPath = path.join(OUT, 'crop-workflow.json');
  fs.writeFileSync(wfPath, JSON.stringify(workflow));
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(2500);
  await page.screenshot({ path: path.join(OUT, 'crop-0-loaded.png') });

  // The picture: the bounding box of magenta pixels (bright or dimmed: red = blue, green 0).
  const found = await page.evaluate(() => {
    const c = document.querySelector('canvas'); const r = c.getBoundingClientRect(); const k = c.width / r.width;
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1, bright = 0;
    for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
      const i = (y * c.width + x) * 4; const R = d[i], G = d[i + 1], B = d[i + 2];
      if (R > 90 && G < 30 && Math.abs(R - B) < 12) { if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y; if (R > 200) bright++; }
    }
    return x1 < 0 ? null : { x: r.left + x0 / k, y: r.top + y0 / k, w: (x1 - x0 + 1) / k, h: (y1 - y0 + 1) / k, bright };
  });
  check('the input picture shows before any run', !!found && found.w > 100 && Math.abs(found.w - found.h) <= 2, JSON.stringify(found));
  if (!found) { await browser.close(); process.exit(1); }
  const at = (x, y) => ({ x: found.x + x / 768 * found.w, y: found.y + y / 768 * found.h });
  async function drag(a, b) {
    await page.mouse.move(a.x, a.y); await page.mouse.down();
    for (let i = 1; i <= 8; i++) await page.mouse.move(a.x + (b.x - a.x) * i / 8, a.y + (b.y - a.y) * i / 8);
    await page.mouse.up(); await page.waitForTimeout(200);
  }
  async function region() {
    const [download] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }), page.click('.toolbar button:has-text("Export")')]);
    const saved = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    return saved.nodes.find(n => n.type === 'ImageCropV2').widgets_values[0];
  }
  const near = (r, want, tol) => ['x', 'y', 'width', 'height'].every(k => Math.abs(r[k] - want[k]) <= tol);

  await drag(at(0.2 * 768, 0.2 * 768), at(0.35 * 768, 0.3 * 768));
  let r = await region();
  check('a drag inside moves it: ComfyUI {115, 77, 512, 512} (to a pixel: this picture is drawn at another size)', near(r, { x: 115, y: 77, width: 512, height: 512 }, 2), JSON.stringify(r));
  await drag(at(0.05 * 768, 0.05 * 768), at(0.6 * 768, 0.7 * 768));
  const r2 = await region();
  check('a drag outside does nothing', JSON.stringify(r2) === JSON.stringify(r), JSON.stringify(r2));
  // The bottom-right corner (+100, +50), no ratio: the corner moves alone.
  await drag(at(r.x + r.width, r.y + r.height), at(r.x + r.width + 100, r.y + r.height + 50));
  const r3 = await region();
  check('a corner resizes it, kept inside the picture (768)', r3.x === r.x && r3.y === r.y && r3.x + r3.width <= 768 && Math.abs(r3.height - (r.height + 50)) <= 3, JSON.stringify(r3));
  await page.screenshot({ path: path.join(OUT, 'crop-1-edited.png') });
  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await browser.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})();
