// Compare Images in the real page on the server engine: two Load Image (from Outputs) nodes (a red and a blue picture put into the server's output folder)
// into Compare Images, Queue, then hover the picture at 25 % and 75 % of its width: left of the pointer is red (A), right of it blue (B).
//   STUDIO_URL (default http://localhost:8188/); OUTPUT_DIR: the server's output folder (default %LOCALAPPDATA%/SpawnGen/output).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUTPUT_DIR = process.env.OUTPUT_DIR || path.join(process.env.LOCALAPPDATA, 'SpawnGen', 'output');

function solidPng(w, h, [r, g, b]) {
  const crc = (buf) => { let c, crcTable = []; for (let n = 0; n < 256; n++) { c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; crcTable[n] = c >>> 0; }
    let x = 0xffffffff; for (const v of buf) x = crcTable[(x ^ v) & 0xff] ^ (x >>> 8); return (x ^ 0xffffffff) >>> 0; };
  const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]);
    const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([len, td, c]); };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  const raw = Buffer.alloc((w * 3 + 1) * h); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) raw.set([r, g, b], y * (w * 3 + 1) + 1 + x * 3);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

(async () => {
  fs.writeFileSync(path.join(OUTPUT_DIR, 'compare_a.png'), solidPng(64, 64, [220, 30, 30]));
  fs.writeFileSync(path.join(OUTPUT_DIR, 'compare_b.png'), solidPng(64, 64, [30, 30, 220]));
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'compare-workflow.json'));
  await page.waitForTimeout(1200);
  await page.locator('button:has-text("Queue")').click();
  await page.waitForTimeout(4000);
  await page.mouse.move(1500, 850);
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, 'compare-idle.png') });

  // The picture is the large square inside Compare Images: find its red / blue area on the canvas.
  const sample = async (x, y) => page.evaluate(([x, y]) => {
    const c = document.querySelector('canvas'); const r = c.getBoundingClientRect();
    const ctx = c.getContext('2d'); const k = c.width / r.width;
    return Array.from(ctx.getImageData(Math.round((x - r.left) * k), Math.round((y - r.top) * k), 1, 1).data);
  }, [x, y]);
  // Scan a row through the middle of the node for the picture's extent.
  let xs = [];
  for (let x = 300; x < 1300; x += 4) { const p = await sample(x, 560); if ((p[0] > 150 && p[2] < 90) || (p[2] > 150 && p[0] < 90)) xs.push(x); }
  if (!xs.length) for (let y = 300; y < 800 && !xs.length; y += 20) for (let x = 300; x < 1300; x += 4) { const p = await sample(x, y); if ((p[0] > 150 && p[2] < 90) || (p[2] > 150 && p[0] < 90)) xs.push(x); }
  check('the picture is drawn', xs.length > 10, xs.length ? `${xs[0]}..${xs[xs.length - 1]}` : 'not found');
  if (xs.length > 10) {
    const left = xs[0], right = xs[xs.length - 1];
    let row = 560; for (let y = 300; y < 800; y += 10) { const p = await sample(left + 4, y); if (p[0] > 150 || p[2] > 150) { row = y + 20; break; } }
    for (const f of [0.25, 0.75]) {
      const px = left + (right - left) * f;
      await page.mouse.move(px, row);
      await page.waitForTimeout(300);
      const l = await sample(px - 12, row), r = await sample(px + 12, row);
      check(`hovering at ${f * 100} %: A (red) left of the split, B (blue) right of it`, l[0] > 150 && l[2] < 90 && r[2] > 150 && r[0] < 90, `left ${l.slice(0, 3)} right ${r.slice(0, 3)}`);
      await page.screenshot({ path: path.join(OUT, `compare-${f * 100}.png`) });
    }
  }
  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await browser.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})();
