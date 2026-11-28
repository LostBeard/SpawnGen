// Painter and subgraphs in the real page (the paintings live per workflow level; the levels not on the canvas are JSON):
//  (A) a dot painted on the root's Painter, then Convert Mask to Image packed into a subgraph and opened: Queue from INSIDE must upload the
//      root's painting (it is written into the root level's JSON) - the MASK result has the dot;
//  (B) the Painter itself packed into a subgraph and opened: Clear, a dot elsewhere, back out by the breadcrumb, Queue at the root - the
//      painting made inside was uploaded on the way out, so the MASK result has the new dot and not the old one.
// Real GPU (the browser engine runs the nodes).  STUDIO_URL (default http://localhost:5300/).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5300/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

const prompt = {
  '1': { class_type: 'Painter', inputs: { mask: '', width: 512, height: 512, bg_color: '#336699' } },
  '2': { class_type: 'SaveImage', inputs: { images: ['1', 0], filename_prefix: 'painter-image' } },
  '3': { class_type: 'MaskToImage', inputs: { mask: ['1', 1] } },
  '4': { class_type: 'SaveImage', inputs: { images: ['3', 0], filename_prefix: 'painter-mask' } },
};

(async () => {
  const wfPath = path.join(OUT, 'painter-subgraph-prompt.json');
  fs.writeFileSync(wfPath, JSON.stringify(prompt));
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
  await page.addInitScript(() => {
    window.__resultBlobs = [];
    const make = URL.createObjectURL;
    URL.createObjectURL = function (b) { const u = make.call(URL, b); if (b instanceof Blob) window.__resultBlobs.push(b); return u; };
  });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text().slice(0, 300)); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message.slice(0, 300)));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const status = () => page.evaluate(() => (document.querySelector('.runbar .status') || {}).textContent || '');
  const nodes = async () => parseInt(((await page.textContent('.status')).match(/(\d+) nodes/) || [])[1] || '-1', 10);
  const run = async () => {
    await page.click('button:has-text("Queue")');
    const t0 = Date.now();
    await page.waitForTimeout(3000);
    while (Date.now() - t0 < 300000 && /Running|Queued|Loading|Starting/i.test(await status())) await page.waitForTimeout(500);
    await page.waitForTimeout(2500);
    return status();
  };
  const findCanvas = () => page.evaluate(() => {
    const c = [...document.querySelectorAll('canvas')].sort((a, b) => b.width * b.height - a.width * a.height)[0];
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    const rows = new Array(c.height).fill(0), cols = new Array(c.width).fill(0);
    for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
      const i = (y * c.width + x) * 4;
      if (d[i] === 51 && d[i + 1] === 102 && d[i + 2] === 153) { rows[y]++; cols[x]++; }
    }
    const y0 = rows.findIndex(n => n > 100), y1 = rows.length - 1 - [...rows].reverse().findIndex(n => n > 100);
    const x0 = cols.findIndex(n => n > 100), x1 = cols.length - 1 - [...cols].reverse().findIndex(n => n > 100);
    const r = c.getBoundingClientRect(), k = r.width / c.width;
    return x0 < 0 || y0 < 0 ? null : { x: r.left + x0 * k, y: r.top + y0 * k, w: (x1 - x0 + 1) * k, h: (y1 - y0 + 1) * k };
  });
  const dot = async (box, x, y) => { const K = box.w / 512; await page.mouse.move(box.x + x * K, box.y + y * K); await page.mouse.down(); await page.mouse.up(); await page.waitForTimeout(250); };
  // the MASK result of the last run (the black picture), its value at a few canvas points
  const maskAt = async (from, points) => page.evaluate(async ([from, points]) => {
    for (const b of window.__resultBlobs.slice(from).reverse()) {
      if (b.type !== 'image/png') continue;
      const bmp = await createImageBitmap(b); const c = new OffscreenCanvas(bmp.width, bmp.height); const x = c.getContext('2d'); x.drawImage(bmp, 0, 0);
      const d = x.getImageData(0, 0, bmp.width, bmp.height).data;
      if (d[(50 * bmp.width + 50) * 4 + 2] !== 0) continue; // the IMAGE (#336699 away from the paint); the MASK is black there
      return points.map(([X, Y]) => d[(Y * bmp.width + X) * 4]);
    }
    return null;
  }, [from, points]);
  const blobs = () => page.evaluate(() => window.__resultBlobs.length);
  // the node whose title text is at a screen point gets a right click; its menu's Open Subgraph
  const openSubgraphAt = async (x, y) => {
    await page.mouse.click(x, y, { button: 'right' });
    await page.waitForTimeout(300);
    if (!(await page.locator('.menu li:has-text("Open Subgraph")').count())) { await page.keyboard.press('Escape'); return false; }
    await page.click('.menu li:has-text("Open Subgraph")');
    await page.waitForTimeout(800);
    return true;
  };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(2500);
  await page.screenshot({ path: path.join(OUT, 'painter-sg-0.png') });

  // (A) a dot on the root's Painter; Convert Mask to Image packed and opened; Queue from inside
  let box = await findCanvas();
  check('(A) the root painter canvas', !!box, JSON.stringify(box));
  await dot(box, 128, 128);
  await page.mouse.click(800, 481); // Convert Mask to Image's title (painter-1-painted.png layout)
  await page.keyboard.press('Control+Shift+E');
  await page.waitForTimeout(600);
  check('(A) Convert Mask to Image packed (4 nodes stay: Painter, 2 x Save Image, the subgraph)', (await nodes()) === 4, `${await nodes()} nodes`);
  await page.screenshot({ path: path.join(OUT, 'painter-sg-1-packed.png') });
  check('(A) the subgraph opens', await openSubgraphAt(800, 481));
  check('(A) inside: breadcrumbs', await page.locator('.crumbs').count() === 1);
  let from = await blobs();
  let st = await run();
  check('(A) Queue from inside runs', !/fail|error/i.test(st), st.trim().slice(0, 120));
  // the result pictures are read after going back to the root: inside the subgraph the root's Save Image previews are not drawn, so the page
  // may not fetch them at all (2026-11-27: on the live AOT sites no picture had been fetched yet while the canvas showed the inside)
  await page.click('.crumbs button');
  await page.waitForTimeout(2500);
  check('(A) back at the root', await page.locator('.crumbs').count() === 0);
  let m = await maskAt(from, [[128, 128], [384, 384], [450, 50]]);
  check('(A) the root painting was uploaded from inside the subgraph (MASK: dot 255, elsewhere 0)', !!m && m[0] === 255 && m[1] === 0 && m[2] === 0, JSON.stringify(m));

  // (B) the Painter packed into a subgraph and opened: Clear, a dot at (384, 384), out by the breadcrumb, Queue at the root
  await page.mouse.click(320, 177); // the Painter's title
  await page.keyboard.press('Control+Shift+E');
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(OUT, 'painter-sg-2-packed.png') });
  check('(B) the Painter packed', (await nodes()) === 4, `${await nodes()} nodes`);
  check('(B) the Painter subgraph opens', await openSubgraphAt(320, 177));
  box = await findCanvas();
  check('(B) the inner painter canvas', !!box, JSON.stringify(box));
  if (box) {
    // Clear: the Brush's 8th row under the canvas (row 430 world wide = box.w)
    const s = box.w / 430;
    await page.mouse.click(box.x + box.w / 2, box.y + box.h + (4 + 7 * 28 + 12) * s);
    await page.waitForTimeout(200);
    await dot(box, 384, 384);
    await page.screenshot({ path: path.join(OUT, 'painter-sg-3-inside.png') });
  }
  await page.click('.crumbs button');
  await page.waitForTimeout(1200);
  from = await blobs();
  st = await run();
  check('(B) Queue at the root runs', !/fail|error/i.test(st), st.trim().slice(0, 120));
  m = await maskAt(from, [[128, 128], [384, 384], [450, 50]]);
  check('(B) the painting made inside was uploaded on the way out (MASK: new dot 255, cleared old dot 0)', !!m && m[0] === 0 && m[1] === 255 && m[2] === 0, JSON.stringify(m));

  check('no page errors', problems.length === 0, problems.slice(0, 5).join(' || '));
  await browser.close();
  const passed = results.filter(Boolean).length;
  console.log(`${passed}/${results.length} passed`);
  process.exit(passed === results.length ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
