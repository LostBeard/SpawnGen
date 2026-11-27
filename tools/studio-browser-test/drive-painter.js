// Painter in the real page: a Painter node (Background #336699) with Save Image on its IMAGE and on its MASK (through Convert Mask to Image);
// real pointer input paints a dot and a line with the default brush (size 20, hard, white, 100 %), the Eraser takes a dot out of the line, and
// the workflow runs. Checked: the mask input names the upload ("painter-<id>-<ms>.png [input]"), the MASK result is the painting's alpha
// where it was painted (dot and line opaque, the erased gap 0, untouched 0), the IMAGE is white there and #336699 elsewhere, a second run
// without new strokes uploads nothing new, and no page errors. Real GPU (the browser engine runs the nodes).
//   STUDIO_URL (default http://localhost:5300/).
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
  const wfPath = path.join(OUT, 'painter-prompt.json');
  fs.writeFileSync(wfPath, JSON.stringify(prompt));
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true, deviceScaleFactor: 1 });
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
  const run = async () => {
    await page.click('button:has-text("Queue")');
    const t0 = Date.now();
    await page.waitForTimeout(3000);
    while (Date.now() - t0 < 300000 && /Running|Queued|Loading|Starting/i.test(await status())) await page.waitForTimeout(500);
    await page.waitForTimeout(2500);
    return status();
  };
  const exported = async () => {
    const [download] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }),
      page.evaluate(() => [...document.querySelectorAll('.toolbar button')].find(b => b.textContent.trim() === 'Export').click())]);
    const wf = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    return wf.nodes.find(n => n.type === 'Painter');
  };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(2500);

  // the painter's canvas on screen: the box of #336699 pixels in the editor's own canvas (the Background under an empty painting)
  const findCanvas = () => page.evaluate(() => {
    const c = [...document.querySelectorAll('canvas')].sort((a, b) => b.width * b.height - a.width * a.height)[0];
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    // rows and columns with many #336699 pixels: the canvas (the Background row's small swatch has the colour too, and is left out)
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
  const box = await findCanvas();
  check('the painter canvas is drawn (Background #336699)', !!box && Math.abs(box.w - box.h) < 3, JSON.stringify(box));
  if (!box) { await page.screenshot({ path: path.join(OUT, 'painter-0.png') }); console.log('problems: ' + problems.slice(0, 5).join(' || ')); await browser.close(); process.exit(1); }
  const K = box.w / 512;
  const at = (x, y) => [box.x + x * K, box.y + y * K];

  // a dot at (128, 128), a line from (100, 300) to (400, 300), then the Eraser through the line at (250, 300)
  await page.mouse.move(...at(128, 128)); await page.mouse.down(); await page.mouse.up(); await page.waitForTimeout(200);
  await page.mouse.move(...at(100, 300)); await page.mouse.down(); await page.mouse.move(...at(400, 300), { steps: 20 }); await page.mouse.up(); await page.waitForTimeout(200);
  const afterStrokes = await findCanvas();
  check('strokes are drawn on the canvas (less Background showing, same box)', !!afterStrokes, JSON.stringify(afterStrokes));
  // the Tool row's Eraser: the first row under the canvas box (its centre 4 + 12 world units below it; the row is 430 world units wide, as the
  // box), the right half of its control (which starts at 30 % of the row)
  await page.mouse.click(box.x + box.w * (0.3 + 0.7 * 0.75), box.y + box.h + 16 * (box.w / 430));
  await page.waitForTimeout(300);
  await page.mouse.move(...at(250, 300)); await page.mouse.down(); await page.mouse.up(); await page.waitForTimeout(200);
  await page.screenshot({ path: path.join(OUT, 'painter-1-painted.png') });

  const before = await exported();
  check('a stroke changes no widget value (mask still empty)', (before.widgets_values || [])[0] === '', JSON.stringify(before.widgets_values));

  const blobsBefore = await page.evaluate(() => window.__resultBlobs.length);
  const st = await run();
  check('the run finished', !/fail|error/i.test(st), st.trim().slice(0, 160));
  const after = await exported();
  const mask = (after.widgets_values || [])[0] || '';
  check('mask names the upload', /^painter-\d+-\d+\.png \[input\]$/.test(mask), mask);

  // the run's pictures, decoded in the page: [IMAGE, MASK] in the order the Save Image nodes ran (told apart by their content)
  const pics = await page.evaluate(async from => {
    const out = [];
    for (const b of window.__resultBlobs.slice(from)) {
      if (!/^image\/png$/.test(b.type)) continue;
      const bmp = await createImageBitmap(b);
      const c = new OffscreenCanvas(bmp.width, bmp.height); const x = c.getContext('2d'); x.drawImage(bmp, 0, 0);
      const d = x.getImageData(0, 0, bmp.width, bmp.height).data;
      const px = (X, Y) => { const i = (Y * bmp.width + X) * 4; return [d[i], d[i + 1], d[i + 2]]; };
      out.push({ w: bmp.width, h: bmp.height, dot: px(128, 128), line: px(150, 300), lineEnd: px(395, 300), erased: px(250, 300), empty: px(450, 50), justOutsideDot: px(140, 128) });
    }
    return out;
  }, blobsBefore);
  console.log('pictures: ' + JSON.stringify(pics));
  const image = pics.find(p => p.empty[0] === 51 && p.empty[2] === 153), maskPic = pics.find(p => p.empty[0] === 0 && p.empty[1] === 0 && p.empty[2] === 0);
  check('IMAGE: white where painted, #336699 elsewhere, the erased gap shows the Background', !!image && image.w === 512 && image.dot.join() === '255,255,255'
    && image.line.join() === '255,255,255' && image.erased.join() === '51,102,153' && image.justOutsideDot.join() === '51,102,153', JSON.stringify(image));
  check('MASK: 255 where painted, 0 erased and untouched', !!maskPic && maskPic.dot[0] === 255 && maskPic.line[0] === 255 && maskPic.lineEnd[0] === 255
    && maskPic.erased[0] === 0 && maskPic.justOutsideDot[0] === 0, JSON.stringify(maskPic));

  // a second run without new strokes: the same file, no new upload
  await run();
  const again = ((await exported()).widgets_values || [])[0];
  check('a second run without strokes keeps the same upload', again === mask, again);

  // ComfyUI parity of a stamp: Clear, Cursor Size 50 and Hardness 50 through the sliders, one click at canvas (256, 256), run; the MASK must be
  // the alpha ComfyUI's own widget painted for the same click (out/painter-brush/dot-size50-hard050-b.png), within Save Image's float floor.
  const s = box.w / 430, rowY = i => box.y + box.h + (4 + i * 28 + 12) * s;   // row i's centre (0 Tool, 1 Cursor Size, 2 colour, 3 Hardness)
  const trackX = f => box.x + (129 + 6 + f * 243) * s;                          // a slider's track: from 30 % of the row + 6, 243 wide
  await page.mouse.click(box.x + (129 + 301 * 0.25) * s, rowY(0));              // back to the Brush (the Eraser hides colour and Hardness)
  await page.waitForTimeout(200);
  await page.mouse.click(box.x + box.w / 2, rowY(7));                           // Clear (the Brush's 8th row)
  await page.mouse.click(trackX(49 / 199), rowY(1));                            // Cursor Size 50
  await page.mouse.click(trackX(0.5), rowY(3));                                 // Hardness 50
  await page.mouse.move(...at(256, 256)); await page.mouse.down(); await page.mouse.up(); await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(OUT, 'painter-2-dot.png') });
  const from2 = await page.evaluate(() => window.__resultBlobs.length);
  await run();
  const fixture = fs.readFileSync(path.join(__dirname, '..', '..', 'tests', 'SpawnGen.Editor.Tests', 'Fixtures', 'painter-brush', 'dot-size50-hard050-b.png')).toString('base64');
  const compare = (from) => page.evaluate(async ([from, fx]) => {
    const pixels = async blob => { const bmp = await createImageBitmap(blob); const c = new OffscreenCanvas(bmp.width, bmp.height); const x = c.getContext('2d'); x.drawImage(bmp, 0, 0); return x.getImageData(0, 0, bmp.width, bmp.height).data; };
    const want = await pixels(await (await fetch('data:image/png;base64,' + fx)).blob());
    let best = null;
    for (const b of window.__resultBlobs.slice(from)) {
      if (b.type !== 'image/png') continue;
      const got = await pixels(b);
      if (got[(50 * 512 + 50) * 4] !== 0) continue; // the MASK picture is black away from the dot
      let worst = 0, off = 0, painted = 0, sx = 0, sy = 0, sw = 0, wx = 0, wy = 0, ww = 0, gx0 = 512, gx1 = 0;
      for (let i = 0; i < 512 * 512; i++) {
        const a = want[i * 4 + 3], g = got[i * 4], x = i % 512 + 0.5, y = Math.floor(i / 512) + 0.5;
        if (a || g) painted++; const d = Math.abs(a - g); if (d) off++; if (d > worst) worst = d;
        sx += g * x; sy += g * y; sw += g; wx += a * x; wy += a * y; ww += a;
        if (g && Math.floor(i / 512) === 256) { gx0 = Math.min(gx0, i % 512); gx1 = Math.max(gx1, i % 512); }
      }
      // where the two dots' centres are (alpha-weighted), and our dot's extent on row 256 (ComfyUI's: 225..286)
      best = { worst, off, painted, ours: [+(sx / sw).toFixed(3), +(sy / sw).toFixed(3)], comfy: [+(wx / ww).toFixed(3), +(wy / ww).toFixed(3)], row256: [gx0, gx1] };
    }
    return best;
  }, [from, fixture]);
  let cmp = await compare(from2);
  console.log('first dot: ' + JSON.stringify(cmp));
  // The canvas sits at a fractional screen position the pixel search cannot see: the first dot's centre says by how much. The click is
  // repeated with that offset taken out, so the stamp lands on the pixel corner (256, 256) as the ComfyUI probe's click did.
  if (cmp && (Math.abs(cmp.ours[0] - 256) > 0.01 || Math.abs(cmp.ours[1] - 256) > 0.01)) {
    const [dx, dy] = [cmp.ours[0] - 256, cmp.ours[1] - 256];
    await page.mouse.click(box.x + box.w / 2, rowY(7)); // Clear
    const [px, py] = at(256 - dx, 256 - dy);
    await page.mouse.move(px, py); await page.mouse.down(); await page.mouse.up(); await page.waitForTimeout(300);
    const from3 = await page.evaluate(() => window.__resultBlobs.length);
    await run();
    cmp = await compare(from3);
  }
  check('a soft stamp matches ComfyUI\'s own (size 50, hardness 50, at 256,256)', !!cmp && cmp.painted > 2900 && cmp.worst <= 1, JSON.stringify(cmp));

  check('no page errors', problems.length === 0, problems.slice(0, 5).join(' || '));
  await browser.close();
  const passed = results.filter(Boolean).length;
  console.log(`${passed}/${results.length} passed`);
  process.exit(passed === results.length ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
