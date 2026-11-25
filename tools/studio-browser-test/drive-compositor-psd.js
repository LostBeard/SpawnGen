// Export PSD of the compositor editor in the real page (real GPU: the PSD writer runs as kernels on the browser engine's WebGPU device).
// Same graph as drive-compositor.js; the top layer gets Multiply, a 30 degree rotation, width 300, a horizontal flip and a Unicode name is
// already in the graph; the Background is shown; then the PSD button is pressed and the downloaded file is read back with psd-dump.py and
// checked against what ComfyUI's own export was measured to write for the same edit (probe-comfy-compositor-psd.js): the records, their
// rectangles, keys, opacities, flags, names, channel order and compression. With psd-tools installed for PYTHON, psd_readback.py also
// checks every layer's pixels are readable by an independent reader.
//   STUDIO_URL (default http://localhost:5280/), PYTHON (default python). Output: out/spawngen-compositor-export.psd.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PYTHON = process.env.PYTHON || 'python';

const prompt = {
  '1': { class_type: 'LoadImage', inputs: { image: 'example.png' } },
  '2': { class_type: 'AddLayer', inputs: { image: ['1', 0], name: 'coat' } },
  '3': { class_type: 'AddLayer', inputs: { image: ['1', 0], layers: ['2', 0], name: 'Ünïcødé 層 ✓', x: 120, y: 60, opacity: 0.8 } },
  '4': { class_type: 'ImageCompositor', inputs: { layers: ['3', 0], compositor: {} } },
};

(async () => {
  const wfPath = path.join(OUT, 'compositor-psd-prompt.json');
  fs.writeFileSync(wfPath, JSON.stringify(prompt));
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const status = () => page.evaluate(() => (document.querySelector('.runbar .status') || {}).textContent || '');

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(2000);
  await page.click('button:has-text("Queue")');
  const t0 = Date.now();
  await page.waitForTimeout(3000);
  while (Date.now() - t0 < 240000 && /Running|Queued|Loading|Starting/i.test(await status())) await page.waitForTimeout(500);
  await page.waitForTimeout(2500);
  await page.mouse.click(1445, 429); // the node's Open Compositor row (drive-compositor.js)
  await page.waitForSelector('[data-testid="compositor"]', { timeout: 10000 });
  await page.waitForTimeout(800);

  const exportPsd = async file => {
    const [download] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), page.click('[data-testid="compositor-export-psd"]')]);
    await download.saveAs(file);
    return download.suggestedFilename();
  };
  const dump = file => execFileSync(PYTHON, [path.join(__dirname, 'psd-dump.py'), file], { env: { ...process.env, PYTHONIOENCODING: 'utf-8' } }).toString('utf8');
  const layersOf = text => {
    const out = [];
    const lines = text.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      const m = lines[i].match(/^\s+layer (\d+): "(.*)" rect left (-?\d+) top (-?\d+) right (-?\d+) bottom (-?\d+) \((\d+)x(\d+)\) channels \[(.*)\]/);
      if (!m) continue;
      const b = lines[i + 1].match(/blend b'(.{4})' opacity (\d+) clipping (\d+) flags (0x[0-9a-f]+)/);
      let luni = null;
      for (let k = i + 2; k < lines.length && /^\s+info /.test(lines[k]); k++) { const u = lines[k].match(/info luni \(\d+ bytes\): (.*)$/); if (u) luni = u[1].replace(/\0/g, '').trim(); }
      out.push({ name: m[2], left: +m[3], top: +m[4], right: +m[5], bottom: +m[6], channels: [...m[9].matchAll(/\((-?\d+),/g)].map(c => +c[1]), key: b[1], opacity: +b[2], flags: b[4], luni });
    }
    return out;
  };

  // 1. the unedited composition (ComfyUI's plain export: Background hidden 0x0a, coat 0,0..512,512, the top layer 120,60..632,572 at 204)
  const plain = path.join(OUT, 'spawngen-compositor-export-plain.psd');
  const name = await exportPsd(plain);
  check('Export PSD downloads a file named spawngen-layers-<date>-<time>.psd', /^spawngen-layers-\d{8}-\d{6}\.psd$/.test(name), name);
  let text = dump(plain);
  let L = layersOf(text);
  check('header: 8BPS version 1, RGB, 8 bit, the canvas size', /header: b'8BPS' version 1 channels [34] 632x572 depth 8 mode 3/.test(text), text.split(/\r?\n/)[0]);
  check('three records, bottom first: Background, coat, the top layer (its name whole in Unicode)', L.length === 3 && L[0].luni === 'Background' && L[1].luni === 'coat' && L[2].luni === 'Ünïcødé 層 ✓', L.map(l => l.luni).join(' | '));
  check('the hidden Background has flags 0x0a and covers the canvas', L[0] && L[0].flags === '0x0a' && L[0].right === 632 && L[0].bottom === 572 && L[0].key === 'norm' && L[0].opacity === 255, JSON.stringify(L[0]));
  check('the layers sit where the run placed them, with their opacity as a byte', L[1] && L[2] && L[1].left === 0 && L[1].top === 0 && L[1].right === 512 && L[1].bottom === 512 && L[1].opacity === 255 && L[1].flags === '0x08'
    && L[2].left === 120 && L[2].top === 60 && L[2].right === 632 && L[2].bottom === 572 && L[2].opacity === 204 && L[2].key === 'norm', JSON.stringify(L.slice(1)));
  check('picture layers store alpha first, then red, green, blue; all RLE', L[1] && JSON.stringify(L[1].channels) === '[-1,0,1,2]' && !/channel compressions: \[[^\]]*0[^\]]*\]/.test(text) && /composite image: compression 1/.test(text));

  // 2. the edit ComfyUI's export was measured with: Multiply, 30 degrees, width 300, flipped, Background shown
  await page.selectOption('[data-testid="compositor-blend"]', 'multiply'); await page.waitForTimeout(200);
  const type = async (id, value) => { await page.fill(`[data-testid="${id}"]`, value); await page.press(`[data-testid="${id}"]`, 'Enter'); await page.waitForTimeout(200); };
  await type('compositor-rotation', '30'); await type('compositor-w', '300');
  await page.click('[data-testid="compositor"] button[title="Flip horizontal"]'); await page.waitForTimeout(200);
  await page.click('[data-testid="compositor-background"] button[title="Toggle visibility"]'); await page.waitForTimeout(300);
  const edited = path.join(OUT, 'spawngen-compositor-export.psd');
  await exportPsd(edited);
  text = dump(edited);
  L = layersOf(text);
  // ComfyUI's file for this edit: "apple" rect left 12 top 19 right 528 bottom 613 (516x594), blend 'mul ', opacity 204; Background flags 0x08
  check('the turned layer is stored in the bounding box ComfyUI stores: 12, 19 to 528, 613', L[2] && L[2].left === 12 && L[2].top === 19 && L[2].right === 528 && L[2].bottom === 613, JSON.stringify(L[2]));
  check('Multiply is written as "mul " at opacity 204', L[2] && L[2].key === 'mul ' && L[2].opacity === 204);
  check('the shown Background has flags 0x08 and the flattened picture has no transparency (3 channels)', L[0] && L[0].flags === '0x08' && /header: b'8BPS' version 1 channels 3 /.test(text), (L[0] || {}).flags);
  await page.screenshot({ path: path.join(OUT, 'compositor-psd-editor.png') });

  // 3. an independent reader, when it is installed
  let readback = 'psd-tools is not installed for ' + PYTHON + ': layer pixels not read back by an independent reader';
  try {
    readback = execFileSync(PYTHON, [path.join(__dirname, 'psd-readback.py'), edited, path.join(OUT, 'spawngen-compositor-export-layers')], { env: { ...process.env, PYTHONIOENCODING: 'utf-8' } }).toString('utf8').trim();
    check('psd-tools reads every layer and the flattened picture', /^OK /m.test(readback), readback.split(/\r?\n/).slice(-1)[0]);
  } catch (e) {
    if (e.status === 3) console.log('NOTE ' + readback);
    else check('psd-tools reads every layer and the flattened picture', false, String(e.stdout || e.message).slice(0, 300));
  }
  check('no browser errors', problems.length === 0, problems.slice(0, 3).join(' || ').slice(0, 300));
  console.log(`${results.filter(Boolean).length}/${results.length} passed`);
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
