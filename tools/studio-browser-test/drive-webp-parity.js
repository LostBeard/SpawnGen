// Save Animated WEBP (lossy) writes the same picture data in the browser as on the desktop: one workflow run twice from a Studio served by
// the SpawnGen server, once on the server's engine (the desktop GPU or CPU, whose lossy files are ComfyUI's byte for byte in the unit tests)
// and once on "This browser (WebGPU)", and the two files compared chunk by chunk (the EXIF chunk left out: it carries each run's own
// workflow text). Found 2026-11-24: on an AMD RX 590, Chrome's WebGPU gave a different file on every encode (SpawnDev.Codecs issue #2: a
// shader loop stopped early on that driver; fixed in Codecs 1.1.5), and nothing here compared the browser's bytes with anything.
// The workflow uses only exact operations (Load Image, Flip Image, Batch Images), so both engines encode identical pixels.
// STUDIO_URL (default http://localhost:8188/, the server). Needs the server running and a GPU.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = path.join(__dirname, 'out', 'webp-parity');
fs.mkdirSync(OUT, { recursive: true });
// its own name in the inputs: a server's input folder may already hold another example.png (2026-11-24: a 768x768 one, and the two engines encoded different pictures)
const INPUT_NAME = 'webp-parity-input.png';
const PICTURE = path.join(__dirname, '..', '..', 'src', 'SpawnGen.Studio', 'wwwroot', 'examples', 'example.png');
let failed = 0;
const check = (ok, what) => { console.log((ok ? 'PASS ' : 'FAIL ') + what); if (!ok) failed++; };

// RIFF chunks of a WebP file as [fourcc, bytes], the EXIF chunk dropped (and its flag bit in VP8X cleared, so only that chunk's presence differs).
function chunks(file) {
  if (file.toString('latin1', 0, 4) !== 'RIFF' || file.toString('latin1', 8, 12) !== 'WEBP') throw new Error('not a WebP file');
  const out = [];
  for (let at = 12; at + 8 <= file.length;) {
    const id = file.toString('latin1', at, at + 4), size = file.readUInt32LE(at + 4);
    let data = Buffer.from(file.subarray(at + 8, at + 8 + size));
    if (id === 'VP8X') data[0] &= ~0x08;
    if (id !== 'EXIF') out.push([id, data]);
    at += 8 + size + (size & 1);
  }
  return out;
}
const sha = b => crypto.createHash('sha256').update(b).digest('hex').slice(0, 16);

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error' && !/powerPreference/.test(m.text())) problems.push(m.text().slice(0, 300)); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message.slice(0, 300)));
  await page.addInitScript(() => {
    window.__blobs = [];
    const make = URL.createObjectURL;
    URL.createObjectURL = function (b) { const u = make.call(URL, b); if (b instanceof Blob) window.__blobs.push(b); return u; };
  });
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  console.log('ADAPTER ' + await page.evaluate(async () => { const a = await navigator.gpu?.requestAdapter(); return a ? `${a.info.vendor} ${a.info.description} ${a.info.backend || ''}` : 'none'; }));
  const engines = await page.locator('select.engine option').allTextContents();
  const browserEngine = engines.find(e => /browser/i.test(e)), serverEngine = engines.find(e => !/browser/i.test(e) && !/No engine/.test(e));
  check(!!browserEngine && !!serverEngine, `both engines are offered (${engines.join(' | ')})`);
  if (!browserEngine || !serverEngine) { await browser.close(); process.exit(1); }
  const runs = async () => parseInt((((await page.locator('.toolbar').first().textContent()) || '').match(/History \((\d+)\)/) || [])[1] || '0', 10);
  const prompt = {
    '1': { class_type: 'LoadImage', inputs: { image: INPUT_NAME } },
    '2': { class_type: 'ImageFlip', inputs: { image: ['1', 0], flip_method: 'y-axis: horizontally' } },
    '3': { class_type: 'ImageBatch', inputs: { image1: ['1', 0], image2: ['2', 0] } },
    '4': { class_type: 'ImageBatch', inputs: { image1: ['3', 0], image2: ['1', 0] } },
    '5': { class_type: 'SaveAnimatedWEBP', inputs: { images: ['4', 0], filename_prefix: 'webp-parity', fps: 8, lossless: false, quality: 90, method: 'default' } },
  };
  const promptFile = path.join(OUT, 'webp-parity.prompt.json');
  fs.writeFileSync(promptFile, JSON.stringify(prompt));
  const inputFile = path.join(OUT, INPUT_NAME);
  fs.copyFileSync(PICTURE, inputFile);

  const files = {};
  for (const [which, engine] of [['server', serverEngine], ['browser', browserEngine]]) {
    await page.selectOption('select.engine', { label: engine }); await page.waitForTimeout(1500);
    // The picture into this engine's inputs through the toolbar's Upload (sent to the server, or kept in the page for the browser engine).
    // Not a drop: example.png carries a workflow, and a dropped PNG with a workflow opens as that workflow instead of becoming an input.
    await page.setInputFiles('.toolbar input[type=file][accept^=".png,.jpg"]', inputFile); await page.waitForTimeout(2500);
    await page.setInputFiles('input[accept=".json,.png"]', promptFile); await page.waitForTimeout(1500);
    const blobsBefore = await page.evaluate(() => window.__blobs.length);
    const before = await runs(), t0 = Date.now();
    await page.click('button:has-text("Queue")');
    let ran = false, toast = '';
    for (let i = 0; i < 1200 && !ran; i++) {
      await page.waitForTimeout(500);
      toast = ((await page.locator('.toast').first().textContent({ timeout: 100 }).catch(() => '')) || toast);
      if (/Run failed|not found|invalid|missing|cannot/i.test(toast)) break;
      const st = (await page.locator('.status').first().textContent().catch(() => '')) || '';
      ran = (await runs()) > before && !/Running|Queued/.test(st);
    }
    await page.waitForTimeout(1500);
    check(ran, `the workflow runs on ${engine} (${((Date.now() - t0) / 1000).toFixed(0)} s)` + (ran ? '' : ' | ' + toast.trim().slice(0, 200)));
    if (!ran) continue;
    // the saved file: the browser engine hands the page a Blob; the server's is fetched from its /view route (the newest history entry's output)
    let bytes = await page.evaluate(async from => {
      const blob = window.__blobs.slice(from).reverse().find(b => b.type === 'image/webp');
      if (!blob) return null;
      const a = new Uint8Array(await blob.arrayBuffer()); let s = '';
      for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode.apply(null, a.subarray(i, i + 0x8000));
      return btoa(s);
    }, blobsBefore);
    if (!bytes && which === 'server') bytes = await page.evaluate(async () => {
      const history = await (await fetch('history?max_items=1')).json();
      const entry = Object.values(history)[0];
      const image = Object.values(entry?.outputs || {}).flatMap(o => o.images || [])[0];
      if (!image) return null;
      const r = await fetch(`view?filename=${encodeURIComponent(image.filename)}&subfolder=${encodeURIComponent(image.subfolder || '')}&type=${image.type || 'output'}`);
      const a = new Uint8Array(await r.arrayBuffer()); let s = '';
      for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode.apply(null, a.subarray(i, i + 0x8000));
      return btoa(s);
    });
    check(!!bytes, `the ${which} engine's file was read`);
    if (!bytes) continue;
    files[which] = Buffer.from(bytes, 'base64');
    fs.writeFileSync(path.join(OUT, `webp-parity.${which}.webp`), files[which]);
  }

  if (files.server && files.browser) {
    const a = chunks(files.server), b = chunks(files.browser);
    console.log(`server:  ${files.server.length} B, chunks ${a.map(([id, d]) => `${id}:${d.length}:${sha(d)}`).join(' ')}`);
    console.log(`browser: ${files.browser.length} B, chunks ${b.map(([id, d]) => `${id}:${d.length}:${sha(d)}`).join(' ')}`);
    check(a.some(([id]) => id === 'ANMF') && a.filter(([id]) => id === 'ANMF').length === 3, 'the server wrote a 3-frame animation');
    const same = a.length === b.length && a.every(([id, d], i) => b[i][0] === id && d.equals(b[i][1]));
    check(same, 'the browser engine wrote the same picture data as the server engine, chunk for chunk (EXIF aside)');
  }
  check(problems.length === 0, 'no browser errors' + (problems.length ? ': ' + problems.slice(0, 3).join(' || ') : ''));
  await browser.close();
  console.log(failed ? `${failed} FAILED` : 'ALL PASS');
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
