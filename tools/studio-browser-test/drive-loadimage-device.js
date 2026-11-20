// Load Image on the in-page engine's GPU decode (SpawnDev.Codecs on WebGPU, the file uploaded straight from JS), checked against the
// spec-correct pictures of the decode corpus. Each corpus file is dropped on the canvas (kept in JS), then a prompt runs
//   Load Image -> Save Image (Advanced, 16-bit PNG)            (the IMAGE)
//   Load Image -> Mask To Image -> Save Image (Advanced, 16-bit) (the MASK)
// and both PNGs are saved to out/loadimage-device/ for check_loadimage_device.py, which compares them with
// tests/SpawnGen.Nodes.Tests/Fixtures/decode_corpus/reference (PNG to the 16-bit step, JPEG exactly as libjpeg: the host decoder is 2-3/255 off
// libjpeg, the device decoder equal to it on the CPU and OpenCL, so a JPEG also tells which path ran). STUDIO_URL (default local :5280).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out', 'loadimage-device');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const CORPUS = process.env.CORPUS_DIR || path.join(__dirname, '..', '..', 'tests', 'SpawnGen.Nodes.Tests', 'Fixtures', 'decode_corpus');
const FILES = (process.env.FILES || 'png_rgb8.png,png_rgba8.png,png_palette8_trns.png,png_gray8.png,png_gray16.png,png_rgb16.png,png_rgb8_interlaced.png,' +
  'png_rgba8_orientation6.png,jpeg_420.jpg,jpeg_orientation6.jpg,jpeg_cmyk_k.jpg').split(',');

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text().slice(0, 300)); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message.slice(0, 300)));
  await page.addInitScript(() => {
    window.__blobs = [];
    const make = URL.createObjectURL;
    URL.createObjectURL = function (b) { const u = make.call(URL, b); if (b instanceof Blob) window.__blobs.push(b); return u; };
  });
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  console.log('ADAPTER', await page.evaluate(async () => { const a = await navigator.gpu?.requestAdapter(); return a ? `${a.info.vendor} ${a.info.description} ${a.info.backend || ''}` : 'none'; }));
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  const runs = async () => parseInt((((await page.locator('.toolbar').first().textContent()) || '').match(/History \((\d+)\)/) || [])[1] || '0', 10);
  const results = {};
  for (const file of FILES) {
    // the file into the page's inputs, kept in JS (a drop: Studio.OpenFileAsync -> UploadFileAsync)
    const b64 = fs.readFileSync(path.join(CORPUS, file)).toString('base64');
    const transfer = await page.evaluateHandle(({ name, b64 }) => {
      const bin = atob(b64); const data = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) data[i] = bin.charCodeAt(i);
      const dt = new DataTransfer(); dt.items.add(new File([data], name)); return dt;
    }, { name: file, b64 });
    await page.dispatchEvent('.stage', 'drop', { dataTransfer: transfer }); await page.waitForTimeout(1200);
    const stem = file.replace(/\W/g, '_');
    // one prompt per output, so the one PNG a run makes is that output (the opened tab's Load Image preview is made before Queue)
    const outputs = {
      image: { '1': { class_type: 'LoadImage', inputs: { image: file } },
               '2': { class_type: 'SaveImageAdvanced', inputs: { images: ['1', 0], filename_prefix: stem + '_image', format: 'png', 'format.bit_depth': '16-bit', 'format.input_color_space': 'sRGB' } } },
      mask: { '1': { class_type: 'LoadImage', inputs: { image: file } },
              '3': { class_type: 'MaskToImage', inputs: { mask: ['1', 1] } },
              '4': { class_type: 'SaveImageAdvanced', inputs: { images: ['3', 0], filename_prefix: stem + '_mask', format: 'png', 'format.bit_depth': '16-bit', 'format.input_color_space': 'sRGB' } } },
    };
    let done = true, made = 0, toast = '';
    for (const [what, prompt] of Object.entries(outputs)) {
      const promptFile = path.join(OUT, `${stem}.${what}.prompt.json`);
      fs.writeFileSync(promptFile, JSON.stringify(prompt));
      await page.setInputFiles('input[accept=".json,.png"]', promptFile); await page.waitForTimeout(1500);
      const blobsBefore = await page.evaluate(() => window.__blobs.length);
      const before = await runs(); await page.click('button:has-text("Queue")');
      let ran = false;
      for (let i = 0; i < 240 && !ran; i++) {
        await page.waitForTimeout(500);
        const st = (await page.locator('.status').first().textContent().catch(() => '')) || '';
        ran = (await runs()) > before && !/Running|Queued/.test(st);
      }
      await page.waitForTimeout(1500);
      const pngs = await page.evaluate(async from => Promise.all(window.__blobs.slice(from).filter(b => b.type === 'image/png').map(async b => {
        const a = new Uint8Array(await b.arrayBuffer()); let s = '';
        for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode.apply(null, a.subarray(i, i + 0x8000));
        return btoa(s);
      })), blobsBefore);
      toast = (await page.locator('.toast').count()) ? (await page.textContent('.toast')).trim() : '';
      if (pngs.length === 1) fs.writeFileSync(path.join(OUT, `${file}.${what}.png`), Buffer.from(pngs[0], 'base64'));
      done = done && ran && pngs.length === 1; made += pngs.length;
    }
    results[file] = { ran: done, pngs: made, toast: toast.slice(0, 200) };
    console.log(`${file}: ran=${done} pngs=${made}${toast ? ' | ' + toast.slice(0, 120) : ''}`);
  }
  fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify({ results, problems }, null, 1));
  console.log(problems.length ? 'BROWSER ERRORS: ' + problems.join(' || ') : 'no browser errors');
  await browser.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
