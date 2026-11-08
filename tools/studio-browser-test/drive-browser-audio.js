// Save Audio's FLAC on both engines in the real page: uploads a 1 s stereo tone (WAV), queues Load Audio -> Save Audio on each engine, then checks the
// History player loads the result (FLAC, about 1 s) and that the browser's file (SpawnDev.Codecs on WebGPU) has the same FLAC frames as the server's.
// FORMAT=opus runs Save Audio (Opus) at 128k instead (Ogg Opus; the serial is random per file, so only the player check applies).
// STUDIO_URL (default http://localhost:8188/), ENGINES (default server,browser).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const STUDIO = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = process.env.PROFILE_DIR || path.join(__dirname, 'out', 'webgpu-profile');
const OUT = path.join(__dirname, 'out');
const FORMAT = process.env.FORMAT || 'flac';
fs.mkdirSync(OUT, { recursive: true });

function toneWav() {
  const rate = 44100, n = rate, ch = 2, data = Buffer.alloc(n * ch * 2);
  for (let i = 0; i < n; i++)
    for (let c = 0; c < ch; c++) data.writeInt16LE(Math.round(12000 * Math.sin(2 * Math.PI * (c ? 495 : 330) * i / rate)), (i * ch + c) * 2);
  const h = Buffer.alloc(44);
  h.write('RIFF', 0); h.writeUInt32LE(36 + data.length, 4); h.write('WAVE', 8); h.write('fmt ', 12); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20);
  h.writeUInt16LE(ch, 22); h.writeUInt32LE(rate, 24); h.writeUInt32LE(rate * ch * 2, 28); h.writeUInt16LE(ch * 2, 32); h.writeUInt16LE(16, 34);
  h.write('data', 36); h.writeUInt32LE(data.length, 40);
  return Buffer.concat([h, data]);
}

// Where the FLAC frames start: after the last metadata block (the prompt comment differs per run, the audio must not).
function framesStart(b) {
  let i = 4;
  for (;;) {
    const last = b[i] & 0x80, len = b.readUIntBE(i + 1, 3);
    i += 4 + len;
    if (last) return i;
  }
}

(async () => {
  const wavPath = path.join(OUT, 'drive_tone.wav');
  fs.writeFileSync(wavPath, toneWav());
  const context = await chromium.launchPersistentContext(PROFILE, {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'],
  });
  const page = context.pages()[0] || await context.newPage();
  const problems = [];
  page.on('console', m => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(STUDIO);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  const files = {};
  for (const engine of (process.env.ENGINES || 'server,browser').split(',')) {
    await page.selectOption('select.engine', { label: engine === 'browser' ? 'This browser (WebGPU)' : 'Server (CUDA / OpenCL)' });
    await page.setInputFiles('input[accept*=".wav"]', wavPath);
    await page.waitForTimeout(1000);
    // A unique prefix per run: the server cannot answer from its cache, and the newest History entry is this run's.
    const prompt = {
      '1': { class_type: 'LoadAudio', inputs: { audio: 'drive_tone.wav' } },
      '2': FORMAT === 'opus'
        ? { class_type: 'SaveAudioOpus', inputs: { audio: ['1', 0], filename_prefix: `audio/drive_${engine}_${Date.now()}`, quality: '128k' } }
        : { class_type: 'SaveAudio', inputs: { audio: ['1', 0], filename_prefix: `audio/drive_${engine}_${Date.now()}` } },
    };
    const promptPath = path.join(OUT, `audio-prompt-${engine}.json`);
    fs.writeFileSync(promptPath, JSON.stringify(prompt));
    await page.setInputFiles('input[accept=".json,.png"]', promptPath);
    await page.waitForTimeout(1200);
    if (!(await page.locator('.hist-entry').count())) await page.click('button:has-text("History")');
    const started = Date.now();
    await page.locator('button:has-text("Queue")').click();
    for (let i = 0; i < 600; i++) { await page.waitForTimeout(500); if (i > 3 && !/Running|Queued/.test(await page.textContent('.status'))) break; }
    await page.waitForTimeout(1200);
    if (!(await page.locator('.hist-entry').count())) await page.click('button:has-text("History")');
    await page.waitForTimeout(600);
    const entry = page.locator('.hist-entry').first();
    const label = await entry.locator('.hist-engine').textContent().catch(() => '');
    const src = await entry.locator('audio').first().getAttribute('src', { timeout: 5000 }).catch(() => null);
    const okEngine = engine === 'browser' ? /browser/i.test(label) : /Server/i.test(label);
    if (!src || !okEngine) {
      check(`ran on the ${engine}`, false, `engine label "${label}", src ${src ? 'present' : 'missing'}; error: ${await page.locator('.hist-error').first().textContent().catch(() => 'none')}`);
      continue;
    }
    // The player loads it: Chrome decodes the FLAC and reports its length.
    const duration = await entry.locator('audio').first().evaluate(a => new Promise(res => {
      if (a.readyState >= 1) return res(a.duration);
      a.addEventListener('loadedmetadata', () => res(a.duration), { once: true });
      a.addEventListener('error', () => res(-1), { once: true });
      a.load();
      setTimeout(() => res(-2), 8000);
    }));
    const bytes = src.startsWith('data:') ? Buffer.from(src.split(',')[1], 'base64')
      : Buffer.from(await (await page.request.get(new URL(src, STUDIO).href)).body());
    files[engine] = bytes;
    fs.writeFileSync(path.join(OUT, `drive-audio-${engine}.${FORMAT}`), bytes);
    const magic = FORMAT === 'opus' ? 'OggS' : 'fLaC';
    check(`ran on the ${engine}: ${FORMAT} the player loads`, bytes.subarray(0, 4).toString() === magic && Math.abs(duration - 1) < 0.05,
      `${bytes.length} bytes, ${Number(duration).toFixed(3)} s, ${((Date.now() - started) / 1000).toFixed(1)} s to run, src ${src.slice(0, 24)}`);
  }
  if (FORMAT === 'flac' && files.server && files.browser) {
    const a = files.server.subarray(framesStart(files.server)), b = files.browser.subarray(framesStart(files.browser));
    check('the browser encodes the same FLAC frames as the server', a.equals(b), `${a.length} vs ${b.length} bytes of frames`);
  }
  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await context.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
