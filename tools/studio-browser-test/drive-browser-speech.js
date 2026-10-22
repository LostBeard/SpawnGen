// Kokoro text to speech in the BROWSER on WebGPU: queues Kokoro Text to Speech -> Save Audio on "This browser", then reads the sound the History panel offers
// (length, loudness) and saves it as out/browser-speech.wav. The model (330 MB) downloads on first use; keep PROFILE_DIR to reuse it.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const URL = process.env.STUDIO_URL || 'http://localhost:5300/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = process.env.PROFILE_DIR || path.join(__dirname, 'out', 'webgpu-profile');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const context = await chromium.launchPersistentContext(PROFILE, {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'],
  });
  const page = context.pages()[0] || await context.newPage();
  const problems = [];
  const started = Date.now();
  const t = () => `+${((Date.now() - started) / 1000).toFixed(0)}s`;
  page.on('console', m => { if (m.type() === 'error') { problems.push(m.text()); console.log(t(), 'console error:', m.text().slice(0, 300)); } });
  page.on('pageerror', e => { problems.push('PAGEERROR ' + e.message); console.log(t(), 'PAGEERROR', e.message.slice(0, 300)); });
  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  const prompt = {
    '1': { class_type: 'KokoroTextToSpeech', inputs: { text: 'Hello from SpawnGen. This is a test of speech.', voice: 'af_heart', speed: 1 } },
    '2': { class_type: 'SaveAudio', inputs: { audio: ['1', 0], filename_prefix: 'speech' } },
  };
  const promptPath = path.join(OUT, 'speech-prompt.json');
  fs.writeFileSync(promptPath, JSON.stringify(prompt));
  await page.setInputFiles('input[accept=".json,.png"]', promptPath);
  await page.waitForTimeout(1500);
  await page.click('button:has-text("History")');
  await page.locator('button:has-text("Queue")').click();
  let last = '';
  for (let i = 0; i < 1800; i++) {
    await page.waitForTimeout(1000);
    const s = (await page.textContent('.status')).trim().split(' \u00b7 ')[0];
    if (s !== last && /Running|Queued/.test(s)) { console.log(t(), s); last = s; }
    if (i > 5 && !/Running|Queued/.test(s)) break;
  }
  await page.waitForTimeout(1500);
  const src = await page.locator('.hist-entry audio').first().getAttribute('src', { timeout: 5000 }).catch(() => null);
  if (!src) { console.log(t(), 'no audio in the history; status:', (await page.textContent('.status')).trim(), '| error:', await page.locator('.hist-error').first().textContent().catch(() => 'none')); }
  else {
    const b64 = src.split(',')[1];
    fs.writeFileSync(path.join(OUT, 'browser-speech.wav'), Buffer.from(b64, 'base64'));
    const wav = Buffer.from(b64, 'base64');
    const rate = wav.readUInt32LE(24), samples = (wav.length - 44) / 2;
    let sum = 0; for (let i = 0; i < samples; i++) { const v = wav.readInt16LE(44 + i * 2) / 32768; sum += v * v; }
    console.log(t(), `saved out/browser-speech.wav: ${rate} Hz, ${(samples / rate).toFixed(2)} s, rms ${Math.sqrt(sum / samples).toFixed(3)}`);
  }
  console.log('errors:', problems.length ? problems.join('\n') : 'none');
  await context.close();
  process.exit(src && problems.length === 0 ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
