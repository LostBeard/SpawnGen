// Sound in the real page, on the in-page engine (no GPU, no model): upload a WAV, Load Audio -> Save Audio, the History panel offers an audio player for
// the saved file and the node says which file it wrote. A tone is made here and uploaded.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

function wav(seconds, rate, freq) {
  const n = Math.floor(seconds * rate), data = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) data.writeInt16LE(Math.round(Math.sin(2 * Math.PI * freq * i / rate) * 12000), i * 2);
  const h = Buffer.alloc(44);
  h.write('RIFF', 0); h.writeUInt32LE(36 + data.length, 4); h.write('WAVEfmt ', 8); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22);
  h.writeUInt32LE(rate, 24); h.writeUInt32LE(rate * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34); h.write('data', 36); h.writeUInt32LE(data.length, 40);
  return Buffer.concat([h, data]);
}

(async () => {
  fs.writeFileSync(path.join(OUT, 'tone.wav'), wav(0.5, 16000, 440));
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu', '--enable-unsafe-webgpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  await page.setInputFiles('input[accept=".png,.jpg,.jpeg,.wav"]', path.join(OUT, 'tone.wav'));
  await page.waitForTimeout(800);
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'audio-workflow.json'));
  await page.waitForTimeout(1200);
  await page.click('button:has-text("History")');
  await page.waitForSelector('.history');
  await page.locator('button:has-text("Queue")').click();
  for (let i = 0; i < 40; i++) { await page.waitForTimeout(500); if (i > 2 && await page.locator('button:has-text("Queue")').isVisible()) break; }
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, 'audio.png') });
  const players = await page.locator('.hist-entry audio').count();
  check('the history entry has an audio player', players === 1, `${players}`);
  const src = await page.locator('.hist-entry audio').first().getAttribute('src');
  check('it plays a WAV data URL of the saved sound', /^data:audio\/wav;base64,UklGR/.test(src || ''), (src || '').slice(0, 40));
  const duration = await page.locator('.hist-entry audio').first().evaluate(a => new Promise(r => { if (a.duration) return r(a.duration); a.onloadedmetadata = () => r(a.duration); a.load(); setTimeout(() => r(-1), 3000); }));
  check('the browser decodes it: half a second long', Math.abs(duration - 0.5) < 0.02, String(duration));
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
