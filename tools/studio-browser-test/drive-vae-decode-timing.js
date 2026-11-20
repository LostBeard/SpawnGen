// One 512 x 512 VAE decode timed in the browser on the native SdVae (Load VAE: vae-ft-mse-840000-ema-pruned.safetensors) and on the ONNX
// checkpoint's VAE (sd_turbo: schmuell/sd-turbo-ort-web), alternating, RUNS each (default 5), for Tuvok's "which conv path is slow" question.
// Empty Latent Image -> Latent Multiply (a new factor every run, so nothing is cached) -> VAE Decode -> Preview Image. Prints ?nodetimes=1's
// VAE Decode and Preview Image times (a device node returns before the GPU is done: compare decode + preview) and the run total.
// PEROP=1 adds ?perop=1 (the ONNX graph's ops, each waited for). STUDIO_URL (default local :5280); models in out/site-profile.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const BASE = process.env.STUDIO_URL || 'http://localhost:5280/';
const URL = BASE + (BASE.includes('?') ? '&' : '?') + 'nodetimes=1' + (process.env.PEROP === '1' ? '&perop=1' : '');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = path.join(__dirname, 'out');
const RUNS = parseInt(process.env.RUNS || '5', 10);

function prompt(kind, factor) {
  const vae = kind === 'native'
    ? { '1': { class_type: 'VAELoader', inputs: { vae_name: 'vae-ft-mse-840000-ema-pruned.safetensors' } } }
    : { '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'sd_turbo' } } };
  return {
    ...vae,
    '2': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
    '3': { class_type: 'LatentMultiply', inputs: { samples: ['2', 0], multiplier: factor } },
    '4': { class_type: 'VAEDecode', inputs: { samples: ['3', 0], vae: ['1', kind === 'native' ? 0 : 2] } },
    '5': { class_type: 'PreviewImage', inputs: { images: ['4', 0] } },
  };
}

(async () => {
  const context = await chromium.launchPersistentContext(path.join(OUT, 'site-profile'), {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'],
  });
  const page = context.pages()[0] || await context.newPage();
  const problems = [];
  let lines = [];
  page.on('console', m => {
    const t = m.text();
    if (/^\[(nodetimes|perop)\]/.test(t)) lines.push(t);
    else if (m.type() === 'error' && !/powerPreference/.test(t)) problems.push(t.slice(0, 200));
  });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message.slice(0, 200)));
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(4000);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  const status = async () => (await page.locator('.status').first().textContent({ timeout: 60000 }).catch(() => '')) ?? '';
  const results = { native: [], onnx: [] };
  for (let r = 0; r <= RUNS; r++) {                          // run 0 of each loads the model: not counted
    for (const kind of ['native', 'onnx']) {
      const file = path.join(OUT, `vae-timing-${kind}.json`);
      fs.writeFileSync(file, JSON.stringify(prompt(kind, 1 + (r + 1) * 1e-3)));
      await page.setInputFiles('input[accept=".json,.png"]', file);
      await page.waitForTimeout(1200);
      lines = [];
      const t0 = Date.now();
      await page.locator('button:has-text("Queue")').click();
      await page.waitForTimeout(300);
      while (Date.now() - t0 < 30 * 60000) { if (!/Running|Queued/.test(await status())) break; await page.waitForTimeout(100); }
      await page.waitForTimeout(400);
      const ms = id => { const l = lines.find(x => x.startsWith(`[nodetimes] node ${id}:`)); return l ? parseFloat(l.split(':')[1]) : NaN; };
      const total = (lines.find(x => x.startsWith('[nodetimes] run:')) || '').match(/run: (\d+)/);
      const row = { decode: ms('4'), preview: ms('5'), run: total ? +total[1] : NaN };
      console.log(`${kind} run ${r}: VAE Decode ${row.decode} ms, Preview ${row.preview} ms, run ${row.run} ms${r === 0 ? ' (load, not counted)' : ''}`);
      if (r > 0) results[kind].push(row);
      if (r === RUNS && process.env.PEROP === '1' && kind === 'onnx') console.log(lines.filter(l => l.startsWith('[perop]')).join('\n'));
    }
  }
  const med = a => { const s = a.filter(x => !isNaN(x)).sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };
  for (const kind of ['native', 'onnx'])
    console.log(`SUMMARY ${kind}: decode ${med(results[kind].map(x => x.decode))} ms, preview ${med(results[kind].map(x => x.preview))} ms, decode + preview ${med(results[kind].map(x => x.decode + x.preview))} ms, run ${med(results[kind].map(x => x.run))} ms (medians of ${results[kind].length})`);
  console.log('errors: ' + (problems.length ? problems.join(' | ') : 'none'));
  await context.close();
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
