// GLSL Shader in the real page on both engines: Load Image (apple.png) -> GLSL Shader (SpawnGen's features_math.frag test shader) -> Save Image, run on
// the server (OpenCL / CUDA) and on "This browser (WebGPU)", then compare the two saved pictures pixel by pixel (read from the History panel).
//   STUDIO_URL (default http://localhost:8188/), PROFILE_DIR (persistent WebGPU profile, default out/webgpu-profile).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = process.env.PROFILE_DIR || path.join(OUT, 'webgpu-profile');
const SHADER = fs.readFileSync(path.join(__dirname, '..', '..', 'tests', 'SpawnGen.Glsl.Tests', 'Fixtures', 'features_math.frag'), 'utf8');
// A comment that changes per run keeps the server from answering from its cache (it caches like ComfyUI).
const UNIQUE = `
// run ${Date.now()}
`;

const workflow = {
  last_node_id: 3, last_link_id: 2,
  nodes: [
    { id: 1, type: 'LoadImage', pos: [40, 100], size: [260, 300], flags: {}, order: 0, mode: 0, inputs: [],
      outputs: [{ name: 'IMAGE', type: 'IMAGE', links: [1], slot_index: 0 }, { name: 'MASK', type: 'MASK', links: [] }], properties: {}, widgets_values: ['apple.png'] },
    { id: 2, type: 'GLSLShader', pos: [360, 100], size: [380, 320], flags: {}, order: 1, mode: 0,
      inputs: [{ name: 'images.image0', type: 'IMAGE', link: 1 }],
      outputs: [{ name: 'IMAGE0', type: 'IMAGE', links: [2], slot_index: 0 }, { name: 'IMAGE1', type: 'IMAGE', links: [] }, { name: 'IMAGE2', type: 'IMAGE', links: [] }, { name: 'IMAGE3', type: 'IMAGE', links: [] }],
      properties: {}, widgets_values: [SHADER + UNIQUE, 'from_input'] },
    { id: 3, type: 'SaveImage', pos: [800, 100], size: [300, 300], flags: {}, order: 2, mode: 0, inputs: [{ name: 'images', type: 'IMAGE', link: 2 }], outputs: [],
      properties: {}, widgets_values: ['glsl_check'] },
  ],
  links: [[1, 1, 0, 2, 0, 'IMAGE'], [2, 2, 0, 3, 0, 'IMAGE']],
  groups: [], config: {}, extra: {}, version: 0.4,
};

(async () => {
  const wfPath = path.join(OUT, 'glsl-workflow.json');
  fs.writeFileSync(wfPath, JSON.stringify(workflow));
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

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  const pictures = {};
  for (const engine of (process.env.ENGINES || 'server,browser').split(',')) {
    await page.selectOption('select.engine', { label: engine === 'browser' ? 'This browser (WebGPU)' : 'Server (CUDA / OpenCL)' });
    await page.setInputFiles('input[accept*=".wav"]', path.join(__dirname, 'apple.png'));
    await page.waitForTimeout(800);
    await page.setInputFiles('input[accept=".json,.png"]', wfPath);
    await page.waitForTimeout(1200);
    const started = Date.now();
    page.once('console', m => {});
    await page.locator('button:has-text("Queue")').click();
    await page.waitForTimeout(600);
    const toast = await page.locator('.toast').first().textContent({ timeout: 500 }).catch(() => '');
    if (toast) console.log(`  ${engine} toast: ${toast.slice(0, 300)}`);
    for (let i = 0; i < 240; i++) {
      await page.waitForTimeout(500);
      if (i > 3 && !/Running|Queued/.test(await page.textContent('.status'))) break;
    }
    const seconds = ((Date.now() - started) / 1000).toFixed(1);
    if (!(await page.locator('.hist-thumbs img').count())) await page.click('button:has-text("History")');
    await page.waitForTimeout(800);
    pictures[engine] = await page.evaluate(async (engine) => {
      // The newest run is the first entry (after any queued ones); it must be this engine's and have finished with a picture.
      const entry = document.querySelector('.hist-list .hist-entry:not(.queued)');
      const label = entry?.querySelector('.hist-engine')?.textContent ?? '';
      if (!entry || !(engine === 'browser' ? /browser/i.test(label) : /Server/i.test(label))) return { error: 'no run on this engine; page says: ' + (document.querySelector('.toast')?.textContent ?? '') };
      if (!entry.classList.contains('done')) return { error: entry.textContent.slice(0, 200) };
      let img = null;
      for (let i = 0; i < 40 && !img; i++) { img = entry.querySelector('.hist-thumbs img'); if (!img) await new Promise(r => setTimeout(r, 250)); }
      if (!img) return { error: 'no picture' };
      await img.decode();
      const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
      const g = c.getContext('2d'); g.drawImage(img, 0, 0);
      return { w: c.width, h: c.height, data: Array.from(g.getImageData(0, 0, c.width, c.height).data) };
    }, engine);
    const ok = pictures[engine] && !pictures[engine].error;
    check(`ran on the ${engine}`, ok, ok ? `${pictures[engine].w}x${pictures[engine].h}, ${seconds} s` : pictures[engine]?.error);
    if (!ok) pictures[engine] = null;
    await page.screenshot({ path: path.join(OUT, `glsl-${engine}.png`) });
    await page.click('button:has-text("History")').catch(() => {});
  }
  const a = pictures.server, b = pictures.browser;
  if (a && b && a.w === b.w && a.h === b.h) {
    let max = 0, sum = 0;
    for (let i = 0; i < a.data.length; i++) { const d = Math.abs(a.data[i] - b.data[i]); sum += d; if (d > max) max = d; }
    check('the browser (WebGPU) picture equals the server picture', max <= 1, `largest ${max}/255, mean ${(sum / a.data.length).toFixed(3)}/255`);
  } else check('both pictures exist at the same size', false);
  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await context.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})();
