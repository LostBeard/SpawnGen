// Editor speed with a big graph: opens a generated workflow of N nodes (a grid of Primitive Int -> Math chains), pans and zooms with the wheel and the mouse, and reports
// the time per canvas frame measured in the page (requestAnimationFrame deltas). Headless Chrome draws in software, so read the numbers as relative. Measured 2026-10-22:
// the interpreter build (dotnet run) needs 40 s to open 400 nodes and drops to ~2 frames per second; the AOT build on GitHub Pages opens 400 nodes in under a second and pans
// at 60 fps (STUDIO_URL=https://lostbeard.github.io/SpawnGen/). Use an AOT build when judging speed.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const N = parseInt(process.env.NODES || '400', 10);

(async () => {
  const nodes = [], links = [];
  let id = 0, link = 0;
  const cols = Math.ceil(Math.sqrt(N / 2)) * 2;
  for (let i = 0; i < N; i += 2) {
    const a = ++id, b = ++id, l = ++link;
    const col = (i / 2) % (cols / 2), row = Math.floor((i / 2) / (cols / 2));
    nodes.push({ id: a, type: 'PrimitiveInt', pos: [col * 520, row * 160], size: [200, 60], flags: {}, order: 0, mode: 0, inputs: [], outputs: [{ name: 'INT', type: 'INT', links: [l] }], properties: {}, widgets_values: [i] });
    nodes.push({ id: b, type: 'MathNode', pos: [col * 520 + 260, row * 160], size: [220, 120], flags: {}, order: 1, mode: 0, inputs: [{ name: 'a', type: 'FLOAT', link: l, widget: { name: 'a' } }], outputs: [{ name: 'float', type: 'FLOAT', links: [] }, { name: 'int', type: 'INT', links: [] }], properties: {}, widgets_values: [0, 2, 'add'] });
    links.push([l, a, 0, b, 0, 'INT']);
  }
  const wf = { last_node_id: id, last_link_id: link, nodes, links, groups: [], config: {}, extra: {}, version: 0.4 };
  fs.writeFileSync(path.join(OUT, 'perf-workflow.json'), JSON.stringify(wf));

  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); else if (/[perf]/.test(m.text())) console.log(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  const t0 = Date.now();
  await page.setInputFiles('input[accept=".json,.png"]', path.join(OUT, 'perf-workflow.json'));
  await page.waitForFunction(() => /nodes/.test(document.querySelector('.status')?.textContent || ''), null, { timeout: 60000 });
  await page.waitForTimeout(500);
  console.log(`opened ${N} nodes in ${Date.now() - t0} ms; status: ${(await page.textContent('.status')).trim()}`);

  const measure = async (name, action) => {
    await page.evaluate(() => { window.__frames = []; let last = performance.now(); const tick = t => { window.__frames.push(t - last); last = t; if (window.__measuring) requestAnimationFrame(tick); }; window.__measuring = true; requestAnimationFrame(tick); });
    await action();
    await page.evaluate(() => { window.__measuring = false; });
    const frames = await page.evaluate(() => window.__frames.slice(1));
    frames.sort((a, b) => a - b);
    const avg = frames.reduce((s, x) => s + x, 0) / Math.max(1, frames.length);
    console.log(`${name}: ${frames.length} frames, avg ${avg.toFixed(1)} ms, median ${frames[Math.floor(frames.length / 2)]?.toFixed(1)} ms, worst ${frames[frames.length - 1]?.toFixed(1)} ms`);
    return avg;
  };
  const results = [];
  results.push(await measure('fit view (everything visible)', async () => { await page.keyboard.press('f'); await page.waitForTimeout(1500); }));
  results.push(await measure('wheel zoom + pan, everything visible', async () => {
    await page.mouse.move(800, 450);
    for (let i = 0; i < 30; i++) { await page.mouse.wheel(0, i % 2 ? 120 : -120); await page.waitForTimeout(30); }
    await page.mouse.down({ button: 'middle' });
    for (let i = 0; i < 40; i++) await page.mouse.move(800 + i * 5, 450 + i * 3);
    await page.mouse.up({ button: 'middle' });
  }));
  await page.screenshot({ path: path.join(OUT, 'perf.png') });
  console.log('errors:', problems.length ? problems.join(' | ') : 'none');
  await browser.close();
  process.exit(problems.length === 0 ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
