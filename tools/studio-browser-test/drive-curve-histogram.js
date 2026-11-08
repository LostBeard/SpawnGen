// The Curve Editor's histogram in the real page: Load Image (apple.png) -> Image Histogram -> Curve Editor -> Preview Any, run on the server; the plot then
// shows the picture's histogram behind the curve (ui "histogram"), and still after a second, fully cached run (has_intermediate_output).
// STUDIO_URL (default http://localhost:8188/).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

const workflow = {
  last_node_id: 4, last_link_id: 3,
  nodes: [
    { id: 1, type: 'LoadImage', pos: [40, 100], size: [260, 300], flags: {}, order: 0, mode: 0, inputs: [],
      outputs: [{ name: 'IMAGE', type: 'IMAGE', links: [1], slot_index: 0 }, { name: 'MASK', type: 'MASK', links: [] }], properties: {}, widgets_values: ['apple.png'] },
    { id: 2, type: 'ImageHistogram', pos: [340, 100], size: [220, 160], flags: {}, order: 1, mode: 0, inputs: [{ name: 'image', type: 'IMAGE', link: 1 }],
      outputs: [{ name: 'rgb', type: 'HISTOGRAM', links: [2], slot_index: 0 }], properties: {}, widgets_values: [] },
    { id: 3, type: 'CurveEditor', pos: [600, 100], size: [320, 300], flags: {}, order: 2, mode: 0, inputs: [{ name: 'histogram', type: 'HISTOGRAM', link: 2 }],
      outputs: [{ name: 'curve', type: 'CURVE', links: [3], slot_index: 0 }], properties: {}, widgets_values: [{ points: [[0, 0], [1, 1]], interpolation: 'linear' }] },
    { id: 4, type: 'PreviewAny', pos: [960, 100], size: [260, 160], flags: {}, order: 3, mode: 0, inputs: [{ name: 'source', type: '*', link: 3 }], outputs: [],
      properties: {}, widgets_values: [] },
  ],
  links: [[1, 1, 0, 2, 0, 'IMAGE'], [2, 2, 0, 3, 0, 'HISTOGRAM'], [3, 3, 0, 4, 0, '*']],
  groups: [], config: {}, extra: {}, version: 0.4,
};

(async () => {
  const wfPath = path.join(OUT, 'curve-histogram-workflow.json');
  fs.writeFileSync(wfPath, JSON.stringify(workflow));
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'Server (CUDA / OpenCL)' });
  await page.setInputFiles('input[accept*=".wav"]', path.join(__dirname, 'apple.png'));
  await page.waitForTimeout(800);
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(1200);

  // The plot: from the curve's (0, 0) point (the lowest light pixel right of the histogram node) along the drawn diagonal to (1, 1).
  const sample = async (x, y) => page.evaluate(([x, y]) => {
    const c = document.querySelector('canvas'); const r = c.getBoundingClientRect(); const k = c.width / r.width;
    return Array.from(c.getContext('2d').getImageData(Math.round((x - r.left) * k), Math.round((y - r.top) * k), 1, 1).data);
  }, [x, y]);
  const light = p => p[0] > 200 && p[1] > 200 && p[2] > 200;
  let corner = null;
  for (let y = 760; y > 300 && !corner; y -= 2) for (let x = 740; x < 1120 && !corner; x += 2) if (light(await sample(x, y))) corner = { x, y };
  let top = null;
  if (corner) {
    let x = corner.x, y = corner.y;
    for (let step = 0; step < 800; step++) {
      let moved = false;
      for (const [nx, ny] of [[x + 1, y - 1], [x + 1, y], [x, y - 1]]) if (light(await sample(nx, ny))) { x = nx; y = ny; moved = true; break; }
      if (!moved) break;
    }
    top = { x, y };
  }
  check('found the plot', !!corner && !!top && top.x - corner.x > 150, corner && top ? `(${corner.x},${corner.y})-(${top.x},${top.y})` : 'not found');
  if (!corner || !top) { await browser.close(); process.exit(1); }
  const grab = () => page.evaluate(([x0, y0, x1, y1]) => {
    const c = document.querySelector('canvas'); const r = c.getBoundingClientRect(); const k = c.width / r.width;
    const d = c.getContext('2d').getImageData(Math.round((x0 - r.left) * k), Math.round((y0 - r.top) * k), Math.round((x1 - x0) * k), Math.round((y1 - y0) * k)).data;
    return Array.from(d.filter((_, i) => i % 4 === 0));
  }, [corner.x + 2, top.y + 2, top.x - 2, corner.y - 2]);
  const changed = (a, b) => a.reduce((n, v, i) => n + (Math.abs(v - b[i]) > 6 ? 1 : 0), 0);
  const before = await grab();
  await page.screenshot({ path: path.join(OUT, 'curve-histogram-before.png') });

  const run = async () => {
    await page.locator('button:has-text("Queue")').click();
    await page.waitForTimeout(600);
    for (let i = 0; i < 120; i++) { await page.waitForTimeout(500); if (i > 3 && !/Running|Queued/.test(await page.textContent('.status'))) break; }
    await page.mouse.move(5, 880); // away from the nodes: no hover highlight
    await page.waitForTimeout(800);
  };
  await run();
  const after = await grab();
  await page.screenshot({ path: path.join(OUT, 'curve-histogram-after.png') });
  const n1 = changed(before, after);
  check('the histogram is drawn behind the curve after a run', n1 > 2000, `${n1} pixels changed`);

  // Clear it from the page's view state by reloading the workflow, then run again: everything is cached, and the histogram comes back.
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(1200);
  await run();
  const again = await grab();
  await page.screenshot({ path: path.join(OUT, 'curve-histogram-cached.png') });
  const n2 = changed(after, again);
  check('a fully cached second run shows it again', changed(before, again) > 2000 && n2 < n1 / 4, `${changed(before, again)} pixels differ from before the runs, ${n2} from the first run`);
  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await browser.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})();
