// The CURVE widget in the real page: a Curve Editor node shows its curve as a plot; pressing on empty space adds a point and dragging moves it; Export writes
// the points in ComfyUI's {"points": [...], "interpolation": ...} form. STUDIO_URL (default http://localhost:8188/).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

const workflow = {
  last_node_id: 1, last_link_id: 0,
  nodes: [{ id: 1, type: 'CurveEditor', pos: [500, 200], size: [320, 300], flags: {}, order: 0, mode: 0, inputs: [],
    outputs: [{ name: 'curve', type: 'CURVE', links: [] }], properties: {}, widgets_values: [{ points: [[0, 0], [1, 1]], interpolation: 'monotone_cubic' }] }],
  links: [], groups: [], config: {}, extra: {}, version: 0.4,
};

(async () => {
  const wfPath = path.join(OUT, 'curve-workflow.json');
  fs.writeFileSync(wfPath, JSON.stringify(workflow));
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', wfPath);
  await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(OUT, 'curve-before.png') });

  // Find the plot: scan for the curve's light point at the plot's bottom-left corner (0, 0).
  const canvas = await page.locator('canvas').boundingBox();
  const sample = async (x, y) => page.evaluate(([x, y]) => {
    const c = document.querySelector('canvas'); const r = c.getBoundingClientRect(); const k = c.width / r.width;
    return Array.from(c.getContext('2d').getImageData(Math.round((x - r.left) * k), Math.round((y - r.top) * k), 1, 1).data);
  }, [x, y]);
  let corner = null, top = null;
  for (let y = canvas.y + 700; y > canvas.y + 100 && !corner; y -= 2)
    for (let x = canvas.x + 300; x < canvas.x + 900 && !corner; x += 2) { const p = await sample(x, y); if (p[0] > 225 && p[1] > 225 && p[2] > 225) corner = { x, y }; }
  // The (1, 1) point: follow the drawn diagonal up and right from the corner while light pixels continue.
  if (corner) {
    let x = corner.x, y = corner.y;
    for (let step = 0; step < 600; step++) {
      const next = [[x + 1, y - 1], [x + 1, y], [x, y - 1]];
      let moved = false;
      for (const [nx, ny] of next) { const p = await sample(nx, ny); if (p[0] > 200 && p[1] > 200 && p[2] > 200) { x = nx; y = ny; moved = true; break; } }
      if (!moved) break;
    }
    top = { x, y };
  }
  check('the curve is drawn with its two points', !!corner && !!top, corner && top ? `(0,0) at ${corner.x},${corner.y}; (1,1) at ${top.x},${top.y}` : 'not found');
  if (corner && top) {
    // Press at x = 0.5, y = 0.2 of the plot (empty space below the diagonal) and drag the new point up to y = 0.8.
    const px = (fx, fy) => ({ x: corner.x + (top.x - corner.x) * fx, y: corner.y - (corner.y - top.y) * fy });
    const a = px(0.5, 0.2), b = px(0.5, 0.8);
    await page.mouse.move(a.x, a.y); await page.mouse.down();
    for (let i = 1; i <= 8; i++) await page.mouse.move(a.x, a.y + (b.y - a.y) * i / 8);
    await page.mouse.up();
    await page.waitForTimeout(400);
    await page.screenshot({ path: path.join(OUT, 'curve-after.png') });
    const [download] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }), page.click('.toolbar button:has-text("Export")')]);
    const saved = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    const value = saved.nodes.find(n => n.type === 'CurveEditor').widgets_values[0];
    const pts = value.points;
    check('the new point is saved where it was dragged', pts.length === 3 && Math.abs(pts[1][0] - 0.5) < 0.03 && Math.abs(pts[1][1] - 0.8) < 0.03, JSON.stringify(value));
  }
  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await browser.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})();
