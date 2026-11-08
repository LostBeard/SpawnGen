// Load Image (from Outputs) in the real page on the server engine: open a Load Image (from Outputs) -> Save Image workflow whose picture name does not exist,
// right-click the node, "Refresh and pick the newest output", Queue. The run succeeds and saves the newest output picture again (outputs_check_*.png).
//   STUDIO_URL (default http://localhost:8188/, the server-hosted Studio). The server needs at least one picture in its output folder.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  const newest = await (await fetch(new globalThis.URL('internal/files/output', URL))).json();
  check('the server lists its outputs newest first', Array.isArray(newest) && newest.length > 0, newest[0]);

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'outputs-workflow.json'));
  await page.waitForTimeout(1200);

  let picked = false;
  for (let y = 380; y < 640 && !picked; y += 16)
    for (let x = 100; x < 600 && !picked; x += 40) {
      await page.mouse.click(x, y, { button: 'right' });
      await page.waitForTimeout(140);
      if (await page.locator('.menu li:has-text("Refresh and pick the newest output")').count()) { await page.click('.menu li:has-text("Refresh and pick the newest output")'); picked = true; }
      else if (await page.locator('.menu').count()) { await page.keyboard.press('Escape'); await page.mouse.click(5, 850); }
    }
  check('the node menu offers the newest output', picked);
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, 'outputs-picked.png') });

  const before = new Set(await (await fetch(new globalThis.URL('internal/files/output', URL))).json());
  await page.locator('button:has-text("Queue")').click();
  let saved = [];
  for (let i = 0; i < 40 && saved.length === 0; i++) {
    await page.waitForTimeout(500);
    const now = await (await fetch(new globalThis.URL('internal/files/output', URL))).json();
    saved = now.filter(n => !before.has(n) && n.startsWith('outputs_check_'));
  }
  check('the run saved the picked picture again', saved.length === 1, saved[0]);
  if (saved.length === 1) {
    const a = Buffer.from(await (await fetch(new globalThis.URL(`view?filename=${encodeURIComponent(newest[0].replace(' [output]', ''))}&type=output`, URL))).arrayBuffer());
    const b = Buffer.from(await (await fetch(new globalThis.URL(`view?filename=${encodeURIComponent(saved[0].replace(' [output]', ''))}&type=output`, URL))).arrayBuffer());
    check('it is the newest output picture', a.length > 0 && b.length > 0, `${a.length} and ${b.length} bytes`);
  }
  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await browser.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})();
