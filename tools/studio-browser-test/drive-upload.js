// Uploading pictures for Load Image (TJ's 2026-11-18 report: Examples > Depth map, Upload, Queue failed on the example's example.png).
// The Depth map example runs out of the box on the shipped example.png; depth-missing.json is the same graph naming not-here.png, for the rest.
// Checks: the node's own "choose file to upload" button uploads and chooses the file; the node shows the chosen picture; a value that is not in
// the list shows as "(not found)" in the dropdown so picking the first real option changes it; the toolbar Upload chooses the file in the
// selected Load Image node; the queue error under a node is shown whole; and the depth map runs on the uploaded picture.
// Look at out/upload-*.png afterwards. Needs the Studio on STUDIO_URL (default http://localhost:5280/) and, for the run, a real WebGPU adapter.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const APPLE = path.join(__dirname, 'apple.png');
const PEAR = path.join(OUT, 'pear.png'); // the same picture under another name, for the toolbar upload
fs.copyFileSync(APPLE, PEAR);

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const toast = async () => (await page.locator('.toast').count()) ? await page.textContent('.toast') : '';
  const openDepth = async () => {
    await page.click('button:has-text("Examples")');
    await page.waitForSelector('.menu');
    await page.click('.menu li:has-text("Depth map")');
    await page.waitForTimeout(1500);
  };
  // The graph that names a picture this engine does not have (the Depth map example with not-here.png), opened with the toolbar's Open.
  const openMissing = async () => {
    await page.setInputFiles('.toolbar input[type=file][accept=".json,.png"]', path.join(__dirname, 'depth-missing.json'));
    await page.waitForTimeout(1500);
  };
  const runs = async () => parseInt(((await page.textContent('.toolbar')).match(/History \((\d+)\)/) || [])[1] || '0', 10);
  const waitForRun = async before => {
    for (let i = 0; i < 300; i++) {
      await page.waitForTimeout(1000);
      if ((await runs()) > before && !/Running|Queued/.test(await page.textContent('.status'))) return true;
    }
    return false;
  };
  // The Depth map example fits the view the same way every time: Load Image's title bar, image combo and upload button are here.
  const TITLE = [330, 432], COMBO = [400, 507], UPLOAD = [370, 533];
  const comboValue = async () => {
    await page.mouse.click(...COMBO);
    await page.waitForSelector('.widget-edit select');
    await page.waitForTimeout(300);
    const v = await page.$eval('.widget-edit select', s => ({ value: s.value, first: s.options[0].text }));
    await page.mouse.click(800, 250); // click away on the canvas: the dropdown closes without a choice
    await page.waitForTimeout(300);
    v.closed = (await page.locator('.widget-edit').count()) === 0;
    if (!v.closed) throw new Error('the dropdown stayed open after a click on the canvas');
    return v;
  };

  await page.goto(URL);
  const adapter = await page.evaluate(async () => { const a = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' }); return a ? `${a.info.vendor} ${a.info.description} ${a.info.backend || ''}` : 'none'; });
  console.log('ADAPTER ' + adapter);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);

  // 0. The example as it ships: example.png is there, shown in the node, and the depth map runs.
  await openDepth();
  let v = await comboValue();
  check('the Depth map example finds the shipped example.png', v.value === 'example.png' && v.first === 'example.png', JSON.stringify(v));
  let before = await runs();
  await page.click('button:has-text("Queue")');
  check('the Depth map example runs as it ships (see out/upload-0-example.png)', await waitForRun(before), await toast());
  await page.screenshot({ path: path.join(OUT, 'upload-0-example.png') });

  // 1. A graph naming a picture this engine does not have: the dropdown says so, and Queue says which, under the node, in full.
  await openMissing();
  await page.screenshot({ path: path.join(OUT, 'upload-1-opened.png') });
  v = await comboValue();
  check('a missing picture shows as "(not found)" in the dropdown, still chosen', v.value === 'not-here.png' && /not-here\.png \(not found\)/.test(v.first), JSON.stringify(v));
  await page.click('button:has-text("Queue")');
  await page.waitForTimeout(1500);
  check('Queue refuses the missing not-here.png and says which', /not-here\.png/.test(await toast()), await toast());
  await page.screenshot({ path: path.join(OUT, 'upload-2-error.png') });

  // 2. The node's own upload button: pick a file, it is uploaded and chosen, and the node shows it.
  const [chooser] = await Promise.all([page.waitForEvent('filechooser', { timeout: 5000 }), page.mouse.click(...UPLOAD)]);
  check('the upload button opens a file picker for pictures', !!chooser, '');
  await chooser.setFiles(APPLE);
  await page.waitForTimeout(1500);
  v = await comboValue();
  check('the uploaded picture is chosen in the node', v.value === 'apple.png', JSON.stringify(v));
  await page.screenshot({ path: path.join(OUT, 'upload-3-node-upload.png') });

  // 3. The missing-picture graph again: picking a listed picture in the dropdown is a real change.
  await openMissing();
  await page.mouse.click(...COMBO);
  await page.waitForSelector('.widget-edit select');
  await page.selectOption('.widget-edit select', 'apple.png');
  await page.waitForTimeout(500);
  v = await comboValue();
  check('choosing a listed picture over a missing one changes the node', v.value === 'apple.png', JSON.stringify(v));

  // 4. Again: the toolbar Upload with Load Image selected chooses the file there.
  await openMissing();
  await page.mouse.click(...TITLE);
  await page.setInputFiles('.toolbar input[type=file][accept^=".png,.jpg"]', PEAR);
  await page.waitForTimeout(1500);
  const said = await toast();
  v = await comboValue();
  check('the toolbar Upload chooses the file in the selected Load Image', v.value === 'pear.png' && /chose it in Load Image #1/.test(said), `${JSON.stringify(v)} / ${said}`);

  // 5. The run: the depth map of the uploaded picture.
  before = await runs();
  await page.click('button:has-text("Queue")');
  await page.waitForTimeout(1000);
  check('Queue takes the graph with the uploaded picture', !/not-here\.png|not in|There is no/.test(await toast()), await toast());
  const ran = await waitForRun(before);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(OUT, 'upload-4-run.png') });
  check('the depth map of the uploaded picture runs to the end (see out/upload-4-run.png)', ran, `toast: ${await toast()} | status: ${await page.textContent('.status')}`);

  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
