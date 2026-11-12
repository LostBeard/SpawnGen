// App Mode end to end through the server: the App Builder turns the starter graph into an app (prompt, seed and steps as inputs, Save Image as
// the output, opens as an app), the app runs with a new prompt and shows the picture, the saved workflow carries ComfyUI's extra.linearData /
// linearMode, and opening that file opens the app. Needs the SpawnGen server (it hosts the Studio), a GPU and SD-Turbo. Coordinates assume
// 1600x900 and the starter graph as the builder fits it beside its panel. Look at out/app-*.png.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const shot = name => page.screenshot({ path: path.join(OUT, name + '.png') });
  const picks = () => page.locator('.app-pick').allTextContents();

  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);

  // A workflow never set up as an app opens the builder.
  await page.click('.toolbar button:has-text("App")'); await page.waitForTimeout(800);
  check('App opens the builder on a workflow without an app', (await page.locator('.app-steps').count()) === 1);
  check('the builder starts at Inputs', /Inputs/.test(await page.textContent('.app-steps button.on')));

  // Inputs: click the prompt, the seed and the steps; a second click on the steps takes them out again.
  await page.mouse.click(465, 280); await page.waitForTimeout(200);
  await page.mouse.click(850, 375); await page.waitForTimeout(200);
  await page.mouse.click(850, 421); await page.waitForTimeout(200);
  let list = await picks();
  check('three inputs picked in click order', list.length === 3 && /text/.test(list[0]) && /seed/.test(list[1]) && /steps/.test(list[2]), list.join(' | '));
  await shot('app-2-inputs');
  await page.mouse.click(850, 421); await page.waitForTimeout(200);
  list = await picks();
  check('a second click takes an input out', list.length === 2, list.join(' | '));
  check('picking does not open the widget editor', (await page.locator('.widget-edit').count()) === 0);

  // Outputs: Save Image is an output node, KSampler is not.
  await page.click('.app-builder-bar button:has-text("Next")'); await page.waitForTimeout(400);
  await page.mouse.click(850, 272); await page.waitForTimeout(200);
  check('a non-output node is refused', (await picks()).length === 0);
  await page.mouse.click(1150, 377); await page.waitForTimeout(200);
  list = await picks();
  check('Save Image picked as the output', list.length === 1 && /Save Image/.test(list[0]), list.join(' | '));
  await shot('app-3-outputs');

  // Preview, then the default view.
  await page.click('.app-builder-bar button:has-text("Next")'); await page.waitForTimeout(400);
  check('Preview shows the app with its two inputs', (await page.locator('.appmode.preview .app-field').count()) === 2);
  await shot('app-4-preview');
  await page.click('.app-builder-bar button:has-text("Next")'); await page.waitForTimeout(400);
  check('Default view offers App and Node graph', (await page.locator('.app-default input[type=radio]').count()) === 2);
  await page.click('.app-default button:has-text("Apply")'); await page.waitForTimeout(600);
  check('Apply shows the app', (await page.locator('.appmode:not(.preview)').count()) === 1);

  // Run with a new prompt.
  await page.fill('.app-field textarea', 'a photo of a green pear on a white plate');
  await page.locator('.app-field textarea').dispatchEvent('change');
  await page.waitForTimeout(300);
  await page.click('.app-run-row button:has-text("Run")');
  const t0 = Date.now();
  while (Date.now() - t0 < 240000 && (await page.locator('.app-results img').count()) === 0) await page.waitForTimeout(1000);
  check('the app shows the picture of the run', (await page.locator('.app-results img').count()) >= 1, `${((Date.now() - t0) / 1000).toFixed(0)} s`);
  check('the run is listed in the strip', (await page.locator('.app-run').count()) >= 1);
  await page.waitForTimeout(800);
  await shot('app-5-result');

  // The saved workflow carries the app.
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('.toolbar button:has-text("Export")')]);
  const savePath = path.join(OUT, 'app-workflow.app.json');
  await download.saveAs(savePath);
  const saved = JSON.parse(fs.readFileSync(savePath, 'utf8'));
  const ld = saved.extra && saved.extra.linearData;
  const prompt = saved.nodes.find(n => n.type === 'CLIPTextEncode' && /pear/.test(String(n.widgets_values[0])));
  check('extra.linearMode is true', saved.extra && saved.extra.linearMode === true);
  check('extra.linearData has the inputs as [node id, widget]', ld && ld.inputs.length === 2 && ld.inputs[0][1] === 'text' && ld.inputs[1][1] === 'seed', JSON.stringify(ld && ld.inputs));
  check('extra.linearData has the output node', ld && ld.outputs.length === 1 && saved.nodes.some(n => String(n.id) === String(ld.outputs[0]) && n.type === 'SaveImage'), JSON.stringify(ld && ld.outputs));
  check('the app input changed the workflow', !!prompt && String(prompt.id) === String(ld.inputs[0][0]));

  // Back to the graph; then opening the saved file opens the app.
  await page.click('.app-in button:has-text("Node graph")'); await page.waitForTimeout(400);
  check('Node graph leaves the app', (await page.locator('.appmode').count()) === 0);
  await page.setInputFiles('.toolbar input[type=file][accept=".json,.png"]', savePath);
  await page.waitForTimeout(1500);
  check('the saved app opens as the app', (await page.locator('.appmode:not(.preview)').count()) === 1);
  check('the tab is named without .app', /^app-workflow$/.test((await page.locator('.tab.active span').textContent()).trim()));
  await shot('app-6-reopened');

  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
