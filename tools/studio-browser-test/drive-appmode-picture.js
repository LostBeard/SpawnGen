// App Builder on the Depth map example (TJ's 2026-11-18 report: Load Image as the input, Save Image as the output, Next on the outputs step threw).
// Builds the app, previews it, applies it, uploads a picture through the app's own upload button and runs it. Look at out/apppic-*.png.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const APPLE = path.join(__dirname, 'apple.png');

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const shot = name => page.screenshot({ path: path.join(OUT, 'apppic-' + name + '.png') });
  const picks = async () => (await page.locator('.app-picks li, .app-pick').allTextContents()).map(s => s.trim()).filter(Boolean);

  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  // The Depth map example naming a picture this engine does not have (the shipped example.png would be found).
  await page.setInputFiles('.toolbar input[type=file][accept=".json,.png"]', path.join(__dirname, 'depth-missing.json')); await page.waitForTimeout(1500);
  await page.click('.toolbar button:has-text("App")'); await page.waitForTimeout(800);
  await shot('1-inputs');
  if (process.env.PROBE) { await browser.close(); return; }
  // Where the builder fits the Depth map example beside its panel (1600x900, measured from out/apppic-1-inputs.png).
  const at = { combo: [220, 513], save: [1080, 440] };
  await page.mouse.click(...at.combo); await page.waitForTimeout(300);
  console.log('INPUT PICKS ' + JSON.stringify(await picks()));
  await page.click('.app-builder-bar button:has-text("Next")'); await page.waitForTimeout(500);
  await page.mouse.click(...at.save); await page.waitForTimeout(300);
  console.log('OUTPUT PICKS ' + JSON.stringify(await picks()));
  await shot('2-outputs');
  await page.click('.app-builder-bar button:has-text("Next")'); await page.waitForTimeout(800);
  await shot('3-preview');
  check('Next on the outputs step shows the preview without an error', problems.length === 0 && (await page.locator('.appmode.preview').count()) === 1, problems.join(' | '));
  const field = page.locator('.appmode.preview .app-field').first();
  check('the preview lists the missing not-here.png as not found, with no picture', /not-here\.png \(not found\)/.test(await field.locator('select').evaluate(s => s.options[s.selectedIndex].text))
    && (await field.locator('img.app-picture').count()) === 0);

  // Default view, Apply: the app itself. Upload a picture with the app's button, run, and the depth map is shown.
  await page.click('.app-builder-bar button:has-text("Next")'); await page.waitForTimeout(400);
  await page.click('.app-default button:has-text("Apply")'); await page.waitForTimeout(800);
  check('Apply shows the app', (await page.locator('.appmode:not(.preview)').count()) === 1);
  // the app's "Upload a picture" button opens the page's file picker (the same one a node's upload button opens, so the file is read in JS)
  const [chooser] = await Promise.all([page.waitForEvent('filechooser', { timeout: 10000 }),
    page.locator('.appmode:not(.preview) .app-field button:has-text("Upload a picture")').first().click()]);
  await chooser.setFiles(APPLE);
  await page.waitForTimeout(1500);
  const app = page.locator('.appmode:not(.preview) .app-field').first();
  check('the app shows the uploaded picture and chose it', (await app.locator('img.app-picture').count()) === 1
    && (await app.locator('select').inputValue()) === 'apple.png', await app.locator('select').inputValue());
  await shot('4-app');
  await page.locator('.appmode:not(.preview) button:has-text("Run")').first().click();
  let shown = false;
  for (let i = 0; i < 300 && !shown; i++) { await page.waitForTimeout(1000); shown = (await page.locator('.appmode:not(.preview) .app-output img, .appmode:not(.preview) .app-results img').count()) > 0; }
  await page.waitForTimeout(1000);
  await shot('5-result');
  check('the app runs and shows the depth map (see out/apppic-5-result.png)', shown);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
