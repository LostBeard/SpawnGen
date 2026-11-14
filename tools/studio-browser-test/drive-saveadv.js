// Save Image (Advanced) in the real page on the server engine: open Load Image -> Save Image (Advanced) (png, 8-bit), switch format to exr in the node,
// check bit_depth now offers exr's choices with exr's default (16-bit float), pick 32-bit float, Queue: the server saves drive_adv_NNNNN.exr and the prompt
// carries exr's values. Needs the server started with an input folder holding probe.png.   STUDIO_URL (default http://localhost:8188/).
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

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'saveadv-workflow.json'));
  await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(OUT, 'saveadv-open.png') });

  // Click down the Save Image (Advanced) node until a combo editor opens whose options contain `has`; returns its options (editor left open).
  const openCombo = async has => {
    for (let y = 440; y < 620; y += 8)
      for (const x of [1000, 1100]) {
        await page.mouse.click(x, y);
        await page.waitForTimeout(120);
        const select = page.locator('.widget-edit select');
        if (await select.count()) {
          const options = await select.locator('option').allTextContents();
          if (options.includes(has)) return options;
          await page.keyboard.press('Escape');
        } else if (await page.locator('.widget-edit input').count()) await page.keyboard.press('Escape');
      }
    return null;
  };

  const formats = await openCombo('exr');
  check('the format combo offers png, exr and avif', JSON.stringify(formats) === JSON.stringify(['png', 'exr', 'avif']), JSON.stringify(formats));
  await page.selectOption('.widget-edit select', 'exr');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(400);
  await page.mouse.click(1000, 800);
  const depths = await openCombo('32-bit float');
  check("bit_depth now offers exr's choices", JSON.stringify(depths) === JSON.stringify(['32-bit float', '16-bit float']), JSON.stringify(depths));
  check("and starts at exr's default", (await page.locator('.widget-edit select').inputValue()) === '16-bit float', await page.locator('.widget-edit select').inputValue());
  await page.selectOption('.widget-edit select', '32-bit float');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(400);
  await page.mouse.click(1000, 800);
  await page.screenshot({ path: path.join(OUT, 'saveadv-exr.png') });

  await page.locator('button:has-text("Queue")').click();
  let entry = null;
  for (let i = 0; i < 40; i++) {
    await page.waitForTimeout(400);
    const hist = await (await fetch(new globalThis.URL('history', URL))).json();
    const last = Object.values(hist).pop();
    if (last?.status?.completed !== undefined && last.prompt?.[2]?.['2']) { entry = last; if (last.status.status_str) break; }
  }
  const inputs = entry?.prompt?.[2]?.['2']?.inputs ?? {};
  check('the prompt carries exr with 32-bit float', inputs.format === 'exr' && inputs['format.bit_depth'] === '32-bit float' && inputs['format.input_color_space'] === 'sRGB',
    JSON.stringify(inputs));
  const images = entry?.outputs?.['2']?.images ?? [];
  check('the run succeeded and saved an .exr', entry?.status?.status_str === 'success' && images.length === 1 && /^drive_adv_\d{5}\.exr$/.test(images[0].filename),
    JSON.stringify(entry?.status?.status_str) + ' ' + JSON.stringify(images));
  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await browser.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})();
