// Custom Combo in the real page on the server engine: open a ComfyUI-style Custom Combo (options Gaussian, Box, Radial; choice Box) wired to Preview as
// Text, check the option rows are drawn with one spare, type a fourth option into the spare row (a new spare appears), Queue: the preview shows "Box".
//   STUDIO_URL (default http://localhost:8188/).
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

  page.on('response', async r => { if (r.url().endsWith('/prompt') && r.request().method() === 'POST') console.log('POST /prompt', r.status(), (r.request().postData() || '').slice(0, 400), (await r.text()).slice(0, 400)); });
  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'customcombo-workflow.json'));
  await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(OUT, 'customcombo-open.png') });

  // Click down the Custom Combo node's widget rows until a text editor opens on an empty option (the spare row).
  let typed = false;
  for (let y = 300; y < 700 && !typed; y += 12) {
    for (const x of [420, 520, 600]) {
      await page.mouse.click(x, y);
      await page.waitForTimeout(150);
      const input = page.locator('.widget-edit input[type="text"]');
      if (await input.count()) {
        if ((await input.inputValue()) === '') { await input.fill('Motion'); await input.press('Enter'); typed = true; break; }
        await input.press('Escape');
      } else if (await page.locator('.widget-edit select').count()) await page.keyboard.press('Escape');
      if (typed) break;
    }
  }
  check('typed a new option into the spare row', typed);
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'customcombo-typed.png') });

  await page.locator('button:has-text("Queue")').click();
  let text = '';
  for (let i = 0; i < 30 && !/Box/.test(text); i++) {
    await page.waitForTimeout(400);
    const hist = await (await fetch(new globalThis.URL('history', URL))).json();
    const last = Object.values(hist).pop();
    text = JSON.stringify(last?.outputs ?? {});
  }
  check('the run passes the choice on', /"Box"/.test(text), text.slice(0, 120));
  const prompt = Object.values(await (await fetch(new globalThis.URL('history', URL))).json()).pop()?.prompt?.[2]?.['1']?.inputs ?? {};
  check('the prompt carries the choice, index and options with one spare', prompt.choice === 'Box' && prompt.index === 1 && prompt.option4 === 'Motion' && prompt.option5 === '',
    JSON.stringify(prompt));
  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await page.screenshot({ path: path.join(OUT, 'customcombo-run.png') });
  await browser.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})();
