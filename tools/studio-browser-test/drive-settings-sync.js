// Settings follow the user, as ComfyUI's do: the Studio keeps them on the server that serves it, so a second browser (its own empty storage)
// opens with the first one's palette, node ids and shortcuts. They are stored under SpawnGen.Settings in the server's settings.
const { chromium } = require('playwright-core');
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const problems = [];
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const newPage = async () => {
    const context = await browser.newContext({ viewport: { width: 1600, height: 900 } }); // its own storage: a second browser
    const page = await context.newPage();
    page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
    page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
    await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
    return page;
  };

  const first = await newPage();
  await require('./reset-settings')(first);
  await first.reload(); await first.waitForSelector('canvas'); await first.waitForTimeout(3500);
  await first.click('.toolbar button:has-text("Settings")'); await first.waitForSelector('.settings');
  await first.selectOption('.settings label:has-text("Colour palette") select', { label: 'Light' });
  await first.locator('.settings label:has-text("Show node ids") input').check();
  await first.click('.settings .set-button'); await first.waitForSelector('.shortcuts');
  await first.fill('.shortcuts .sc-filter', 'bypass');
  await first.locator('.shortcuts .sc-row .sc-keys').click(); await first.waitForTimeout(300);
  await first.keyboard.press('Control+q'); await first.waitForTimeout(600);
  const stored = await first.evaluate(async () => (await fetch('api/settings/SpawnGen.Settings')).json());
  check('the settings are stored on the server', stored && stored.Palette === 'light' && stored.ShowNodeIds === true
    && JSON.stringify(stored.Shortcuts) === '{"Comfy.Canvas.ToggleSelectedNodes.Bypass":["Ctrl+Q"]}', JSON.stringify(stored).slice(0, 200));

  const second = await newPage();
  check('a second browser opens with the light palette', (await second.evaluate(() => document.documentElement.dataset.theme)) === 'light');
  await second.click('.toolbar button:has-text("Settings")'); await second.waitForSelector('.settings');
  check('and the node ids on', await second.locator('.settings label:has-text("Show node ids") input').isChecked());
  await second.click('.settings .set-button'); await second.waitForSelector('.shortcuts');
  await second.fill('.shortcuts .sc-filter', 'bypass');
  check('and the shortcut', (await second.locator('.shortcuts .sc-row kbd').allTextContents()).join() === 'Ctrl+Q');

  await require('./reset-settings')(second);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
