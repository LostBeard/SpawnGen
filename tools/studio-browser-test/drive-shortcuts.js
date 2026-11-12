// Keyboard shortcuts in the real page: Ctrl+Shift+K lists them; a new key for Bypass works and the old one does nothing; the change survives a
// reload; Reset all puts Ctrl+B back; ComfyUI's Alt+C (collapse), Alt+= / Alt+- (zoom), Ctrl+, (settings), Ctrl+O (open) and Escape work.
const { chromium } = require('playwright-core');
const fs = require('fs');
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const canvas = async () => { await page.locator('canvas').focus(); await page.waitForTimeout(100); };
  const press = async keys => { await canvas(); await page.keyboard.press(keys); await page.waitForTimeout(350); };
  const exported = async () => {
    const [d] = await Promise.all([page.waitForEvent('download'), page.click('.toolbar button:has-text("Export")')]);
    return JSON.parse(fs.readFileSync(await d.path(), 'utf8'));
  };
  const zoom = async () => parseInt(((await page.textContent('.status')).match(/(\d+)%/) || [])[1] || '0', 10);
  const row = name => page.locator('.shortcuts .sc-row', { hasText: name });

  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);

  await press('Control+Shift+K');
  check('Ctrl+Shift+K lists the shortcuts', (await page.locator('.shortcuts .sc-row').count()) >= 25);
  await page.fill('.shortcuts .sc-filter', 'bypass');
  check('the list can be searched', (await page.locator('.shortcuts .sc-row').count()) === 1);
  check('Bypass has Ctrl+B', (await row('Bypass').locator('kbd').allTextContents()).join() === 'Ctrl+B');
  await row('Bypass').locator('.sc-keys').click(); await page.waitForTimeout(300);
  check('clicking the keys waits for new ones', (await page.locator('.sc-keys.capturing').count()) === 1);
  await page.keyboard.press('Control+q'); await page.waitForTimeout(300);
  check('the new key is shown', (await row('Bypass').locator('kbd').allTextContents()).join() === 'Ctrl+Q');
  await page.screenshot({ path: __dirname + '/out/shortcuts-1.png' });
  await page.click('.shortcuts .set-head button'); await page.waitForTimeout(300);
  check('the list closes', (await page.locator('.shortcuts').count()) === 0);

  await press('Control+a');
  await press('Control+b');
  let wf = await exported();
  check('the old key does nothing', wf.nodes.every(n => n.mode === 0), JSON.stringify(wf.nodes.map(n => n.mode)));
  await press('Control+q');
  wf = await exported();
  check('the new key bypasses', wf.nodes.every(n => n.mode === 4), JSON.stringify(wf.nodes.map(n => n.mode)));
  await press('Control+z');

  await page.reload(); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  await press('Control+Shift+K');
  await page.fill('.shortcuts .sc-filter', 'bypass');
  check('the change survives a reload', (await row('Bypass').locator('kbd').allTextContents()).join() === 'Ctrl+Q');
  await page.fill('.shortcuts .sc-filter', '');
  await page.click('.shortcuts .sc-foot button:has-text("Reset all")'); await page.waitForTimeout(300);
  await page.fill('.shortcuts .sc-filter', 'bypass');
  check('Reset all puts Ctrl+B back', (await row('Bypass').locator('kbd').allTextContents()).join() === 'Ctrl+B');
  await page.click('.shortcuts .set-head button'); await page.waitForTimeout(300);

  // ComfyUI's keys.
  await press('Control+a');
  await press('Alt+c');
  wf = await exported();
  check('Alt+C collapses the selected nodes', wf.nodes.every(n => n.flags && n.flags.collapsed === true));
  await press('Alt+c');
  const z0 = await zoom();
  await press('Alt+=');
  const z1 = await zoom();
  await press('Alt+-'); await press('Alt+-');
  const z2 = await zoom();
  check('Alt+= zooms in and Alt+- out', z1 > z0 && z2 < z1, `${z0}% ${z1}% ${z2}%`);
  await press('Control+,');
  check('Ctrl+, opens the settings', (await page.locator('.settings').count()) === 1);
  await press('Escape'); await press('Escape');
  check('Escape closes them', (await page.locator('.settings').count()) === 0);
  await canvas();
  const [chooser] = await Promise.all([page.waitForEvent('filechooser', { timeout: 5000 }).catch(() => null), page.keyboard.press('Control+o')]);
  check('Ctrl+O opens the file picker', chooser !== null && chooser.isMultiple() === false);

  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
