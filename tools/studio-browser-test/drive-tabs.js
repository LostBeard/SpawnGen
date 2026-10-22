// Workflow tabs and autosave in the real page: edits survive a reload, a new tab starts a separate workflow, switching restores each one,
// a tab can be renamed and closed, and a file opens in its own tab. Checks read the saved workflow JSON and the tab bar.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const tabNames = async () => (await page.locator('.tabs .tab span').allTextContents()).map(s => s.trim());
  const activeName = async () => ((await page.locator('.tabs .tab.active span').first().textContent()) || '').trim();
  const saved = async () => {
    const [download] = await Promise.all([page.waitForEvent('download'), page.click('button:has-text("Save")')]);
    const p = path.join(OUT, 'tabs-saved.json');
    await download.saveAs(p);
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  };
  const samplerPos = wf => wf.nodes.find(n => n.type === 'KSampler').pos.map(Math.round);
  const open = async () => { await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500); };

  await open();
  await require('./tab-offset')(page);
  check('starts with one tab', JSON.stringify(await tabNames()) === JSON.stringify(['Workflow']), (await tabNames()).join(','));

  // Drag the KSampler by its title (the fitted example puts it at 1040, 234), then wait for the autosave and reload.
  await page.mouse.move(1040, 234);
  await page.mouse.down();
  await page.mouse.move(1000, 300, { steps: 6 });
  await page.mouse.up();
  const moved = samplerPos(await saved());
  await page.waitForTimeout(1600);
  await page.reload();
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  const afterReload = samplerPos(await saved());
  check('an edit survives a reload (autosave)', JSON.stringify(moved) === JSON.stringify(afterReload), `${moved} vs ${afterReload}`);

  // A new tab is a separate workflow.
  await page.click('.tabs .newtab');
  await page.waitForTimeout(500);
  check('+ adds a second tab and makes it active', (await tabNames()).length === 2 && (await activeName()) === 'Workflow (2)', (await tabNames()).join(',') + ' / ' + (await activeName()));
  const fresh = samplerPos(await saved());
  check('the new tab starts from the example, not from the other tab', JSON.stringify(fresh) !== JSON.stringify(moved), `${fresh} vs ${moved}`);

  // Back to the first tab: its edit is still there.
  await page.click('.tabs .tab:nth-child(1)');
  await page.waitForTimeout(500);
  const first = samplerPos(await saved());
  check('switching back restores the first workflow', JSON.stringify(first) === JSON.stringify(moved), `${first} vs ${moved}`);

  // Rename by double click.
  await page.dblclick('.tabs .tab:nth-child(1)');
  await page.waitForSelector('.tabs .tab input');
  await page.fill('.tabs .tab input', 'Portrait');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(300);
  check('double click renames a tab', (await tabNames())[0] === 'Portrait', (await tabNames()).join(','));

  // Open a file: it gets its own tab.
  await page.setInputFiles('input[type=file]', path.join(__dirname, 'comfy-default-workflow.json'));
  await page.waitForTimeout(1200);
  check('opening a file adds a tab named after it', (await tabNames()).length === 3 && /comfy-default-workflow/.test(await activeName()), (await tabNames()).join(','));
  await page.screenshot({ path: path.join(OUT, 'tabs.png') });

  // All three tabs come back after a reload, with the same active one.
  await page.waitForTimeout(1600);
  await page.reload();
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  check('every tab and the active one survive a reload', (await tabNames()).length === 3 && /comfy-default-workflow/.test(await activeName()), (await tabNames()).join(',') + ' / ' + (await activeName()));

  // Close the active tab: the neighbour on the left takes over.
  await page.click('.tabs .tab.active button');
  await page.waitForTimeout(500);
  check('closing a tab removes it and activates a neighbour', (await tabNames()).length === 2 && (await activeName()) === 'Workflow (2)', (await tabNames()).join(',') + ' / ' + (await activeName()));

  // A file that is not a workflow leaves the tab alone.
  fs.writeFileSync(path.join(OUT, 'not-a-workflow.json'), '{"hello": 1}');
  const nameBefore = await activeName();
  await page.setInputFiles('input[type=file]', path.join(OUT, 'not-a-workflow.json'));
  await page.waitForTimeout(800);
  check('a file that is not a workflow adds no tab', (await tabNames()).length === 2 && (await activeName()) === nameBefore, (await tabNames()).join(','));

  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
