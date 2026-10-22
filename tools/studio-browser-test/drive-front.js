// Queue (Front) in the real page: while a run is going a Front button appears; clicking it queues the graph at the front of the waiting runs
// (the History panel lists waiting runs in the order they start). Needs the server: STUDIO_URL=http://localhost:8188/.
const { chromium } = require('playwright-core');
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
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  check('no Front button while idle', (await page.locator('button:has-text("Front")').count()) === 0);
  await page.locator('input[type=number]').first().fill('6');            // a batch of six: one runs, five wait
  await page.click('button.primary');
  await page.waitForSelector('button.danger');                         // Stop shows: a run is going
  check('Front appears while running', (await page.locator('button:has-text("Front")').count()) === 1);
  await page.locator('input[type=number]').first().fill('1');
  await page.click('button:has-text("Front")');                        // front: should start next
  await page.waitForTimeout(400);
  await page.keyboard.press('Control+Shift+Enter');                    // front again via the shortcut
  await page.waitForTimeout(400);
  const queued = await page.evaluate(async () => (await (await fetch('/queue')).json()).queue_pending.map(p => p[0]));
  const top2 = [...queued].sort((a, b) => b - a).slice(0, 2);
  check('the two front runs wait ahead of the older ones (newest first)', queued.length >= 4 && queued[0] === top2[0] && queued[1] === top2[1], queued.join(','));
  await page.evaluate(async () => { await fetch('/queue', { method: 'POST', body: JSON.stringify({ clear: true }) }); await fetch('/interrupt', { method: 'POST' }); });
  await page.waitForTimeout(1500);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
