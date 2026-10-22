// The Examples menu in the real page: it lists the example workflows, and each opens in its own tab with its nodes (no missing-node warning).
// Look at out/examples.png: the hires-fix example laid out automatically.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const nodes = async () => parseInt(((await page.textContent('.status')).match(/(\d+) nodes/) || [])[1] || '-1', 10);

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.click('button:has-text("Examples")');
  await page.waitForSelector('.menu');
  const items = await page.locator('.menu li').allTextContents();
  check('the menu lists the starter and the examples', items.length === 16 && /Starter/.test(items[0]), `${items.length} items`);

  const wanted = [['Hires fix', 9], ['LLM writes the prompt', 9], ['Speak and listen', 4], ['FLUX.2 Klein picture edit', 3], ['Remove the background', 8]];
  for (const [label, count] of wanted) {
    if (!(await page.locator('.menu').count())) { await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu'); }
    const tabs = await page.locator('.tabs .tab').count();
    await page.click(`.menu li:has-text("${label}")`);
    await page.waitForTimeout(1200);
    check(`"${label}" opens in a new tab with its ${count} nodes`, (await page.locator('.tabs .tab').count()) === tabs + 1 && (await nodes()) === count, `${await nodes()} nodes, tabs ${await page.locator('.tabs .tab').count()}`);
    if (label === 'Hires fix') await page.screenshot({ path: path.join(OUT, 'examples.png') });
  }
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
