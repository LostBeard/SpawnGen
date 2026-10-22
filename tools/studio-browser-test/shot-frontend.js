// Adds a Reroute and a Note through node search and saves a screenshot (look at out/frontend-nodes.png).
const { chromium } = require('playwright-core');
const path = require('path');
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('pageerror', e => problems.push(e.message));
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500); await page.mouse.click(200, 700); await page.keyboard.press('f'); await page.waitForTimeout(500);
  for (const [x, y, name] of [[200, 700, 'reroute'], [700, 780, 'note']]) {
    await page.mouse.dblclick(x, y);
    await page.waitForSelector('.search input'); await page.waitForTimeout(400);
    await page.keyboard.type(name); await page.waitForTimeout(300); await page.keyboard.press('Enter'); await page.waitForTimeout(500);
  }
  console.log(await page.textContent('.status'));
  await page.screenshot({ path: path.join(__dirname, 'out', 'frontend-nodes.png') });
  console.log('problems:', problems.length ? problems.join(' | ') : 'none');
  await browser.close();
})();
