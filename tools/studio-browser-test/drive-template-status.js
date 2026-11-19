// The template browser as a visitor meets it: the show modes (Opens, the default; Ready; All) with their counts, a status badge on every
// non-API card, a template from the default list opening with no missing-node note, one with missing nodes saying which; and the Examples menu
// offering every example alike. STUDIO_URL (default http://localhost:5280/).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
let failed = 0;
const check = (ok, what) => { console.log((ok ? 'PASS ' : 'FAIL ') + what); if (!ok) failed++; };

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error' && !/powerPreference|Failed to load resource/.test(m.text())) errors.push(m.text().slice(0, 200)); });
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3000);
  const toast = async () => (await page.locator('.toast').first().textContent({ timeout: 4000 }).catch(() => '')) || '';

  // Examples menu: every example is offered alike (every model runs in the browser; none is "desktop only").
  await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
  const items = (await page.locator('.menu li').allTextContents()).map(s => s.trim());
  check(items.includes('Image to 3D model (TRELLIS.2)') && !items.some(s => /desktop/i.test(s)), 'Examples: Image to 3D is offered like the rest (no desktop label)');
  check(items.includes('Text to video (LTX-Video 2B)') && items.includes('Picture to video (LTX-Video 2B)'), 'Examples: the two LTX-Video examples are listed');
  await page.keyboard.press('Escape'); await page.mouse.click(800, 860);
  await page.waitForTimeout(500);

  await page.click('button:has-text("Templates")');
  await page.waitForSelector('.templates .tpl-card', { timeout: 60000 });
  await page.waitForTimeout(1500);
  const modes = (await page.locator('.templates .tpl-show').allTextContents()).map(s => s.trim());
  console.log('modes: ' + modes.join(' | '));
  const count = label => parseInt((modes.find(m => m.startsWith(label)) || '').match(/\((\d+)\)/)?.[1] ?? '-1', 10);
  const opens = count('Opens'), ready = count('Ready'), all = count('All');
  check(all > 700 && opens > 50 && opens < all && ready <= opens, `counts: Opens ${opens}, Ready ${ready}, All ${all}`);
  check((await page.locator('.templates .tpl-show.on').textContent()).startsWith('Opens'), 'the default mode is Opens');
  const cards = await page.locator('.templates .tpl-card').count();
  const missingShown = await page.locator('.templates .tpl-card .tpl-missing').count();
  const apiShown = await page.locator('.templates .tpl-card .tpl-api').count();
  check(missingShown === 0 && apiShown === 0, `Opens shows no missing-node or API card (${cards} cards)`);
  const badges = await page.locator('.templates .tpl-card .tpl-badge').count();
  check(badges === cards, `every card in Opens has a status badge (${badges} of ${cards})`);

  // A card from the default list opens without a missing-node note.
  const first = page.locator('.templates .tpl-card').first();
  const firstTitle = (await first.locator('.tpl-title').textContent()).trim();
  await first.click();
  const t1 = await toast();
  check(!/do(es)? not have these nodes/.test(t1), `"${firstTitle}" opens with no missing-node note (toast: ${t1.slice(0, 120) || 'none'})`);
  await page.screenshot({ path: path.join(OUT, 'templates-opened.png') });

  // All: a card with missing nodes says which when opened.
  await page.click('button:has-text("Templates")'); await page.waitForSelector('.templates .tpl-card');
  await page.locator('.templates .tpl-show', { hasText: 'All' }).click(); await page.waitForTimeout(800);
  const missingCard = page.locator('.templates .tpl-card:has(.tpl-missing)').first();
  check(await missingCard.count() === 1, 'All shows cards marked with missing nodes');
  const tip = await missingCard.locator('.tpl-missing').getAttribute('title');
  await page.screenshot({ path: path.join(OUT, 'templates-all.png') });
  await missingCard.click();
  const t2 = await toast();
  check(/does not have these nodes yet/.test(t2) && tip && t2.includes(tip.replace(/^.*yet: /, '').split(',')[0]), `a missing-node template names its missing nodes (${t2.slice(0, 140)})`);

  check(errors.length === 0, 'no page errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' || ') : ''));
  await browser.close();
  console.log(failed ? `${failed} FAILED` : 'ALL PASS');
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
