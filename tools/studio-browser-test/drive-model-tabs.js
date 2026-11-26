// Models kept by a GPU memory budget belong to the workflow tabs that used them (real GPU, the browser engine): with "GPU memory for models"
// set, two examples run in two tabs and both tabs' models stay loaded; closing one tab drops the models only it used; the other tab's
// workflow then runs again without loading anything; and with no budget closing a tab drops nothing (the older rule frees models).
//   STUDIO_URL (default http://localhost:5300/), MODEL_MEMORY_GB (default 6).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const BASE = process.env.STUDIO_URL || 'http://localhost:5300/';
const URL = BASE + (BASE.includes('?') ? '&' : '?') + 'nodetimes=1&testhooks=1';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const GB = parseFloat(process.env.MODEL_MEMORY_GB || '6');

(async () => {
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  for (const budget of [GB, 0]) {
    const context = await chromium.launchPersistentContext(path.join(OUT, 'site-profile'), {
      executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
      args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'],
    });
    const page = context.pages()[0] || await context.newPage();
    await page.addInitScript(gb => {
      try { const s = JSON.parse(localStorage.getItem('spawngen.settings') || '{}'); s.ModelMemoryGb = gb; localStorage.setItem('spawngen.settings', JSON.stringify(s)); } catch (e) { }
    }, budget);
    let lines = [];
    const problems = [];
    page.on('console', m => { const t = m.text(); if (m.type() === 'error') problems.push(t.slice(0, 200)); if (t.startsWith('[models]')) lines.push(t.slice(0, 260)); });
    page.on('pageerror', e => problems.push('PAGEERROR ' + e.message.slice(0, 200)));
    await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(4000);
    const read = async (sel, t = 120000) => (await page.locator(sel).first().textContent({ timeout: t }).catch(() => null)) ?? '';
    const runs = async () => parseInt(((await read('.toolbar')).match(/History \((\d+)\)/) || [])[1] || '0', 10);
    const run = async () => {
      const before = await runs(), t0 = Date.now();
      await page.click('button:has-text("Queue")');
      while (Date.now() - t0 < 600000) {
        await page.waitForTimeout(1500);
        const toast = await read('.toast', 300);
        if (/fail|error/i.test(toast) && !/Running/.test(await read('.status'))) return 'ERROR ' + toast.slice(0, 160);
        if ((await runs()) > before && !/Running|Queued/.test(await read('.status'))) return 'ran';
      }
      return 'timed out';
    };
    const open = async label => { await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu'); await page.locator('.menu li', { hasText: label }).first().click(); await page.waitForTimeout(2000); };
    const tabs = async () => (await page.locator('.tab').allTextContents()).map(t => t.replace('×', '').trim());
    const label = budget > 0 ? `budget ${budget} GB` : 'no budget';

    await open('Remove the background');
    const first = await run();
    await open('Depth map');
    const second = await run();
    check(`${label}: two examples run in two tabs`, first === 'ran' && second === 'ran', `${first} / ${second}; tabs: ${(await tabs()).join(' | ')}`);
    const droppedSoFar = lines.filter(l => /dropped|freed/.test(l));
    if (budget > 0) check(`${label}: both tabs' models are still loaded`, !droppedSoFar.some(l => /rmbg/.test(l)), droppedSoFar.join(' || ').slice(0, 300));
    else check(`${label}: the second workflow freed the first one's model (the older rule)`, droppedSoFar.some(l => /rmbg/.test(l)), droppedSoFar.join(' || ').slice(0, 300));

    // close the "Remove the background" tab
    const names = await tabs();
    const at = names.findLastIndex(n => /Remove the background/.test(n));   // the one this drive opened (the profile restores older tabs)
    lines = [];
    await page.locator('.tab').nth(at).locator('button[title="Close this workflow"]').click();
    await page.waitForTimeout(2500);
    const afterClose = lines.filter(l => /dropped/.test(l));
    if (budget > 0) {
      check(`${label}: closing a tab drops the model only it used`, afterClose.some(l => /rmbg/.test(l) && /Released/.test(l)) && !afterClose.some(l => /depth/.test(l)), afterClose.join(' || ').slice(0, 300));
      // the depth tab is the one left open: its workflow runs again (results forgotten) and loads nothing
      lines = [];
      await page.evaluate(() => window.spawngenForgetResults && window.spawngenForgetResults());
      const again = await run();
      check(`${label}: the open tab's workflow runs again on its kept model`, again === 'ran' && !lines.some(l => /dropped/.test(l)), again + ' ' + lines.join(' || ').slice(0, 200));
    } else {
      check(`${label}: closing a tab drops nothing (nothing is tagged)`, afterClose.length === 0, afterClose.join(' || ').slice(0, 300));
    }
    check(`${label}: no browser errors`, problems.filter(p => !/powerPreference/.test(p)).length === 0, problems.slice(0, 3).join(' || '));
    await page.screenshot({ path: path.join(OUT, `model-tabs-${budget > 0 ? 'budget' : 'nobudget'}.png`) }).catch(() => {});
    await context.close();
  }
  console.log(`${results.filter(Boolean).length}/${results.length} passed`);
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
