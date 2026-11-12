// The History panel lists the server's earlier runs after the page (re)opens, as ComfyUI's queue panel lists its /history: runs queued over the
// API before the page was opened show up with their pictures, marked "earlier", and "Load workflow" opens the workflow they were queued with.
// Needs a server-hosted Studio (no GPU: Empty Image -> Save Image). STUDIO_URL defaults to http://localhost:8188/.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const BASE = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

async function queue(color) {
  const workflow = { last_node_id: 2, last_link_id: 1, version: 0.4, links: [[1, 1, 0, 2, 0, 'IMAGE']], groups: [], config: {}, extra: {},
    nodes: [
      { id: 1, type: 'EmptyImage', pos: [100, 100], size: [300, 130], flags: {}, order: 0, mode: 0, inputs: [], outputs: [{ name: 'IMAGE', type: 'IMAGE', links: [1] }], properties: {}, widgets_values: [64, 48, 1, color] },
      { id: 2, type: 'SaveImage', pos: [500, 100], size: [300, 270], flags: {}, order: 1, mode: 0, inputs: [{ name: 'images', type: 'IMAGE', link: 1 }], outputs: [], properties: {}, widgets_values: ['history_probe'] },
    ] };
  const body = {
    prompt: { '1': { class_type: 'EmptyImage', inputs: { width: 64, height: 48, batch_size: 1, color } },
              '2': { class_type: 'SaveImage', inputs: { images: ['1', 0], filename_prefix: 'history_probe' } } },
    client_id: 'drive', extra_data: { extra_pnginfo: { workflow } },
  };
  const r = await fetch(BASE + 'prompt', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const id = (await r.json()).prompt_id;
  for (let i = 0; i < 100; i++) {
    const h = await (await fetch(BASE + 'history/' + id)).json();
    if (h[id]) return id;
    await new Promise(res => setTimeout(res, 100));
  }
  throw new Error('the run did not finish');
}

(async () => {
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  await queue(0x336699);
  await queue(0x993366);

  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  await page.goto(BASE);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(4000);

  const label = await page.locator('button:has-text("History")').textContent();
  check('the History button counts the earlier runs', /History \(\d+\)/.test(label), label);
  await page.click('button:has-text("History")');
  await page.waitForSelector('.history');
  const titles = await page.locator('.hist-title').allTextContents();
  check('the earlier runs are listed, marked earlier, newest first', titles.length >= 2 && titles.every(t => /earlier/.test(t)) && /done/.test(titles[0]), titles.slice(0, 3).join(' | '));
  const nums = titles.map(t => parseInt((/#(\d+)/.exec(t) || [])[1]));
  check('newest first by the server number', nums[0] > nums[1], nums.slice(0, 3).join(','));
  const loaded = await page.evaluate(() => [...document.querySelectorAll('.hist-thumbs img')].slice(0, 2).every(i => i.complete && i.naturalWidth === 64));
  check('their pictures load from the server', loaded);
  await page.screenshot({ path: path.join(OUT, 'server-history.png') });

  const tabsBefore = await page.locator('.tabs .tab').count();
  await page.locator('.hist-entry').first().locator('button:has-text("Load workflow")').click();
  await page.waitForTimeout(1500);
  const tabsAfter = await page.locator('.tabs .tab').count();
  check('Load workflow opens the queued workflow in a new tab', tabsAfter === tabsBefore + 1, `${tabsBefore} -> ${tabsAfter}`);
  const nodes = await page.evaluate(() => document.querySelector('.status')?.textContent || '');
  await page.screenshot({ path: path.join(OUT, 'server-history-loaded.png') });

  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await browser.close();
  const failed = results.filter(r => !r).length;
  console.log(failed ? `${failed} FAILED` : 'ALL PASSED');
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
