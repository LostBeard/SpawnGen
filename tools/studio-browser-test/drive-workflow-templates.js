// ComfyUI's workflow templates in the real page on the server: Templates opens the browser (the server fetches Comfy-Org's MIT templates once and keeps
// them); the cards show with thumbnails; a search finds Z-Image Turbo; a click opens it in a new tab with ComfyUI's graph (read through Export).
//   STUDIO_URL (default http://localhost:8188/). Needs the internet the first time (the server's copies serve after that).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };

  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3000);
  await page.click('.toolbar button:has-text("Templates")');
  await page.waitForSelector('.templates .tpl-card', { timeout: 60000 });
  const count = Number(await page.textContent('.templates .tpl-count'));
  const groups = await page.locator('.templates .tpl-groups button').allTextContents();
  check('the templates load (all of them, in ComfyUI\'s groups)', count > 300 && groups.length > 5, `${count} templates; groups ${groups.join(', ')}`);
  await page.waitForTimeout(2500);
  const thumbs = await page.evaluate(() => [...document.querySelectorAll('.templates .tpl-thumb img')].slice(0, 12).filter(i => i.complete && i.naturalWidth > 0).length);
  check('the cards show their thumbnails', thumbs >= 6, `${thumbs} of the first 12 loaded`);
  await page.screenshot({ path: path.join(OUT, 'templates-0-open.png') });

  await page.fill('.templates .tpl-head input', 'z-image turbo text');
  await page.waitForTimeout(400);
  const titles = await page.locator('.templates .tpl-title').allTextContents();
  check('a search narrows them', titles.length > 0 && titles.length < 20 && titles.some(t => /Z-Image-Turbo/i.test(t)), titles.slice(0, 6).join(' | '));
  const apiMarks = await page.locator('.templates .tpl-api').count();
  await page.locator('.templates .tpl-card', { hasText: 'Z-Image-Turbo: Text to Image' }).first().click();
  await page.waitForTimeout(2500);
  check('the browser closes on opening one', await page.locator('.templates').count() === 0);
  const tabs = await page.locator('.tabs .tab, .tabbar .tab, [class*=tab] span').allTextContents();
  const [download] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }), page.click('.toolbar button:has-text("Export")')]);
  const saved = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
  // The template's graph is a subgraph (as ComfyUI saves it): its nodes are in definitions.subgraphs.
  const all = [...saved.nodes, ...((saved.definitions && saved.definitions.subgraphs) || []).flatMap(g => g.nodes || [])];
  const types = all.map(n => n.type);
  const unet = all.find(n => n.type === 'UNETLoader');
  check('it opens ComfyUI\'s Z-Image Turbo graph', !!unet && JSON.stringify(unet.widgets_values).includes('z_image_turbo') && types.includes('KSampler'),
    `${types.length} nodes: ${[...new Set(types)].join(', ').slice(0, 200)}`);
  await page.screenshot({ path: path.join(OUT, 'templates-1-opened.png') });
  console.log(`(API marks on the search results: ${apiMarks})`);
  check('no page errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await browser.close();
  console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
  process.exit(results.every(Boolean) ? 0 : 1);
})();
