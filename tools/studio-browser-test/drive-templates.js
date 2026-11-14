// Node templates through the server, as ComfyUI's: select two nodes, the canvas menu's Save Selected as Template asks a name and writes the user's
// comfy.templates.json; Node Templates inserts them (with the link between them) where the menu was opened; a taken name asks before replacing;
// Manage renames, exports and imports (a templates file ComfyUI wrote) and deletes; a reload keeps them. Needs a server whose user folder has no
// comfy.templates.json (SpawnGen__UserPath set to an empty folder). Coordinates as drive-menus.js (menus-workflow.json at 1600x900).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const COMFY_FILE = path.join(__dirname, '..', '..', 'tests', 'SpawnGen.Editor.Tests', 'Fixtures', 'comfy.templates.json');

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const counts = async () => { const m = (await page.textContent('.status')).match(/(\d+) nodes, (\d+) links/); return m ? [+m[1], +m[2]] : [-1, -1]; };
  const canvasMenu = async (x = 300, y = 780) => { await page.mouse.click(x, y, { button: 'right' }); await page.waitForTimeout(300); return page.locator('.menu > li').allTextContents(); };
  const stored = () => page.evaluate(async () => {
    const files = await (await fetch('api/userdata?dir=.')).json();
    return files.includes('comfy.templates.json') ? (await fetch('api/userdata/comfy.templates.json')).json() : null;
  });
  const templateNames = async () => { await canvasMenu(); await page.click('.menu li.has-sub:has-text("Node Templates")'); const n = await page.locator('.menu li.has-sub.open .sub li').allTextContents(); await page.mouse.click(300, 820); await page.waitForTimeout(200); return n; };

  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  check('no templates to start with', (await stored()) === null);
  await page.setInputFiles('input[accept=".json,.png"]', path.join(__dirname, 'menus-workflow.json')); await page.waitForTimeout(1200);

  // Select the top Text and Preview as Text, save them as a template.
  await page.mouse.click(590, 313);
  await page.keyboard.down('Shift'); await page.mouse.click(1000, 313); await page.keyboard.up('Shift');
  let items = await canvasMenu();
  check('with a selection the canvas menu offers Save Selected as Template', items.some(t => /Save Selected as Template/.test(t)), items.join(' | '));
  await page.click('.menu li:has-text("Save Selected as Template")'); await page.waitForTimeout(400);
  check('it asks for a name (focused)', await page.evaluate(() => !!document.activeElement && !!document.activeElement.closest('.save-dialog')));
  await page.keyboard.type('Pair'); await page.keyboard.press('Enter'); await page.waitForTimeout(800);
  let file = await stored();
  const data = file && JSON.parse(file[0].data);
  check('comfy.templates.json holds it in ComfyUI\'s shape', file && file.length === 1 && file[0].name === 'Pair' && data.nodes.length === 2 && data.links.length === 1
    && data.links[0].origin_id !== undefined && Array.isArray(data.subgraphs), JSON.stringify(file && file.map(f => f.name)));

  // Insert it.
  await page.mouse.click(300, 820);
  const before = await counts();
  await canvasMenu(300, 150);
  await page.click('.menu li.has-sub:has-text("Node Templates")');
  await page.click('.menu .sub li:text-is("Pair")'); await page.waitForTimeout(500);
  const after = await counts();
  check('Node Templates inserts the nodes and the link between them', after[0] === before[0] + 2 && after[1] === before[1] + 1, `${before} -> ${after}`);
  await page.screenshot({ path: path.join(__dirname, 'out', 'templates-1.png') });

  // The same name again asks first.
  await page.mouse.click(300, 820);
  await page.mouse.click(590, 573);
  await canvasMenu();
  await page.click('.menu li:has-text("Save Selected as Template")'); await page.waitForTimeout(400);
  await page.keyboard.type('Pair'); await page.keyboard.press('Enter'); await page.waitForTimeout(400);
  check('a taken name asks before replacing', /Replace it/.test(await page.textContent('.save-dialog')));
  await page.click('.save-dialog button.primary'); await page.waitForTimeout(800);
  file = await stored();
  check('and replaces it', file.length === 1 && JSON.parse(file[0].data).nodes.length === 1);

  // Manage: rename, import ComfyUI's file, export, delete.
  await canvasMenu();
  await page.click('.menu li.has-sub:has-text("Node Templates")');
  await page.click('.menu .sub li:text-is("Manage...")'); await page.waitForTimeout(400);
  await page.fill('.templates-manage .tpl-row input', 'Text only'); await page.locator('.templates-manage .tpl-row input').dispatchEvent('change'); await page.waitForTimeout(600);
  check('a template is renamed', (await stored())[0].name === 'Text only');
  await page.setInputFiles('.templates-manage input[type=file]', COMFY_FILE);
  for (let i = 0; i < 25 && (file = await stored()).length < 2; i++) await page.waitForTimeout(200); // the import reads the file, then saves
  check('a templates file ComfyUI wrote is imported', JSON.stringify(file.map(f => f.name)) === '["Text only","My pair"]', JSON.stringify(file.map(f => f.name)));
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('.templates-manage button:has-text("Export")')]);
  const exported = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
  check('Export downloads comfy.templates.json', download.suggestedFilename() === 'comfy.templates.json' && exported.length === 2);
  await page.click('.templates-manage .tpl-row button'); await page.waitForTimeout(600);
  check('a template is deleted', JSON.stringify((await stored()).map(f => f.name)) === '["My pair"]');
  await page.click('.templates-manage button:has-text("Close")'); await page.waitForTimeout(300);

  // ComfyUI's template goes in too, and a reload keeps the list.
  const c0 = await counts();
  await canvasMenu(1200, 680);
  await page.click('.menu li.has-sub:has-text("Node Templates")');
  await page.click('.menu .sub li:text-is("My pair")'); await page.waitForTimeout(500);
  const c1 = await counts();
  check('ComfyUI\'s template inserts VAE Decode and Save Image with their link', c1[0] === c0[0] + 2 && c1[1] === c0[1] + 1, `${c0} -> ${c1}`);
  await page.reload(); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  check('a reload keeps the templates', JSON.stringify(await templateNames()) === '["My pair","Manage..."]', JSON.stringify(await templateNames()));

  // Leave none behind for the next run.
  await page.evaluate(async () => { await fetch('api/userdata/comfy.templates.json', { method: 'DELETE' }); });
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
