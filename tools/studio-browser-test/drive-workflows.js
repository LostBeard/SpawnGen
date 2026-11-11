// The workflow library through the server: Save asks a name and keeps the workflow in ComfyUI's userdata workflows/ folder; Ctrl+S saves it again;
// a taken name asks before replacing; the Workflows sidebar lists, renames and deletes; the Apps tab lists .app.json apps, opens them as the app
// and Create app opens the App Builder; a reload keeps which file each tab is; the browser's own store (IndexedDB, for a Studio no server serves)
// keeps the same rules. Every route also answers under /api. Run the server with SpawnGen__UserPath set to an empty folder: this drive writes there.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, acceptDownloads: true });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const shot = name => page.screenshot({ path: path.join(OUT, name + '.png') });
  const saved = () => page.evaluate(async () => { const r = await fetch('api/userdata?dir=workflows&recurse=true'); return r.ok ? r.json() : []; });
  const tabName = async () => (await page.locator('.tab.active span').textContent()).trim();
  const savedNames = () => page.locator('.workflows .lib-cat:has-text("Saved") ~ .wf-item span, .workflows .lib-cat:has-text("Apps") ~ .wf-item span').allTextContents();

  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  const before = await saved();
  check('the library starts empty (an empty user folder)', Array.isArray(before) && before.length === 0, JSON.stringify(before));

  // Workflows sidebar.
  await page.click('.toolbar button:has-text("Workflows")'); await page.waitForTimeout(500);
  check('Workflows opens the sidebar', (await page.locator('.workflows').count()) === 1);
  check('it lists the open tab, not saved', /not saved/.test(await page.textContent('.workflows .wf-item.active')));
  check('and says nothing is saved yet', /No saved workflows yet/.test(await page.textContent('.workflows')));

  // Save: the dialog asks for a name (focused), Enter saves into a folder.
  await page.click('.toolbar button:has-text("Save")'); await page.waitForTimeout(400);
  check('Save asks for a name the first time', (await page.locator('.save-dialog').count()) === 1);
  check('the name box has the focus', await page.evaluate(() => document.activeElement && document.activeElement.closest('.save-dialog') !== null));
  await page.fill('.save-dialog input', 'portraits/first');
  await page.keyboard.press('Enter'); await page.waitForTimeout(800);
  let files = await saved();
  check('Enter saves it in the server library, in a folder', files.length === 1 && files[0] === 'portraits/first.json', JSON.stringify(files));
  check('the tab is named after it', (await tabName()) === 'first');
  check('the sidebar lists it', (await savedNames()).includes('portraits/first'), (await savedNames()).join(' | '));
  const firstText = await page.evaluate(async () => (await fetch('api/userdata/workflows%2Fportraits%2Ffirst.json')).text());
  check('the saved file is the ComfyUI workflow', /"nodes"/.test(firstText) && /"links"/.test(firstText));
  await shot('wf-1-saved');

  // Ctrl+S on the canvas saves it again, without asking.
  await page.mouse.click(700, 760); await page.waitForTimeout(200);
  await page.keyboard.press('Control+s'); await page.waitForTimeout(800);
  check('Ctrl+S saves again without asking', (await page.locator('.save-dialog').count()) === 0 && (await saved()).length === 1);

  // A new tab saved under the same name asks before replacing.
  await page.click('.tabs .newtab'); await page.waitForTimeout(800);
  await page.click('.toolbar button:has-text("Save")'); await page.waitForTimeout(400);
  await page.fill('.save-dialog input', 'portraits/first');
  await page.click('.save-dialog button.primary'); await page.waitForTimeout(600);
  check('a taken name asks before replacing', /already saved/.test(await page.textContent('.save-dialog')) && /Replace/.test(await page.textContent('.save-dialog button.primary')));
  await shot('wf-2-replace');
  await page.fill('.save-dialog input', 'second');
  await page.click('.save-dialog button.primary'); await page.waitForTimeout(800);
  files = await saved();
  check('another name saves beside it', files.length === 2 && files.includes('second.json'), JSON.stringify(files));

  // Rename and delete in the sidebar.
  const row = page.locator('.workflows .wf-item', { hasText: 'portraits/first' });
  await row.hover(); await row.locator('button[title="Rename"]').click(); await page.waitForTimeout(300);
  await page.keyboard.press('Control+a'); await page.keyboard.type('renamed'); await page.keyboard.press('Enter'); await page.waitForTimeout(800);
  files = await saved();
  check('rename moves the file', files.includes('renamed.json') && !files.includes('portraits/first.json'), JSON.stringify(files));
  check('the tab that has it open takes the new name', (await page.locator('.tabs .tab span').allTextContents()).includes('renamed'));
  const del = page.locator('.workflows .lib-cat:has-text("Saved") ~ .wf-item', { hasText: 'second' });
  await del.hover(); await del.locator('button[title="Delete"]').click(); await page.waitForTimeout(300);
  check('delete asks first', (await page.locator('.workflows .wf-item.confirm').count()) === 1);
  await page.click('.workflows .wf-item.confirm button.danger'); await page.waitForTimeout(800);
  files = await saved();
  check('and deletes the file', files.length === 1 && files[0] === 'renamed.json', JSON.stringify(files));
  check('the tab stays open, not saved any more', /not saved/.test(await page.textContent('.workflows .wf-item.active')));

  // An app file (as ComfyUI saves one) in the Apps tab.
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('.toolbar button:has-text("Export")')]);
  const wf = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
  const prompt = wf.nodes.find(n => n.type === 'CLIPTextEncode'), save = wf.nodes.find(n => n.type === 'SaveImage');
  wf.extra = Object.assign(wf.extra || {}, { linearMode: true, linearData: { inputs: [[String(prompt.id), 'text']], outputs: [String(save.id)] } });
  const posted = await page.evaluate(async t => (await fetch('api/userdata/workflows%2FMy%20app.app.json?overwrite=false', { method: 'POST', body: t })).status, JSON.stringify(wf));
  check('an app file is stored through /api/userdata', posted === 200, String(posted));
  await page.click('.workflows .wf-tabs button:has-text("Apps")'); await page.waitForTimeout(600);
  check('the Apps tab lists only apps', JSON.stringify(await savedNames()) === JSON.stringify(['My app']), (await savedNames()).join(' | '));
  await shot('wf-3-apps');
  await page.click('.workflows .wf-item:has-text("My app")'); await page.waitForTimeout(1200);
  check('an app opens as the app', (await page.locator('.appmode:not(.preview)').count()) === 1);
  check('in a tab named after it', (await tabName()) === 'My app');
  check('with its input', (await page.locator('.appmode .app-field').count()) === 1);
  await page.click('.app-in button:has-text("Node graph")'); await page.waitForTimeout(400);
  await page.click('.toolbar button:has-text("Workflows")'); await page.waitForTimeout(400);
  if ((await page.locator('.workflows').count()) === 0) { await page.click('.toolbar button:has-text("Workflows")'); await page.waitForTimeout(400); }
  await page.click('.workflows .wf-tabs button:has-text("Apps")'); await page.waitForTimeout(400);
  await page.click('.workflows .wf-create'); await page.waitForTimeout(600);
  check('Create app opens the App Builder', (await page.locator('.app-steps').count()) === 1);
  await page.click('.app-builder-bar button:has-text("Exit")'); await page.waitForTimeout(400);

  // A reload keeps which file each tab is.
  await page.reload(); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  await page.click('.toolbar button:has-text("Workflows")'); await page.waitForTimeout(600);
  const open = await page.locator('.workflows .lib-cat:has-text("Open") ~ .wf-item').evaluateAll(els => els.map(e => e.title));
  check('after a reload the tabs still know their files', open.includes('renamed.json') && open.includes('My app.app.json'), open.join(' | '));
  await page.click('.workflows .wf-item:has-text("renamed") >> nth=-1'); await page.waitForTimeout(800);
  check('opening a saved workflow that is open switches to its tab', (await tabName()) === 'renamed' && (await page.locator('.tabs .tab').count()) === 3, String(await page.locator('.tabs .tab').count()));

  // The browser's own store keeps the same rules (used when no server serves the Studio).
  const idb = await page.evaluate(async () => {
    const f = window.spawngenFiles, r = [];
    r.push(await f.write('t/a.json', '{"a":1}', false));
    r.push(await f.write('t/a.json', '{"a":2}', false));
    r.push(await f.write('t/a.json', '{"a":3}', true));
    r.push(await f.read('t/a.json'));
    r.push(await f.write('b.app.json', '{}', true));
    r.push(await f.move('t/a.json', 'b.app.json', false));
    r.push(await f.move('t/a.json', 'c.json', false));
    r.push(await f.move('nope.json', 'd.json', true));
    r.push(JSON.parse(await f.list()).map(x => x.path).sort().join(','));
    r.push(await f.remove('c.json'), await f.remove('c.json'), await f.remove('b.app.json'));
    r.push(JSON.parse(await f.list()).length);
    return r;
  });
  check('the browser store writes, refuses a taken name, reads, moves and deletes',
    JSON.stringify(idb) === JSON.stringify(['done', 'exists', 'done', '{"a":3}', 'done', 'exists', 'done', 'missing', 'b.app.json,c.json', 'done', 'missing', 'done', 0]), JSON.stringify(idb));

  const api = await page.evaluate(async () => [(await fetch('api/object_info/SaveImage')).status, (await fetch('api/queue')).status, (await fetch('object_info/SaveImage')).status]);
  check('routes answer with and without /api', api.every(s => s === 200), api.join(','));
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
