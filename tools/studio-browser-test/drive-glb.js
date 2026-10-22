// The 3D viewer in the real page: opens a textured cube .glb (built here) over the page, checks it draws (the canvas is not blank), that dragging orbits
// and the wheel zooms (the picture changes), and that Escape closes it. Look at out/glb-cube.png: a blue/orange checkered cube.
// STUDIO_GLB=<file.glb> opens that file instead of the cube (and only screenshots it): out/glb-file.png.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { cubeGlb } = require('./make-test-glb');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:8188/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1000, height: 700 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const hash = async () => crypto.createHash('md5').update(await page.locator('.gv-canvas').screenshot()).digest('hex');
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(2500);

  if (process.env.STUDIO_GLB) {
    const b64 = fs.readFileSync(process.env.STUDIO_GLB).toString('base64');
    await page.evaluate(b => { window.__viewer = window.spawngenView3D('data:model/gltf-binary;base64,' + b, 'model'); }, b64);
    await page.waitForSelector('.glbviewer[data-ready="1"], .gv-status.error', { timeout: 60000 });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(OUT, 'glb-file.png') });
    const info = await page.evaluate(() => { const r = document.querySelector('.glbviewer'); return { ready: r.dataset.ready, tris: r.dataset.triangles, status: r.querySelector('.gv-status').textContent }; });
    console.log(JSON.stringify(info), problems.join(' | '));
    await browser.close();
    process.exit(info.ready === '1' ? 0 : 1);
  }

  const b64 = cubeGlb().toString('base64');
  await page.evaluate(b => { window.__viewer = window.spawngenView3D('data:model/gltf-binary;base64,' + b, 'test cube'); }, b64);
  await page.waitForSelector('.glbviewer[data-ready="1"]', { timeout: 15000 });
  await page.waitForTimeout(500);
  const info = await page.evaluate(() => ({ meshes: document.querySelector('.glbviewer').dataset.meshes, tris: document.querySelector('.glbviewer').dataset.triangles }));
  check('the cube loads: 1 mesh, 12 triangles', info.meshes === '1' && info.tris === '12', JSON.stringify(info));
  const first = await hash();
  await page.screenshot({ path: path.join(OUT, 'glb-cube.png') });
  const size = (await page.locator('.gv-canvas').screenshot()).length;
  check('the picture is not flat (a flat one compresses to a few KB)', size > 15000, size + ' bytes');
  const box = await page.locator('.gv-canvas').boundingBox();
  await page.mouse.move(box.x + 500, box.y + 300); await page.mouse.down(); await page.mouse.move(box.x + 620, box.y + 340, { steps: 6 }); await page.mouse.up();
  await page.waitForTimeout(200);
  const orbited = await hash();
  check('dragging orbits the model', orbited !== first);
  await page.mouse.wheel(0, -400); await page.waitForTimeout(200);
  check('the wheel zooms', (await hash()) !== orbited);
  await page.screenshot({ path: path.join(OUT, 'glb-cube-orbited.png') });
  await page.keyboard.press('Escape'); await page.waitForTimeout(300);
  check('Escape closes the viewer', (await page.locator('.glbviewer').count()) === 0);
  await page.evaluate(() => { window.__viewer = window.spawngenView3D('data:model/gltf-binary;base64,AAAA', 'broken'); });
  await page.waitForSelector('.gv-status.error');
  check('a file that is not a .glb says so', /not a \.glb/.test(await page.locator('.gv-status').textContent()));
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
