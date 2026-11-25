// The browser clears the site's stored models in the middle of a session (it may, for any site whose storage is not persistent): the next
// run must download what it needs again, not fail. Found 2026-11-24 on the live site: Chrome emptied the origin's storage during a row of
// examples and every later model load failed within 2 s with "A requested file or directory could not be found at the time an operation
// was processed." (SpawnDev.ILGPU.ML held one OPFS directory handle for the page's life; fixed in 5.8.28).
// Runs the Depth map example, empties the origin-private file system from the page (what the browser's own clearing leaves behind), runs
// Remove the background (another model: a fresh load), empties it again and runs Pose and object detection (a third model).
// STUDIO_URL (default http://localhost:5280/). Uses its own profile (out/cache-cleared-profile): it deletes that profile's cached models.
const { chromium } = require('playwright-core');
const path = require('path');
const URL = (process.env.STUDIO_URL || 'http://localhost:5280/') + '?nodetimes=1';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
let failed = 0;
const check = (ok, what) => { console.log((ok ? 'PASS ' : 'FAIL ') + what); if (!ok) failed++; };

(async () => {
  const ctx = await chromium.launchPersistentContext(path.join(__dirname, 'out', 'cache-cleared-profile'), { executablePath: CHROME, headless: true,
    viewport: { width: 1600, height: 900 }, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
  const page = ctx.pages()[0] || await ctx.newPage();
  const logs = [];
  page.on('console', m => logs.push(m.text()));
  // The model cache limit is set here, not left to what an earlier run stored in this profile: CACHE_GB (default 24 here; 0 = no limit, the Studio's own default).
  // Both must pass: with a limit the library also lists and trims the cache, which once had no recovery of its own (ML 5.8.28-local.3).
  await page.addInitScript(gb => { try { localStorage.setItem('spawngen.settings', JSON.stringify({ ModelCacheLimitGb: gb })); } catch (e) { } }, parseFloat(process.env.CACHE_GB || '24'));
  await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3500);
  const read = async sel => (await page.locator(sel).first().textContent({ timeout: 60000 }).catch(() => '')) || '';
  const runs = async () => parseInt(((await read('.toolbar')).match(/History \((\d+)\)/) || [])[1] || '0', 10);
  const runExample = async label => {
    await page.click('button:has-text("Examples")'); await page.waitForSelector('.menu');
    await page.locator('.menu li', { hasText: label }).first().click(); await page.waitForTimeout(1500);
    const before = await runs();
    await page.click('button:has-text("Queue")');
    for (let i = 0; i < 600; i++) {
      await page.waitForTimeout(1000);
      const toast = (await page.locator('.toast').first().textContent({ timeout: 200 }).catch(() => '')) || '';
      if (/Run failed/.test(toast)) return 'failed: ' + toast.trim().split('\n')[0].slice(0, 200);
      if ((await runs()) > before && !/Running|Queued/.test(await read('.status'))) return 'ran';
    }
    return 'timed out';
  };
  const stored = () => page.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    let n = 0; for await (const _ of root.keys()) n++;
    return n;
  });

  check((await runExample('Depth map')) === 'ran', 'Depth map runs (its model is downloaded and stored)');
  const before = await stored();
  check(before > 0, `the site's storage holds the model (${before} entries at its root)`);

  // What a browser clearing the site's storage leaves: the origin-private file system empty, every handle the page held gone stale.
  const left = await page.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const names = []; for await (const name of root.keys()) names.push(name);
    for (const name of names) await root.removeEntry(name, { recursive: true });
    let n = 0; for await (const _ of root.keys()) n++;
    return n;
  });
  check(left === 0, 'the stored models were removed behind the page');

  const other = await runExample('Remove the background');
  console.log('another model after the clearing: ' + other);
  check(other === 'ran', 'an example with another model runs after the clearing (its files download into a new store)');
  // A second clearing, then a third model (Depth map again would come from its cached result and load nothing).
  await page.evaluate(async () => { const root = await navigator.storage.getDirectory(); for await (const name of root.keys()) await root.removeEntry(name, { recursive: true }); });
  const again = await runExample('Pose and object detection');
  console.log('a third model after a second clearing: ' + again);
  check(again === 'ran', 'and again after a second clearing');
  check((await stored()) > 0, 'the store holds models again');
  const lines = logs.filter(l => /could not be found|\[models\]/.test(l)).slice(-6);
  if (lines.length) console.log('console: ' + lines.join(' || ').slice(0, 600));
  await ctx.close();
  console.log(failed ? `${failed} FAILED` : 'ALL PASS');
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
