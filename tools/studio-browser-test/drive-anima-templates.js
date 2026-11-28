// Anima templates in the Studio's template browser (2026-11-28, Lore): with Load Model Patch and Apply Anima LLLite in the build, the four Anima
// templates whose nodes SpawnGen has open with no missing-node mark (preview, base v1, LLLite inpainting, LLLite any-control), and the depth one
// still names its Depth Anything 3 nodes. STUDIO_URL (default http://localhost:5300/), a published Studio.
const { chromium } = require('playwright-core');
const URL = process.env.STUDIO_URL || 'http://localhost:5300/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) failed++; };
(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
    const errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    await page.goto(URL); await page.waitForSelector('canvas'); await page.waitForTimeout(3000);
    await page.click('button:has-text("Templates")');
    await page.waitForSelector('.templates .tpl-card', { timeout: 60000 });
    await page.click('.templates .tpl-show:has-text("All")');
    await page.fill('.templates input[placeholder="Search ComfyUI\'s templates..."]', 'anima');
    await page.waitForTimeout(1500);
    const want = {
      image_anima_preview: null, image_anima_base_v1: null, image_anima_lllite_image_inpainting: null, image_anima_lllite_any_control_to_image: null,
      image_anima_lllite_depth_control_to_image: /DA3/,
    };
    for (const [name, missing] of Object.entries(want)) {
      const card = page.locator(`.templates .tpl-card:has(img[src*="${name}-1."])`);
      if (await card.count() !== 1) { check(false, `${name}: card found (${await card.count()})`); continue; }
      const mark = card.locator('.tpl-missing');
      const tip = (await mark.count()) ? await mark.getAttribute('title') : null;
      if (missing) check(tip && missing.test(tip) && !/AnimaLLLiteApply|ModelPatchLoader/.test(tip), `${name}: still missing only its own nodes (${tip})`);
      else check(!tip, `${name}: no missing nodes (${tip || 'none'})`);
    }
    check(errors.length === 0, `no page errors ${errors.slice(0, 2).join(' | ')}`);
  } catch (e) { console.log('SCRIPT ERROR', e); failed++; }
  finally { await browser.close(); }
  console.log(failed ? `${failed} FAILED` : 'ALL PASS');
  process.exit(failed ? 1 : 0);
})();
