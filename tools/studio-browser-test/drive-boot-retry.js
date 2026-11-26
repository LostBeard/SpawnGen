// The page starts although the host answers a runtime file wrongly once, or the browser has a wrong answer in its cache (found on the live
// site, 2026-11-26: GitHub's CDN answered one framework .wasm with a 503 error page, Chrome kept it in its disk cache and the Studio could
// not start in that browser until its cache was cleared). The drive makes the first request of one framework file fail three ways and
// expects the Studio's canvas each time: an HTTP 503 with an error page, a 200 with the wrong bytes (what a cached error page looks like to
// the integrity check), and a dropped connection. The recovery is SpawnDev.Blazor.BootRecovery's (the team's package for
// dotnet/runtime#135090, referenced by the Studio): this drive checks that SpawnGen's own page, which starts Blazor by hand with a custom
// loadBootResource, really gets it.  STUDIO_URL (default http://localhost:5300/).
const { chromium } = require('playwright-core');
const URL = process.env.STUDIO_URL || 'http://localhost:5300/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const cases = {
    'an HTTP 503 with an error page': route => route.fulfill({ status: 503, contentType: 'text/html; charset=utf-8', body: '<html><body>503 Service Unavailable</body></html>' }),
    'a 200 with the wrong bytes (a cached error page)': route => route.fulfill({ status: 200, contentType: 'application/wasm', body: '<html><body>503 Service Unavailable</body></html>' }),
    'a dropped connection': route => route.abort('connectionreset'),
    'nothing wrong (control)': null,
  };
  for (const [label, fail] of Object.entries(cases)) {
    const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
    const page = await context.newPage();
    let failed = 0, target = null;
    const warnings = [];
    page.on('console', m => { if (m.text().startsWith('[SpawnDev.Blazor.BootRecovery]') || m.text().startsWith('[boot]')) warnings.push(m.text().slice(0, 170)); });
    if (fail) await page.route(/_framework\/System\.Collections\.[^/]*\.wasm(\.gz)?$/, async route => {
      target = route.request().url().split('/').pop();
      if (failed++ === 0) return fail(route);
      return route.continue();
    });
    await page.goto(URL);
    const started = await page.waitForSelector('canvas', { timeout: 90000 }).then(() => true).catch(() => false);
    const text = started ? '' : (await page.evaluate(() => document.body.innerText.slice(0, 120))).replace(/\n/g, ' / ');
    check(`the Studio starts after ${label}`, started && (!fail || (failed >= 2 && warnings.length >= 1)),
      fail ? `${target}: asked ${failed} times; ${warnings[0] || 'no recovery line'}${text ? '; page says: ' + text : ''}` : text);
    await context.close();
  }
  await browser.close();
  console.log(`${results.filter(Boolean).length}/${results.length} passed`);
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
