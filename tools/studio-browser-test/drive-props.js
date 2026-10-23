// The properties panel in the real page: I opens it, it shows the selected node, its form changes the node (one undo step each), the input links
// lead to the source node, several selected nodes change together, and it sits left of the History panel. Look at out/props-*.png.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });
const URL = process.env.STUDIO_URL || 'http://localhost:5280/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const problems = [];
  page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
  page.on('pageerror', e => problems.push('PAGEERROR ' + e.message));
  const results = [];
  const check = (name, ok, detail) => { results.push(ok); console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); };
  const panel = page.locator('.props');
  const field = label => page.locator(`.props label:has(> span:text-is("${label}")) input, .props label:has(> span:text-is("${label}")) select`).first();
  const titleField = () => page.locator('.props label:has-text("Title") input');
  const typeText = async () => (await page.locator('.props .props-kv:has(> span:text-is("Type"))').textContent()).trim();
  const wait = ms => page.waitForTimeout(ms);

  await page.goto(URL);
  await page.evaluate(() => { localStorage.clear(); });
  await page.goto(URL);
  await page.waitForSelector('canvas');
  await require('./tab-offset')(page);
  await wait(3500);

  await page.mouse.click(200, 700); // focus the canvas, nothing selected
  await page.keyboard.press('i');
  await wait(300);
  check('I opens the panel', await panel.count() === 1);
  check('with nothing selected it says how to use it', /Select a node/.test(await panel.textContent()));

  await page.mouse.click(1040, 234); // the KSampler's title bar
  await wait(400);
  check('it shows the selected node', /KSampler/.test(await typeText()), await typeText());
  const steps = field('steps');
  check('every parameter has a field (steps, cfg, sampler_name, seed control)',
    await steps.count() === 1 && await field('cfg').count() === 1 && await field('sampler_name').count() === 1 && await field('control_after_generate').count() === 1);
  const before = await steps.inputValue();
  await steps.fill('7');
  await steps.press('Enter');
  await wait(400);
  await page.screenshot({ path: path.join(OUT, 'props-ksampler.png') });
  check('changing steps in the panel changes the node', (await field('steps').inputValue()) === '7');
  await page.locator('canvas').focus();
  await page.keyboard.press('Control+z');
  await wait(400);
  check('one undo puts it back', (await field('steps').inputValue()) === before, `${before} -> ${await field('steps').inputValue()}`);

  await field('steps').fill('100000');
  await field('steps').press('Enter');
  await wait(300);
  check('numbers are kept inside the widget range', parseInt(await field('steps').inputValue()) <= 10000, await field('steps').inputValue());

  await field('sampler_name').selectOption('dpmpp_2m');
  await wait(300);
  check('combos change from the panel', (await field('sampler_name').inputValue()) === 'dpmpp_2m');

  await titleField().fill('Main sampler');
  await titleField().press('Enter');
  await wait(300);
  check('the title can be changed', (await titleField().inputValue()) === 'Main sampler');
  await page.locator('.props label:has-text("Mode") select').selectOption({ label: 'Bypass' });
  await wait(300);
  check('the mode can be changed', (await page.locator('.props label:has-text("Mode") select').inputValue()) === '4');
  await page.locator('.props label:has-text("Mode") select').selectOption({ label: 'Always' });

  // An input link leads to its source node.
  const modelLink = page.locator('.props .props-kv:has(> span:has-text("model")) button.link');
  check('inputs say where they come from', /Load Checkpoint/.test(await modelLink.textContent()), await modelLink.textContent());
  await modelLink.click();
  await wait(400);
  check('following the link selects the source node', /CheckpointLoaderSimple/.test(await typeText()), await typeText());

  // Several nodes at once.
  await page.locator('canvas').focus();
  await page.keyboard.press('Control+a');
  await wait(400);
  check('several selected nodes are counted', /7 nodes selected/.test(await panel.textContent()));
  await page.locator('.props label:has-text("Colour") select').selectOption('green');
  await wait(400);
  await page.screenshot({ path: path.join(OUT, 'props-many.png') });
  check('one colour for all of them', (await page.locator('.props label:has-text("Colour") select').inputValue()) === 'green');
  await page.locator('canvas').focus();
  await page.keyboard.press('Control+z');
  await wait(400);
  check('and one undo takes it off all of them', (await page.locator('.props label:has-text("Colour") select').inputValue()) === 'default');

  // Side by side with History.
  await page.keyboard.press('h');
  await wait(400);
  const p = await panel.boundingBox(), h = await page.locator('.history').boundingBox();
  check('it moves left of the History panel', p && h && p.x + p.width <= h.x + 1, `${p && p.x + p.width} vs ${h && h.x}`);
  await page.keyboard.press('h');
  await page.keyboard.press('i');
  await wait(300);
  check('I closes it again', await panel.count() === 0);
  check('no browser errors', problems.length === 0, problems.join(' | '));
  await browser.close();
  process.exit(results.every(Boolean) ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
