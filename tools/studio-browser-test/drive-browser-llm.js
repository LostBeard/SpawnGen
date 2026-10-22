// A language model in the BROWSER on WebGPU: Text Generation (LLM) -> Preview Any on "This browser"; the text appears in the Preview Any node
// (out/browser-llm.png). The model (Qwen2.5 0.5B, 530 MB) downloads on first use; keep PROFILE_DIR to reuse it.
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const URL = process.env.STUDIO_URL || 'http://localhost:5300/';
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PROFILE = process.env.PROFILE_DIR || path.join(__dirname, 'out', 'webgpu-profile');
const OUT = path.join(__dirname, 'out');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const context = await chromium.launchPersistentContext(PROFILE, {
    executablePath: CHROME, headless: true, viewport: { width: 1600, height: 900 },
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'],
  });
  const page = context.pages()[0] || await context.newPage();
  const problems = [];
  const started = Date.now();
  const t = () => `+${((Date.now() - started) / 1000).toFixed(0)}s`;
  page.on('console', m => { if (m.type() === 'error') { problems.push(m.text()); console.log(t(), 'console error:', m.text().slice(0, 300)); } });
  page.on('pageerror', e => { problems.push('PAGEERROR ' + e.message); console.log(t(), 'PAGEERROR', e.message.slice(0, 300)); });
  await page.goto(URL);
  await page.waitForSelector('canvas');
  await page.waitForTimeout(3500);
  await page.selectOption('select.engine', { label: 'This browser (WebGPU)' });
  const prompt = {
    '1': { class_type: 'TextGeneration', inputs: { prompt: 'Write one vivid sentence describing a single red apple on a wooden table, for an image generator. Answer with the sentence only.', system_prompt: 'You write short prompts for image generators.', model: 'qwen2.5-0.5b-instruct', max_new_tokens: 60, temperature: 0 } },
    '2': { class_type: 'PreviewAny', inputs: { source: ['1', 0] } },
  };
  const promptPath = path.join(OUT, 'llm-prompt.json');
  fs.writeFileSync(promptPath, JSON.stringify(prompt));
  await page.setInputFiles('input[accept=".json,.png"]', promptPath);
  await page.waitForTimeout(1500);
  await page.locator('button:has-text("Queue")').click();
  let last = '';
  for (let i = 0; i < 1800; i++) {
    await page.waitForTimeout(1000);
    const s = (await page.textContent('.status')).trim().split(' \u00b7 ')[0];
    if (s !== last && /Running|Queued/.test(s)) { console.log(t(), s); last = s; }
    if (i > 5 && !/Running|Queued/.test(s)) break;
  }
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(OUT, 'browser-llm.png') });
  console.log(t(), 'status:', (await page.textContent('.status')).trim());
  console.log('errors:', problems.length ? problems.join('\n') : 'none');
  await context.close();
  process.exit(problems.length === 0 ? 0 : 1);
})().catch(e => { console.error('SCRIPT ERROR', e); process.exit(2); });
