// The scripts' mouse coordinates were chosen before the tab bar (top) and the run status bar (bottom) existed. Call this once the page is up: every
// mouse move, click and double click is then shifted by half the tab bar's height minus half the status bar's (the fitted example graph is centred in
// the canvas, so it moved down by half of what the canvas lost at the top and up by half of what it lost at the bottom), and the same canvas
// positions keep working. Returns the offset.
// The toolbar has since grown past one line at 1600 px (the batch field wraps under the buttons): its extra rows take height from the canvas at
// the top the same way, so half of that extra (the bottom of its last row minus the bottom of its first) is added too.
module.exports = async function offsetMouseByTabBar(page) {
  const half = async selector => { const box = await page.locator(selector).boundingBox(); return box ? box.height / 2 : 0; };
  const toolbarExtra = await page.evaluate(() => {
    const bar = document.querySelector('.toolbar');
    if (!bar) return 0;
    const bottoms = [...bar.children].map(e => e.getBoundingClientRect()).filter(r => r.height > 0).map(r => r.bottom);
    return bottoms.length ? Math.max(...bottoms) - Math.min(...bottoms) : 0;
  });
  const dy = Math.round(await half('.tabs') - await half('.runbar') + toolbarExtra / 2);
  const mouse = page.mouse;
  for (const name of ['move', 'click', 'dblclick']) {
    const original = mouse[name].bind(mouse);
    mouse[name] = (x, y, ...rest) => original(x, y + dy, ...rest);
  }
  return dy;
};
