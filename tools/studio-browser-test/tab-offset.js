// The scripts' mouse coordinates were chosen before the tab bar existed. Call this once the page is up: every mouse move, click and double click is
// then shifted down by half the tab bar's height (the fitted example graph is centred in the canvas, so it moved by half of what the canvas lost),
// and the same canvas positions keep working. Returns the offset.
module.exports = async function offsetMouseByTabBar(page) {
  const box = await page.locator('.tabs').boundingBox();
  const dy = box ? Math.round(box.height / 2) : 0;
  const mouse = page.mouse;
  for (const name of ['move', 'click', 'dblclick']) {
    const original = mouse[name].bind(mouse);
    mouse[name] = (x, y, ...rest) => original(x, y + dy, ...rest);
  }
  return dy;
};
