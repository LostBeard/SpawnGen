// The Studio keeps its settings on the server it is served from (shared by every browser): drives that change settings clear them at the start
// and the end, so they start from the defaults and leave none behind for the next drive. Clearing stores null; the next page load then takes
// its browser's settings (a fresh profile's defaults) and stores those.
module.exports = async page => {
  await page.evaluate(async () => {
    try { await fetch('api/settings/SpawnGen.Settings', { method: 'POST', body: 'null', headers: { 'Content-Type': 'application/json' } }); } catch (e) { }
  });
};
