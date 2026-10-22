# Studio browser test

Drives the running Studio in headless Chrome with real pointer and keyboard events (drag, wheel zoom, double-click search, in-place
widget editing, save/open workflow, context menu) and checks the result through the toolbar status and the saved workflow JSON.
Screenshots go to `out/` (ignored by git).

```powershell
dotnet run --project ../../src/SpawnGen.Studio -c Release --urls http://localhost:5280   # in one terminal
npm install                                                                              # once
npm test                                                                                 # in another
```

Environment: `STUDIO_URL` (default `http://localhost:5280/`), `CHROME_PATH` (default the Windows Chrome install path).
The click coordinates assume a 1600x900 viewport and the starter graph, so they are fixed in the script. Run `drive.js` against the Studio alone (5280) or the server-hosted one via `STUDIO_URL`.

## End to end through the server

`npm run test:run` (against the server, default `http://localhost:8188/`) opens the Studio the server hosts, presses Queue, follows the run
and checks the server history. It needs a GPU and the SD-Turbo model, and takes about 40 s. Look at `out/run-3-done.png` afterwards: the
Save Image node should show the picture.
