# SpawnGen plans

The one place for what SpawnGen is doing next, what it waits on, and why. Read it first after any reset; change it in the same commit as the
work that changes it. What is already done and how well it matches ComfyUI is in [PARITY.md](PARITY.md).

**Goal.** A ComfyUI-class node-graph studio in C#: ComfyUI's workflows and templates open and run the same, on CUDA, OpenCL and WebGPU in the
browser, with no Python. SpawnGen also works out the SpawnDev libraries on a real project: what SpawnGen needs from SpawnDev.ILGPU.ML
(Tuvok) or SpawnDev.Codecs (Geordi) is asked of them, ranked by what it unlocks, never worked around and never treated as a stop.

## Live site bug (2026-11-20): Save Image's PNGs are broken in the browser

The live deploy (a24eb38, Codecs 1.1.2) saves Save Image PNGs that no decoder opens: SpawnDev.Codecs' PngEncoderGpu on WebGPU writes a bad
deflate stream (every chunk CRC valid). CPU and OpenCL encode the same 512x512 picture correctly. Affects every example ending in Save
Image (SD-Turbo, Image to image, ...). Asked of Geordi with the exact failing pixels; when a Codecs -local lands: the browser repro
(drive-site-examples ONLY='SD-Turbo, 1 step|Image to image' against a local publish), then the next deploy. drive-site-examples now fails
any run whose result picture cannot be decoded (the old check passed this). TJ (2026-11-20): no interim host-PNG workaround on the site; wait for the correct fix in Codecs. TRELLIS on the HF Space is retested in that deploy (ML 5.8.25
has the 3-submits-per-block fix; on 5.8.24 it tripped the RX 590's driver watchdog twice).

## Next deploy (TJ, 2026-11-20)

**2026-11-27: main 2d0c2bb = SpawnDev.ILGPU.ML 5.8.37 (resumable browser downloads as OPFS part files; old entries read as they are) +
the Painter widget + the ported mask editor + pages.yml swap for the AOT peak.** Verified on the RX 590 against a local publish: suite
3563 / 0 / 3, two visits of the 17 examples 17 / 17 (30m19s, 29m37s), drives green. Pages run 46093517557 dispatched; then the Space
(`_scratch/deploy-wait-publish.sh`), then `_scratch/live-check.sh 5837` on both URLs (GPU: ask Riker for a window first; the painter,
painter-subgraph and maskeditor drives are in it now). **LIVE and CHECKED 2026-11-27 21:40:** Pages 17 / 17 (sum 1720 s, -0.2 % on
the last live record), Space 5 / 5, every drive on both sites (painter-subgraph after a drive fix: it read results while the canvas
showed the subgraph's inside, where the root's previews are not fetched). The AOT build took 13 min (not an hour); RAM peaked at
15.6 / 16 GB (three clang 5.3 + 4.8 + 3.2 GB, wasm-opt 10.4 GB) and the swap stayed unused: keep it, the margin is thin.

**2026-11-23: main = SpawnDev.ILGPU 5.4.46 + SpawnDev.Codecs 1.1.4 + SpawnDev.ILGPU.ML 5.8.25 (811749f).** Suite 3394 / 0 / 3; the explicit
GPU codec tests 12 / 12 on the RX 590 (OpenCL); in Chrome on the RX 590 (D3D12) against a local non-AOT publish: SD-Turbo 63 s, Image to
image 16 s, Text to video 167 s (models already in OPFS), every result decodes and was looked at. Heavy runs go through
`tools/run-guarded.ps1` (RAM watchdog, kills the run's tree below 2 GB free). In order now:
(1) Tuvok's 5.8.26 gate on branch `ml-local-5826` (worktree `_scratch/wt-ml5826`, ML 5.8.26-local.2 = ML master c6deede1 on ILGPU 5.4.46,
CompilerCachePolicy): FLUX.2 Klein picture edit with ?perop=1, then TRELLIS, in Chrome on the RX 590; then
`CompilerCachePolicy.ModelWarmedUp(accelerator)` after the first complete run of the models SpawnGen builds from components (SD, Z-Image,
Qwen-Image, FLUX.1, ACE-Step) with the browser heap before / after (_DevComms 2026-11-23-0830). (2) Geordi's numbers on Codecs 1.1.4: lossy
Save Animated WEBP per frame in Chrome against 1.1.3, the PNG decode corpus + example.png on branch `load-image-device` (worktree
`_scratch/wt-lid`, main merged in, not pushed until it builds and passes). (3) drive-appmode.js on the real GPU. (4) The deploy and the
full live check on both URLs: with ML 5.8.26 if Tuvok releases it after (1), else as main is.

**Resume Sunday 2026-11-22 (weekly quota reset; TJ asked all to wrap up 2026-11-20 ~19:40).** In order: (1) RX 590 runs of ML
5.8.26-local.1 when Tuvok packs it (after Data's 5.4.43 gate): FLUX.2 Klein picture edit with ?perop=1 (driver watchdog today) and TRELLIS;
(2) Geordi's PNG decode numbers on the RX 590 (Codecs 1.1.4-local.2, branch load-image-device: PNG eXIf + unfilter wavefront; WebGPU 512
PNG was >2 min / TDR on 1.1.3); (3) drive-appmode.js: drop --disable-gpu and run it on the real GPU (TJ 2026-11-20: SwiftShader is never a target; the 1500 s
SwiftShader run froze the PC; the open question below is dropped); (4) the deploy and full live check. NV1080 hung 2026-11-20 ~17:40
after a low-virtual-memory warning (17:13) and two TDRs: heavy runs go through a RAM watchdog (kill this run's tree below 2 GB free).

**2026-11-20 evening: main is ready for it.** device-frames merged on the released libraries: SpawnDev.ILGPU.ML 5.8.25, SpawnDev.Codecs
1.1.3 (the WebGPU / RX 590 PNG fix, so branch codecs-1.1.3-local is retired: it only bumped versions) and SpawnDev.ILGPU 5.4.43 (Data:
WebGL rows sized to the device's MAX_TEXTURE_SIZE, so Chrome without a GPU runs). Suite 3394 / 0 / 3; LTX-Video desktop GPU test on the
RX 590 passes with the frames on the device (cold 96.6 s, warm 50.2 s, the clip decodes). Before the deploy: the Save Image examples and
Text to video in Chrome against a local publish (PNGs decode): **done 2026-11-20 18:40, f2113aa non-AOT on :5300, RX 590: SD-Turbo 95 s,
Image to image 19 s, Text to video 302 s cold, every picture decodes.** Data's --disable-gpu appmode drive on 5.4.43: Data's texture error
is gone (no browser errors), but SD-Turbo gives no picture in 1500 s (RUN_SECONDS=1500) and the page stops answering: find out where it
spends the time (download vs SwiftShader WebGL kernels) before telling Data. Then deploy and the full live check on
both URLs (every example, the UI drives, TRELLIS on the HF Space). FLUX.2 Klein picture edit still trips the RX 590's driver watchdog
(also on the live site, so not this merge): Tuvok's per-block submits, ML 5.8.26-local.1.

Earlier plan (kept for the record): one Pages + Hugging Face Space deploy when SpawnDev.Codecs 1.1.2 is on nuget.org: merge branch codecs-1.1.2-local (Codecs 1.1.2 +
ILGPU 5.4.41), suite, drives; deploy; then every example and the UI drives on both live URLs. It carries the animated-preview / download
fix TJ reported, the device-tensor work, ML 5.8.24 (TRELLIS.2 and Wan Turbo in the browser) and the faster WebP save. ML 5.8.25
(device-frames branch) goes in a later deploy.

**Later 2026-11-23:** Tuvok's gate passed on the RX 590 in Chrome (ML 5.8.26-local.2: FLUX.2 Klein picture edit 105 s, 237 s with
?perop=1, a correct edit; TRELLIS 1020 s non-AOT, GLB saved, mesh not opened; 0 driver resets); he is releasing ML 5.8.26. Branch
`ml-local-5826` has MlRuntime.PromptSucceeded (the warm-up call) and drive-model-switch.js; merge it when 5.8.26 is on nuget.org. Still owed
to Tuvok: the heap across a model switch (SD 1.5 safetensors -> FLUX.2 Klein -> SD 1.5, releases on / off). **`load-image-device` is merged**
(Load Image decodes on the GPU from the file kept in JS): Chrome WebGPU on the RX 590 decodes the 512x512 example picture exactly in
seconds (Codecs 1.1.3: over 2 min / driver watchdog), 11 corpus files exact in the browser, 42 / 42 on OpenCL; suite 3478 / 0 / 3. Owed to
Geordi: lossy Save Animated WEBP per frame in Chrome, 1.1.4 against 1.1.3.

**2026-11-23 evening: main = ML 5.8.26 + ILGPU 5.4.46 + Codecs 1.1.4 (76488c5), deploy-ready apart from the live check.** `ml-local-5826`
is merged (MlRuntime.PromptSucceeded -> CompilerCachePolicy.ModelWarmedUp once after a prompt that loaded a model; in Chrome on the RX 590,
SD 1.5 safetensors -> FLUX.2 Klein -> SD 1.5: 7.0 MiB less .NET heap with the release on, run times equal; `drive-model-switch.js`). Suite
3479 / 0 / 3. Browser engine: after a lost device the open is retried and a failed open is not kept (a driver reset during FLUX.2 Klein's
VAE decode had left the page failing every run; drive-device-lost.js 9 / 9, red-checked). Save Animated WEBP in Chrome: 32.0 s for 25
frames on Codecs 1.1.4 (58.7 s on 1.1.3); Geordi is cutting the per-frame dispatch count next. **Open:** FLUX.2 Klein's VAE decode can
still trip the RX 590's driver watchdog in Chrome (1 loss in 5 SD -> FLUX.2 passes on 5.8.26-local.2); Tuvok: GraphExecutor submits by
estimated work instead of every 64 nodes; my 10-run count on 5.8.26 is the "before", rerun on his -local. **Next:** the deploy (Pages AOT +
HF Space) and the full live check on both URLs; drive-appmode.js on the real GPU.

**2026-11-23 night: deployed.** Pages run 45305247628 (40f1dfb, ML 5.8.26) and the Hugging Face Space (commit c65b306). Live Pages check
on the RX 590: 17 / 17 examples ran, every screenshot looked at (TRELLIS's mesh not opened). The first pass after a deploy is slow (every
shader compiled anew): Text to video 331 s, then 121 s warm. **Not yet run:** the Space's examples, the UI drives on both URLs.
**Device losses on the RX 590 in Chrome (ML 5.8.26, also the live site):** FLUX.2 Klein lost the device in 4 of 27 first runs of a page
(about 20 s in: the text encoder, one command buffer for the whole encoder) and FLUX 3 / FLUX.2 in the VAE decode (GraphExecutor, up to 64
convs per command buffer). Tuvok's fixes are in ML 5.8.27-local.2 (branch `ml-local-5827`, worktree `_scratch/wt-ml5826`, with Codecs
1.1.5-local.2): 0 losses in 20 FLUX.2 runs so far; 20 more fresh first runs and 10 of FLUX 3 decide his release (`drive-loss-count.js`,
FRESH=1). After a real driver reset Chrome gives the page no device at all (plain JS refused for 46 s and more; a new browser works):
SpawnGen's retry cannot recover that page, and its message ("Queue again") is wrong then: say "restart the browser" once the reload
probe in drive-loss-count has shown whether a reload is enough.
**Lossy WebP on AMD WebGPU is suspect (Geordi, 2026-11-23):** on the RX 590 in Chrome, repeated encodes of the same frame give different
bytes (his probe; identical on a 4070). SpawnGen's clips decode and look right; the byte-parity test covers CPU and OpenCL only. Owed:
his new probe + the libwebp-identity tests on the RX 590; then a browser parity check of Save Animated WEBP. Save Animated WEBP node, 25
frames in Chrome: 58.7 s (1.1.3), 32.0 s (1.1.4), 28.1 s (1.1.5-local.2); a tiny dispatch costs 62 us here, so the time is not dispatch count.

**2026-11-24 00:50: main = ML 5.8.27 (released after the RX 590 counts: FLUX.2 Klein 0 losses in 30 first runs, 4 in 27 before; FLUX 3
Klein 0 in 10); Pages run 45337313371 building it.** When it is green: tools/hf-space/publish.sh 45337313371, then on both URLs every
example and the UI drives (not run on the 5.8.26 deploy either). The 5.8.26 Space check: 16 / 17, TRELLIS lost the device at 88 s as the
17th example in one page and ran alone in 890 s; Tuvok asked for 5 fresh TRELLIS runs on 5.8.27 with stage names
(_scratch/trellis-losscount-5827.log). Geordi's lossy WebP on AMD WebGPU: fails libwebp identity at every commit back to v1.1.3 (36 of
222), so it never was right on this card in the browser; his garbage-buffer test result is owed (_scratch/codecs-garbage.log).

**2026-11-24 02:30: ML 5.8.27 is live on Pages (run 45339199113; the first dispatch failed, nuget.org had not listed 5.8.27 yet: check
the index before pushing a bump).** The Hugging Face Space is still the 5.8.26 build (c65b306): `hf upload` is refused, "Storage limit
reached for this space (Max: 1 GB)", 959 MB used. Asked TJ: squash the Space's history (irreversible) or another way; do NOT squash
without his word. Running: every example on the live Pages 5.8.27 build (_scratch/live-pages-5827.log); then the UI drives on Pages.
TRELLIS on 5.8.27, 5 fresh runs: no device loss; one run failed with "guide level 1 has 5341 voxels, decoder has 0" while a Codecs test
shared the GPU (sent to Tuvok). Lossy WebP on AMD WebGPU: cause found by Geordi with the RX 590 stage hashes (the setup kernel's level-cost
table stops after part of the first of 96 tables: a data-dependent while loop on this driver); his fixed-loop fix (Codecs 3c54c1c, local,
a WORKAROUND awaiting TJ) passes libwebp identity here. When a Codecs release has it: bump, and add a browser byte-parity check of Save
Animated WEBP to the drives.

**2026-11-24 07:05: a workflow that reuses none of the earlier one's models frees them first (browser engine).** Cause of the "every
example in a row" device losses on the RX 590, found with stage capture on the live 5.8.27 build: FLUX.2 Klein lost the device in its
first denoise step as the 6th example (both SD checkpoints still on the 8 GB card; D3D12 pages memory around each command buffer and the
2 s watchdog fires), TRELLIS about 90 s in as the 17th (3 of 4 full passes), while both run alone every time. Fix:
PromptExecutor.UnrelatedPromptStarting (ResultCache.MayHoldModels / DropModelEntries) -> MlRuntime.FreeHandedOutModels, wired in the
browser engine; UnrelatedPromptTests + ModelResidencyTests, red-checked; suite 3484 / 0 / 3. Local publish, RX 590 Chrome: 17 / 17 in one
page, no driver reset; drive-model-switch shows the SD checkpoint kept for its own workflow and freed before FLUX.2 Klein. One pass is not
a rate: repeat the full pass on the live build after the deploy. Not done: the same rule on the desktop server (a desktop card has the
same limit; decide with TJ, since it trades reloads for memory); freeing by measured need rather than by workflow. Also open: one
"A requested file or directory could not be found" at the end of a TRELLIS run after the SD examples (1 in 2, live), unexplained.

**2026-11-24 08:30: the model-freeing fix is live and checked (Pages run 45389993898): 17 / 17 examples in one page on the RX 590, no
driver reset, UI drives 10 / 10 on the live URL.** Found on the way: Chrome cleared the test profile's origin storage in the middle of a
pass and every later model load failed ("A requested file or directory could not be found"): the library kept one OPFS directory handle
for the page's life. Tuvok fixed it in ML 5.8.28-local.2; branch `ml-local-5827` (worktree `_scratch/wt-ml5826`) has that version and
`drive-cache-cleared.js` (8 / 8 there, red on 5.8.27): merge when ML 5.8.28 is on nuget.org (check the index first), then redeploy. The
browser engine now asks for persistent storage once (a headless test profile is refused; the console line says which). **Open, measured:**
after a full pass the origin's OPFS holds 110 files, 42.25 GB, while navigator.storage.estimate reports usage 3.60 GB of a 14.3 GB quota:
StreamingModelSource plans its caching against that estimate, so it keeps storing far past the quota (and the browser then clears
everything). Next: plan against the bytes really stored (sum of the cached files), with Tuvok (PlanCacheAsync is the library's). The
Hugging Face Space is still the 5.8.26 build (storage limit; TJ to decide). Desktop server: model freeing not wired (TJ to decide).

**2026-11-24 10:30: two branches wait on releases; main is ahead of the live site by small things.** The live site (run 45389993898) has had
three clean full passes with the model-freeing fix (one local, two live: 17 / 17 each; before it 1 clean pass in 4). main since then:
persistent-storage request, the restart-the-browser message after a refused device (RunFailureAdvice, failure messages stay 15 s). To merge:
`ml-local-5827` (ML 5.8.28-local.4: cleared-storage recovery on every cache path, the model cache limit setting, default 24 GB;
drive-cache-cleared 8 / 8 with the cap and without, drive-cache-limit 9 / 9) when ML 5.8.28 is on nuget.org; `codecs-115-local` (Codecs
1.1.5-local.3: lossy WebP on AMD WebGPU fixed; drive-webp-parity.js, the browser engine's file = the server engine's, red on 1.1.4) when
Codecs 1.1.5 is on nuget.org. Check the nuget index before each bump; then one deploy and the full live check (examples, UI drives, the
three new drives against a local publish). A new Studio setting must be applied in the startup block AND in ApplySettings.
Decisions waiting on TJ: the Hugging Face Space (storage limit, still the 5.8.26 build); model freeing on the desktop server; the
default cache limit; Geordi's parallel-encode design for slow cards (RX 590: two encodes together only 1.10x).

**2026-11-24 13:50: main = ML 5.8.28, live (Pages run 45447954796), and checked: 17 / 17 examples in one page, drive-cache-cleared (cap
and no cap), drive-cache-limit, drive-device-lost, UI drives (upload and appmode each failed once on the live URL and passed on rerun:
a file-picker click that timed out, one net::ERR_CONNECTION_RESET console line; not explained).** `ml-local-5827` is merged: the model cache
limit (Settings > Models, default 24 GB), recovery when the browser clears the stored models. After it: the limit also trims a store
that is already over it at the start of the next run (the library only evicts for a new download; the live profile stayed at 42 GB
under the 24 GB default): Pages run 45462410630, live pass _scratch/live-pages-trim.log to read (expect the store near 24 GB after it).
Chrome made the live origin's storage persistent in the test profile (persisted: true). Still to merge: `codecs-115-local` when Codecs
1.1.5 is on nuget.org (Geordi's release sweep), then redeploy and run drive-webp-parity against a server from main.

**2026-11-24 15:15: one copy of each published file, both sites deployed (Pages run 45474706459, Space commit f5bdce2).** TJ: one version of the
app, raw or ONE compressed form; projects carry CompressionEnabled=false; the Space keeps only the current build. Pages: 121 MB of raw
files (it gzips on the wire: the 90.7 MB AOT runtime arrives as 23.9 MB); the Space: runtime files as .gz only (43 MB in all; it sends
files uncompressed), listed by gzboot.json, which index.html reads before Blazor.start (`tools/gzboot.sh`). `tools/hf-space/publish.sh` now
ends with keep_only_current.py (squash + delete the unused LFS files; squashing alone freed nothing): the Space went 959 MB -> 42 MB.
Live pass under the 24 GB cache limit before this deploy: 17 / 17, store 42.25 -> 22.04 GB. Running: live checks of both URLs after the
one-copy deploy (_scratch/live-pages-onecopy.log, live-space-onecopy.log: the Space's first run in .gz-only form with the AOT build).
Still to merge: `codecs-115-local` when Codecs 1.1.5 is on nuget.org.

**2026-11-24 17:30: the model cache limit is OFF by default (TJ), live on both sites (Pages run 45495259517, Space 11e9eab) and checked.**
Downloaded models are kept until the user removes them in the Models panel (what WebLLM and Transformers.js do; the protection against
the browser clearing a site is persistent storage, which the engine asks for). StudioSettings.ModelCacheLimitGb (0 = keep all) replaces
ModelCacheGb, whose default the first version had written into saved settings. Live: drive-cache-limit (with the default-behaviour
phase), drive-cache-cleared, history and upload drives pass on both URLs; SD-Turbo, Image to image, FLUX.2 Klein, Text to video pass on Pages.
Running for Data: ILGPU 5.4.47 vs 5.4.48-local.1 (every kernel in its own IR context) in SpawnGen on the RX 590:
`_scratch/ilgpu-measure.sh`, builds in worktrees `_scratch/wt-i5447` / `wt-i5448l1` served on :5310 / :5311, logs `_scratch/ilgpu-measure/`.
Remaining for SpawnGen: merge `codecs-115-local` when Codecs 1.1.5 is on nuget.org. Open with TJ: model freeing on the desktop server.

**2026-11-24 22:00: main = ILGPU 5.4.48 / ML 5.8.28 / Codecs 1.1.4, live on both sites (Pages run 45526929598) and checked.** 5.4.47 must be
skipped: it emitted WGSL with an unfilled placeholder for 28 Codecs kernels (Save Image's PNG encoder, the WebP encoder), found here when
measuring Data's builds; 5.4.48 is 5.4.47 with that fixed. Local: suite 3491 / 0 / 3, GPU codec tests 53 / 53 (OpenCL), 17 / 17 examples,
cache / device-lost / ten UI drives. Live: Pages 17 / 17, five UI drives on each URL, five examples on the Space.
NV1080's L: drive filled during the first local pass (the test browser profiles held about 118 GB of models; a FLUX 3 run then failed with
"Failed to write data to data pipe"): old profiles removed, and `_scratch/opfs-clear.js <url>` empties one origin's stored models in
out/site-profile. Check `df -h /l` before a long browser pass. Remaining: merge `codecs-115-local` when Codecs 1.1.5 is on nuget.org, run
drive-webp-parity on the released pair (ILGPU 5.4.48 + Codecs 1.1.5; identical on the pre-release pair), deploy. Open with TJ: model
freeing on the desktop server.

**2026-11-25 00:45: main = ILGPU 5.4.48 / ML 5.8.28 / Codecs 1.1.5 (`codecs-115-local` merged), pushed; Pages run 45554502911 building.**
Codecs 1.1.5 makes lossy WebP correct on AMD WebGPU (a shader loop stopped early on that driver; SpawnDev.Codecs issue #2). On the released
pair, RX 590: `drive-webp-parity.js` (needs the SpawnGen server on :8188) passes, the browser engine's Save Animated WEBP = the server
engine's chunk for chunk; suite 3491 / 0 / 3; GPU codec tests 53 / 53 (OpenCL); 17 / 17 examples and every drive against a local publish.
After the build: tools/hf-space/publish.sh, then the live check of both URLs (_scratch/live-pages-c115.log, live-space-c115.log).
Nothing else waits on a library now. NV1080 test rig: out/site-profile keeps a full model set per origin (105 GB with three); after a
local full pass run `node _scratch/opfs-clear.js http://localhost:5300/`, and check `df -h /l` before long passes.
Open with TJ: model freeing on the desktop server (browser only today).

**2026-11-25 01:40: live and checked on both sites: ILGPU 5.4.48 / ML 5.8.28 / Codecs 1.1.5 (Pages run 45554502911).** Pages: 17 / 17 examples
in one page, five UI drives. Space: five examples, UI drives (drive-upload's file picker did not open once and passed on four reruns: the
second time at that step on a live URL; the drive now records what was under the click). No library merge is pending. Not taken yet:
ILGPU 5.4.49 (a failed kernel compile no longer breaks later ones; same WGSL as 5.4.48 per Data): bump it with the next change. From here
the work returns to the sections below (tensors on the GPU, parity).

## Now: the compositor editor (Create Layered Image), started 2026-11-25

The last of ComfyUI's frontend editors SpawnGen still shows as JSON text. The node side is done (it replays a saved composition, PARITY.md);
what is missing is the editor. Measured black-box on the reference install (`tools/studio-browser-test/probe-comfy-compositor.js`: Load Image
-> Add Layer x2 -> Create Layered Image, run once, open the editor; out/comfy-compositor-2-editor.png, comfy-compositor-editor.json):
- The node, before a run: "Run the workflow to generate a composite" and an Open Compositor button; the editor needs one run for the inputs.
  The widget's value is still `{}` after a run (nothing saved until the editor is used).
- The editor is a full-window dialog titled Compositor, with Restore and Close in its header.
- Left: Layers, top-most first, each with a visibility toggle, a thumbnail, the name and a line "N 80%" (blend mode letter and opacity);
  at the bottom Background (its own visibility toggle, off at first; a colour swatch; the canvas size, "632×572" for a 512 layer at
  (0, 0) and one at (120, 60)) and Move layer up / down.
- Centre: a toolbar (Select, Hand, Zoom with a percentage menu, Undo, Redo, Export PSD) and the canvas on a checkerboard, the selected
  layer framed with eight handles and a rotation handle above the top edge.
- Right, for the selected layer: Position X / Y, Dimensions W / H, Alignment (left, horizontal centre, right, top, vertical centre,
  bottom), Rotation in degrees, Flip horizontal / vertical, Opacity in percent, Blend mode (a combo, "Normal").
**Measured 2026-11-25 (`probe-comfy-compositor-steps.js`, out/comfy-compositor-steps.json; 36 recorded steps):**
- The value is `{}` until the first edit, then is rewritten after EVERY action (not on Close):
  `{"version":1,"canvas":{"w":632,"h":572},"inputs":[<one fingerprint per layer, input order>],"background":{"color":"#ffffff",
  "opacity":1,"visible":false},"layers":[{"name","visible","opacity" (0..1),"blend","transform":{"x","y","w","h","rotation" (RADIANS)},
  "flipH","flipV"}, ...]}`; layers stay in INPUT order, and an `"order":[1,0]` key (input indices, bottom first: settled, coat moved above apple gave [1,0]) appears only while the
  drawing order differs from the input order (Move layer down back to the start removes it).
- Fields, in this order: X, Y, W, H, rotation (degrees), opacity (percent). The list line is the blend letter and the opacity ("N 55%").
- Dragging the body moves by the drag divided by the zoom (+45, +27 screen px at "90%" -> x 170.00027, y 90.00018: the zoom is not
  exactly 0.9, and positions are stored unrounded; the fields show them rounded).
- A corner drag resizes freely (+54, +9 screen -> W +60, H +10: the aspect is NOT kept); with Shift it is kept (+45, +9 -> 542 x 542:
  the drag projected on the diagonal). An edge handle changes one side (right edge +45 -> W 563, H unchanged). W came out 563 where
  512 + 50 was expected, H 562: the rounding of sizes needs a finer measurement.
- Undo / Redo: Ctrl+Z and the buttons, one action per step.
- Alignment is to the CANVAS (632 x 572): left x 0, horizontal centre 60, right 120, top y 0, vertical centre 30, bottom 60.
- Flips set flipH / flipV and leave the fields alone. Typed values are taken as they are (rotation 30 -> 0.5235987755982988 rad,
  opacity 55 -> 0.55, X 10, W 300, H 100: W and H independently).
- Clicking a row selects that layer (fields follow). Move layer up / down change the drawing order. The row toggle sets `visible`; the
  Background toggle sets background.visible. Restore returns the initial composition (apple back to 80%).
**Second round (`probe-comfy-compositor-steps2.js`, controls clicked by their place on screen; out/comfy-compositor-steps2.json):**
- Blend list, 26 in this order: Normal, Multiply, Screen, Overlay, Darken, Lighten, Color Dodge, Color Burn, Hard Light, Soft Light,
  Difference, Exclusion, Linear Dodge, Linear Burn, Vivid Light, Pin Light, Linear Light, Hard Mix, Subtract, Divide, Grain Extract,
  Grain Merge, Hue, Saturation, Color, Luminosity. Stored lower-case ("multiply"); the list line shows the first letter ("M 80%").
- Resizing puts the handle ON the pointer (a press 0.45 px off the corner is not kept as an offset: this is the 563 of round 1): the corner moved by (-90, -90), (-45, -45), (-9, -9) screen px at 90% gave 412, 462, 502
  in the fields; the value keeps the unrounded size (w 412.4543, h 411.8987) and the top-left stays put.
- The rotation handle turns the layer about its centre: dragged from above the top edge to the right side -> field 90, stored
  1.5688117998196385 rad (the exact angle of the pointer; x, y, w, h unchanged).
- Arrow keys and Delete do nothing. Escape closes the editor; so does Close; both keep every edit (the value is already written).
- A click on empty space selects the Background: the right panel then shows only the canvas width and height (632, 572) as number
  fields. The Hand tool pans the view and changes no value.
- Restore: after open -> edit -> Restore the layer was back at its first values (round 1); after Close -> reopen -> Restore nothing
  changed (round 2, value identical). Reading: Restore returns to the composition as it was when the editor was opened. To confirm.
- Still unmeasured: the zoom menu's entries, the background colour picker, editing the canvas size, the `order` key's direction.
**Done 2026-11-25: step 2, the logic.** `SpawnGen.Editor/CompositorEditor.cs` (the document, selection, every measured action, undo / redo,
Restore, hit testing, the value) with `CompositorEditorTests` (16, from the measured values; red-checked). SpawnGen's own choices where nothing
was measured are marked in the class remarks (resizing a rotated layer, a side under one pixel, typed sizes that are not positive).
**Done 2026-11-25: step 3, the dialog.** `SpawnGen.Studio/Pages/CompositorDialog.razor` (drawn from CompositorEditor; the layer pictures are
the run's ui compositor_layers placed with CSS) and `Studio.Compositor.cs` (the node's Open Compositor row opens it on `EditorNode.CompositorRun`,
set from the run's ui; every action writes the widget, one graph undo step each). `drive-compositor.js`: 33/33 on the RX 590 through the
browser engine (every measured action against ComfyUI's stored values through Export, close / reopen / Restore, and a second run that
shows the edit). The zoom menu was read from the probe's screenshot: 25%, 50%, 100%, 200%, Fit; the editor opens at min(1, fit) x 0.9.
Left: step 4, Export PSD (decide then), and the unmeasured corners named in CompositorEditor's remarks.
**Step 4, Export PSD: measured 2026-11-25, waiting on where the writer lives.** `probe-comfy-compositor-psd.js` pressed the button and
`psd-dump.py` printed the file: name comfyui-layers-YYYYMMDD-HHMMSS.psd; 8BPS version 1, RGB, 8 bit, canvas size; layer records bottom
first: Background (full canvas, a solid fill, `SoCo`), then each layer already rasterised into the bounding box of its rotated shape (it
may run past the canvas), blend key (`mul ` for Multiply), opacity 0..255 (204 for 80 %), a `luni` Unicode name, flags 0x08; the picture
layers are also smart objects (`PlLd`, `SoLd`, the originals in `lnk2`); all channels RLE; an RLE composite. A file format writer is
codec work, so Geordi was asked whether SpawnDev.Codecs takes a PSD writer (_DevComms 2026-11-25-0340-Lore-TO-Geordi-codecs-psd-writer-request.md);
if not, it is written in SpawnGen. Either way the Studio side is: render each layer as the run does, call the writer, download.
**Next, in order:** (1) the few items still unmeasured above, as the logic reaches them; (2) the logic in
SpawnGen.Editor (CompositorEditor: the document, the selection, the actions, undo) with tests from those measurements, writing the JSON the
node already replays; (3) the dialog in the Studio (the picture comes from the node's ui: compositor_inputs / compositor_bboxes) and a
browser drive that repeats the probe's actions; (4) Export PSD last (a writer for layered PSD: decide then).

**2026-11-25 03:25: the compositor editor and ILGPU 5.4.49 are live and checked** (Pages run 45566462502, the Space published from it and
squashed). On the RX 590: Pages 17/17 examples, drive-compositor 33/33 and the five UI drives; the Space drive-compositor 33/33, five
examples and the five UI drives. Packages live: ILGPU 5.4.49, ML 5.8.28, Codecs 1.1.5. Next from the backlog: Export PSD for the
compositor (decide the writer), then the next item of "SpawnGen work that needs no library".

**2026-11-25 13:05: ML 5.8.30 is live and checked; Export PSD is done on a branch.** Packages live on both sites: ILGPU 5.4.49, ML 5.8.30,
Codecs 1.1.5 (Pages run 45651218126: its deploy step failed in GitHub's incident of that morning and passed on a rerun). On the RX 590:
Pages 17/17 examples (the engine's new "[gpu] backend" line confirms the build), the Space five examples, drive-compositor 33/33 on
both, the UI drives on both. **Open: drive-upload's file picker** did not open on the first Pages runs after the deploy (2 of 4, then 5
of 5 passed); the drive now prints the page's pointer events and input.click calls when it happens: read them next time, the cause is
not known. ML's WebGL background-removal bug (fixed in 5.8.30) could never be reached from SpawnGen: without WebGPU the Studio offers no
browser engine (measured with navigator.gpu hidden).
**Export PSD** (branch `codecs-psd-local`, Codecs 1.1.6-local.2, where TJ put the PSD writer: PsdEncoderGpu, kernels):
`ImageCompositor.Export` (each layer as a run draws it, the Background record, the flattened picture), `CompositorPsd` (Nodes.ML), the PSD
button in the dialog (`Studio.Compositor.cs`; it needs WebGPU in the browser, the writer runs on the GPU). Suite 3519; 
`drive-compositor-psd.js` 11/11 on the real GPU (the records equal ComfyUI's measured export, the turned layer's box 12,19..528,613
included); psd-tools 1.27 and GIMP 2.10.30 read the file (`psd-readback.py`, `psd-gimp.scm`); Photoshop, Krita and Photopea not tried.
**Merged, deployed and checked 2026-11-25 19:35** (Codecs 1.1.6 on nuget.org at about 18:45; main also took ML 5.8.31; Pages run
45722034653, the Space published from it). On the RX 590: Pages 17/17 examples, the Space five; on both sites drive-compositor-psd 11/11,
drive-compositor 33/33 and the UI drives. Packages live: ILGPU 5.4.49, ML 5.8.31, Codecs 1.1.6. The Painter node is in this build too
(its widget is not: `mask` shows as a text row).
**The upload drive's lost file picker, read from the page's own events:** on the failing run the page HAD called input.click() on its
connected file input, with user activation, 1 ms after the pointer-up, and still no chooser event reached the drive. So the page did
its part. The drive now adds a chooser listener when the page is created (Playwright turns interception on with the first listener,
a round trip that a click started at the same moment can beat); 6 of 6 passed after that, which is not proof for a failure seen about
once in five runs: if it comes back with the listener in place, the race is not the cause.

## Now: models in GPU memory, by the user's choice (TJ, 2026-11-25)

TJ: a workflow whose tab is still open is probably still in use, so its models should stay loaded when the GPU has room (some users
have 128 GB or more of unified memory); models go only when room is needed; and it must be user configurable.
**What exists today (read from the code, 2026-11-25), which is NOT eviction on need:** (a) the browser engine frees every handed-out
model when a prompt starts that reuses none of the earlier prompt's models (`PromptExecutor.UnrelatedPromptStarting` ->
`MlRuntime.FreeHandedOutModels`; added 2026-11-24 because three workflows' models on the 8 GB RX 590 lost the device); (b) inside a
workflow the big transformers take turns by fixed rules written for an 8 GB card (`ModelResidency.ReleaseTransformers`); (c) pipeline
nodes' models are freed when another prompt loads something. Nothing measures memory: WebGPU gives a page no figure for free or
total GPU memory, and running out shows as a lost device, not as an error to react to. The desktop server frees nothing between
workflows. The engine does not know about tabs: one result cache per engine.
**Plan:** (1) a setting, "Models in GPU memory": *One workflow at a time* (today's browser behaviour, the safe choice for small
cards) or *Keep while they fit*, with a "GPU memory for models (GB)" budget the user sets (the desktop server can offer the
device's own figure where ILGPU reports one; a browser cannot). (2) With a budget: every loaded model has a size (its weights on the
device: from the ML library; ask Tuvok for a figure per model if none is exposed), models of every open tab stay loaded, and before a
load that would pass the budget the least recently used models that the running prompt does not use are freed until it fits. (3) The
Studio tells the engine which workflows are open; closing a tab frees the models only that tab used. (4) The fixed take-turns rules
of (b) follow the same budget instead of assuming 8 GB. (5) Both engines (this also answers the open question about the desktop
server). (6) Tests: unit tests of the keep / evict decisions with sizes; browser drives on the RX 590 for both choices (three
workflows in three tabs: with a large budget nothing reloads, with a small one the oldest goes, one-at-a-time as today); the
full example pass stays clean under the default. Default: *One workflow at a time* until the user sets a budget (Lore's choice:
a page cannot know the card is large, and guessing wrong loses the device).

**2026-11-25, after TJ pointed at the libraries ("all of our stuff has to take advantage of hardware we don't have"):** ILGPU has
`MemoryPressure` (`IDeviceMemoryReclaimer`, `RegisterReclaimer`): allocations reclaim from registered holders, least recently used
first. In ML only the buffer pools register (idle scratch, never weights); the proactive check needs `Accelerator.TryGetFreeMemory`,
which only CUDA has, so in a browser nothing fires by itself and the budget must come from the app (Tuvok, from the code). Sizes
exist per weight class but not uniformly. **So the eviction belongs in ML, not in SpawnGen:** Tuvok proposed, and I asked for, an
ML interface (DeviceBytes, IsLoaded, LastUsed, Unload when idle, reload on use) on every model kind plus an LRU manager per
accelerator with a byte budget (the device's own free memory where a backend reports it); SpawnGen then pins what the running
prompt uses, releases a closed tab's models, shows evictions, offers the budget as a setting, and drops its fixed rules. Waiting
on: TJ's yes for the ML feature, Tuvok's short design, Data's view for ILGPU. Until it exists SpawnGen's behaviour stays as it is.

**2026-11-25 evening: the library pieces exist; SpawnGen's migration starts (branch `model-residency`).** TJ approved the ML feature
(a model over the budget loads with a warning). ILGPU 5.4.50: `Accelerator.DeviceMemoryBudgetBytes` and free memory on AMD OpenCL
(checked on the RX 590: 8128 MiB free, minus exactly what is allocated). ML 5.8.32-local.1: `ModelResidency` (per accelerator:
`AcquireAsync(key, load, expectedBytes, tag)` leases, `BudgetBytes`, `ReleaseAsync(tag)`, `ReleaseAllAsync`, `Invalidate`, `Register`
for engines that load themselves, `Evicted`, `Log`). **SpawnGen's steps:**
- **A, the generic paths:** `MlRuntime` owns a manager; with a budget set (or a backend that reports free memory), checkpoints and
  every `GetOrLoadAsync` model are acquired from it, the lease held until the prompt ends (a new `PromptQueue.AfterRun`); the
  result cache drops its model entries at each prompt start so loader nodes run again and lease (they return the kept model), and
  on `Evicted`; without a budget everything stays as today ("one workflow at a time"). Setting: "GPU memory for models (GB, 0 =
  one workflow at a time)" in the Studio; the server takes the device's own figure (CUDA, AMD OpenCL) or a configured one. My own
  `ModelResidency` static class (the take-turns rules) is renamed `TransformerTurns`.
- **B, the self-loading transformers** (FLUX.1 / FLUX.2 / Z-Image / Qwen-Image / ACE-Step / LTX-2 diffusion models, which load on
  first use and unload by the fixed 8 GB rules): they join the manager with `Register` (their resident bytes, an evict, room
  ensured before a load), and the fixed rules apply only without a budget. Until B they are not counted: say so in reports.
- **C, tabs:** the Studio tags prompts with their tab; closing a tab calls `ReleaseAsync(tag)`.
- **Gate (Tuvok's and mine), RX 590:** the 17 examples with a budget below the card's size, no device loss, and a second pass that
  shows reuse in the log; the same pass with no budget unchanged from today; unit tests of the runtime's keep / lease / drop
  decisions on the CPU accelerator; what a large budget does is exercised by setting one larger than the card (models kept until
  the device itself gives out is NOT a pass: on 8 GB that case can only be shown with small models).

**2026-11-26 02:50: steps A and B are done on branch `model-residency`, and the RX 590 gate passes.** (ML 5.8.32-local.6, ILGPU 5.4.50.)
- `MlRuntime`: with a budget (or a device that reports free memory) every model is leased from the library's manager until
  `EndPromptAsync` (`PromptQueue.AfterRun`); loader nodes run again each prompt (`PromptExecutor.ReacquireModels`) and get the kept
  model; a dropped model takes the cached results that may hold it. `MakeRoomAsync` replaces the fixed take-turns rule at its 20
  call sites (room by need; the old rule only when the device still lacks it, and always without a budget). A pipeline that loads in
  its first run asks for room right after its acquire, which is where the manager measures it from. Sizes are the manager's (kept
  in the browser's storage, `spawngen.modelsizes`). Setting: "GPU memory for models (GB, 0 = one workflow at a time)"; server:
  `SpawnGen:ModelMemoryGb`, else the device's own figure. Tests: `ModelBudgetTests` (11) on the CPU accelerator.
- **Gate (17 examples x 2 passes, 6 GB budget, local publish):** the first attempt lost the device in pass 2 (the FLUX pipelines were
  measured when created, 0.2 GB each, and two were kept side by side; FLUX 3 took 1100 s first). After the fix: 34/34 on local.3 and
  34/34 on local.5, no device loss, pass 2 times as pass 1. No budget: 17/17 on local.5 and on local.6, times unchanged.
- **Known flaw, being fixed in the libraries:** a model measured over its run is recorded with the run's idle scratch included, so
  kept models sum to more than is allocated and room is made earlier than needed (safe, not tight). ML master already takes free
  memory from ILGPU's count alone; ILGPU 5.4.51 adds reclaimable bytes so the size is what the model holds. Rerun the gate on that
  build with the test profile's saved sizes cleared.
- **What the gate cannot show:** two large models kept side by side (an 8 GB card cannot hold them); that is unit-tested with a
  large budget only. The first real evidence needs a larger machine.
- **Step C done 2026-11-26 (tabs):** a prompt carries its tab (`extra_data.spawngen_workflow`), the models it loads are tagged with
  it (`MlRuntime.PromptTag`), and closing a tab calls `IEngineClient.ReleaseWorkflowAsync` (the browser engine directly; a SpawnGen
  server through `POST /spawngen/release`): the idle models only that tab used are dropped. `drive-model-tabs.js` 9/9 on the RX 590
  (6 GB: two tabs' models both stay, closing one drops only its model, the other reruns on its kept model; no budget: nothing
  changes). The server keeps the measured sizes in `model-sizes.json` next to its models folder. Suite 3540.
- **Left:** file sizes as expected bytes for first loads (`IModelSource.GetSizeAsync`); the gate again on the ML build with the
  measuring fix (saved sizes cleared first); a server run on this PC under OpenCL (the device's own figure is the budget there:
  not run yet, only the unit tests cover the server path); then the merge when ML 5.8.32 is on nuget.org.

**2026-11-26 12:35: models in GPU memory by the user's budget is merged and LIVE, and the page recovers from a cached boot error.**
Packages live on both sites: ILGPU 5.4.51, ML 5.8.32, Codecs 1.1.6, SpawnDev.Blazor.BootRecovery 1.0.1 (Pages run 45853326418, the
Space published from it). On the RX 590: Pages 17/17 examples, the Space five; on both sites drive-boot-retry 4/4, drive-model-tabs
9/9 (with a 6 GB budget), drive-compositor 33/33, drive-compositor-psd 11/11 and the five UI drives. Before the merge, on the
released ML 5.8.32 locally: no budget 17/17, the 6 GB two-pass gate 34/34, suite 3540.
- The setting defaults to 0 (one workflow at a time), so nothing changes for a user until they set a budget.
- **The startup failure found by the live check:** GitHub's CDN answered one framework file with a 503 page and the browser kept
  it in its cache (the .NET loader asks with force-cache): the page could not start in that browser. This is dotnet/runtime#135090,
  which the team's SpawnDev.Blazor.BootRecovery package already fixes; I first wrote my own retry in index.html, TJ pointed at the
  package, and the Studio now references it instead. SpawnGen's own loader of <name>.gz files (the Space) asks with force-cache
  too, so the package covers it, checks the inflated bytes against the file's integrity value (it did not before) and retries
  once past the cache. `drive-boot-retry.js` fails one file's first request three ways (503 page, wrong bytes, dropped
  connection): without recovery 1/4 on both forms of the site, with it 4/4.
- **Next on this topic:** ML 5.8.33 (a local build is out): the desktop file store's sidecar fix (a transient lock made a complete
  model look missing: seen twice on the desktop server here, not reproduced since) and each model measured in its own window.
  Branch for it: saved sizes cleared, the gate again, Kokoro's and the preview decoder's sizes read, a desktop server run.
- Still only unit-tested: two large models kept side by side (no machine here can hold them).

**Bug found 2026-11-26: VAE Encode on the Stable Diffusion ONNX checkpoints is not deterministic.** Seen first as image to image and
inpainting coming out 1/255 different in a few hundred pixels between two runs of one build on the desktop server, while every
workflow without an encode was identical byte for byte. `VaeEncodeRepeatTests` (explicit, OpenCL on the RX 590): six encodes of one
picture differ in nearly every latent value by up to 0.015 (sd_turbo and sd_v1-5). Cause: both encoder ONNX graphs contain a
RandomNormalLike node and output `latent_sample` (the posterior sampled inside the graph). ComfyUI's VAE Encode, measured black box
(`_scratch/comfy_encode_twice.py`: two encodes of one picture, Save Latent): 0 of 65536 values differ. So this is a parity bug
(a fixed seed must give the same picture) on those two checkpoints. Asked Tuvok for a session option that fixes the random
operators (zeros, which gives the posterior mean; or a caller's seed); when it exists VAE Encode uses it and the test above
becomes a normal one. Not checked yet: whether ComfyUI's encode is the mean or a fixed-seed sample (compare with the mean then).

**2026-11-26 19:30: ML 5.8.33 is live and checked; the VAE Encode fix is ready on a branch.** Live on both sites: ILGPU 5.4.53, ML 5.8.33,
Codecs 1.1.6, BootRecovery 1.0.1 (Pages run 45894398836). RX 590: Pages 17/17, the Space five, boot-retry 4/4, model-tabs 9/9,
compositor 33/33, PSD 11/11 and the UI drives on both. 5.8.33 brings the desktop file store's sidecar fix and each model measured
in its own window (Kokoro 0.32 GB, not 2.56); its new desktop convolution kernel was compared on the RX 590 under OpenCL against
5.8.32: every deterministic output identical.
**VAE Encode** (branch `ml-5834`, ML 5.8.34-local.1): the SD ONNX encoder session runs with `RandomAtMean`. `VaeEncodeRepeatTests`:
0 differing latent values (was about 16363 of 16384); eight desktop-server examples run twice: all 9 pictures identical, image to
image and inpainting included; suite 3540. Merge when ML 5.8.34 is on nuget.org. Still to measure: the latents against ComfyUI's
for an SD VAE (the reference install has none).

**2026-11-26 23:55: ML 5.8.34 is live and checked; VAE Encode on the SD ONNX checkpoints is deterministic.** Live on both sites:
ILGPU 5.4.54, ML 5.8.34, Codecs 1.1.6, BootRecovery 1.0.1 (Pages run 45928935715). RX 590: Pages 17/17, the Space five, boot-retry
4/4, model-tabs 9/9, compositor 33/33, PSD 11/11 and the UI drives on both. Before the merge, locally: two visits of all 17 (17/17
each), `VaeEncodeRepeatTests` 3/3 with 0 differing latent values, suite 3540. The encoder session runs with `RandomAtMean`
(`OnnxCheckpointModels`), so image to image and inpainting repeat exactly for a fixed seed (eight desktop-server examples run
twice: all 9 pictures identical). 5.8.34 also brings the browser shader cache: the first example of a visit (it includes the
engine starting) took 51 s on a first visit and 30 s on the second. Open: the latents against ComfyUI's for an SD VAE were not
measured (the reference install has none); by the encoder graph's arithmetic the result is the posterior mean.

## Now: Painter (the node and its editor), started 2026-11-25

The last of ComfyUI's frontend editors (5 templates use it). SpawnGen has neither the node nor the editor. From the reference server's
/object_info/Painter (public schema; its source is not read): display name Painter, category image, inputs `mask` (STRING, widgetType
PAINTER, image_upload, default "", no socket), `width` and `height` (INT, hidden, default 512, 64..4096 step 64, no socket), `bg_color`
(COLOR, default #000000, no socket), optional `image` (IMAGE, "Optional base image to paint over"); outputs IMAGE, MASK.
**Measured 2026-11-25 (`probe-comfy-painter.js`, out/comfy-painter.json, the node at its default size 450 x 550):** under the slots a
square canvas (bitmap 512 x 512, shown 426 wide), then rows: Tool (Brush | Eraser), Cursor Size (slider, 20), Color Picker (a colour
input #ffffff, its hex text, a percent number field 100), Hardness (slider, 100%), Width (slider, 512), Height (slider, 512),
Background (colour input #000000 and its hex), and a Clear button across the node. A stroke changes no widget value and sends
nothing; when the prompt is built the painting is uploaded (POST /api/upload/image) and `mask` becomes
"painter-<node id>-<milliseconds>.png [input]"; width, height and bg_color go in the prompt as they are.
**The node's outputs, measured 2026-11-25 (`probe-comfy-painter-run.js`, `png-stats.py`; files in out/painter-fixtures/):** the upload is
an RGBA PNG of the canvas size, transparent where nothing is painted, straight alpha (a 50 % red stroke is (255, 0, 0, 128); the
eraser leaves alpha 0). MASK is the painting's alpha (128 for that stroke). IMAGE is the painting over bg_color with that alpha
((255, 0, 0, 128) over #336699 -> (153, 50, 76)), or over the linked image when there is one (the output then has the image's size:
512 x 512 with width 256, height 384 set). Without an image, width x height other than the painting's size gives an output of
width x height with the painting resized (256 x 384 from 512 x 512: which filter is still to settle, on pixels). With mask "" the
IMAGE is bg_color at width x height and the MASK is zero. Still to measure: the resize filter; a base image of another size than
the painting; a batch of base images; brush geometry (size, hardness falloff, spacing) for the editor's own strokes.
**Settled 2026-11-25 (`painter-resize-check.py`):** the resize is Pillow's Lanczos on the painting's RGBA bytes (0 difference on mask
and image; every other Pillow filter and torch interpolate mode differs); a linked picture of another size (300 x 200) gives an
output of ITS size with the painting stretched to it; of a batch only the first picture is used (one output picture).
**Done 2026-11-25: step 3, the node.** `SpawnGen.Nodes.Core/PainterNode.cs` and `PainterTests` (8) on Fixtures/painter (the measured
files): every saved byte of IMAGE and MASK equal to ComfyUI's for both paintings, over black, over #336699, resized, and over a
picture; red-checked. The schema gained `hidden`, `widgetType` and `socketless` on inputs (InputAttribute, WidgetSpec, object_info
both ways). Until the widget exists the node shows `mask` as a text row. Next: step 4, the widget (strokes: brush geometry, hardness
falloff and spacing still to measure), then the Studio and its drive, then the five templates.
**The brush, measured 2026-11-26 (`probe-comfy-painter-brush.js`, `painter-brush-read.py`; paintings in out/painter-brush/):**
- Sliders: Cursor Size 1..200 (20), Hardness 0..100 (100), Width and Height 64..4096 (512). The node keeps nothing of them in
  its properties: brush settings are the widget's own state.
- Cursor Size is the DIAMETER in canvas pixels: a click at (256, 256) paints x 246..265 for size 20, 231..280 for 50, 156..355 for
  200 (so the disc is centred on the pixel corner at 255.5 + 0.5). Size 1 paints 1 x 2 pixels at alpha 112. Edges are antialiased.
- Hardness: 100 is a hard disc. Lower hardness both softens and WIDENS the stamp: size 50 at hardness 50 covers 62 pixels (alpha
  rising 10, 24, 37, ... by about 13.6 a pixel to 255, then solid), at hardness 0 it covers 74 (rising about 6.8 a pixel to 250 at
  the centre). Reading: alpha falls linearly from an inner radius r x hardness to an outer radius r x (1 + 0.48 (1 - hardness))
  (31 and 37 for r = 25): the 0.48 comes from two points only, measure more hardness values before fixing the formula.
- The percent field is the stroke's opacity: 50 gives alpha 128 everywhere in the stroke. WITHIN one stroke nothing accumulates
  (back and forth over itself stays 128); SEPARATE strokes composite source-over (two overlapping dots or crossing lines: 192).
- A stroke is a continuous line with round caps (from x 100 to 400 at size 50: 75..424), also when the pointer moves fast (straight
  segments between pointer positions). The colour is stored as it is (#ff0000 -> 255, 0, 0), straight alpha.
- The eraser removes alpha (a hard eraser dot through a line leaves 0, with an antialiased rim).
- **Settled 2026-11-27** (same probe, now locating the canvas on every stroke; out/painter-brush/*-b.png, erase-*, edge-dot-b,
  size-change-*, shrink-then-grow):
  - **Hardness falloff:** alpha = 255 x clamp((R_out - d) / (R_out - R_in), 0, 1), rounded, with R_in = r x hardness and
    R_out = r x (1 + (1 - hardness) / 2), d = distance from the stamp centre (the pixel corner under the pointer) to the pixel
    centre. Exact along the centre row for r 25 at hardness 0, 10, 25, 50, 75, 90 and r 50 at 0 and 50 (e.g. r 25, hardness 90: 51
    119 187; hardness 0: centre 250, not 255); over the whole stamp 2 to 7 % of the pixels are one step UNDER it in a symmetric
    pattern (Chrome dithers gradients) and the last pixel before R_out loses up to 4 to the circle's antialiased edge, never over
    (PainterWidgetTests, red-checked). SpawnGen draws the same canvas gradient, so the browser gives it the same dither. The 0.48 of 11-26 was this 0.5 read off whole pixels. At 100 it is the
    hard antialiased disc. It reads as a radial gradient, opaque to r x hardness, transparent at R_out.
  - **The Eraser has its own controls:** Cursor Size, Width, Height only (no colour row, no opacity field, no Hardness: in Eraser mode
    slider 1 is Width, which is what broke the 11-26 probe). It ignores the brush's opacity (50 % set: erases to 0) and hardness (0
    set: the same pixels as the hard eraser): a hard, full eraser of Cursor Size.
  - **Edge:** a stamp at the canvas edge is clipped (x 0..26 for a 50 dot at x 2), nothing wraps.
  - **Width / Height on a painting:** the canvas bitmap takes the new size at once (256 x 640, shown 426 wide), the painting stays
    at its pixels from the top-left corner, NOT scaled; what falls outside is gone: shrinking to 256 and growing back to 512 keeps
    only the dot inside 256. The uploaded file is the new size.
  - **Soft strokes** (line-soft-*, synth-soft-seg*): a soft line builds up along its path (centre 254-255 at hardness 0, the
    cross-section wider and softer than one stamp) and the per-stroke opacity applies once to the whole stroke (opacity 50: every
    value half, interior 127-128). With exact synthetic pointer events the painting does NOT depend on how the path is cut into
    events (1, 2, 30 and 300 segments: byte-identical), and it repeats every 75 px along the line (r 25, hardness 0; columns x and
    x + 75 identical). Real pointer input through Playwright gave stronger or weaker strokes with the pointer's speed, so ComfyUI's
    result also depends on how the browser delivers the events (frames); a human stroke cannot be matched pixel for pixel anyway.
    Model fitted (`fit_spacing.py` in this session's scratchpad, not kept): stamps of the falloff above at a fixed spacing,
    composited source-over inside the stroke, then the stroke at its opacity: best spacing about r / 5 (5.36 or 5.0), mean error
    0.25 / 255, max 31 at a few rim pixels. NOT exact: the rule inside a stroke is still open (try: max instead of source-over, a
    spacing from R_out, positions on the event path's arc length). SpawnGen uses the r / 5 source-over model with that tolerance
    stated in its tests until it is settled.
  - **A linked image** (`probe-comfy-painter-linked.js`, out/painter-linked/): the widget shows the upstream Load Image's picture at
    once (no run needed) as an `<img>` under the canvas; the canvas bitmap holds only the strokes (transparent elsewhere). The
    Width, Height and Background rows are HIDDEN while an image is linked, a "W x H" label sits under the canvas, and the canvas
    takes the picture's size (a 64 x 512 picture: canvas 64 x 512) with the painting cropped from the top-left as above. The
    pointer is drawn as a ring of the brush size.
- SpawnGen will not match the antialiased rim pixel for pixel (the browser's own rasteriser draws it): the geometry, the falloff,
  the opacity rule and the eraser are what its strokes must match, with a stated tolerance at the rim.
**Step 4, the widget: how it is built (2026-11-27).** The Studio is the only UI (Desktop is a Photino window on it), so the painting
lives in the page as JS canvases end to end (AGENTS rule 4) and is driven from C# through SpawnJS wrappers (no raw JS; SpawnJS already
has CreateRadialGradient, OffscreenCanvas, ConvertToBlob, GlobalCompositeOperation, GlobalAlpha). Parts:
(a) `SpawnGen.Editor/PainterWidget.cs`, pure logic with tests: brush settings (Tool, Cursor Size 1..200 = 20, colour #ffffff,
opacity 0..100 = 100, Hardness 0..100 = 100; view state per node, as ComfyUI keeps them outside the node), the stamp (R_in, R_out,
the gradient stops the canvas gets), stamp placement along the pointer path at a fixed spacing carried across events, the canvas size
(the linked picture's, else Width x Height), resize keeping the top-left, which rows show (Eraser: no colour / Hardness; a linked
image: no Width / Height / Background, a "W x H" label), the layout, the upload name `painter-<node id>-<ms>.png` and the `mask` value
"<name> [input]". Tests: the stamp formula against ComfyUI's dot fixtures (every ramp pixel exact), the event-count independence,
the soft-line model against synth-soft-seg001 within the measured tolerance.
(b) EditorView: a press on the canvas paints (a stroke), the rows edit the settings, Clear clears.
(c) Studio: a `PainterSurface` per node (the painting canvas + the stroke canvas; a stroke draws at full alpha on the stroke canvas,
then lands on the painting at the opacity, source-over; the Eraser draws hard stamps destination-out), NodeRenderer draws it over the
linked picture or the node's background; on queue each Painter with strokes is uploaded (ConvertToBlob -> UploadFileAsync, bytes stay
in JS) and its `mask` set before the prompt is compiled; a loaded workflow's `mask` file is drawn back into the canvas.
(d) The drive: paint dots and a line through real pointer events, run, compare IMAGE / MASK with ComfyUI's fixtures.
Also owed: the older mask editor (index.html spawngenMaskEditor) is hand-written JS; port it onto the same C# surface after (c).
**Done 2026-11-27: (a) a8d961d, (b) 9ae9423, (c) + (d) in the commit after.** (c) `SpawnGen.Studio/Canvas/PainterSurface.cs` keeps each
painting in two OffscreenCanvases (painting, stroke) per scope (tab + subgraph path) and node, NodeRenderer draws the widget (the linked
picture or bg_color under the painting, the brush ring, the rows), the colour / opacity / Background edit in place, Queue uploads every
changed painting first (ConvertToBlob -> UploadFileAsync, the bytes never enter .NET) and sets `mask`; a workflow naming an uploaded
painting draws it back from the engine's input folder. (d) `drive-painter.js` 10/10 on the RX 590 against a local publish (red-checked:
without the upload 4 fail): real pointer strokes, the Eraser, IMAGE / MASK pixels, no new upload without new strokes, and a soft stamp
(size 50, hardness 50, clicked on the pixel corner) equal to ComfyUI's own painting of the same click within 1 / 255 (Save Image's float
floor), 3064 painted pixels on both.
**Subgraph levels, done 2026-11-27:** Queue also uploads the paintings of the outer levels of an open subgraph and writes their mask
values into those levels' JSON on the stack (RootJson is built from it); leaving a level (breadcrumb, Escape, App mode, the builder) first
uploads that level's paintings, so they are in the definition it is written back into. `drive-painter-subgraph.js` 13/13 (red-checked:
without both, its two upload checks fail). Left as it is: with NO engine at all, a level left with strokes keeps them in the page until
it is opened again and Queue is pressed (nothing can run without an engine anyway).
**Step 6, the templates (checked 2026-11-27 against template-needs.json, commit 2fbb6d07 of Comfy-Org/workflow_templates): SIX use
Painter, and none can run yet for reasons outside Painter.** Three are paid cloud API nodes (api_bria_eraser, api_bria_genfill:
BriaEraser / BriaGenFill; api_bytedance_seedream_5_0_pro_image_edit: ByteDanceSeedreamNodeV3): SpawnGen has no API nodes (a decision
for TJ: they need the user's Comfy account and send pictures to a cloud). Three need model families SpawnGen does not have:
image_anima_lllite_image_inpainting (AnimaLLLiteApply, ModelPatchLoader; Anima base + LLLite inpainting + Qwen3 0.6B + Qwen-Image VAE),
image_moss_image_color_depth_control (MOSSImageControl, LoadMoGeModel / MoGeInference / MoGeRender, SkipLayerGuidanceDiTSimple; MOSS +
MoGe 3 ViT-g + Qwen3.5 9B int8), video_wan2_1_infinitetalk (WanInfiniteTalkToVideo, AudioEncoderLoader / AudioEncoderEncode,
ModelPatchLoader; Wan 2.1 I2V 14B + wav2vec2). Ranked by what they unlock: (i) Anima LLLite inpainting (one image model on known parts:
Qwen-Image VAE, a Qwen3 encoder; ModelPatchLoader also serves InfiniteTalk), (ii) InfiniteTalk (Wan 2.1 14B is already parked on Y:;
audio encoders open talking-head video), (iii) MOSS + MoGe (two new families). Each goes through the usual path: ComfyUI reference run,
what the library lacks to Tuvok, nodes here, parity.
**Mask editor ported, 2026-11-27:** `Pages/MaskEditorDialog.razor` replaces index.html's hand-written spawngenMaskEditor (removed): the
same tools and result, canvases driven through SpawnJS, and its three over-half rules (an existing mask on open, Invert, Save) are
canvas filters (inline SVG feComponentTransfer, discrete) instead of pixel loops, so no pixel enters .NET. drive-maskeditor.js now runs
on the real GPU (it had --disable-gpu = SwiftShader), reads the run's result from the page's blobs (the History panel lost its
thumbnails, which had made it stale), and checks reopen + Invert: 8 / 8; red-checked (the open-time filter off: 4 fail).
**Still open on Painter itself:** the rule inside a soft stroke (above).

## Next: the Anima family (planned 2026-11-27)

Five templates (template-needs.json): image_anima_preview, image_anima_base_v1 (t2i, turbo LoRA), image_anima_lllite_image_inpainting
(the Painter template), _any_control_to_image (Canny), _depth_control_to_image (Depth Anything 3). What Anima is (public model cards, not
ComfyUI's GPL source): a Cosmos-Predict2-2B text-to-image DiT (CosmosTransformer3DModel, 2B, 28 layers) with Qwen3 0.6B in place of
Cosmos's T5-11B, decoded by the Qwen-Image VAE. Licences: Anima's weights are CircleStone Labs Non-Commercial v1.2 (+ NVIDIA Open Model
License as a Cosmos derivative): users download them, SpawnGen ships none; tell TJ before an Anima example goes on the site. Clean
architecture reference: Diffusers' CosmosTransformer3DModel (Apache-2.0, portable with attribution like InvokeAI); ComfyUI only as a black
box (its outputs, measured).
SpawnGen already has: the Qwen-Image VAE (qwen_image_vae.safetensors), a Qwen3 0.6B encoder (ACE-Step's Qwen3-Embedding-0.6B path: check
whether Anima's qwen_3_06b_base is the same architecture / which hidden layer it takes), KSampler, LoRA loading, ComfySwitchNode,
ResizeImageMaskNode, MaskPreview, ImageCompare, ResolutionSelector (already measured; a 280-point grid re-checked 2026-11-27: same
rule). Missing: the Cosmos DiT, CLIPLoader recognising the Anima encoder file (type stable_diffusion), ModelPatchLoader,
AnimaLLLiteApply (ControlNet-LLLite patches), and for the depth template DA3 (Depth Anything 3) nodes.
**Step 1 done 2026-11-27: the files** (in _scratch/comfyui-ref/models, SHA-256 = HF's): anima-base-v1.0 (bd43b7cf..., 4.18 GB bf16),
anima-preview3-base (14fffe8a..., 4.18 GB), qwen_3_06b_base (cd2a5120..., 1.19 GB), anima-turbo-lora-v0.2 (1b55e40b..., 149 MB),
anima-lllite-inpainting-v2 (5242e677..., 66 MB); qwen_image_vae (a70580f0...) is byte-identical to the one SpawnGen already has.
**What their headers say** (tensor names and shapes only):
- The DiT (`net.*`, 685 tensors): 28 blocks, width 2048, 16 heads x 128 with q / k RMS norms, self-attention and cross-attention
  (context width 1024), MLP 2048 -> 8192 -> 2048, adaLN-LoRA modulation (per block three pairs of [256, 2048] for self-attn, cross-attn
  and MLP; t_embedder 2048 -> 6144; t_embedding_norm), patch embed 68 -> 2048 (= (16 latent channels + 1 mask channel) x 2 x 2), final
  layer 2048 -> 64 (16 x 2 x 2). That is Cosmos-Predict2-2B's MiniTrainDIT.
- PLUS `net.llm_adapter`: 6 transformer blocks at width 1024 (self-attn, cross-attn, MLP 4096 with bias, q / k norms of 64), an embedding
  of 32128 rows (T5's vocabulary), out_proj 1024 -> 1024. Reading: the prompt also goes in as T5 token ids whose embeddings attend to the
  Qwen3 hidden states, and the adapter's output is the DiT's 1024-wide context. So the text side needs Qwen3's tokenizer AND T5's
  (to be confirmed on ComfyUI's black box: which ids, which Qwen3 layer, padding / length).
- The encoder: plain Qwen3-0.6B (28 layers, 1024 wide, q 16 x 128, kv 8 x 128, MLP 3072, vocab 151936): the same shapes as the
  Qwen3-Embedding-0.6B SpawnGen runs for ACE-Step (different weights).
- The LLLite patch (metadata `anima-preview/control-net-lllite`, version 2, 1056 tensors): targets `self_attn_q_pre, self_attn_kv_pre,
  mlp_fc1_pre` in the DiT blocks; a conditioning net on a 4-channel input (picture + mask, presumably) with 4 res blocks -> 64 features;
  per target `down` 2048 -> 64, `mid` 128 -> 64, `up` 64 -> 2048, `cond_to_film` 64 -> 128 and a `depth_embed`. Kohya's ControlNet-LLLite
  (sd-scripts, Apache-2.0) is the clean reference for the scheme.
**Step 2, run 1 done 2026-11-27 (`_scratch/probe/anima/queue_run1.py`, dumps in `_scratch/probe/anima/run1`, ComfyUI --cpu
--cache-none with 6 threads: Riker's Chrome keeps 2):** the preview template at 512 x 512, 4 steps, cfg 4, er_sde / simple, seed 42 ->
a clean picture (comfyui-ref/output/anima_run1_00001_.png); 47 s a step on the CPU. CLIPLoader type `stable_diffusion` detects the
Anima encoder from the file. Measured:
- CLIPTextEncode gives cond [1, N, 1024] = Qwen3 hidden states for the N Qwen token ids (no BOS / EOS: 24 ids for the positive, 23 for
  the negative), attention_mask [1, N], pooled None, and **t5xxl_ids** [M] int32 (T5 vocabulary, ending with EOS 1) with t5xxl_weights
  all 1.0 (M = 26 / 27): which Qwen layer / norm the cond is, is to be found against these dumps.
- The model is called with c_crossattn [2, 512, 1024] (the adapter's output, padded to 512 rows), latents [2, 16, 1, 64, 64] (16
  channels, a 1-frame video shape, 512 / 8), timestep = sigma (first 0.9999667, not 1).
- simple scheduler, 4 steps: sigmas 1, 0.9, 0.75, 0.5, 0 = flow matching with shift 3.
- VAEDecode loads it as WanVAE (the Qwen-Image VAE SpawnGen has).
**Run 3 (taps for Tuvok, `queue_run3.py`, `_scratch/probe/anima/run3-taps`, copied to `_DevComms/anima-run3-taps-2026-11-27`):**
the adapter runs T5 ids -> embed -> in_proj -> 6 blocks (each cross-attending to the Qwen3 states, RoPE on both sides, no masks) ->
out_proj -> norm (that order) = [1, M, 1024]; ComfyUI pads it with zero rows to 512 and batches [uncond, cond]; the DiT's patch
embed takes the latent plus a 17th channel of zeros; RoPE is built as rotation matrices [1024, 64, 2, 2], no absolute position
embedding. Asked of Tuvok 2026-11-27 (_DevComms 2215 + 2310): the DiT and the adapter in SpawnDev.ILGPU.ML; he starts after ML 5.8.38.
**Tokenizers settled 2026-11-27:** ComfyUI's ids for both prompts are exactly Qwen/Qwen3-0.6B-Base's tokenizer.json with no special
tokens (24 / 23 ids) and google-t5/t5-base's tokenizer.json (T5 v1.1 shares its SentencePiece vocabulary; EOS 1 appended, 26 ids);
google/t5-v1_1-xxl has no tokenizer.json. SpawnGen side: Qwen2Tokenizer on the Qwen3 file, the T5 tokenizer SpawnGen / ML already use
for LTX-Video, each checked against run 1's ids in a test when the encoder node is written.
**The conditioning settled 2026-11-27:** cond = Qwen3-0.6B's last hidden state AFTER the final norm (Hugging Face transformers' Qwen3Model,
Apache-2.0, on qwen_3_06b_base.safetensors with Qwen/Qwen3-0.6B-Base's config, CPU fp32, run 1's 24 ids: 6.9e-7 relative to ComfyUI's
pos cond; every other layer is off by 1 or more). That is the path SpawnGen already runs for ACE-Step's caption (Qwen3-Embedding-0.6B,
final norm): Anima's text encoder = that encoder on the Anima file, the Qwen3-0.6B-Base tokenizer without special tokens, plus the T5
ids for the adapter.
**Waiting on Tuvok's API before the encoder node:** CLIPLoader type stable_diffusion must send a Qwen3 language-model file (the header
test Qwen3TextModel.Is) to an Anima encoder instead of SdTextEncoder; its CONDITIONING has to carry the T5 ids as well as the Qwen3 states
(SpawnGen's TextEncoding holds one tensor + pooled today: add the ids as a CondEntry option, as ComfyUI's t5xxl_ids, unless the library
takes the adapter's output directly, in which case the encoder runs the adapter and the cond is c_crossattn [N, 1024] + zero rows to
512). Decide with the library's shape of the adapter (inside the DiT call or its own module).
**Decided 2026-11-27 (Tuvok):** the library has `AnimaLlmAdapter` as its own module ((Qwen3 final-norm states [N, 1024], T5 ids [M]) ->
context [M, 1024]) and the DiT takes the context UNPADDED (it pads to 512 with zero rows itself; one context per batch row, CFG rows
[neg, pos]). The adapter's weights are `net.llm_adapter.*` in the DiT file (UNETLoader's), not in the Qwen3 file (CLIPLoader's), so in
SpawnGen: CLIPTextEncode on the Anima Qwen3 gives a CONDITIONING of the Qwen3 states + the T5 ids (a CondEntry option, as ComfyUI's
t5xxl_ids); the Anima model (UNETLoader) runs the adapter once per conditioning entry (cached across steps) and feeds the DiT.
**Tuvok's float64 reference matches every tap (2026-11-28; SpawnDev.ILGPU.ML tools/anima-ref/ANIMA-SPEC.md, ad2a4579: READ IT before
wiring the sampler):** worst stage 6.4e-6 relRMS, call_0 / call_1 end to end 2.4e-6 / 1.2e-6, adapter pos / neg 1.2e-6 / 1.3e-6.
Facts for SpawnGen's side: (1) the dumps' call "output" is ComfyUI's DENOISED x - sigma * v; the DiT returns the velocity v and takes
sigma itself as its timestep (not sigma x 1000): SpawnGen's Anima model object converts (flow model, shift 3, like the other flow
models); (2) the context is padded to exactly 512 zero rows, NOT masked (unpadded is 150 % off): the library's DiT does it; (3) the
adapter's cross-attention ropes both sides (T5 positions on q, Qwen3 positions on k); its in_proj is identity. Next on his side: the
C# port on ML's GPU kernels, gated on call_0 / call_1 and pos / neg.
**ANIMA RUNS IN SPAWNGEN (2026-11-28, branch `anima-local` on ML 5.8.39-local.1):** `AnimaModels.cs` (CLIPLoader stable_diffusion on a
Qwen3-0.6B file -> AnimaTextEncoderModel: final-norm Qwen3 rows + T5 ids in the CONDITIONING; UNETLoader on a file with
net.llm_adapter.* -> AnimaDiffusionModel: the adapter once per conditioning entry, the DiT per step returning v; Wan 2.1 latent format,
shift-3 flow). AnimaTests on the RX 590 (OpenCL): conditioning 3.2e-6 / 5.0e-6, model calls 7.5e-6..1.8e-5 (3.3 s a call at 512 x 512),
the preview template graph (512 x 512, 4 steps, er_sde, cfg 4, seed 42) final latent 2.2e-4, picture 62.7 dB, 46 s.
The tokenizers: Qwen3-0.6B-Base's plain; T5 v1.1's tokenizer.json from LostBeard's LTX repo (t5-base's old-format file made the library's
reader throw: Tuvok fixed that in 5.8.39).
**Two sampler gaps found on the way, fixed for every model (measured, runs 6 to 9):** (1) 15 samplers (dpmpp_sde / 2m_sde / 2m_sde_heun /
3m_sde and their _gpu, er_sde, exp_heun_2_x0 (+ _sde), sa_solver (+ pece), seeds_2 / 3) take a FLOW model's first step from the model's
sigma at percent 1e-4 when the schedule starts at 1 (0.99996668 at shift 3, 0.9999 at shift 1, 0.99998333 at shift 6), x staying as noised
at 1; the other 30 call the model at 1 (`Samplers.OffsetsFirstSigma`). (2) The noise the samplers add on the way comes from the SEED + 1
stream (the initial noise is the seed's): ddpm, er_sde, euler_ancestral, exp_heun_2_x0_sde, lcm, res_multistep_ancestral, sa_solver (+ pece),
seeds_2 / 3 land on ComfyUI's next input to ~1.5e-7 with it (SamplingCore). Before both, the Anima graph was 0.73 / 13.2 dB off.
**Still open from run 9:** the Brownian-tree samplers (dpmpp_sde, dpmpp_2m_sde, dpmpp_3m_sde) match neither stream: SpawnGen's Brownian
path is a stand-in for torchsde's tree (Apache-2.0: portable) -> port it, gate on run 9; dpmpp_2s_ancestral gives NaN from sigma 1 on a
flow model (ComfyUI does not); the cfg_pp ancestral variants and dpm_2_ancestral are not settled by run 9's replay (needs the uncond
estimate / a later call).
**LLLite DONE (2026-11-28, ML 5.8.39-local.2):** `AnimaLLLiteNodes.cs`: Load Model Patch (model_patches folder, MODEL_PATCH) and Apply
Anima LLLite (UNetOptions.AnimaLLLite, in force inside its sigma window like Deep Shrink); the Anima run loads the patch once, gives it the
picture once (the IMAGE goes to the device as [H, W, C] and is reordered there by the library's transpose; InpaintPicture with the mask),
and sets dit.LLLite per call. Against runs 4 / 5 on the RX 590: with the patch 6.4e-6 / 5.0e-6, without (the control) 2.6e-5 / 8.3e-6;
red-checked (the run not setting it: 0.90). Refused until measured: a picture of another size than the latent x 8, a second LLLite on one
model, the 3-channel control LLLites (any-test / depth: their ComfyUI runs first), a start / end window other than the run's (the window
logic is ControlNet's convention, not measured on LLLite).
**MERGED to main 2026-11-28 (714f68c, ML 5.8.39 from nuget.org; NOT deployed):** the Anima family + LLLite. The inpainting template
(run 11: as the template wires it, turbo OFF, 512 x 512, 4 steps cfg 4) gives ComfyUI's latent to 2.2e-5 and the picture at 77.6 dB; the
template is really txt2img with the LLLite (EmptyLatentImage, no img2img), the turbo toggle (ON by default) needs the LoRA below.
**LoRA on branch `anima-lora` (f5383d6, ML 5.8.40-local.1; merge when 5.8.40 is on nuget.org):** UseWeightsAsync takes LoraUse steps,
AnimaLora per file (strength x B A unmerged, DiT blocks only; the same file twice adds strengths), the run sets exactly its handle's LoRAs.
Run 10 through KSampler 8 steps: final latent 1.0e-3 (1.0), 8.3e-4 (0.5), 3.5e-5 without; red check 0.90 / 0.66. The with-LoRA latent
is ~30x further from ComfyUI's than the control: SETTLED by Tuvok (float64 replay of the 8 steps): the turbo trajectory
amplifies any per-call difference ~50x (float64 itself lands 3.8e-4 from ComfyUI with the LoRA, 5.8e-6 without), the LoRA path adds
nothing; the gap is the precise GPU kernels' fp32 error (~5x ComfyUI CPU's) amplified. Bound 1e-2 stays. Gaps seen on the way: Save Latent refused a
video-shaped (5-D) latent (FIXED ccb4e6c: byte for byte ComfyUI's file, which also corrected the header order);
KNOWN FLAKE (2026-11-28): Server.Tests A_node_that_throws_is_reported_with_its_message_and_the_queue_carries_on can time out (its 15 s
wait for the good prompt's execution_success) when the suite runs slow (148 s / 198 s runs, the Server and Nodes test processes sharing
the RX 590); 3 of 3 quiet runs (124 s) green. If it shows on a quiet machine, look at the GPU picture save under contention first. a null literal counts as a missing input in SpawnGen (ComfyUI accepts "compare_view": null).
Browser pass on main DONE (10ff22d): Painter 10/10, Painter subgraph 13/13, mask editor, template status, and the new
drive-anima-templates (4 Anima templates without missing nodes, depth names only DA3; red-checked on the live site). ANIMA IN THE BROWSER (2026-11-28, drive-browser-anima.js): the preview graph
(512 x 512, 4 steps er_sde cfg 4) on WebGPU gives ComfyUI's picture at 62.3 dB (desktop 62.7), ~8 s a step on the RX 590, ~3 min with
the first-use downloads (6 GB at ~50 MB/s from circlestone-labs/Anima). DEPLOYED 2026-11-28 (Pages run 46148945676 from 155f2aa, AOT; Space published); LIVE CHECK ALL GREEN: Pages 17 / 17 examples, Space 5 / 5, 12 UI drives on each site 0 fail. Not yet run in the browser: the LLLite
inpainting template (same engine path; its turbo default refuses until the LoRA merges).
**Then on Anima:** the turbo-ON templates end to end once anima-lora merges; LLLite control-picture patches; Anima in the browser (4.2 GB DiT).
(b, done above) LoRA on the Anima DiT (the base template's turbo LoRA: LoraLoaderModelOnly):
ML-side, applied at load from the LoRA file (agreed with Tuvok 2026-11-28; run 10 for his gate in `_DevComms/anima-run10-turbo-lora-2026-11-28`):
the file is PEFT lora_A / lora_B rank 32 without alpha, blocks.* (896) + llm_adapter.blocks.* (120); in ComfyUI the adapter's output is
UNCHANGED by it (0.0 at strength 1 and 0.5) while the DiT's call moves 1.01 / 0.236: apply it to the DiT blocks only;
(c) when ML 5.8.39 is on nuget.org: merge, the browser (the 4.2 GB DiT on WebGPU), the five templates.
**LLLite data ready ahead of time (2026-11-28, runs 4 and 5, `_DevComms/anima-run4-lllite-2026-11-28`):** anima-base-v1.0 +
anima-lllite-inpainting-v2 on compositor-a.png with a Painter rectangle mask, cfg 1, 2 steps, seed 7, and the same graph WITHOUT
AnimaLLLiteApply as the control; model calls, block 0 / 27 taps, final latents. In ComfyUI the patch is transformer_options patches
(`post_input` among them), not DiT modules. Effect: blocks.0 out 27 %, call output 123 % off the control; with LLLite the 2-step
picture already follows the input picture. Given to Tuvok for after the DiT.
**Steps, in order:** (1) the model files (sizes, hashes) onto L: for the ComfyUI reference (CPU runs ~ RAM: one model at a time,
--cache-none, the RAM / disk watchdog; nothing heavy while Riker runs live DayZ); (2) ComfyUI reference runs of image_anima_preview /
base_v1 with dumps of the encoder output, one DiT call, the sampler's latents and the VAE decode; (3) what SpawnDev.ILGPU.ML lacks for
the Cosmos DiT (ask Tuvok with the dumps; ranked: the DiT blocks first) and the Qwen3 hidden-state path; (4) SpawnGen nodes on the
library parts, parity on the dumps; (5) ModelPatchLoader + AnimaLLLiteApply (the inpainting template first: it is the Painter one);
(6) the templates open and run, then the browser.
**Steps, in order:** (1) black-box probe of the node in ComfyUI's Node 2.0 view (`probe-comfy-painter.js`): what the widget shows, its
tools (brush, eraser, sizes, colours, hardness, opacity, clear, undo), what a stroke stores in `mask` (an uploaded file name? where
and when is it uploaded), how width / height / bg_color are set from the widget, what it shows with an image linked; (2) black-box
runs through /prompt for the node's outputs: no painting, a painted file with and without `image`, sizes that differ from the
painting, bg_color, what MASK is (the painting's alpha?); fixtures from those; (3) the node in SpawnGen.Nodes.Core with tests on the
fixtures; (4) the widget's logic in SpawnGen.Editor with tests from the measurements; (5) the Studio side and a browser drive that
paints, runs and checks the outputs; (6) the five templates open and run. The ComfyUI reference server runs on :8199 while measuring.

## Tensors stay on the GPU (2026-11-19, TJ: "the RIGHT WAY TO DO IT so WELL WORTH IT")

**Why.** AGENTS.md rule 4 (zero-copy is law; browser bulk data stays in JS, never the .NET heap) binds SpawnGen's own types. Today IMAGE,
LATENT and MASK are `HostTensor`s (a managed `float[]`), so the site's Text to video example moves every frame GPU -> .NET floats (54 MB
for 25 x 512x352) -> a .NET pixel loop (SaveAnimatedWebpNode.Rgba) -> back to the GPU encoder -> the file to a .NET `byte[]` -> a JS Blob.
The libraries already take and give GPU buffers (Codecs 1.1.1: `ArrayView1D` frames in and the file out for WebP / FLAC / Opus); only
SpawnGen's types and the ML pipelines' host-only video results force the copies.

**Design.**
1. `HostTensor` (IMAGE / LATENT / MASK) gets device residency: an owned ILGPU buffer on the runtime's accelerator, a host array, or both.
   `Data` is the host array and throws while the tensor is device-only; `ToHostAsync()` / `ToDeviceAsync(accelerator)` copy once and keep
   both. Tensors are immutable once a node returns them. SpawnGen.Engine references SpawnDev.ILGPU.
2. The executor brings a node's tensor inputs to the host (awaited, so WebGPU's async readback works) unless the node is marked
   `[Node(DeviceTensors = true)]`: every host node keeps working unchanged, and each node converted to device work drops its copy.
3. Ownership: the result cache counts references to each tensor across its entries (Save nodes pass their input through) and disposes the
   device buffer when the last one goes (eviction, Clear, a lost device). A tensor no entry holds is disposed when the prompt ends.
4. Results: `IResultStore.SaveAsync` takes the file as a device buffer too. The desktop / server write it to disk (a terminal file sink).
   The browser's store keeps every result as a JS `Uint8Array` (written once from the device with `CopyToHostUint8ArrayAsync`); the page's
   blob URL is made from it in JS; .NET reads a result back only when a node loads one (Load Image (from Outputs)).
5. The ML video pipelines (LTX-Video, Wan 2.2, LTX-2) return their frames on the device: asked of Tuvok (the library returns a buffer the
   caller owns, IMAGE's layout [F, H, W, 3] in 0..1). Then VAE Decode, the samplers' latents and Load Image (Codecs' GPU decoders) the same.

**Order.** (a) device residency + executor + cache ownership (engine tests: host nodes unchanged, a device chain never touches the host,
buffers freed); (b) the browser's JS result store and device-buffer saves; (c) Save Animated WEBP from the device (RGBA kernel, Codecs'
`ArrayView1D` animation encoder); (d) the LTX / Wan / LTX-2 nodes on the library's device frames; (e) Save Image / Preview Image (Codecs'
GPU PNG), Save Video / WEBM / Audio, VAE Decode / Encode, Load Image; (f) every remaining node that touches `.Data` in bulk, by use.
Each step: the Text to video example timed in Chrome on the RX 590 before and after (node time, total, page heap), the suite, the drives.
**Status (2026-11-19 evening):** (a) done (90ade44, DeviceTensorTests red-checked); (b) + (c) done (0728817: JS-side browser results,
Save Animated WEBP on the device, ComfyUI parity byte for byte on CPU and RX 590). Waiting: Tuvok's DeviceFrames (ML 5.8.25-local.2) for (d);
Geordi's exact-size EncodeAnimationToDeviceAsync / ToUint8ArrayAsync (replaces the worst-case output buffers). Input files: done 2026-11-20 (A):
uploads, drops, the mask editor's PNG and the example pictures stay in JS (IInputStore.OpenAsync / SaveAsync(StoredBytes), the engines'
UploadFileAsync(Blob), the drop handled in C#, a PNG's workflow found chunk by chunk). (B) on branch load-image-device (74774e4): Load Image decodes on the
device (IPictureDecoder / GpuPictureDecoder over Codecs), exact on the decode corpus on CPU, OpenCL and WebGPU (RX 590). Merge after: Codecs'
WebGPU PNG decode finishes a 512x512 picture (today >2 min / driver watchdog; Geordi, after 1.1.3), the server tests' slowdown is fixed (done on the branch: one shared
GPU runtime, 152 s -> 22 s), and Codecs 1.1.4-local.1 (PNG eXIf orientation) for CodecsDecodeCorpusTests. Load Image's host decoders
were made correct first (efb0832: 16-bit, tRNS, interlaced, PNG orientation, CMYK; TJ: decoding is held to correctness, not ComfyUI's PyAV).
(e) part 1 done (996f1aa): Save Image / Preview Image / every SaveTempAsync preview encode a device picture with Codecs' GPU PNG
(DevicePictureTests, red-checked). Audit after it (2026-11-19): IVaeModel is float[] in / out with a .NET NHWC loop (VaePictures.FromVae),
and UNetRun.PredictAsync is float[] in / out, called every step and CFG branch while the sampler math runs in .NET. By bytes moved the
order is: (d) video frames (Tuvok DeviceFrames), (f) VAE decode / encode on the device (asked of Tuvok: every VAE, NHWC 0..1 out), (g) the
model forward on device views (asked of Tuvok) + SpawnGen's sampler math as device kernels (latents stay on the device between steps).

## The site is the doorstep (2026-11-19, TJ)

TJ: the GitHub Pages site is where people decide whether to try SpawnGen; if it does not work there, no one tries the desktop app. After every
Pages deploy: run EVERY example to the end on the live site in Chrome on the real GPU (`tools/studio-browser-test/drive-site-examples.js`,
default URL the live site; look at each screenshot, out/site-ex-<n>.png), the template browser drive (`drive-template-status.js`), the
upload / App drives; report what was not covered. Found and fixed 2026-11-19 (first full pass, 14 of 15 examples ran):
- Inpainting ran but redrew nothing (example.png has no transparency, so the mask was empty): example_inpaint.png (coat transparent) ships
  beside example.png; ExamplesTests fails any example that uses Load Image's MASK on a picture without one.
- Image to 3D (TRELLIS.2) refused in the browser ("range streaming not wired into this node yet"). TJ: ALL ML runs in the browser and on
  the desktop; no model is desktop-only (the ML demo itself runs image-to-3D and text-to-3D). Fixed at the engine, not per node: the browser
  engine's model source is StreamingModelSource (the ML demo's TRELLIS.2 recipe applied per file: cache in OPFS what fits, stream the rest
  with range requests), so TRELLIS.2 and LTX-2 run in the browser with no special case. A "desktop app" label and an LTX-2 browser refusal
  added earlier the same day were reverted.
- No video examples: Text to video and Picture to video (LTX-Video 2B 0.9.8 distilled, the existing node; about 8 GB of weights on first run).
- Templates showed ComfyUI's whole catalog (772) alike, most with "(missing)" nodes: wwwroot/template-needs.json (tools/templates-manifest)
  lists each template's nodes and model files at a pinned Comfy-Org/workflow_templates commit (the Studio and the server open that commit);
  the browser shows Opens (every node here, the default; 100), Ready (every model file too; 6 in a browser), All, with a badge per card.
- The Speech example saved with Save Audio (FLAC) (DEPRECATED): now Save Audio (Advanced), flac; its note no longer says the listener needs a
  desktop GPU (Parakeet ran in the browser).
- Running every example in a row on the live site lost the WebGPU device at FLUX 3 Klein (it runs alone: 162 s), and every later example then
  failed until a reload. Two fixes: MlRuntime.GetOrLoadPipelineAsync frees the per-node pipelines a prompt has not used before anything new
  loads (PromptScope; ModelResidencyTests), and the browser engine watches GPUDevice.lost and opens a new device for the next run, clearing
  cached outputs (drive-device-lost.js, a real loss through ?testhooks=1; red-checked). Not freed yet: models handed out as MODEL / CLIP /
  VAE (SD checkpoints): they need an unload that reloads on demand, if the full sequence still runs out of GPU memory.
- Hugging Face counts every /resolve/ range request (3,000 per 5 min anonymous); Tuvok is moving ML's three readers to resolve once and range
  the CDN URL (library-bug-reports.md GAP-ML-HF-RESOLVER-RATE).
- Hosted model files (2026-11-19): wwwroot/model-catalog.json (tools/templates-manifest, the same pinned template commit) lists the 403
  model files ComfyUI's templates name with the Hugging Face repo, path and models folder ComfyUI fetches each from; both engines register
  them (HostedModels: the loaders of diffusion_models / text_encoders / vae / controlnet / latent_upscale_models / checkpoints / loras list
  them; a file in the models folder wins). In the browser the template list's Ready went from 6 to 97 of 100, and ComfyUI's SDXL Turbo
  template ran to the end in the local browser Studio with its 6.9 GB checkpoint fetched from Hugging Face (228 s, drive-template-run.js).
- Hugging Face Space (2026-11-19, TJ's idea: traction where ML people are): https://huggingface.co/spaces/LostBeard/spawngen (static SDK,
  app at lostbeard-spawngen.static.hf.space). It serves the GitHub Pages AOT build with the base href "/" and tools/hf-space/README.md as its
  card (the models it uses: those models' pages list the Space). tools/hf-space/publish.sh [pages-run] publishes a Pages run (gh + hf). The
  Space redirects / to /index.html: the Studio answers that route too. Run the site drives against the Space URL after each publish.
- TJ opened Wan 2.2 14B Text to Video (marked ready): a subgraph node with its inputs "connected to nothing". Two fixes: (1) Ready needs every
  model to be one SpawnGen has run (tools/templates-manifest/verified-models.json, 29 files with evidence; Ready 15 of 100 in the browser,
  the rest "not run in SpawnGen yet"); (2) ComfyUI's promoted widgets: a subgraph input that feeds an inner widget shows on the subgraph node
  as that widget with the inner value and the port's label, its value goes to the inner widget when the graph runs, and saving writes the
  values into the definition's inner nodes with the instance's widgets_values empty, as ComfyUI does (PromotedWidgetTests on the template,
  red-checked). 49 of the 100 templates whose nodes all exist use promoted widgets.
Open, measured: FLUX 3 Klein's first run in the browser took 20 min (download). Fixed 2026-11-20: Upscale, Remove the background and Pose and object detection had nodes that grew over their neighbours once pictures showed (ApiPromptImport reserves the square picture; ExamplesTests checks every example). Not covered: a tall portrait picture in a user's own API prompt can still grow past the square room. Wan 2.2 5B Turbo now runs in the browser (ML 5.8.24, RX 590 Chrome: 1280x704 x 9 in 421 s warm, ab6f7ef);
next decide whether it is an example (its weights are ~12 GB); the template "Ready" set for the browser (6) grows only with hosted model files.

## Resume here (state at 2026-11-14 end of day)

- **2026-11-16: ml-local-flux1 merged into main** on SpawnDev.ILGPU.ML 5.8.18 + SpawnDev.ILGPU 5.4.34 (nuget.org): Z-Image, FLUX.1
  (checkpoint and split, Krea), Generate Text, Qwen-Image, Qwen-Image Edit 2509 (Text Encode Qwen Image Edit Plus) all on main; GPU checks on
  the release unchanged (RX 590). **Qwen-Image Edit 2511 runs on main** (ml-local-2511 merged 2026-11-16 on
  SpawnDev.ILGPU.ML 5.8.19, nuget.org): calls <= 9.7e-4 (the ~1e-3 is float32 in ComfyUI's own run), graph 52.4 dB;
  codecs-unreleased merges when SpawnDev.Codecs is released. The lines below are the state before the merge.
- main = this commit: the box canvas, the crop editor, the templates browser, AeVae / InstructPixToPix, PLANS. All suites pass.
  2026-11-15: main on SpawnDev.ILGPU 5.4.33 (nuget.org) with ML 5.8.17; suites pass (3173 / 3 skipped).
  2026-11-15: ml-local-flux1 = 131d35d on ML 5.8.18-local.5 + ILGPU 5.4.33: Krea = ComfyUI (calls 2.35e-6 / 2.39e-5, 65.7 dB; bounds
  tightened), Generate Text passes min_p / presence_penalty. Waiting on Tuvok: the library refuses min_p 1.0 (ComfyUI allows it), so
  Sampling_takes_min_p_and_presence_penalty fails until then. codecs-unreleased = 796db3e on ILGPU 5.4.33.
  main builds with 0 warnings (2026-11-15): the Studio assembly is browser-only (SupportedOSPlatform), the preview-frame parser moved to
  SpawnGen.Editor.PreviewFrames, CA2014 / CS9124 / CS8602 / CS8604 / CS8620 fixed. Keep it at 0.
  2026-11-15: ml-local-flux1 = 0b3deab: **Qwen-Image runs** (split graph, = ComfyUI on the RX 590: conditioning 4.4e-6, calls 1.7e-4, VAE 8e-7,
  graph 72.5 dB, 356 s at 256 px), QwenImageTests. The library's Qwen text encoder takes exactly 512 ids (prompts over 473 tokens refused;
  asked of Tuvok, docs/library-bug-reports.md GAP-ML-QWEN-TEXT-LENGTH). Next on the branch: Qwen-Image Edit 2509 (TextEncodeQwenImageEditPlus:
  vision rows, reference latents) when Tuvok's 2509 path is in a -local; the dumps are his (_DevComms 2026-11-15-1425 / 1450).
  2026-11-15 later: ml-local-flux1 = 8402f6f on ML 5.8.18-local.7: Qwen-Image prompts of any length; **Text Encode Qwen Image Edit Plus**
  (encode side = ComfyUI: conditioning 4.4e-5 / 7.3e-5, reference latents 1.2e-6; vision and reference resize rules measured, QwenImageEditTests).
  **Qwen-Image Edit 2509 runs** (7151226, ML 5.8.18-local.8): model calls with 1 / 2 references = ComfyUI (<= 7.9e-4), the 2509 graph 48.8 dB
  (its negative's sigma-0.75 call is ~140x input-sensitive: measured, see QwenImageEditTests). Merge ml-local-flux1 when ML 5.8.18 is on nuget.org. Big model files live on L: (comfyui-ref/models).
  Sent Tuvok 2026-11-15: 2509 dumps (encode 1 / 2 pictures, both sampling runs), CLIP vision H (_DevComms 2026-11-15-1500: pixel input
  bit-exact). Wan 2.1 i2v 14B: encode side sent (_DevComms 2026-11-15-1535: umT5, WanImageToVideo's concat latent / mask / vision output); the sampling
  stage does not fit NV1080's 32 GB in ComfyUI (the fp8 14B load alone): **parked** (TJ, 2026-11-16: no machine with more RAM); the encode side
  stays gated in the library. Next family (Tuvok): **Qwen-Image Edit 2511** (fp8mixed, 20.5 GB; encode = 2509's; index_timestep_zero
  references): ComfyUI runs in _scratch/probe/qwenedit (stages e2511_1 / e2511_2 / d2511_1). Merge ml-local-flux1 when ML 5.8.18 is on nuget.org
  (Tuvok is cutting it 2026-11-16).
- Branch `ml-local-flux1` (worktree `_scratch/wt-flux1`) = 70b6802 on SpawnDev.ILGPU.ML 5.8.18-local.3 + SpawnDev.ILGPU 5.4.32: Z-Image,
  FLUX.1 (checkpoint and split), Generate Text, measured on the RX 590 (Flux1Tests, Flux1SplitTests, ZImageTests, TextGenerateTests,
  AeVaeTests tightened). Next there: **Qwen-Image** ((d) in the -local.2 list below; Tuvok's API in his messages / Docs; dumps in
  _DevComms 2026-11-14-0800); then MinP / PresencePenalty when the next -local (ILGPU 5.4.33) lands, and Krea's bounds tighten then.
  Merge to main when ML 5.8.18 is on nuget.org.
- Branch `codecs-unreleased` (worktree `_scratch/wt-codecs`) = e889bfa: main merged, Codecs 1.0.2-local.19 + ILGPU 5.4.32, suites pass.
  Merge when SpawnDev.Codecs is released.
- Models for the GPU tests: SPAWNGEN_MODELS_ROOT = the ComfyUI reference install's models folder (big files live there on L:, not on C:) (_scratch/comfyui-ref/models: Z-Image,
  FLUX.1 checkpoint, Krea, Qwen-Image files, qwen_3_4b, clip_l, t5 fp8_scaled, ae); FLUX.2 / LTX use SpawnGen's own models folder.
- The ComfyUI reference server is stopped; restart it for probes (reference-comfyui-ref-tooling memory). Probe nodes in
  comfyui-ref/custom_nodes/spawngen_dump.py now include SpawnGenHookTokenIds, SpawnGenHookVaeModules and SpawnGenHookModelModules.

## Now (in order)

0. **ACE-Step 1.5 (music) in SpawnGen** (2026-11-17, SpawnDev.ILGPU.ML 5.8.20-local.2; branch `ml-local-acestep`). **Status 2026-11-17:** nodes done and measured (PARITY.md 3b: template graph 8.4e-5 latent / 2.1e-4 audio with ComfyUI's table as a fixture, 1.6e-2 with the MIT silence latent); waiting on Tuvok publishing 5.8.20 to nuget.org, then merge to main. Later: VAE Encode Audio (measure its posterior noise first), fractional durations (measure ComfyUI's prompt printing), a batch dimension on AUDIO. ComfyUI's
   audio_ace_step_1_5_split graph: UNETLoader acestep_v1.5_turbo, DualCLIPLoader type ace (qwen_0.6b_ace15 + qwen_1.7b_ace15), VAELoader
   ace_1.5_vae, TextEncodeAceStepAudio1.5, EmptyAceStep1.5LatentAudio, ConditioningZeroOut, ModelSamplingAuraFlow 3, KSampler, VAEDecodeAudio.
   Design: the encode node makes the conditioning ComfyUI's way (caption states [1, n, 1024] + lyric embeddings + audio codes from the
   1.7B LM when generate_audio_codes is on); the model runs ONE call per sampler step (AceStepConditionEncoder for the encoder states, the
   context [source | ones], AceStepDiT.Forward for the velocity), so every SpawnGen sampler / scheduler works; latents are 3-D [b, 64, frames]
   (25 frames per second, 1920 samples each); the VAE decodes to stereo 48 kHz. The silence latent is ACE-Step's MIT silence_latent.pt
   (ComfyUI's own table is GPL data: never shipped; it is a test fixture only), so whole runs sit ~1.6e-2 from ComfyUI's, the parts tight.
   Measure first (black box): ConditioningZeroOut on ACE conditioning (lyrics, codes zeroed?); EmptyAceStep1.5LatentAudio's frame count for
   fractional seconds; the model's own schedule without AuraFlow; that LATENT = the model's space (final latent vs the last call). Then:
   the nodes, tests against Fixtures from _DevComms 2026-11-17-0010 / 1100 / 1230, a browser check, PARITY.md.

Measured 2026-11-14 (`_scratch/count_missing_noapi.py`, the 254 templates without paid API nodes): what a template still lacks is almost
always a model family (Tuvok) or Save / Load Video (Geordi's release); SpawnGen's own nodes unlock little alone. So the host side of the
families Tuvok builds next is done ahead of him, measured against ComfyUI, and the rest waits.

1. **FLUX.1 / Z-Image, SpawnGen's side, before the library's models land** (FLUX.1: 8 templates need no other node; Z-Image: 6):
   - ~~Load VAE / VAE Encode / VAE Decode with ComfyUI's ae.safetensors~~ done (1233d7d, ML 5.8.17; encode differs at a few single latent
     cells, invisible after decoding: the library's fp16 weight storage, proven by Tuvok; full precision for fp32 files in the next ML release,
     then tighten AeVaeTests' A and C bounds to 2e-3).
   - ~~InstructPixToPixConditioning~~ done (4 templates: FLUX.1 canny / depth, HiDream E1; measured against ComfyUI with ae).
   - CLIPTextEncodeFlux (flux_schnell_full_text_to_image): schema and conditioning layout now, runs when the library's T5 / CLIP-L land.
   - **In progress on branch `ml-local-flux1`** (worktree `_scratch/wt-flux1`, SpawnDev.ILGPU.ML 5.8.18-local.1 = ML master 33128b23, which
     has FLUX.1 dev, Z-Image Turbo and Generate Text from ComfyUI's files, gated by Tuvok on CUDA against SpawnGen's dumps; calls in the
     library's Docs/flux1.md). Order: Z-Image (8 templates; its 6B transformer suits 8 GB), Generate Text (qwen_3_4b), FLUX.1 (12B, blocks
     stream past a VRAM budget). Each measured against ComfyUI on the RX 590 (the first AMD numbers). Merge when the ML release is out.
     - Z-Image done on the branch (5a246db): = ComfyUI on the RX 590 (conditioning 3.9e-6, model calls 1.9e-5, the template graph's picture
       68.6 dB at 256 px, 121 s). Those AMD figures showed CUDA's tensor-core path rounding activations to bf16 (5.6e-3 there): Tuvok fixes it
       in the shared GEMM (hi + lo bf16 halves). Waiting: -local.2 with the per-call caption length (cfg above 1 without reopening).
     - Generate Text waits for the library's 8-bit load option (bf16 -> Q8_0 on the device, ~4.3 GB): qwen_3_4b's 8.05 GB of bf16 does not
       fit an 8 GB card.
     - FLUX.1 dev done on the branch (eca3faf): Load Checkpoint takes ComfyUI's all-in-one flux1-dev-fp8.safetensors; = ComfyUI on the RX 590
       (T5 1.4e-5, pooled CLIP-L 1.4e-6, model calls 4.7e-6 / 8.5e-6, the checkpoint graph's picture 69.4 dB at 256 px, 45 s).
     - Qwen-Image dumps sent (_DevComms 2026-11-14-0800): LATENT [b, 16, 1, H/8, W/8]; a system-prompt template whose first 34 tokens
       ComfyUI drops; cond and uncond as separate calls. Tuvok ports it next.
     - **-local.2 done on the branch (2026-11-14):** Z-Image opens once per picture size and passes each call's caption length, precise
       text encoders (604024a); the split FLUX.1 graph (Krea fp8_scaled + clip_l + t5 fp8_scaled) checked on the RX 590: calls 1.3e-3 / 2.6e-3,
       T5 4.7e-3, clip_l 1.35e-6, 55.6 dB, the library's two stated gaps (ILGPU's bf16 rounding in kernels, fixed in 5.4.33; the T5's
       folded scales), the bounds to tighten when they land; Generate Text (135e333, ebd178c): Q8 on the RX 590, ComfyUI's sentence with
       one word changed, the thinking split as ComfyUI's; min_p / presence_penalty asked of the library (on ML master 241611cb, in the next -local with ILGPU 5.4.33: then pass GenerationConfig.MinP and
       PresencePenalty from sampling "on"; seeded draws use the library's RNG, not torch's). **Next: Qwen-Image (d) below.**
     - The -local.2 list (Tuvok, 2026-11-14), (a) to (c) done: (a) Z-Image: open the transformer once with maxCaptionTokens and pass each call's length;
       set Flux2TextEncoder.PreciseActivations = true for the lumina2 encoder (CUDA; OpenCL is fp32 already); (b) the split FLUX.1 graph's
       fp8_scaled files (scale_weight read), and UmT5Encoder.PreciseActivations = true for FLUX.1's T5 (CUDA); (c) Generate Text with the automatic Q8_0 load (13 of 23 greedy tokens = ComfyUI's with thinking
       off, 159 / 159 with it on: say so in the node's docs); (d) Qwen-Image: WanVae in Wan 2.1 mode (LATENT raw [16, 1, h, w], the VAE takes
       the normalised latent: convert at the node, as Wan 2.2), text = Qwen2.5-VL's last hidden state after the final norm, rows from 34 on
       (Flux2Weights / Flux2TextEncoder with layer 28 and finalNorm, PreciseActivations on), QwenImageTransformer (timestep sigma, output the
       velocity, FLUX.1's patchify on the one frame; on 8 GB most blocks stream: expect minutes per step at 1328 px).
     - Next on the branch: the split FLUX.1 graph (Load Diffusion Model + DualCLIPLoader flux + Load VAE ae: flux_dev_full_text_to_image and
       the Krea / Fill / Kontext templates; T5 and CLIP-L from ComfyUI's own t5xxl / clip_l files), schnell, then Generate Text when Q8 lands.
2. **Generate Text (ComfyUI's TextGenerate, 26 templates)**: a CLIP loader's text encoder file run as a language model (prompt
   enhancers in the FLUX 3, Qwen-Image 2.1, Krea 2, Moss, Ernie, HiDream O1 and Stable Audio 3 templates; four plain LLM templates). The library
   generates from GGUF files and Gemma4 only, so it is asked of Tuvok: generation from ComfyUI's own files, qwen_3_4b first (SpawnGen already
   loads it for FLUX.2 Klein; all 36 layers and the final norm are in the file, the output head is tied to the embeddings). ComfyUI's behaviour
   measured (black box, `_scratch/probe/textgen`): llm_qwen3_text_gen's graph, greedy (sampling_mode off), "Describe a red apple in one
   sentence." gives "A red apple is a round, juicy fruit with a smooth, shiny red skin and a sweet, tart flavor."; with thinking on, the
   thinking text comes out on `thinking` without its tags and max_length counts it (160 tokens of thinking left generated_text empty).
   Token-exact references sent (prompt ids as templated, greedy ids, both modes; _DevComms 2026-11-14-0600 folder). Tuvok queued it after
   Z-Image, fed through his GGUF decode engine if the safetensors fit it. Then SpawnGen's node: the schema (sampling on / off as a dynamic combo, image / video / audio inputs refused until the model takes them).
3. **ComfyUI's 3D nodes**, after the families that use them: outside the paid-API templates the missing 3D nodes appear only in
   Pixal3D / TRELLIS.2 from ComfyUI's split files (2 templates: Save 3D (Advanced), Preview 3D (Advanced) and the MESH chain), Hunyuan3D 2.x
   (4: Voxel to Mesh) and TripoSplat (1: the splat nodes). The list of 35 missing nodes and their schemas: `/object_info` of the reference install.

## Waiting on others

| what | who | then |
|---|---|---|
| SpawnDev.Codecs release on nuget.org | Geordi | merge `codecs-unreleased`: Load / Save Video, Save WEBM, Save Audio FLAC / Opus / MP3 (240 + 68 + 36 template uses) |
| the model families below | Tuvok | SpawnGen's nodes for each, measured against ComfyUI |
| GPU lossless animated WebP ~60x slower than libwebp on the RX 590 (25 x 512x352: 1.0.1 345 s, 1.0.2-local.18 373 s and 48% larger; libwebp 5.9 s); _DevComms 2026-11-14-0500. Geordi's diagnosis: a one-thread LZ77 tokenizer run 7-10 times a frame (a parallel one is in progress); local.18's size from transparent pixels cleared to black (to get libwebp's predicted colour); the lossy encoder is slow too (2.15 s a frame on a 4070). Fixed in Codecs 919c2a7 (local.19 after Data's ILGPU gate): on a 4070 the full-frame call 4.26 -> 0.24 s a frame, same bytes; the WebPAnimEncoder call 3.71 -> 0.36 s a frame, 4.79 -> 3.77 MB (libwebp 3.24); lossy next. **local.19 on the RX 590 (fe78d9a on codecs-unreleased): the node's call 373 -> 31.1 s, 3.54 MB; the full-frame call 345 -> 19.4 s, same bytes (libwebp 5.9 s); lossy and the last 9% of size still Geordi's** | Geordi, high: every video save | time his build on the RX 590 (`VideoNodeTests.Save_Animated_WEBP_encode_timing_for_a_clip`, frames in the _DevComms folder) |
| AVIF writer (Save Image (Advanced) avif) | Geordi, low: no template uses it | the avif option |

## Model families asked of SpawnDev.ILGPU.ML

Counted from ComfyUI's template package (comfyui_workflow_templates_json 0.1.122: 635 templates, the 328 with paid API nodes left out), strict
file-name matches, a template counted once per family, 2026-11-13 (`_scratch/count_missing.py` and the family count in the session notes).
Recount only when the template package changes. Every family is loaded from ComfyUI's own files through its split loaders; SpawnGen measures
ComfyUI's LATENT, conditioning and sigmas and sends the dumps before the library builds. Requests: _DevComms `*-Lore-TO-Tuvok-*`
(2026-11-13-2045 FLUX.2, 2026-11-13-2350 this list).

| family | templates | status |
|---|---|---|
| FLUX.2 Klein 4B (split graph) | 5 | done, on main since 2026-11-14 with ML 5.8.17 (text to image, editing, base cfg, 1024) |
| Qwen-Image | 43 | t2i, Edit 2509 and Edit 2511 (1-3 pictures) on main (ML 5.8.19), = ComfyUI |
| Wan 2.1 / 2.2 14B | 24 | requested; Tuvok's 3rd, with CLIP vision |
| CLIP vision (ViT-H, SigLIP 384, ViT-G) | 21 | ViT-H dumps sent 2026-11-15; with Wan 14B |
| LTX-2 / 2.3 / 2.5 (19-22B audio + video) | 18 | requested; Tuvok's 4th. 2026-11-17: LTX-2 19B fp8 runs as a ComfyUI reference on NV1080 (encode 39 GB, transformer 49 GB private: the mapped checkpoint pages; ~3 min a call at 64x64x9); Gemma 3 12B fp4_mixed nvfp4 decode measured bit-exact; first call dumps sent (_DevComms 2026-11-17-1700), then the encode / call / decode taps (1900). TJ: weights licences do not bind our code; Tuvok ports from diffusers' ltx2 (Apache-2.0). SpawnGen's host nodes done (PARITY.md 3b); the model nodes wired and matching ComfyUI on branch ml-local-ltx2 (merge with SpawnDev.ILGPU.ML's next release). Next: batched CFG in the guider (one weight stream per step for streamed models), then LTXVImgToVideoInplace / LTXVPreprocess / LTXVAddGuide once the library has LTX-2's VAE encoder, LoRAs and the latent upscaler (asked of Tuvok 2026-11-17). LTX-2.5 is gated on Hugging Face | **LTXVAddGuide done 2026-11-18** (24 templates use it; PARITY.md 3b has the measured rules and figures). Template coverage (2026-11-19, against SpawnGen's object_info): video_ltx2_t2v / i2v / canny_to_video have every node; depth_to_video lacks LotusConditioning (the Lotus depth model: a model family to request). LTXVImgToVideoInplace runs through the sampler on ML 5.8.23 (Ltx2Guide.InPlace; 2026-11-19). Waiting: the canny / depth IC-LoRA files (ltx-2-19b-ic-lora-*-control, not downloaded). Get IC-LoRA Parameters / AddGuide's iclora_parameters are only in the LTX-2.3 templates (model not in the library yet). | FLUX.1 (dev, schnell, Kontext, Fill) | 13 | requested; Tuvok's 1st (ae.safetensors first): dumps in Now 3 |

**Browser engine page freeze during a model's first load (measured 2026-11-19, low):** the page stops answering while SpawnDev.ILGPU.ML builds the model after the download (Depth Anything V2: DepthEstimationPipeline.CreateFromHubAsync). 32.4 s in the local non-AOT Studio (dotnet run, .NET interpreted), but at most 1.6 s on the AOT build users get (GitHub Pages: whole first run 16 s). Only the dev build is affected badly; the 1.6 s hitch could go if the library yielded during the build.

| ACE-Step | 9 | 1.5 (7 templates; MIT) ComfyUI dumps sent 2026-11-17 (_DevComms 2026-11-17-0010 / 1100); Tuvok: codes off / on, VAE, LM teacher-forced = ComfyUI. **Licence-driven gap:** ComfyUI's silence latent is ComfyUI's own (GPL) table, never shipped; the MIT silence_latent.pt puts a whole run 1.6e-2 (latent) / 3.4e-2 (audio) from ComfyUI's (1.8e-4 with ComfyUI's table as a test fixture). VAEEncodeAudio samples the posterior: measure its noise before claiming parity. Package after the LM sampling loop. v1 (3, Apache-2.0) after |
| HunyuanVideo | 9 | requested |
| Z-Image | 8 | requested; with FLUX.1 (shared ae, qwen_3_4b) |
| HiDream | 7 | requested |
| Generate Text (TextGenerate) from ComfyUI's text encoder files | 26 | requested 2026-11-14: qwen_3_4b first (Now 2); Tuvok: after Z-Image |
| Hunyuan3D 2.x (image to 3D, VAE to voxels) | 4 | not yet asked (Now 3) |
| TRELLIS.2 / Pixal3D from ComfyUI's split files (SpawnGen has the library's one-node TRELLIS.2) | 2 | not yet asked (Now 3) |
| TripoSplat (image to Gaussian splat) | 1 | not yet asked (Now 3) |
| FLUX.2 Klein 9B, FLUX 3 (ComfyUI's files), FLUX.2 dev | 4, 4, 3 | after the above |

Already in the library and in SpawnGen: SD 1.x / 2.x / SDXL (single file and split), SD-Turbo, FLUX.2 Klein 4B and FLUX 3 Klein (one node
each, the library's exports), Wan 2.2 TI2V 5B (one node and the split graph), LTX-Video 2B (Now 1), TRELLIS.2, Kokoro, Parakeet, small LLMs.

## SpawnGen work that needs no library (after Now)

1. Save 3D (Advanced), Load 3D, Preview 3D: ComfyUI's 3D viewport widget (LOAD_3D: camera, model info, render size).
2. ComfyUI's frontend editors SpawnGen still shows as JSON text (the curve editor is done): crop editor, bounding-box canvas, compositor
   editor; Painter (5 templates). **The bounding-box canvas is done** (Create Bounding Boxes, 11 templates; next: the crop editor, then the
   compositor). **Crop editor done** (b8370f4 logic, then the Studio). **Compositor editor done 2026-11-25** (see "Now: the compositor editor" above; Export PSD left). Its measurements: (probe-comfy-bboxes.js with NODE_TYPE=ImageCropV2, Node 2.0
   view): the widget is the input picture with the crop rectangle on it ("No input image connected" without one), a Ratio dropdown (Custom and
   presets) with a lock, and X / Y / Width / Height fields (min 0 / 0 / 1 / 1, max 8192, step 1); the value {x, y, width, height}. With a picture
   (probe-comfy-crop.js, Load Image 768 x 768 -> Crop Image): the upstream picture shows at once (no run needed) with the rectangle and four
   corner handles; a drag inside moves it by the drag in picture pixels, rounded (0.15 / 0.10 of the picture: +115, +77); a drag outside it
   does nothing (no new rectangle); Ratio: 1:1, 3:4, 4:3, 16:9, 9:16, Custom; choosing one keeps the width and sets the height (512 -> 512 x
   288 for 16:9); with a ratio a corner drag keeps it (the drag projected on the ratio's direction: (+50, +100) on 512 x 288 -> 593 x 334);
   corners stop at a 16-pixel side (a top-left corner dragged past the other: 28 x 16 at 16:9). The lock (aria "Lock aspect ratio" ->
   "Unlock aspect ratio" when on) keeps the current aspect for corner drags the same way ((+100, +10) on 400 x 200 -> 484 x 242: the drag
   projected on the aspect, the height from the rounded width); typed values are taken as they are (not clamped to the picture, the lock
   ignored: width 900 at x 100 on a 768 picture stays); drags stay inside the picture (a move past the edge stops at 768 - size, a corner at
   the edge: width 668 from x 100). Measured 2026-11-14. Then: CropWidget logic and tests in SpawnGen.Editor, the row (the linked Load Image's picture), EditorView,
   NodeRenderer, a browser test. ComfyUI's editor measured
   black-box in its Node 2.0 view (`tools/studio-browser-test/probe-comfy-bboxes*.js`, 2026-11-14): a canvas at the boxes' aspect, a Grid
   toggle (on) and Clear all (no confirmation); dragging draws a box (any direction, stored normalised), with Grid on its corners snap to a
   32 canvas-pixel grid at any canvas size (round to nearest), off they round to whole pixels; dragging inside a box moves it, a corner resizes
   it, by the drag rounded the same way; a click selects, Delete removes the selected box; boxes carry a "01" number tag. The selected box's
   panel: obj / text toggle (text adds a "text to render (verbatim)" field), a description field, color_palette swatches with "+" (adds
   #ffffff; a click opens a colour picker with a hex field; right-click removes). Steps: the logic in SpawnGen.Editor (BoxesWidget) with tests
   from these measurements, the row in NodeLayout, pointer modes in EditorView, drawing in NodeRenderer (background from the node's ui
   background_images), the text / colour fields in Studio.razor, and a browser test that repeats the probe's actions and expects ComfyUI's states.
3. Text Overlay (a font engine; also the labels of Create Bounding Boxes' preview and Draw BBoxes).
4. **ComfyUI's templates browser: done** (drive-workflow-templates.js: 761 templates in ComfyUI's 12 groups, thumbnails, search, opening one): ComfyUI's templates are MIT (Comfy-Org/workflow_templates; the comfyui_workflow_templates
   packages say MIT): index.json (12 groups, ~640 templates with title, description, tags, models, size, openSource) and per template
   <name>.json and <name>-1.webp. ComfyUI's server serves them at /templates/<file> (measured). SpawnGen: the server serves /templates/<file>,
   fetched once from raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/ and cached (file names checked); the browser engine
   reads the raw URL; the Studio's Templates dialog: the groups, search, cards (thumbnail, title, description, an API mark for paid ones),
   a click opens the template as a new tab. THIRD-PARTY-NOTICES gets the templates' MIT notice. Then: an installer (signing only if free);
   translations last.

## Recently done

2026-11-14: FLUX.2 Klein split graph merged to main on ML 5.8.17 (VAE 1.1e-4, conditioning 1.3e-6, text to image 36.9 dB, edit
44.4 dB vs ComfyUI on the RX 590); LTX stage timing and the WebP encode timing (725af90); FLUX.1 / Z-Image VAE (ae.safetensors) measured
against ComfyUI (1233d7d).

2026-11-13: Empty SD3 Latent Image and the empty-latent channel rule (42b77d2); Save Image (Advanced) PNG 8/16 and EXR, per-option dynamic
inputs, numeric combo options, tooltips in /object_info (411eb0f); FLUX utility nodes (dcf0d75); Trim Video (874e737); Save 3D Model as
ComfyUI's SaveGLB (e599083); on `codecs-unreleased`: main merged and MP3 saving (9f01b20).

## How the work is done

- ComfyUI behaviour is measured black-box on the reference install (`_scratch/comfyui-ref`, port 8199) with SpawnGen's probe nodes
  (`custom_nodes/spawngen_dump.py`: dump / load tensors, latents, conditioning, sigmas, 3D files); its GPL source is never read. Vary an
  input between repeated prompts: ComfyUI returns identical prompts from its cache.
- Every change: tests against the measured fixtures, a red-check of each guard, the whole suite, a `_TestStats` record, a pushed commit.
  Anything that changes ML output: open the picture or clip and look at it.
