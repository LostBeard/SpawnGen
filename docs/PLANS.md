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
   compositor). **Crop editor done** (b8370f4 logic, then the Studio; next: the compositor). Its measurements: (probe-comfy-bboxes.js with NODE_TYPE=ImageCropV2, Node 2.0
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
