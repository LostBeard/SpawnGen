# Library bug reports (from SpawnGen integration)

Found while wiring **SpawnGen** (Blazor WASM + Photino desktop) against NuGet **SpawnDev.ILGPU.ML 5.4.0** / **SpawnDev.ILGPU 5.4.1**, Chrome WebGPU, AMD RX 590 (Ellesmere, 8 GB), Windows 10.

Consumer app: https://github.com/LostBeard/SpawnGen
Reference demo that works: the SpawnDev.ILGPU.ML demo app, `/generate` page

---

## BUG-ML-1 — Exact tiled VAE hard-fails on non-512² (256²)

**Repo / area:** `SpawnDev.ILGPU.ML` — `TiledVaeUpDecoder` / `ImageGenerationPipeline.ExactTiledVaeDecodeAsync`  
**Severity:** Low–Medium (512² browser path is tested/supported; 256² looks untested)  
**Package:** SpawnDev.ILGPU.ML **5.4.0**  
**Does not block:** SpawnGen / demo image gen at **512²**

### Summary
`ImageGenerationPipeline.VaeTileGrid = 2` (exact / seam-free tiled VAE) assumes a **512×512** output (latent **64×64**, mid-block **512×64×64**). At **256×256** (latent **32×32**) it throws instead of tiling at the correct spatial size or refusing with a clear “512 only” check up front.

### Actual
```text
mid-block output must be 512x64x64=2097152, got 524288.
```
(`524288 = 512×32×32`)

Thrown from:
```csharp
// TiledVaeUpDecoder.DecodeUpBlocksAsync
const int C = 512, H = 64, W = 64;
if (midOut.Length != (long)C * H * W)
    throw new ArgumentException($"mid-block output must be {C}x{H}x{W}=...");
```

### Expected
Either:
1. Support ExactTiledVae for 256² (and other multiples) by deriving `H,W` from the captured mid-block / latent size, or  
2. Validate `imageSize` / mid-block shape **before** decode and throw a clear message: e.g. `VaeTileGrid>0 requires 512² (got 256)`.

### Repro
1. Browser or desktop: load SD-Turbo via `ImageGenerator` / `ImageGenerationPipeline`.
2. Set `pipeline.VaeTileGrid = 2`.
3. Generate with `Size = 256`.
4. Observe exception at VAE phase after denoise.

### Notes
- Property docs on `VaeTileGrid` still say `0 = auto (tile 2x2 when output >= 512)`. Implementation default path uses **full-res** (`exactTile = 0`) unless `VaeTileGrid > 0` or `VAE_TILE_EXACT` is set — docs and code disagree (see BUG-ML-3).

---

## BUG-ML-2 — WebGPU SD-Turbo: `DXGI_ERROR_DEVICE_HUNG` on RX 590 / Chrome 155 (UNet @ 512² Release; also VAE @ 256²)

**Repo / area:** `SpawnDev.ILGPU.ML` GraphExecutor / SD-Turbo VAE decode on WebGPU  
**Severity:** Medium (repro on 256² full-res; team has verified **512²** browser image gen)  
**Package:** SpawnDev.ILGPU.ML **5.4.0**, SpawnDev.ILGPU **5.4.1**  
**Browser:** Chrome **155.x** (WebGPU)  
**GPU:** AMD Radeon RX 590 (Ellesmere), 8 GB  
**Host RAM:** ~16 GB (Chrome default GPU-process job limit ~8 GB)  
**Does not block:** continuing SpawnGen on **512²** + tiled VAE

### Summary
**Also seen on Release @ 512² (UNet):** Chrome GPU log (`SPAWNGEN_CHROME_LOG`) reported `DXGI_ERROR_DEVICE_HUNG` / TDR during UNet (`up_blocks.*`), not a clear allocator OOM. Per-op sync made the hang worse. Prefer moderate `MaxPendingReleaseBytes` + tiled VAE; do not enable `GraphExecutor.PerOpSync` for this path.

With accelerators probed correctly (`AllAcceleratorsAsync` + preferred WebGPU), SD-Turbo **downloads/uploads weights fine** (VRAM moves; not a WASM-heap weight path). At **256²** (full-res VAE), denoise completes then **VAE decode** loses the device:

```text
[GE node-162 sync] WebGPU device has been lost and cannot accept commands.
|| reclaims-this-process=9 (244 MiB freed)
|| last 40 ops: … GroupNormalization | Sigmoid | Mul | Conv | … | Resize | Conv
```

Same class of failure as prior changelog entries (VAE `up_blocks` / Chrome GPU-process OOM), still repro on 5.4.0 at **256²**.

### What we already tried (consumer side)
| Mitigation | Result |
|------------|--------|
| `AllAcceleratorsAsync` + `CreatePreferredAcceleratorAsync` (was missing; silent CPU before) | Fixes false “WebGPU” / flat VRAM; real WebGPU upload works |
| Stable Chrome profile (OPFS warm) | Downloads seconds, not tens of minutes |
| `--enable-features=WinSboxHighGPUJobMemoryLimits` | Still device-lost at VAE node-162 |
| `VaeTileGrid = 2` at 256² | Hits BUG-ML-1 |
| `VaeTileGrid = 2` at 512² | Device lost **earlier** (during UNet step 1; `reclaims-this-process=0`) |

### Desktop control
Same machine, **Photino host + OpenCL / Ellesmere**, SD-Turbo 256² via `ImageGenerator`: **success** (preview 256×256, second queued run OK). So GPU + models are fine; browser WebGPU VAE path is the failure.

### Timeline excerpt (256², warm OPFS, high GPU mem flag on)
```text
+35.7s  step 0/3
+44.3s  step 1/3
+55.9s  step 2/3
+88.6s  [GE node-162 sync] WebGPU device has been lost …
```

### Expected
256² SD-Turbo on WebGPU completes without device loss on an 8 GB GPU (demo `/generate` claims this class of machine works), or documents a hard minimum / required `VaeTileGrid` / Chrome flags.

### Ask for ML team
- Confirm whether 5.4.0 still expects ExactTiled auto for browser 512 and what the supported path is for **256**.
- Whether reclaim (`reclaims-this-process=9`) should have prevented this peak, or UNet/TE must `ReleaseWorkingMemory()` before VAE on WebGPU.
- Any known Chrome 155 + AMD Polaris interaction.

---

## BUG-ML-3 — `VaeTileGrid` XML docs ≠ implementation default

**Repo / area:** `ImageGenerationPipeline.VaeTileGrid`  
**Severity:** Medium (docs)  
**Package:** 5.4.0

### Summary
XML docs:
> `0 = auto (tile 2x2 when the output is >=512 on a side, …)`

Implementation (`Generate` / decode path):
```csharp
else if (VaeTileGrid > 0) exactTile = VaeTileGrid;
else exactTile = 0;   // full-res GPU-resident (VaeTileGrid 0 [default] or -1)
```
So **default 0 is full-res**, not auto-tile. Older changelog text described auto-tile; current code comments say full-res is intentional after the GPU-resident VAE change.

### Expected
Update the public property docs (and README) to match: `0` / `-1` = full-res; `N>0` = exact NxN tile; call out **512²-only** for ExactTiled until BUG-ML-1 is fixed.

---

## Not bugs (consumer notes only)

**`MLContext.CreateContext()` → CPU debug backend.** Documented: that path is for the debug/CPU accelerator. Prefer `AllAccelerators` / `AllAcceleratorsAsync` + `CreatePreferredAcceleratorAsync` when you want the best available device (WebGPU in browser, Cuda/OpenCL on desktop). SpawnGen’s early flat-VRAM “upload” was us calling `CreateContext()` wrong, not a library defect.

**Photino `GetService<T>()` — async methods only.** Sync getters and `IProgress<T>` over the wire cannot work on this RPC bridge by design. SpawnGen uses a local façade + `IHostMlBackend`. Do not file against SpawnJS/Photino.

---

## Environment snapshot

| Item | Value |
|------|--------|
| OS | Windows 10 (19045) |
| GPU | AMD RX 590 (OpenCL device name: Ellesmere), 8 GB |
| .NET | net10.0 / SDK with 10.0.3 runtime |
| SpawnDev.ILGPU.ML | **5.4.0** (NuGet) |
| SpawnDev.ILGPU | **5.4.1** (transitive) |
| Browser | Chrome 155.x WebGPU |
| Build | **Release only** (Debug ILGPU IR/IL differs; not comparable) |
| Desktop control | Photino + OpenCL Ellesmere — SD-Turbo OK |
| Consumer | SpawnGen (this repository) |


---

## Suggested owners

| ID | Likely owner |
|----|----------------|
| BUG-ML-1, BUG-ML-2, BUG-ML-3 | ILGPU.ML |

---

## BUG-ML-4 — OpenCL (RX 590): token-embedding Gather returns row id+8192; SD-Turbo output is noise

**Area:** `SpawnDev.ILGPU.ML` Gather op (or SpawnDev.ILGPU OpenCL codegen). **Package:** ML 5.4.0 / ILGPU 5.4.1. Found 2026-10-22.

`ImageGenerationPipeline` on desktop OpenCL (AMD RX 590, "Ellesmere") produces plain noise. Bisect against the CPU accelerator: the first
differing node (of 1621) is the CLIP token-embedding `Gather` ([49408,1024] table, ids [1,77]); the GPU returns exact table rows but the wrong
ones, id + 8192 for every non-BOS id (320 -> 8512, 1125 -> 9317; BOS 49406 correct). Full evidence and repro test:
`StageDiagnostics.Clip_first_diverging_node_gpu_vs_cpu` in `tests/SpawnGen.Nodes.Tests` (explicit test; prints the per-node diff).
Status: sent to Tuvok and Data 2026-10-22; no workaround in SpawnGen.

**Update (2026-10-22, later): likely root cause is in SpawnDev.ILGPU 5.4.1, not the Gather op.** Data's check
(`CopyFromStreamAsync` into an OpenCL buffer, a small console check) fails on the RX 590 with 46,397,466 mismatches at 16 MiB chunks and
50,331,648 at 1 MiB chunks: later chunks' data lands in earlier places (the reused host `byte[]` is overwritten while the non-blocking
OpenCL write is still pending). The +8192 rows = 32 MiB = two 16 MiB chunks. Owner: Data (SpawnDev.ILGPU). Rerun this program and the
SD-Turbo graph against his fixed build when he names the version.

**Resolved (2026-10-22):** fixed in SpawnDev.ILGPU **5.4.2** (forks 2.4.2), released to nuget.org the same day. The stream check reports 0 mismatches
at 16 and 1 MiB chunks, and the library pipeline and the SpawnGen node graph both render a correct apple scene on the RX 590 (OpenCL, seed 7,
1 step, 512). `SpawnGen.Nodes.ML` references `SpawnDev.ILGPU` 5.4.2 directly, so SpawnGen no longer needs any pre-release feed.

---

## BUG-ML-FLUX2VAE-DECODE — FLUX.2 VAE decodes wrong on AMD OpenCL (RX 590); encode is right

**Repo / area:** `SpawnDev.ILGPU.ML` — `StableDiffusion.SdVae` with ComfyUI's `flux2-vae.safetensors`
**Package:** SpawnDev.ILGPU.ML **5.8.17-local.1** (master c4437406); reported 2026-11-14 (_DevComms 2026-11-14-0050-Lore-TO-Tuvok)
**Blocks:** SpawnGen's FLUX.2 VAE Decode on the RX 590 (branch `ml-local-flux2`)

Against ComfyUI's own results (Fixtures/flux2_vae): encode 5.75e-4 / 9.30e-4 relative RMS (fine, as the library's CUDA gate); decode 2.0e-1 /
3.0e-1, green pulled down (channel means 0.504 / 0.462 / 0.510 vs 0.505 / 0.507 / 0.509). A plain `SdVae` encode then decode of the same picture
on the same device is 18.8 dB (ComfyUI's round trip: 38.2 dB), so the decoder itself is at fault there. SD 1.5 / SDXL VAEs decode correctly on
the same card. Also: `SdVae` does not run on the CPU accelerator (group size 256 > 64). Repro: `Flux2VaeTests.Encode_and_decode_match_ComfyUIs_FLUX2_VAE`.

**Cause (proven 2026-11-14 with Tuvok's decoder taps):** the first wrong stage is the decoder's first nearest x2 upsample. Its neighbour table
divides on the device, and AMD OpenCL's float division is not correctly rounded (OpenCL allows 2.5 ulp): `(float)24 / 48` = 0.49999997
(0x3EFFFFFF) on the RX 590, so 13,110 of 27,648 table entries point at the wrong pixel. Power-of-two ratios divide exactly, so SD's 512 px
decodes never showed it. Fix in ML (Tuvok): divide on the host, multiply on the device (5.8.17-local.2). Raised with Data for ILGPU's
OpenCL backend: `-cl-fp32-correctly-rounded-divide-sqrt` would make every kernel's `/` match .NET.

---

## GAP-ML-QWEN-TEXT-LENGTH — Qwen-Image's text encoder runs a fixed 512 rows

**Repo / area:** `SpawnDev.ILGPU.ML` — `Flux2.Flux2TextEncoder` (SeqLen = 512) as Qwen-Image's Qwen2.5-VL encoder
**Package:** SpawnDev.ILGPU.ML **5.8.18-local.6**; reported 2026-11-15 (SendMessage to Tuvok)
**Limits:** SpawnGen's CLIP Text Encode with Load CLIP type qwen_image (branch `ml-local-flux1`)

`Encode` takes exactly 512 ids. ComfyUI encodes a prompt of any length, so SpawnGen refuses a Qwen-Image prompt over 473 tokens (512 minus the
template's 34 + 5) instead of matching it, and every encode computes 512 rows however short the prompt (46 for "a red apple on a wooden table").
Qwen-Image Edit 2509 adds the pictures' vision rows to the same sequence (192 + 198 for two small pictures, measured), so three pictures
already pass 512. Asked: a length per call (or per open), as QwenImageTransformer has. With the padding, SpawnGen matches ComfyUI's
conditioning at 4.4e-6 / 7.9e-6 on the RX 590 (QwenImageTests).

**Resolved (2026-11-15):** SpawnDev.ILGPU.ML **5.8.18-local.7** takes the length per encoder (`maxTokens`), each call any length up to it,
unpadded = padded bit for bit. SpawnGen sizes the encoder to the prompt: no length limit; conditioning 4.2e-6 / 8.2e-6 on the RX 590.

## SpawnDev.ILGPU.ML: an LTX-2 frame mask without guide frames (2026-11-19, asked of Tuvok)

ComfyUI's LTXVImgToVideoInplace keeps the start picture IN PLACE (latent frame 0, noise mask [1 - strength, 1, ...]); the model then runs that
frame at sigma x mask and holds it at the clean latent. SpawnDev.ILGPU.ML 5.8.22 takes per-frame masks only inside an `Ltx2Guide` with at least one
guide frame (`Ltx2AvModel.DenoiseAsync`: "guide frames 0 of N latent frames"). Asked: allow 0 guide frames (the mask as per-frame timestep
multipliers, no boxes, no bias), or a frame-mask parameter without a guide. SpawnGen refuses the case with a clear message until then; the
ComfyUI taps for the test are in Fixtures/ltx2/i2vcall_* (A_start_picture_sample_matches_ComfyUI, strength 1 and 0.6).

## GAP-ML-HF-RESOLVER-RATE (2026-11-19, sent to Tuvok)

Every range request SpawnDev.ILGPU.ML sends to Hugging Face is a /resolve/ call, which the Hub counts against a "resolvers" budget:
measured `ratelimit-policy: "fixed window";"resolvers";q=3000;w=300` (3,000 per 5 minutes, anonymous) on a 302 to the signed CDN URL.
FetchRangeStream (16 MiB ReadAhead) and HttpModelDownloader's segments (16 MiB, adaptive down to 512 KiB) appear to hit /resolve/ per chunk, so
one 27 GB file is ~1,700 calls per load. TJ's observation: likely the cause of past HF throttling (why models.spawndev.com is used so much).
Suggested: resolve once, range the CDN URL (`Response.url` after the 302), re-resolve on expiry; respect the `ratelimit` header. Matters to
SpawnGen since StreamingModelSource (2026-11-19) streams what the browser cannot store.
