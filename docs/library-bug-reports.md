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
