# SpawnGen plans

The one place for what SpawnGen is doing next, what it waits on, and why. Read it first after any reset; change it in the same commit as the
work that changes it. What is already done and how well it matches ComfyUI is in [PARITY.md](PARITY.md).

**Goal.** A ComfyUI-class node-graph studio in C#: ComfyUI's workflows and templates open and run the same, on CUDA, OpenCL and WebGPU in the
browser, with no Python. SpawnGen also works out the SpawnDev libraries on a real project: what SpawnGen needs from SpawnDev.ILGPU.ML
(Tuvok) or SpawnDev.Codecs (Geordi) is asked of them, ranked by what it unlocks, never worked around and never treated as a stop.

## Now (in order)

1. **LTX-Video 2B node**: done 2026-11-13 (verified by eye on the RX 590: 512x352 x 25 in 349 s warm). Follow-up: a per-stage timing run (T5 /
   DiT steps / decode) for Tuvok; the warm run re-streams the T5 every time.
2. **FLUX.2 Klein split graph** on a branch against SpawnDev.ILGPU.ML 5.8.17-local.1 (Tuvok: flux2-vae, Klein 4B bf16 / fp8, qwen_3_4b, and
   the LATENT kernels matching ComfyUI's layout: `VaeToLatent` / `LatentToVae` / `LatentToTokens`). Nodes: Load VAE (flux2-vae), VAE Encode /
   Decode for FLUX.2 LATENT ([b, 128, H/16, W/16], patchified and batch-normalised, centre crop to 16), Load Diffusion Model (Klein 4B), Load CLIP
   type flux2 (qwen_3_4b), CLIP Text Encode, the model's sampling (Flux2Scheduler is done), Set Reference Latent in the model (edit).
   Check against ComfyUI on the reference install, then the template `image_flux2_klein_text_to_image` end to end on the RX 590.
   Merge to main when ML 5.8.17 is on nuget.org.
   Status (branch `ml-local-flux2`, worktree `_scratch/wt-flux2`):
   - Load VAE / VAE Encode / Decode: = ComfyUI's on the RX 590 with ML 5.8.17-local.2 (2d3a202: decode 1.1e-4). The local.1 decode was wrong on
     AMD OpenCL (inexact float division in the upsample table): fixed in ML (Tuvok) and, for every kernel, in ILGPU 5.4.32 (Data, proven on the RX 590).
   - Load CLIP type flux2 (2c71ffe): = ComfyUI's conditioning on the RX 590 (1.3e-6 relative RMS).
   - Load Diffusion Model (flux-2-klein-*) and its own schedule (ab58f40): ComfyUI's distilled text-to-image graph runs on the RX 590, the same
     picture as ComfyUI's (36.9 dB at 256 x 256, fp8 file; 30 s).
   - Reference latents / editing (51ada8c): ComfyUI's edit graph = ComfyUI's green apple on the RX 590 (44.4 dB).
   - Next: the base model with cfg 5, the 1024 x 1024 template once, then merge when ML 5.8.17 is on nuget.org.
3. **Dumps for Tuvok** (his order; same format as the FLUX.2 VAE dumps: raw float32 files plus a note in _DevComms):
   1. ~~ae.safetensors (FLUX.1 / Z-Image VAE)~~ sent 2026-11-14 (_DevComms 2026-11-14-0005): [b, 16, H/8, W/8], values look raw, centred crop to 8.
   2. ~~FLUX.1 dev~~ sent 2026-11-14 (_DevComms 2026-11-14-0300): T5 [1,256,4096] + pooled y [1,768], guidance, model space (raw - 0.1159) x 0.3611, timestep sigma, result = denoised. schnell on request.
   3. ~~Z-Image Turbo~~ sent 2026-11-14 (_DevComms 2026-11-14-0400): one Qwen3-4B hidden state [1, N, 2560] unpadded + mask, AuraFlow shift 3,
      the ae latent format, result = denoised. All three dump requests done; Tuvok builds FLUX.1, then Z-Image.

## Waiting on others

| what | who | then |
|---|---|---|
| SpawnDev.Codecs release on nuget.org | Geordi | merge `codecs-unreleased`: Load / Save Video, Save WEBM, Save Audio FLAC / Opus / MP3 (240 + 68 + 36 template uses) |
| SpawnDev.ILGPU.ML 5.8.17 on nuget.org (the branch is on 5.8.17-local.2) | Tuvok | merge the FLUX.2 branch |
| the model families below | Tuvok | SpawnGen's nodes for each, measured against ComfyUI |
| AVIF writer (Save Image (Advanced) avif) | Geordi, low: no template uses it | the avif option |

## Model families asked of SpawnDev.ILGPU.ML

Counted from ComfyUI's template package (comfyui_workflow_templates_json 0.1.122: 635 templates, the 328 with paid API nodes left out), strict
file-name matches, a template counted once per family, 2026-11-13 (`_scratch/count_missing.py` and the family count in the session notes).
Recount only when the template package changes. Every family is loaded from ComfyUI's own files through its split loaders; SpawnGen measures
ComfyUI's LATENT, conditioning and sigmas and sends the dumps before the library builds. Requests: _DevComms `*-Lore-TO-Tuvok-*`
(2026-11-13-2045 FLUX.2, 2026-11-13-2350 this list).

| family | templates | status |
|---|---|---|
| FLUX.2 Klein 4B (split graph) | 5 | library done (5.8.17-local.1); SpawnGen nodes: Now 2 |
| Qwen-Image | 43 | requested; Tuvok's 2nd |
| Wan 2.1 / 2.2 14B | 24 | requested; Tuvok's 3rd, with CLIP vision |
| CLIP vision (ViT-H, SigLIP 384, ViT-G) | 21 | requested; with Wan 14B |
| LTX-2 / 2.3 / 2.5 (19-22B audio + video) | 18 | requested; Tuvok's 4th |
| FLUX.1 (dev, schnell, Kontext, Fill) | 13 | requested; Tuvok's 1st (ae.safetensors first): dumps in Now 3 |
| ACE-Step | 9 | requested |
| HunyuanVideo | 9 | requested |
| Z-Image | 8 | requested; with FLUX.1 (shared ae, qwen_3_4b) |
| HiDream | 7 | requested |
| FLUX.2 Klein 9B, FLUX 3 (ComfyUI's files), FLUX.2 dev | 4, 4, 3 | after the above |

Already in the library and in SpawnGen: SD 1.x / 2.x / SDXL (single file and split), SD-Turbo, FLUX.2 Klein 4B and FLUX 3 Klein (one node
each, the library's exports), Wan 2.2 TI2V 5B (one node and the split graph), LTX-Video 2B (Now 1), TRELLIS.2, Kokoro, Parakeet, small LLMs.

## SpawnGen work that needs no library (after Now)

1. Save 3D (Advanced), Load 3D, Preview 3D: ComfyUI's 3D viewport widget (LOAD_3D: camera, model info, render size).
2. ComfyUI's frontend editors SpawnGen still shows as JSON text: crop editor, bounding-box canvas, compositor editor; Painter (5 templates).
3. Text Overlay (a font engine; also the labels of Create Bounding Boxes' preview and Draw BBoxes).
4. ComfyUI's templates browser; an installer (signing only if free); translations last.

## Recently done

2026-11-13: Empty SD3 Latent Image and the empty-latent channel rule (42b77d2); Save Image (Advanced) PNG 8/16 and EXR, per-option dynamic
inputs, numeric combo options, tooltips in /object_info (411eb0f); FLUX utility nodes (dcf0d75); Trim Video (874e737); Save 3D Model as
ComfyUI's SaveGLB (e599083); on `codecs-unreleased`: main merged and MP3 saving (9f01b20).

## How the work is done

- ComfyUI behaviour is measured black-box on the reference install (`_scratch/comfyui-ref`, port 8199) with SpawnGen's probe nodes
  (`custom_nodes/spawngen_dump.py`: dump / load tensors, latents, conditioning, sigmas, 3D files); its GPL source is never read. Vary an
  input between repeated prompts: ComfyUI returns identical prompts from its cache.
- Every change: tests against the measured fixtures, a red-check of each guard, the whole suite, a `_TestStats` record, a pushed commit.
  Anything that changes ML output: open the picture or clip and look at it.
