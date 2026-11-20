# SpawnGen plans

The one place for what SpawnGen is doing next, what it waits on, and why. Read it first after any reset; change it in the same commit as the
work that changes it. What is already done and how well it matches ComfyUI is in [PARITY.md](PARITY.md).

**Goal.** A ComfyUI-class node-graph studio in C#: ComfyUI's workflows and templates open and run the same, on CUDA, OpenCL and WebGPU in the
browser, with no Python. SpawnGen also works out the SpawnDev libraries on a real project: what SpawnGen needs from SpawnDev.ILGPU.ML
(Tuvok) or SpawnDev.Codecs (Geordi) is asked of them, ranked by what it unlocks, never worked around and never treated as a stop.

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
Open, measured: FLUX 3 Klein's first run in the browser took 20 min (download); the Upscale example's nodes overlap. Next for the site: run the Wan 2.2 5B Turbo node in the browser (the library serves it; never tried
there), then decide whether it is an example; the template "Ready" set for the browser (6) grows only with hosted model files.

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
