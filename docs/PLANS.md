# SpawnGen plans

The one place for what SpawnGen is doing next, what it waits on, and why. Read it first after any reset; change it in the same commit as the
work that changes it. What is already done and how well it matches ComfyUI is in [PARITY.md](PARITY.md).

**Goal.** A ComfyUI-class node-graph studio in C#: ComfyUI's workflows and templates open and run the same, on CUDA, OpenCL and WebGPU in the
browser, with no Python. SpawnGen also works out the SpawnDev libraries on a real project: what SpawnGen needs from SpawnDev.ILGPU.ML
(Tuvok) or SpawnDev.Codecs (Geordi) is asked of them, ranked by what it unlocks, never worked around and never treated as a stop.

## Now (in order)

Measured 2026-11-14 (`_scratch/count_missing_noapi.py`, the 254 templates without paid API nodes): what a template still lacks is almost
always a model family (Tuvok) or Save / Load Video (Geordi's release); SpawnGen's own nodes unlock little alone. So the host side of the
families Tuvok builds next is done ahead of him, measured against ComfyUI, and the rest waits.

1. **FLUX.1 / Z-Image, SpawnGen's side, before the library's models land** (FLUX.1: 8 templates need no other node; Z-Image: 6):
   - ~~Load VAE / VAE Encode / VAE Decode with ComfyUI's ae.safetensors~~ done (1233d7d, ML 5.8.17; encode differs at a few single latent
     cells, invisible after decoding: the library's fp16 weight storage, proven by Tuvok; full precision for fp32 files in the next ML release,
     then tighten AeVaeTests' A and C bounds to 2e-3).
   - ~~InstructPixToPixConditioning~~ done (4 templates: FLUX.1 canny / depth, HiDream E1; measured against ComfyUI with ae).
   - CLIPTextEncodeFlux (flux_schnell_full_text_to_image): schema and conditioning layout now, runs when the library's T5 / CLIP-L land.
   - Then, as Tuvok lands them: the FLUX.1 transformer through Load Diffusion Model / DualCLIPLoader (dumps sent 2026-11-14-0300), then
     Z-Image (2026-11-14-0400).
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
| GPU lossless animated WebP ~60x slower than libwebp on the RX 590 (25 x 512x352: 1.0.1 345 s, 1.0.2-local.18 373 s and 48% larger; libwebp 5.9 s); _DevComms 2026-11-14-0500. Geordi's diagnosis: a one-thread LZ77 tokenizer run 7-10 times a frame (a parallel one is in progress); local.18's size from transparent pixels cleared to black (to get libwebp's predicted colour); the lossy encoder is slow too (2.15 s a frame on a 4070). Fixed in Codecs 919c2a7 (local.19 after Data's ILGPU gate): on a 4070 the full-frame call 4.26 -> 0.24 s a frame, same bytes; the WebPAnimEncoder call 3.71 -> 0.36 s a frame, 4.79 -> 3.77 MB (libwebp 3.24); lossy next | Geordi, high: every video save | time his build on the RX 590 (`VideoNodeTests.Save_Animated_WEBP_encode_timing_for_a_clip`, frames in the _DevComms folder) |
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
| Qwen-Image | 43 | requested; Tuvok's 2nd |
| Wan 2.1 / 2.2 14B | 24 | requested; Tuvok's 3rd, with CLIP vision |
| CLIP vision (ViT-H, SigLIP 384, ViT-G) | 21 | requested; with Wan 14B |
| LTX-2 / 2.3 / 2.5 (19-22B audio + video) | 18 | requested; Tuvok's 4th |
| FLUX.1 (dev, schnell, Kontext, Fill) | 13 | requested; Tuvok's 1st (ae.safetensors first): dumps in Now 3 |
| ACE-Step | 9 | requested |
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
   compositor). ComfyUI's editor measured
   black-box in its Node 2.0 view (`tools/studio-browser-test/probe-comfy-bboxes*.js`, 2026-11-14): a canvas at the boxes' aspect, a Grid
   toggle (on) and Clear all (no confirmation); dragging draws a box (any direction, stored normalised), with Grid on its corners snap to a
   32 canvas-pixel grid at any canvas size (round to nearest), off they round to whole pixels; dragging inside a box moves it, a corner resizes
   it, by the drag rounded the same way; a click selects, Delete removes the selected box; boxes carry a "01" number tag. The selected box's
   panel: obj / text toggle (text adds a "text to render (verbatim)" field), a description field, color_palette swatches with "+" (adds
   #ffffff; a click opens a colour picker with a hex field; right-click removes). Steps: the logic in SpawnGen.Editor (BoxesWidget) with tests
   from these measurements, the row in NodeLayout, pointer modes in EditorView, drawing in NodeRenderer (background from the node's ui
   background_images), the text / colour fields in Studio.razor, and a browser test that repeats the probe's actions and expects ComfyUI's states.
3. Text Overlay (a font engine; also the labels of Create Bounding Boxes' preview and Draw BBoxes).
4. ComfyUI's templates browser; an installer (signing only if free); translations last.

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
