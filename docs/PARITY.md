# ComfyUI parity

What "on par with ComfyUI" means here, what is done, and what is next. Status: **done** (verified), **partial**, **todo**.
"Library" means it depends on what SpawnDev.ILGPU.ML can run; SpawnGen asks the library team (with a repro or a request) instead of working around it.

## 1. Where it runs (SpawnGen's own goal, beyond ComfyUI)

| Item | Status |
|---|---|
| Desktop app, native CUDA/OpenCL (Photino window) | **done** (RX 590 / OpenCL verified) |
| Server with ComfyUI-shaped API (headless) | **done** |
| Browser, static files, WebGPU engine in the page (hosted on GitHub Pages: https://lostbeard.github.io/SpawnGen/, AOT build by manual workflow) | **done**: SD-Turbo, SD 1.5, FLUX.2 Klein and the vision models verified on WebGPU |
| Same workflow file on all three | **done** (ComfyUI workflow JSON) |
| ComfyUI HTTP API for other clients: /prompt /queue /history /view /upload/image /ws /interrupt /object_info (and /object_info/{node}) /models /models/{folder} /embeddings /extensions /free /system_stats | **done** |

## 2. Graph editor

| Feature | Status |
|---|---|
| Pan / zoom / select / box select / drag / resize / collapse | **done** |
| Link drag with type feedback, drop on node, move/remove links | **done** |
| Undo / redo, copy / paste (with internal links, Ctrl+Shift+V keeps inputs) | **done** |
| Mute / bypass | **done** |
| Node search (double click / release link on canvas) | **done**, no category tree yet |
| Widgets: number / text / multiline / combo / toggle | **done**; number drag-scrub (8 px per step) and end arrows **done** |
| Right-click menus | **partial** (node, link, canvas) |
| ComfyUI workflow JSON open / save | **done** (stock workflow opens) |
| Reroute, Note, Markdown Note (editor-only, compiled away); legacy PrimitiveNode folded into its widget on open (Primitive Int/Float/String are real nodes) | **done** |
| Groups (boxes): Ctrl+G around the selection, drawn, drag the title to move with the nodes inside, resize, rename (double click), colours, delete, undo, saved in the workflow | **done** |
| Subgraphs: select nodes, Ctrl+Shift+E (or the context menu) packs them into one node; Ctrl+Shift+E unpacks; "Open Subgraph" edits the inside on the canvas between Subgraph Inputs / Outputs boundary nodes, with breadcrumbs back out (the edit changes every instance); saved in **ComfyUI's subgraph format** (definitions.subgraphs, links from -10 / to -20, revision); all 116 ComfyUI blueprints open and save back with no lost links (ComfyCorpusTests) | **done**; promoted widgets (proxyWidgets), adding/removing ports from inside, and results of output nodes inside a subgraph shown in the node todo |
| Node library sidebar (Nodes button / N key: categories, filter, click or drag onto the canvas) | **done** |
| Minimap (bottom right: nodes, groups, the view; click or drag to move the view; Map button) | **done** |
| Properties panel | **done**: Properties (I, toolbar, node menu "Properties Panel"): title, type, id, category, description, mode, colour, pinned, collapsed, "Node name for S&R", every parameter as a form field (numbers clamped to the widget range, combos, text, multiline, booleans, the seed control), inputs with the node each comes from and outputs with where they go (click to go there); several selected nodes: mode, colour and pin together; each change is one undo step. Node `properties` (S&R name, cnr_id, ver, ...) are kept from opened workflows |
| File name prefix variables | **done**: `%date:yyyy-MM-dd hh:mm:ss%` and `%Node.widget%` (node found by its S&R name, else its title) filled when the prompt is made, like ComfyUI's frontend; `%width%`, `%height%`, `%year%` ... `%second%` filled when saving, like ComfyUI's backend |
| Multiple workflows (tabs: new, rename, close, a file opens in its own tab), autosave to browser storage with the view; restored on reload | **done**; recent files todo |
| Mask editor (Load Image / Load Image (as Mask): right-click > Open in Mask Editor; brush size, paint / erase, clear, invert; saves a picture with the painted area transparent, which Load Image turns into its mask) | **done** |
| Examples menu: 14 ready workflows (text to image, SD 1.5, image to image, hires fix, inpainting, FLUX.2 Klein and its edit, LLM-written prompt, upscale, remove background, depth, detection, speech), each opens in its own tab; every one is validated by a test | **done** |
| Drop a generated PNG (ComfyUI or SpawnGen) or a workflow / API-format JSON on the canvas to open it; PNGs carry prompt + workflow | **done** |
| Settings (remembered in the browser): link render modes (spline / straight / linear / hidden), snap to grid with its size, minimap | **done** |
| Themes, shortcut customisation, shortcut parity check | **themes done**: Settings > Colour palette (Dark, Light) recolours the canvas (grid, nodes, category title tints, widgets, minimap) and the page (CSS variables, applied before the app boots so there is no dark flash), saved with the settings; socket and link colours stay ComfyUI's in every palette. More palettes are data in `Theme.cs`. Shortcut customisation and parity check todo |
| Node colours (ComfyUI presets via the context menu, saved as color / bgcolor, undo, copy), custom titles | **done**; pins (P or the menu: nodes and groups cannot be moved, resized, collapsed or deleted; saved as flags.pinned) done; node id badges (Settings > Show node ids) done; source badges todo |

## 3. Execution

| Feature | Status |
|---|---|
| API-format prompts, validation with per-node errors | **done** |
| Cached re-runs (only what changed) | **done** |
| Queue, interrupt, progress, per-node status, history | **done** |
| Run result images inside nodes | **done** |
| "Control after generate" per seed widget: fixed / increment / decrement / randomize, applied after each queue, saved in the workflow | **done** |
| Batch count (queue N times) | **done**; cancel queued items done (Cancel in the History panel); queue front done (Front button while a run is going, Ctrl+Shift+Enter, `front: true` on POST /prompt); reorder todo |
| ComfyUI execution semantics: list inputs/outputs (runs per item, `INPUT_IS_LIST`, `OUTPUT_IS_LIST`), lazy inputs (`check_lazy_status`), execution blockers, Autogrow inputs (`images.image0` ...), dynamic combos (`resize_type.width` ...), on-demand evaluation with cache signatures from the graph | **done**; node expansion and the loop nodes (StartLoop / EndLoop) todo |
| Latent preview during sampling (latent2rgb of the denoised estimate, shown in the sampler node each step; verified by eye: an apple emerging at step 2 of 8) | **done**; TAESD-quality previews todo |
| Samplers and guidance: all 45 ComfyUI sampler names and 9 schedulers in ComfyUI's order, ComfyUI's guidance pipeline (pre-cfg / cfg / post-cfg patches, denoise mask, cfg overrides), guiders (CFG, Basic, Dual CFG, Perp-Neg), CFG patches (RescaleCFG, CFGNorm, CFGZeroStar, APG, Mahiro, TCFG, FreSca, Epsilon Scaling, RenormCFG, Differential Diffusion), ModelSamplingDiscrete; 159 cases replayed against ComfyUI's own sampling code | **done**; every sampler draws on the RX 590 (SD-Turbo); compared with ComfyUI itself on CPU: heun, dpm_2_ancestral, dpmpp_3m_sde look alike, res_multistep_cfg_pp / res_multistep_ancestral_cfg_pp / dpm_fast differ and are being fixed. Seeds give different pictures than ComfyUI (the noise generator is not torch's) |
| History panel (H): every run of the session with its saved pictures, full-size view, load a run's workflow into a new tab, clear; queue shows running / queued | **done** (session only; queued runs listed with Cancel; gallery of past server runs and reorder todo) |

## 3b. Model support (library)

| Item | Status |
|---|---|
| SD-Turbo (ONNX): txt2img, img2img | **done** |
| SD 1.5 (ONNX export a420/stable-diffusion-v1-5_onnx, fp32 UNet with external data, real CFG 7, dpmpp_2m/karras 8 steps) | **done**, verified by eye on the RX 590 desktop: 512x512 apple in 133 s including load; and in the browser (WebGPU, RX 590, fp32 UNet as is): the 256x256 picture looks identical to the desktop one (the weights come through the SpawnDev mirror, then Hugging Face on a 404) |
| SD 2.x, SDXL (SDXL: two text encoders, pooled/size conditioning) ONNX checkpoints | todo: SDXL needs a node-level change |
| Single-file `.safetensors` checkpoints (ComfyUI's main format) | todo: **library** (needs a UNet/CLIP/VAE builder from weights) |
| LoRA, ControlNet, IP-Adapter, embeddings, upscale models | todo: **library** (LoRA could be applied to loaded weights from SpawnGen) |
| Image Classification (MobileNetV2, 1000 ImageNet classes: labels text + top label) and Match Labels to Image (CLIP ViT-B/32 zero-shot: your own descriptions ranked; 0.6 GB download) | **done**, verified on the RX 590 (OpenCL) with the bus photo: minibus first / "a photo of a bus" first; browser not run; models download on first use |
| Upscale (ESPCN x3: Load Upscale Model + Upscale Image using Model), Remove Background (RMBG 1.4: image + mask), Depth Anything V2 small, Style Transfer (5 styles), Pose Estimation (MoveNet: skeleton + keypoint text), Object Detection (YOLOv8n: boxes + list; found and fixed a library letterbox padding bug in 5.5.1) | **done**, verified by eye on the RX 590 on the desktop (OpenCL) and, except RMBG which ran but was not looked at, in the browser (WebGPU); models download on first use (desktop: the models folder's hub cache; browser: OPFS) |
| FLUX.2 Klein 4B (text to image, picture edit with reference images; 4-bit or 8-bit weights; one node) | **done**, verified by eye on the RX 590 desktop (OpenCL): 512x512 apple in 142 s after the model files were downloaded; browser (WebGPU, RX 590) also verified: 256x256 apple in 40 s with the files cached |
| Audio: AUDIO type, Load Audio / Save Audio / Preview Audio (WAV), Audio Info; Kokoro Text to Speech (English voices; verified: 3.4 s speech, desktop OpenCL and browser WebGPU identical in length and loudness); Speech to Text (Parakeet TDT 0.6B v3; verified: heard "Hello from Spongen. This is a test of speech." from Kokoro's sentence on the RX 590, desktop only: 2.4 GB encoder). History panel plays the sound | **done** |
| Text Generation (LLM): small local GGUF chat models (Qwen2.5 0.5B / 1.5B, Qwen3 0.6B, SmolLM2 360M, Gemma 3 270M) as a node whose text feeds any STRING input; verified: Qwen2.5 0.5B wrote "A single red apple sits gracefully on a wooden table..." and that prompt drew the picture with SD-Turbo (desktop OpenCL); also runs in the browser (WebGPU), 34 s with the model cached. Result text wraps inside Preview Any | **done** |
| FLUX 3 Klein 5.6B (text to image from a scene caption, up to 10 reference pictures; 4-bit or 8-bit) | **done**, verified by eye on the RX 590 (OpenCL): 512x512, 16 steps in 208 s including the 6 GB download; browser not run |
| TRELLIS.2 image to 3D (Image to 3D (TRELLIS.2) node: picture + optional mask or RMBG cut-out -> textured PBR .glb; Save / Preview 3D Model; the Studio opens the .glb in its own WebGL viewer from History or the node's right-click menu) | **done on the desktop**: verified by eye on the RX 590 (OpenCL): an apple picture became a 99,500-triangle textured apple, 486 s warm at 512 / 100k faces (683 s with the 10 GB download); the browser engine says so and is todo (range-streamed weights) |
| Wan video | **library** has the pipeline; the model repo is not readable here (main PC only), todo: expose as nodes |

## 4. Node library (host-side nodes, SpawnGen's to write)

Measured, not estimated: `tools/parity/compare_object_info.py` compares SpawnGen's `/object_info` with a running ComfyUI's (master of 2026-10-22,
API nodes left out) input by input: sections, widget order, types, defaults, ranges, combo options, outputs and their names, display names, categories.
**208 shared nodes, 200 identical**; the 8 that differ do so by design (the background-removal model list, RecordAudio's recorder widget, the
MatchType display type, Resolution Selector's preview-only widget). `tools/parity/apply_names.py` keeps names and categories in step.

Behaviour is checked against ComfyUI's own code: fixtures made by running ComfyUI's node code on real torch / kornia / Pillow / numpy (image ops,
compositing, Canny, colour spaces, Pad Image, Color Transfer, SDR to HDR, Image Quantize byte for byte with Pillow's median cut and dithers,
common_upscale incl. Pillow's 8-bit Lanczos), Python-exact string, regex, `str.format` and simpleeval ports (about 3,500 CPython vectors),
torch's interpolate and bislerp for latents, and the sampling replay above.

Still missing, by kind:
1. **Need the library** (model builders from safetensors): the split loaders (UNETLoader, VAELoader, CLIPLoader, DualCLIPLoader), LoRA, ControlNet,
   CLIP vision, model families (SD3, FLUX.1, LTX, Wan, Hunyuan, Qwen, Cosmos ...). Most ComfyUI templates and blueprints use these.
2. **Need a codec**: video (LoadVideo / SaveVideo / CreateVideo / GetVideoComponents), WebP, MP3 / Opus / FLAC audio saving.
3. **Need SpawnGen work**: GLSL Shader (14 blueprints; WebGL2 in the browser, a GL ES backend on the desktop), Custom Combo (frontend-defined options),
   ImageCropV2 (an object-valued bounding-box widget and a crop editor), the loop nodes (node expansion), Text Overlay (a font engine).

## 5. Platform

| Item | Status |
|---|---|
| C# node packs in place of Python custom nodes: DLLs in the server's custom_nodes folder are loaded at start, the Studio learns them from /object_info and runs them on the server (docs/CUSTOM_NODES.md, samples/SpawnGen.SampleNodes) | **done**; C# scripting node (code typed in the editor) and browser-side packs todo |
| Themes, internationalisation | themes **done** (Dark, Light); internationalisation todo |
| Model manager: checkpoints the host knows (checkpoints.json in the models folder pointing at any Optimum-style ONNX Stable Diffusion repo, or a folder in the models folder), listed in Load Checkpoint while the server engine is selected; download on first use (desktop), cache (browser OPFS) | **partial**: ONNX checkpoints only (single-file .safetensors needs the library). The Models panel lists downloaded files by size (desktop hub cache or browser OPFS), deletes one or all, and shows download progress (also in the status line during runs); verified in the page (drive-models.js) |
| Hosted static browser build on GitHub Pages (https://lostbeard.github.io/SpawnGen/), AOT compiled by a manual workflow (about an hour): 400 nodes open in under a second and pan at 60 fps (the interpreter build needs 40 s) | **done** |
| Packaging: self-contained desktop builds (Photino app as an installer / single file) | todo |
