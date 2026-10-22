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
| Subgraphs: select nodes, Ctrl+Shift+E (or the context menu) packs them into one node whose inputs and outputs are the links that crossed the boundary; the graph runs as if they were unpacked (nested too, muted works); Ctrl+Shift+E on the node unpacks it; saved in the workflow (own layout in definitions.subgraphs, ComfyUI shows them as missing nodes); pack and unpack are one undo step | **done (v1)**; entering a subgraph to edit it in place, promoting inner widgets to the node, results of output nodes inside a subgraph shown in the node (they do reach the History panel), ComfyUI subgraph file format todo |
| Node library sidebar (Nodes button / N key: categories, filter, click or drag onto the canvas) | **done** |
| Minimap (bottom right: nodes, groups, the view; click or drag to move the view; Map button) | **done** |
| Properties panel | todo |
| Multiple workflows (tabs: new, rename, close, a file opens in its own tab), autosave to browser storage with the view; restored on reload | **done**; recent files todo |
| Mask editor (Load Image / Load Image (as Mask): right-click > Open in Mask Editor; brush size, paint / erase, clear, invert; saves a picture with the painted area transparent, which Load Image turns into its mask) | **done** |
| Examples menu: 14 ready workflows (text to image, SD 1.5, image to image, hires fix, inpainting, FLUX.2 Klein and its edit, LLM-written prompt, upscale, remove background, depth, detection, speech), each opens in its own tab; every one is validated by a test | **done** |
| Drop a generated PNG (ComfyUI or SpawnGen) or a workflow / API-format JSON on the canvas to open it; PNGs carry prompt + workflow | **done** |
| Settings (remembered in the browser): link render modes (spline / straight / linear / hidden), snap to grid with its size, minimap | **done** |
| Themes, shortcut customisation, shortcut parity check | todo |
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
| List inputs/outputs (`INPUT_IS_LIST`), lazy inputs, loops / node expansion | todo (engine) |
| Latent preview during sampling (latent2rgb of the denoised estimate, shown in the sampler node each step; verified by eye: an apple emerging at step 2 of 8) | **done**; TAESD-quality previews todo |
| Samplers ipndm_v, gradient_estimation, euler_cfg_pp, euler_ancestral_cfg_pp, dpmpp_2m_cfg_pp, gradient_estimation_cfg_pp | **done**, apples verified by eye on the RX 590 (SD-Turbo, 4 steps, cfg 1.4); gradient_estimation is noticeably harsh at 4 steps (its gamma-2 extrapolation wants more) |
| History panel (H): every run of the session with its saved pictures, full-size view, load a run's workflow into a new tab, clear; queue shows running / queued | **done** (session only; queued runs listed with Cancel; gallery of past server runs and reorder todo) |

## 3b. Model support (library)

| Item | Status |
|---|---|
| SD-Turbo (ONNX): txt2img, img2img | **done** |
| SD 1.5 (ONNX export a420/stable-diffusion-v1-5_onnx, fp32 UNet with external data, real CFG 7, dpmpp_2m/karras 8 steps) | **done**, verified by eye on the RX 590 desktop: 512x512 apple in 133 s including load; and in the browser (WebGPU, RX 590, fp32 UNet as is): the 256x256 picture looks identical to the desktop one (the weights come through the SpawnDev mirror, then Hugging Face on a 404) |
| SD 2.x, SDXL (SDXL: two text encoders, pooled/size conditioning) ONNX checkpoints | todo: SDXL needs a node-level change |
| Single-file `.safetensors` checkpoints (ComfyUI's main format) | todo: **library** (needs a UNet/CLIP/VAE builder from weights) |
| LoRA, ControlNet, IP-Adapter, embeddings, upscale models | todo: **library** (LoRA could be applied to loaded weights from SpawnGen) |
| Upscale (ESPCN x3: Load Upscale Model + Upscale Image using Model), Remove Background (RMBG 1.4: image + mask), Depth Anything V2 small, Style Transfer (5 styles), Pose Estimation (MoveNet: skeleton + keypoint text), Object Detection (YOLOv8n: boxes + list; found and fixed a library letterbox padding bug in 5.5.1) | **done**, verified by eye on the RX 590 on the desktop (OpenCL) and, except RMBG which ran but was not looked at, in the browser (WebGPU); models download on first use (desktop: the models folder's hub cache; browser: OPFS) |
| FLUX.2 Klein 4B (text to image, picture edit with reference images; 4-bit or 8-bit weights; one node) | **done**, verified by eye on the RX 590 desktop (OpenCL): 512x512 apple in 142 s after the model files were downloaded; browser (WebGPU, RX 590) also verified: 256x256 apple in 40 s with the files cached |
| Audio: AUDIO type, Load Audio / Save Audio / Preview Audio (WAV), Audio Info; Kokoro Text to Speech (English voices; verified: 3.4 s speech, desktop OpenCL and browser WebGPU identical in length and loudness); Speech to Text (Parakeet TDT 0.6B v3; verified: heard "Hello from Spongen. This is a test of speech." from Kokoro's sentence on the RX 590, desktop only: 2.4 GB encoder). History panel plays the sound | **done** |
| Text Generation (LLM): small local GGUF chat models (Qwen2.5 0.5B / 1.5B, Qwen3 0.6B, SmolLM2 360M, Gemma 3 270M) as a node whose text feeds any STRING input; verified: Qwen2.5 0.5B wrote "A single red apple sits gracefully on a wooden table..." and that prompt drew the picture with SD-Turbo (desktop OpenCL); also runs in the browser (WebGPU), 34 s with the model cached. Result text wraps inside Preview Any | **done** |
| FLUX 3 Klein 5.6B (text to image from a scene caption, up to 10 reference pictures; 4-bit or 8-bit) | **done**, verified by eye on the RX 590 (OpenCL): 512x512, 16 steps in 208 s including the 6 GB download; browser not run |
| TRELLIS.2, video (Wan) | **library** has pipelines; todo: expose as nodes |

## 4. Node library (host-side nodes, SpawnGen's to write)

Done: Load Image (IMAGE + MASK from the alpha, PNG and JPEG incl. progressive and EXIF orientation) and Load Image (as Mask), Save/Preview Image, Empty Latent, CLIP Text Encode, KSampler and KSampler (Advanced) with 24 sampler names (euler, euler_ancestral, heun, dpm_2, dpm_2_ancestral, lms, dpmpp_2s_ancestral, dpmpp_sde, dpmpp_2m, dpmpp_2m_sde, dpmpp_3m_sde and their _gpu names, ipndm, ipndm_v, res_multistep, res_multistep_ancestral, ddim, lcm, gradient_estimation, euler_cfg_pp, euler_ancestral_cfg_pp, dpmpp_2m_cfg_pp, gradient_estimation_cfg_pp) and 8 schedulers (normal, simple, sgm_uniform, ddim_uniform, karras, exponential, beta, linear_quadratic), CLIP Text Encode with ComfyUI prompt emphasis ((word:1.3), (word), nested, escaped brackets; verified by eye: a stronger green weight gives a greener apple) and the conditioning nodes, VAE Encode/Decode, Load Checkpoint; and 40 host-side nodes (resampling, image ops, masks, latent ops, primitives, tested against hand-computed values).

Todo, in priority order:
1. **Core image** (done: scale x3, crop, pad for outpaint, blend, composite, invert, batch, from-batch, repeat, blur, sharpen, empty image, flip, rotate, quantize, morphology): color adjust, WebP loading, still todo.
2. **Masks** (done: solid, invert, crop, composite, feather, grow, threshold, image<->mask, load image as mask, image color to mask).
3. **Latent** (done: upscale, upscale by, composite, composite masked, batch, batch seed behavior, save / load latent (.latent safetensors files, ComfyUI layout), from-batch, repeat, rotate, flip, crop, blend, set noise mask, add, subtract, multiply, interpolate): VAE Decode (Tiled) and VAE Encode (Tiled) done (ComfyUI widgets; the library VAE already tiles exactly, so the tile settings do not change the work); bislerp upscale todo; VAE Encode (for Inpainting) done, verified by eye (a vase painted into an apple, the rest untouched).
4. **Sampling** (done: KSampler, KSampler Advanced, 11 samplers, 6 schedulers, denoise, noise masks, SamplerCustom with KSamplerSelect, BasicScheduler, Karras/Exponential/Polyexponential/VP/SDTurbo schedulers, SplitSigmas(+Denoise), FlipSigmas, SetFirstSigma, RandomNoise / DisableNoise, BasicGuider / CFGGuider, SamplerCustomAdvanced; SamplerCustom and SamplerCustomAdvanced == KSampler byte-for-byte on a turbo step), plus ipndm_v, gradient_estimation and the euler / euler_ancestral / dpmpp_2m / gradient_estimation cfg_pp samplers (the CFG++ ones get the unconditional estimate from the sampling core; ipndm_v is lms's variable-step integral; the first three are unit-tested equal to their plain forms when there is no guidance): uni_pc, deis, heunpp2, dpm_fast / dpm_adaptive, er_sde, seeds, sa_solver, dpmpp_2s_ancestral_cfg_pp, (dpmpp_sde and the other SDE samplers draw independent noise instead of k-diffusion's Brownian tree: same statistics, different pictures per seed. dpmpp_3m_sde needs more than a few steps: at 4 SD-Turbo steps it is noisy.)
5. **Conditioning** (done: combine, concat, average, set area, set area percentage, set area strength, set mask, zero out, timestep range): ControlNet apply, unCLIP and the model-specific ones need library support.
6. **Primitives and utilities** (done: int, float, string, multiline string, boolean): Preview Any (text result shown in the node) done; math, string ops, switch, "save text" todo.
7. **Loaders**: separate VAE / CLIP loaders, LoRA loader (once supported), upscale model loader.

## 5. Platform

| Item | Status |
|---|---|
| C# node packs in place of Python custom nodes: DLLs in the server's custom_nodes folder are loaded at start, the Studio learns them from /object_info and runs them on the server (docs/CUSTOM_NODES.md, samples/SpawnGen.SampleNodes) | **done**; C# scripting node (code typed in the editor) and browser-side packs todo |
| Themes, internationalisation | todo (settings dialog exists) |
| Model manager: checkpoints the host knows (checkpoints.json in the models folder pointing at any Optimum-style ONNX Stable Diffusion repo, or a folder in the models folder), listed in Load Checkpoint while the server engine is selected; download on first use (desktop), cache (browser OPFS) | **partial**: ONNX checkpoints only (single-file .safetensors needs the library). The Models panel lists downloaded files by size (desktop hub cache or browser OPFS), deletes one or all, and shows download progress (also in the status line during runs); verified in the page (drive-models.js) |
| Hosted static browser build on GitHub Pages (https://lostbeard.github.io/SpawnGen/), AOT compiled by a manual workflow (about an hour): 400 nodes open in under a second and pan at 60 fps (the interpreter build needs 40 s) | **done** |
| Packaging: self-contained desktop builds (Photino app as an installer / single file) | todo |
