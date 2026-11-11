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
| Queue, interrupt, progress, per-node status, history; ComfyUI's websocket protocol: JSON events (ending each prompt with `executing` for no node), previews as binary frames (PREVIEW_IMAGE, or PREVIEW_IMAGE_WITH_METADATA after a client's `feature_flags`) | **done** |
| Run result images inside nodes | **done** |
| "Control after generate" per seed widget: fixed / increment / decrement / randomize, applied after each queue, saved in the workflow | **done** |
| Batch count (queue N times) | **done**; cancel queued items done (Cancel in the History panel); queue front done (Front button while a run is going, Ctrl+Shift+Enter, `front: true` on POST /prompt); reorder todo |
| ComfyUI execution semantics: list inputs/outputs (runs per item, `INPUT_IS_LIST`, `OUTPUT_IS_LIST`), lazy inputs (`check_lazy_status`), execution blockers, Autogrow inputs (`images.image0` ...), dynamic combos (`resize_type.width` ...), on-demand evaluation with cache signatures from the graph | **done**; node expansion (nodes that add nodes to the graph while it runs, with display ids and loop scopes), MatchType inputs and outputs, and the loop nodes (Start Loop / End Loop in simple, For and List modes, nested, carried values, accumulate, termination inputs) with ComfyUI's loop validation: seven graphs run through ComfyUI give the same previews and the same loop errors (LoopTests) |
| Latent preview during sampling (latent2rgb of the denoised estimate, shown in the sampler node each step; verified by eye: an apple emerging at step 2 of 8) | **done**; preview method per prompt as ComfyUI (`extra_data.preview_method`: auto / latent2rgb, taesd, none; Settings > Sampler preview): taesd decodes each step with taesd_decoder / taesdxl_decoder from models/vae_approx (verified on the RX 590: SD 1.5, 12 steps, a preview per step that ends close to the picture) |
| Samplers and guidance: all 45 ComfyUI sampler names and 9 schedulers in ComfyUI's order, ComfyUI's guidance pipeline (pre-cfg / cfg / post-cfg patches, denoise mask, cfg overrides), guiders (CFG, Basic, Dual CFG, Perp-Neg), CFG patches (RescaleCFG, CFGNorm, CFGZeroStar, APG, Mahiro, TCFG, FreSca, Epsilon Scaling, RenormCFG, Differential Diffusion, Video Linear / Triangle CFG Guidance), UNet patches (FreeU, FreeU_V2, PerturbedAttentionGuidance, Self-Attention Guidance, HyperTile, PatchModelAddDownscale, TomePatchModel), ModelSamplingDiscrete; 159 cases replayed against ComfyUI's own sampling code | **done**; every sampler draws on the RX 590 (SD-Turbo); compared with ComfyUI itself (CPU, same SD-Turbo weights, same start latent): euler, res_multistep_cfg_pp and dpm_fast land within 0.2-0.7 % rms of ComfyUI's result (ComfyUIRenderParityTests); this found and fixed SD 2.x CLIP padding (token 0, not the end token) and latent file scaling. FreeU 0.42 %, FreeU_V2 0.28 %, PerturbedAttentionGuidance 0.43 %, Self-Attention Guidance 0.22 %, HyperTile 0.35 %, PatchModelAddDownscale (deep shrink) 0.32 % of ComfyUI's SD 1.5 results (pictures 0.08-0.13/255; ML 5.8.6). TomePatchModel (ToMe) 0.40 % after one step with top-left destinations on both sides (ComfyUI draws its own with torch.randint, which cannot be repeated; SpawnGen draws them from a per-run generator, so a graph repeats its picture); its merge choices are discrete, so the plain pipeline's small gap flips a few of them and the gap grows with steps (5 % after 10). HyperTile follows ComfyUI's node, which tiles only q (each tile attends to a band of rows of the untiled k / v). Deep shrink's area / bislerp methods are not on the GPU yet. The starting noise is torch's CPU randn for the seed, as in ComfyUI, so deterministic samplers match a seed's picture; the noise the ancestral / SDE samplers draw during the run is SpawnGen's own CPU stream for the seed (ComfyUI draws it on the device, and dpmpp_sde's Brownian tree is a stand-in for torchsde's), so those samplers give different pictures for a seed |
| History panel (H): every run of the session with its saved pictures, full-size view, load a run's workflow into a new tab, clear; queue shows running / queued | **done** (session only; queued runs listed with Cancel; gallery of past server runs and reorder todo) |

## 3b. Model support (library)

| Item | Status |
|---|---|
| SD-Turbo (ONNX): txt2img, img2img | **done** |
| SD 1.5 (ONNX export a420/stable-diffusion-v1-5_onnx, fp32 UNet with external data, real CFG 7, dpmpp_2m/karras 8 steps) | **done**, verified by eye on the RX 590 desktop: 512x512 apple in 133 s including load; and in the browser (WebGPU, RX 590, fp32 UNet as is): the 256x256 picture looks identical to the desktop one (the weights come through the SpawnDev mirror, then Hugging Face on a 404) |
| SD 2.x, SDXL (SDXL: two text encoders, pooled/size conditioning) | **done on branch `sd-checkpoint`** (waits for the ILGPU.ML release with SdCheckpoint): SDXL base from its .safetensors, both towers with ComfyUI's per-tower padding, pooled output, size conditioning (defaults = latent size), CLIP Text Encode (SDXL); verified against ComfyUI's own render at 1024 (rms 0.0001 of 5.84). SD 2.x: sgm-layout files (sd_turbo.safetensors) wait for the library's detection fix |
| Single-file `.safetensors` checkpoints (ComfyUI's main format) | **done on branch `sd-checkpoint`**: `<models>/checkpoints/**/*.safetensors` listed like ComfyUI; SD 1.x and SDXL verified against ComfyUI (SD 1.5 cfg 7.5: 0.19% rms, fp32 file held as fp16); ONNX exports still work through the same nodes |
| Long prompts (more than 75 tokens), CLIP Set Last Layer | **done on branch `sd-checkpoint`**: ComfyUI's chunking (whole pieces, split at 8+ tokens), token ids identical to ComfyUI's on every chunk of both towers, conditioning relRMS ~1.5e-6 (Fixtures/clip); CLIP Set Last Layer on single-file checkpoints (ONNX exports only give their trained layer) |
| LoRA, ControlNet, IP-Adapter, embeddings, upscale models | LoRA **done on branch `sd-checkpoint`**: Load LoRA (Model and CLIP) and Load LoRA (model only) from `<models>/loras` on single-file checkpoints (kohya, PEFT, LDM keys; merged on the GPU, originals restored when the set changes, so inputs never change), verified against ComfyUI (SD 1.5 + LCM-LoRA 0.27% rms). ControlNet, IP-Adapter, embeddings: todo (library) |
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
**290 shared nodes, 286 identical**; the 4 that differ do so by design: the background-removal model list (SpawnGen ships one),
RecordAudio's recorder widget, Resolution Selector's preview-only widget, and Compare Images' view widget, for which SpawnGen states the `{}` its
editor sends (ComfyUI's frontend supplies it; both servers require the input and accept `{}`). Compare Images draws picture A with B over it right of
a split that follows the pointer (tools/studio-browser-test/drive-compare.js). `tools/parity/apply_names.py` keeps names and categories in step.

Behaviour is checked against ComfyUI's own code: fixtures made by running ComfyUI's node code on real torch / kornia / Pillow / numpy (image ops,
compositing, Canny, colour spaces, Pad Image, Color Transfer, SDR to HDR, Image Quantize byte for byte with Pillow's median cut and dithers,
common_upscale incl. Pillow's 8-bit Lanczos; bounding boxes, Crop Image (ImageCropV2), Draw BBoxes, Crop By Bounding Boxes, the tile list split and
merge, and the compositor: Pillow's rectangle drawing, bicubic rotate and premultiplied Lanczos, all 26 blend modes in numpy's float32, layer fingerprints
and saved compositions), Python-exact string, regex, `str.format` and simpleeval ports (about 3,500 CPython vectors), torch's interpolate and bislerp
for latents, and the sampling replay above.

The dataset nodes (comfy_extras/nodes_dataset.py: Load Image / Image-Text (from Folder), Save Image / Image-Text (to Folder), the resize, crop,
normalize, brightness, contrast, shuffle, deduplicate, grid and merge nodes, and the text steps) are checked against ComfyUI's node classes and a
ComfyUI server (Fixtures/dataset): pictures pixel for pixel (Pillow's 8-bit steps, numpy's legacy generator for Random Crop and the shuffles, the
average hash bit for bit), the tensor maths bit for bit, captions, Preview as Text output, file names (overwrite and increment numbering) and caption
bytes exactly. WebP files in a folder decode on the GPU (SpawnDev.Codecs, the first frame like Pillow's convert("RGB")); JPEGs go through
SpawnGen's decoder, which is close to but not bit-exact with Pillow's libjpeg.

Load Image reads PNG (animated PNGs too: frames composited by the APNG / PNG specifications with the frame choices measured from ComfyUI's
PyAV path, including its showing a frame disposed to the background unblended; deliberate difference: "over" blending uses the PNG specification's
formula rounded to nearest, so a blended value can differ from ComfyUI's by 1/255) and JPEG itself and WebP (lossy, lossless, alpha, animated), GIF (animated) and BMP through SpawnDev.Codecs on the host's GPU
(OpenCL / CUDA on the desktop, WebGPU in the browser): every frame is one picture of the batch, transparency gives the mask, and a picture without
transparency gets ComfyUI's empty 64 x 64 mask (Load Image (as Mask) wraps Load Image, as ComfyUI's does). Load Image (from Outputs) lists the output pictures newest first as ComfyUI's
`/internal/files/output` does (the server serves that route too); the Studio refreshes the list after each run, and the node's right-click menu
"Refresh and pick the newest output" stands for ComfyUI's refresh button (control_after_refresh "first"); tools/studio-browser-test/drive-outputs.js. Checked against ComfyUI's own Load Image
(Fixtures/codecs, make_load_image.py) on the CPU accelerator and on the RX 590: batch and mask layout as ComfyUI's, pixels and alpha exactly libwebp's /
Pillow's. Deliberate difference: since ComfyUI #13594 (2026-04-28) its Load Image decodes through PyAV (ffmpeg), which converts most pixel formats
through swscale to float: WebP and BMP colours come out 1 to 2.4 / 255 off the decoded values, fully opaque WebP / GIF pixels get a mask of 1/255
instead of 0, and JPEGs are upsampled by ffmpeg instead of libjpeg (4:2:0 pictures: mean 5/255, up to 68/255 off Pillow's). SpawnGen keeps the
reference decoders' values (what Pillow, libwebp and the browsers show).
Load Image, Load Image (as Mask), Save Image, Preview Image and Compare Images were rebuilt clean-room (2026-11-11) and checked against ComfyUI's
nodes and server (Fixtures/image_nodes, tools/parity/make_image_node_fixtures.py; ImageNodeParityTests): RGB, RGBA and palette PNGs (with and without
tRNS: a palette picture always gets a full-size mask) and an EXIF-rotated JPEG exactly / within 3/255; Load Image (as Mask) gives every frame; Save
Image's names (`%batch_num%` is the picture's index while the counter is the one for the prefix as written, so those names start at 00001 each run, as
ComfyUI's do), its PNG text (prompt, then each extra_pnginfo entry) and the preview prefixes (ComfyUI_temp_ / comfy.compare.a_temp_ / b_temp_ with
the same five letters, comfy.bboxes.background_temp_ for the first background picture only). Preview Image's files carry the prompt; the previews
of Compare Images, Preview Mask and the other newer nodes carry no text, as ComfyUI's. Deliberate differences: ComfyUI (PyAV) fails with
"Invalid argument" on grey, grey + alpha, 1-bit, grey JPEG and 16-bit PNGs; SpawnGen loads them as Pillow does (a 16-bit picture by its high
bytes); PngReader now reads 1-, 2- and 4-bit grey and palette PNGs (PNG spec 7.2, 13.12). Temporary files are numbered as ComfyUI numbers them
(prefix_00001_.png), in both of SpawnGen's stores.
Deliberate differences (SpawnGen is more forgiving where ComfyUI fails): Convert Image to Mask with channel "alpha" on an image without alpha gives a mask of ones (ComfyUI: IndexError); Image Blur / Image Sharpen with a radius as large as the picture keep mirroring the edges (ComfyUI: PyTorch's reflect-padding error).
Deliberate difference: Convert Image Color Space computes every pixel in double precision and rounds once, so it gives the exact values where
ComfyUI's float32 arithmetic differs by up to about 1e-6 of a pixel's largest channel (PQ, LogC3 and ACEScct decoded to large linear values; checked
in float64, 2026-11-08). Results in the 0..1 range agree within 2e-5.
Documented difference: Convert SDR to HDR follows ComfyUI's measured pipeline exactly (sRGB decode, Rec.2020, one gain from Rec.709 luminance,
HLG), but ComfyUI's calibrated luminance curve has no published description, so SpawnGen interpolates a table measured from ComfyUI
(tools/parity/make_sdr_to_hdr_table.py: 101 target luminances by 256 levels). Output values agree to about 2e-5 on the fixture and within about
2e-4 between table knots (under a tenth of an 8-bit level); target luminances between 0 and 0.001 nits use the 0.001 nit curve (2026-11-11).
Documented differences, Create Layered Image (ImageCompositor, rebuilt clean-room 2026-11-11): the layer fingerprints the editor matches saved
compositions by are SpawnGen's own (16 hex digits of SHA-256 over the same inputs ComfyUI's depend on: the 8-bit pixels with the mask applied and the
initial placement, size, rotation, opacity, blend mode, flips and visibility). ComfyUI's hash could not be derived from its behaviour, so a composition
saved by ComfyUI's editor reports stale in SpawnGen (and the other way round) and is drawn from the layer stack; one saved in SpawnGen replays. The test maps
ComfyUI's fingerprints to SpawnGen's position by position and checks that the two tell the same layers apart. The hue, saturation and color blend modes
convert through HLS / HSV in double precision and agree within 2e-7 (the order of ComfyUI's operations there is not known; the other 23 modes are
bit-exact). Rotated layers are placed by the reach of their turned corners, which matches every fixture case; a few exact 360-degree turns and
fractional saved sizes of rotated layers can sit one pixel off ComfyUI's placement.
The model-free audio nodes (Trim Audio Duration, Split / Join Audio Channels, Concatenate Audio, Merge Audio, Adjust Audio Volume, Empty Audio, Audio
Equalizer (3-Band)) are checked against ComfyUI's node classes (Fixtures/comfy/comfy_audio_nodes.json.gz): the equalizer and the sample-exact nodes bit for
bit, resampling (torchaudio's sinc kernel) within 2e-6. Differences: SpawnGen's AUDIO is one clip without a batch dimension; a missing input passes the
other one on; Save Audio writes 16-bit WAV (ComfyUI: FLAC); an unknown merge method is a clear error where ComfyUI fails with UnboundLocalError.
Deliberate difference: Save Text writes line breaks as given (`
`); ComfyUI opens its file in Python's text mode, so on Windows it writes `

` (and `

` becomes `


`), on Linux `
` (measured, clean-room batch C).
Deliberate difference (a ComfyUI bug not copied): measured with ComfyUI, its text steps put out each result as a list of one text, ["text"]
(previews show JSON, chained text steps and Save Image-Text captions fail); SpawnGen's put out the text itself.
Documented difference: VAE Decode (Tiled) and VAE Encode (Tiled) blend their tiles with SpawnGen's own linear ramps over the overlaps, so the
seams can differ slightly from ComfyUI's tiling; the temporal settings are accepted and unused (SpawnGen's VAEs are image VAEs).
Documented difference: Save Checkpoint, ModelSave and VAESave write the prompt and the workflow into the file's metadata; ComfyUI also writes
Stability's modelspec keys (architecture, title and so on), which SpawnGen leaves out.
Documented difference: the beta scheduler (and Beta Sampling Scheduler) rounds a symmetric shape's median timestep where the median lands exactly on
a half (0.5 times an odd count); ComfyUI decides that tie by scipy's last-bit error, which SpawnGen reproduces for the measured shapes (alpha = beta =
0.1, 0.5, 0.6, 1.5, 2.0, and every shape scipy gets exactly); another symmetric shape can land one timestep away from ComfyUI's at that one step.

Still missing, by kind:
1. **Need the library**: CLIP vision, model families (SD3, FLUX.1, LTX, Wan, Hunyuan, Qwen, Cosmos ...). Most ComfyUI templates and blueprints use
   these. Done on branch `sd-checkpoint`: single-file checkpoints, LoRA, and the split loaders (Load Diffusion Model, Load VAE, Load CLIP, Load CLIP (Dual)) for
   SD 1.x / SDXL from `<models>/diffusion_models|unet`, `vae`, `text_encoders|clip`, verified against ComfyUI with its own saved component files (SD 1.5 0.19%,
   SDXL 0.002% and picture 0.00/255) and a real VAE (vae-ft-mse-840000, picture 0.08/255). SD 2.x diffusion models loaded alone (eps / v read from the weights) and
   LoRA on components loaded alone also verified (SD-Turbo split = single file, SD 1.5 split + LCM-LoRA = ComfyUI). ControlNet for SD 1.x and SDXL (diffusers-format ControlNets too; Load ControlNet Model,
   Apply ControlNet, Apply ControlNet (DEPRECATED), chained ControlNets, percent range) verified against ComfyUI with a canny ControlNet (0.18% / 0.17%,
   pictures 0.04/255). Load VAE's tiny VAEs (taesd, taesdxl from <models>/vae_approx) match
   ComfyUI's pictures (0.10/255, 0.04/255). Detect Edges (Canny) gives ComfyUI's edge maps pixel for pixel. InpaintModelConditioning, VAE Encode and VAE Encode
   (for Inpainting) prepare the pictures and masks exactly as ComfyUI's nodes do (crop, greying, mask growth; InpaintParityTests); with an ordinary model the
   run is an img2img confined to the mask (verified on the RX 590); inpainting models (9-channel UNets: the mask and the masked picture's latent as extra
   input, ComfyUI's concat_cond) match ComfyUI with SD 1.5 inpainting (0.08 %, picture 0.02/255). Model merging (ModelMergeSimple / Blocks / Subtract / Add /
   SD1 / SD2 / SDXL, CLIPMergeSimple / Subtract / Add) as weight steps replayed on the GPU in graph order with the LoRAs, and saving (Save Checkpoint,
   ModelSave, VAESave; fp16 LDM names) verified: a merge with a LoRA'd copy = ComfyUI (0.13 %), and the saved checkpoint loads back as that merge.
   Open: CLIPSave (ComfyUI writes Hugging Face names, one file per tower: asked the library), Save Checkpoint of parts loaded separately, fp8 weight_dtype.
2. **Need a codec** (SpawnDev.Codecs; requests with Geordi 2026-11-08 in this order: animated lossy WebP, WebM writer, FLAC of any length, Ogg Opus, VP9 / AV1 inter frames): Save Animated WEBP, SaveWEBM, video (LoadVideo / SaveVideo / CreateVideo / GetVideoComponents), MP3 / Opus / FLAC audio saving. WebP / GIF / BMP loading is done.
3. **Need SpawnGen work**: done: GLSL Shader. src/SpawnGen.Glsl compiles GLSL ES 3.00 fragment shaders (the subset ComfyUI's blueprints use: functions
   and overloads, out / inout, loops, vectors and swizzles, int / uint, texture / texelFetch / textureSize and the usual built-ins) to .NET IL at run time
   and runs them as ILGPU kernels, so the same shader runs on CUDA, OpenCL, WebGPU in the browser and the CPU, with no GL. Against ComfyUI's own
   renderer (ANGLE on the RX 590, Fixtures in tests/SpawnGen.Glsl.Tests): all 21 blueprint cases within 0.33/255 (most 0.000) on the CPU and OpenCL,
   Film Grain included since SpawnDev.ILGPU 5.4.17 (Data's uint constant-conversion fix); browser WebGPU = server OpenCL pixel for pixel
   (drive-glsl.js). Curve Editor (CURVE, ComfyUI's monotone cubic / linear curves, = ComfyUI's LUTs to 1e-12) feeds u_curve; the Studio draws and edits it
   as ComfyUI's curve widget does (drag, add, double-click to remove, interpolation label) with the last run's histogram behind it (drive-curve*.js). Custom Combo is done: the editor
   keeps ComfyUI's frontend-defined option widgets (option1 ..., one spare, hidden index; saved as [choice, index, options], all 116 blueprints
   round-trip; drive-customcombo.js) and the node skips value checks as ComfyUI's validate_inputs does.
   Text Overlay (a font engine; the same engine would draw the number tags and descriptions in Create Bounding Boxes' preview and the labels of Draw
   BBoxes, which are left out until then). Done on branch `bbox-nodes`: object-valued widgets (BOUNDING_BOX, BOUNDING_BOXES, COMPOSITOR) in prompts,
   workflows and object_info, with Crop Image (ImageCropV2), Create Bounding Boxes, Draw BBoxes, Crop By Bounding Boxes, Split Image into List of Tiles,
   Merge List of Tiles to Image, Add Layer, Layers From Bounding Boxes and Create Layered Image; the Studio edits those widgets as JSON text until it has
   ComfyUI's crop editor, bounding-box canvas and compositor editor (todo).

## 5. Platform

| Item | Status |
|---|---|
| C# node packs in place of Python custom nodes: DLLs in the server's custom_nodes folder are loaded at start, the Studio learns them from /object_info and runs them on the server (docs/CUSTOM_NODES.md, samples/SpawnGen.SampleNodes) | **done**; C# scripting node (code typed in the editor) and browser-side packs todo |
| Themes, internationalisation | themes **done** (Dark, Light); internationalisation todo |
| Model manager: checkpoints the host knows (checkpoints.json in the models folder pointing at any Optimum-style ONNX Stable Diffusion repo, or a folder in the models folder), listed in Load Checkpoint while the server engine is selected; download on first use (desktop), cache (browser OPFS) | **partial**: ONNX checkpoints, plus single-file .safetensors from `<models>/checkpoints` on branch `sd-checkpoint`. The Models panel lists downloaded files by size (desktop hub cache or browser OPFS), deletes one or all, and shows download progress (also in the status line during runs); verified in the page (drive-models.js) |
| Hosted static browser build on GitHub Pages (https://lostbeard.github.io/SpawnGen/), AOT compiled by a manual workflow (about an hour): 400 nodes open in under a second and pan at 60 fps (the interpreter build needs 40 s) | **done** |
| Packaging: self-contained desktop builds (Photino app as an installer / single file) | todo |
