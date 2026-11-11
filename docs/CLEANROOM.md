# Clean-room rewrite

SpawnGen is MIT. Much of its node code was written while reading ComfyUI's Python (GPL-3), which an MIT project cannot contain. TJ decided
(2026-11-08): rewrite that code clean-room, building behavior only from measurements, specifications and public documentation; leave the git history
as it is. Every new piece of SpawnGen follows the same rules.

## Rules

What a clean-room author may use:

- The node's **interface**: its entry in ComfyUI's `/object_info` (names, types, defaults, options, tooltips) and SpawnGen's schema attributes.
- **Black-box behavior**: SpawnGen's tests and fixtures (outputs made by running ComfyUI), and new measurements made by running ComfyUI or PyAV and
  comparing outputs (`_scratch/comfyui-ref`, `_scratch/comfyui-venv`). Running ComfyUI is fine; reading its source is not.
- **Specifications and papers**: PNG/APNG, RFCs, ISO standards, the papers a method comes from.
- **Public documentation**: docs.comfy.org, ComfyUI's node descriptions and tooltips, PyTorch / NumPy / Pillow / Python documentation.
- **Permissively licensed code** (MIT, BSD, Apache-2.0, zlib, HPND, PSF, CC0), with its notice kept in THIRD-PARTY-NOTICES.md: for example
  k-diffusion (MIT) for samplers, PyTorch (BSD) for tensor semantics, Pillow (HPND) for image operations.

What a clean-room author must not use:

- ComfyUI's source (any `.py` in a ComfyUI checkout, its git history, quotes of it), FFmpeg's source, or any other GPL / LGPL / AGPL / unlicensed code.
- SpawnGen's current implementation of the code being rewritten, or its git history (it was written while reading ComfyUI).
- Messages, notes or summaries that describe ComfyUI's or FFmpeg's internals rather than measured behavior.
- Python tracebacks from ComfyUI: they print its source lines. Measurement scripts catch exceptions and print only the type and message, with
  stderr captured. If source lines appear anyway, the author reports them and the affected code is redone by another fresh author.

Code comments describe measured behavior or the specification ("measured with ComfyUI: ...", "PNG spec 12.x"), never "port of ComfyUI's ...".

## Process (one batch at a time)

1. Lore picks a batch: a few files, with the tests that cover them.
2. The author is a session that has not read ComfyUI's source or the old implementation: a fresh Lore session (from batch I on; a new session
   starts with none of that in its context) or a fresh agent. It works on a branch or in a git worktree, deletes the batch's implementation files
   without reading them, then writes new ones from the allowed sources until the batch's tests pass. Where a test fixture is unclear, it measures
   ComfyUI by running it (in-process calls of the node class or the HTTP API, with tracebacks and warnings suppressed).
3. Lore verifies: the full test suite, behavior checks in the Studio where the nodes have UI, `object_info` parity, and a provenance review of the new
   code (no ComfyUI identifiers or structure beyond the public interface, comments cite measurements or specs).
4. The batch is committed to `main` as "Clean-room: ..." and marked done below.

A test may be loosened only where the clean implementation is correct by the specification and ComfyUI's output differs (then it becomes a documented
deliberate difference in PARITY.md), never to hide a bug.

## Inventory

Status: **rewrite** (written while reading ComfyUI), **review** (built from specifications or permissive sources; check provenance and notices),
**done**.

| Area | Files | Status |
|---|---|---|
| Node files, Nodes.Core | AudioNodes, AudioOps, BoundingBox*, ColorTransferNodes, Compositor*, DatasetNodes, Image* (Nodes, SaveExtraNodes) | rewrite |
| Done | StringNodes (12 nodes; pilot, 2026-11-08: schemas = ComfyUI's object_info, all tests, text chain identical to ComfyUI) | done |
| Done | PrimitiveNodes, UtilityNodes, FlowNodes, FormulaNodes, FrontendNodes (batch A, 2026-11-08: Convert Number and If/Else Switch now take ComfyUI's typed inputs) | done |
| Done | TextNodes, BatchNodes (with the Batching helper), RebatchNodes (batch C, 2026-11-08: schemas unchanged) | done |
| Done | MaskNodes, ColorNodes (batch B, 2026-11-08: 254 random cases bit-identical to ComfyUI; schemas unchanged) | done |
| Done | LoopNodes, CurveNodes, Model3DNodes, ImageUtilityNodes (batch D, 2026-11-08: schemas unchanged; curve widget and histogram page checks pass) | done |
| Done | CompositingNodes (redo F, 2026-11-08: 336 random cases bit-identical to ComfyUI, no source seen; schemas unchanged) | done |
| Done | TileNodes, PadImageNodes (with TorchMath, PyTorch's documented lerp), ImageFilterNodes (batch E, 2026-11-08: Canny pixel-exact on the fixtures, matched against kornia's public functions (Apache-2.0); schemas unchanged) | done |
| Done | ImageOps (batch G, 2026-11-08: 16 nodes + Pixels; 430 random cases bit-exact or within 3.4e-6; the soft_light test was wrong and is corrected to ComfyUI's measured 0.8189) | done |
| Done | LatentOps, LatentNodes (batch H, 2026-11-08: 19 nodes; 689 random cases bit-exact except bilinear / bislerp resizes, which come from the resize helpers) | done |
| Done | LatentOperationNodes, LatentSliceNodes, LatentFiles (batch K, 2026-11-08, e69116b: the .latent header key order differs from ComfyUI's, both valid) | done |
| Done | ImageColorNodes (+ ColorSpaces), ImageExtraOps (batch I, 2026-11-08, written by a fresh Lore session: 8 nodes, schemas = ComfyUI's object_info; Quantize byte-exact on all 233 fixture cases, Bayer spread 512 / palette size measured; YUV, Morphology, Flip, Rotate, Colour to Mask exact; Convert Image Color Space from IEC 61966-2-1, BT.2100, ST 2084, ARRI LogC3, ACEScct S-2016-001 and Bradford, with tone mapping and gamut fitting measured: extended Reinhard (Reinhard et al. 2002) on the batch peak, luminance-keeping fit; computed in double, within 1e-6 of each pixel's largest channel of ComfyUI's float32) | done |
| Not merged | Branch ml-local (attention-multiply nodes, deep shrink area and bislerp) was written before the decision while reading ComfyUI: it stays unmerged. Rebuild these clean-room with the Nodes.ML batches; the ComfyUI renders on that branch (Fixtures/sampling/comfyui_sd15/euler_attn_*, euler_deepshrink_area / _bislerp) can serve as their fixtures. ML 5.8.7 (which has the library side) is on main. | rewrite |
| Done | CommonUpscale, Resample, PilLanczos (own file now), TorchInterpolate, ResizeNodes, ImageTransformNodes without Crop Image to Mask (batch J, 2026-11-08, fresh Lore session: PilLanczos from Pillow 13.0.0's Resample.c / Convert.c (MIT-CMU); PyTorch interpolate kernels; ComfyUI's crop, lanczos byte path, Stitch, Resize And Pad, Scale to Max and Resize Image/Mask rules and bislerp measured black-box. All 84 common_upscale, 96 Stitch, 40 Resize And Pad, 30 Scale to Max fixture cases reproduced; new fixture comfy_resize_nodes.json.gz (389 ComfyUI cases from tools/parity/make_resize_fixtures.py) passes; zero-size crop now fails like ComfyUI; schemas unchanged) | done |
| Done | ImageCropToMaskNode (Crop Image to Mask, 2026-11-11, fresh Lore session: rebuilt from black-box measurement; TRELLIS.2's preprocessing (MIT) read for orientation only. Found: the picture is first shrunk to max(width, height) with int(W k) sizes, the mask is inverted when its edges are mostly opaque and its centre half mostly not, values under 0.05 are dropped, grow is a square max/min, the window is the 8-bit > 204 box grown to the aspect ratio and pad and rounded to even sides, an empty box inverts the shrunk mask and then falls back to stretching the whole composite. New fixture comfy_image_nodes_crop_to_mask.json.gz (50 ComfyUI cases from tools/parity/make_crop_to_mask_fixtures.py) and the 7 older crop cases all pass byte-exact; schema unchanged) | done |
| Note for the GLSL batch | GLSLShader's default shader text has CRLF line endings (git checkout of a raw string); ComfyUI's is LF | to fix in the rewrite |
| Node files, Nodes.ML | AudioModelNodes, ClassificationNodes, ClipNodes, ComponentLoaderNodes, Conditioning, ControlNetNodes, CustomSampling, FluxNodes, GlslShaderNode, GuidanceNodes, LatentOperationCfg, LoraNodes, MergeNodes, MlNodes, SamplerNodes, SamplingNodes, Trellis2Node, UNetPatchNodes, VisionNodes | rewrite |
| Sampling stack | Samplers, SamplersAdvanced, Schedulers, SamplingCore, Guidance, GuidancePatches, ModelSampling, GitsTable, BetaMath, PromptWeights, LatentFormat, LatentPreview, WeightSteps, ModelParts, Checkpoint | rewrite (k-diffusion, MIT, and the papers are allowed sources) |
| Helpers shaped after ComfyUI | AudioDsp, Stores (naming and counter), ImageDecoder | rewrite. Found in batch G: Batching.Images with zero-size inputs throws IndexOutOfRange |
| Format readers and writers | PngReader (+Apng: blend done from the spec, 2026-11-08), PngWriter (+Apng), JpegReader, WavFile | review |
| Pillow / PyTorch behavior | PilDraw, PilTransform, PilQuantize, TorchRandom | review (Pillow HPND, PyTorch BSD: notices). (Batch F found the old TorchInterpolate's bilinear up to 1.2e-7 off torch; TorchInterpolate was rewritten in batch J) |
| Python semantics | Python/Py* | review (Python documentation; CPython is PSF) |
| Runtime and model loading | MlRuntime, GpuHelpers, model sources, CheckpointCatalog, OnnxCheckpointModels, SafetensorsCheckpointModels, CodecsImages, CodecsAudio (branch) | review |
| Engine, Editor, Studio, Server, Runtime, Glsl | SpawnGen's own design, compatible with ComfyUI's formats and API; the executor's caching and validation were modeled on ComfyUI's | review the executor and validator; the rest is SpawnGen's own |
| THIRD-PARTY-NOTICES.md | notices for permissive sources used (Pillow, libjpeg-turbo / IJG, PyTorch; add k-diffusion when the samplers use it) | done, extend as sources are added |
