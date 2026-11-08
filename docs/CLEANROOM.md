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

Code comments describe measured behavior or the specification ("measured with ComfyUI: ...", "PNG spec 12.x"), never "port of ComfyUI's ...".

## Process (one batch at a time)

1. Lore (who has read ComfyUI's source and so does not write rewritten code) picks a batch: a few files, with the tests that cover them.
2. A fresh agent that has not seen this project's history works in a git worktree. It deletes the batch's implementation files without reading them,
   then writes new ones from the allowed sources until the batch's tests pass. Where a test fixture is unclear, it measures ComfyUI by running it.
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
| Node files, Nodes.Core | AudioNodes, AudioOps, BatchNodes, BoundingBox*, ColorNodes, ColorTransferNodes, CompositingNodes, Compositor*, CurveNodes, DatasetNodes, FlowNodes, FormulaNodes, FrontendNodes, Image* (Nodes, Ops, ExtraOps, ColorNodes, FilterNodes, SaveExtraNodes, TransformNodes, UtilityNodes), Latent* (Nodes, Ops, OperationNodes, SliceNodes, Files), LoopNodes, MaskNodes, Model3DNodes, PadImageNodes, PrimitiveNodes, RebatchNodes, ResizeNodes, TextNodes, TileNodes, UtilityNodes | rewrite |
| Done | StringNodes (12 nodes; pilot, 2026-11-08: schemas = ComfyUI's object_info, all tests, text chain identical to ComfyUI) | done |
| Node files, Nodes.ML | AudioModelNodes, ClassificationNodes, ClipNodes, ComponentLoaderNodes, Conditioning, ControlNetNodes, CustomSampling, FluxNodes, GlslShaderNode, GuidanceNodes, LatentOperationCfg, LoraNodes, MergeNodes, MlNodes, SamplerNodes, SamplingNodes, Trellis2Node, UNetPatchNodes, VisionNodes | rewrite |
| Sampling stack | Samplers, SamplersAdvanced, Schedulers, SamplingCore, Guidance, GuidancePatches, ModelSampling, GitsTable, BetaMath, PromptWeights, LatentFormat, LatentPreview, WeightSteps, ModelParts, Checkpoint | rewrite (k-diffusion, MIT, and the papers are allowed sources) |
| Helpers shaped after ComfyUI | AudioDsp, CommonUpscale, Resample, Stores (naming and counter), ImageDecoder | rewrite |
| Format readers and writers | PngReader (+Apng: blend done from the spec, 2026-11-08), PngWriter (+Apng), JpegReader, WavFile | review |
| Pillow / PyTorch behavior | PilDraw, PilTransform, PilQuantize, TorchRandom, TorchInterpolate | review (Pillow HPND, PyTorch BSD: notices) |
| Python semantics | Python/Py* | review (Python documentation; CPython is PSF) |
| Runtime and model loading | MlRuntime, GpuHelpers, model sources, CheckpointCatalog, OnnxCheckpointModels, SafetensorsCheckpointModels, CodecsImages, CodecsAudio (branch) | review |
| Engine, Editor, Studio, Server, Runtime, Glsl | SpawnGen's own design, compatible with ComfyUI's formats and API; the executor's caching and validation were modeled on ComfyUI's | review the executor and validator; the rest is SpawnGen's own |
| THIRD-PARTY-NOTICES.md | notices for permissive sources used (Pillow, libjpeg-turbo / IJG, PyTorch; add k-diffusion when the samplers use it) | done, extend as sources are added |
