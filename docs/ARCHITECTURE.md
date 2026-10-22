# SpawnGen architecture

SpawnGen is a ComfyUI-class node-graph generative-ML studio written in C#. ML runs through
[SpawnDev.ILGPU.ML](https://www.nuget.org/packages/SpawnDev.ILGPU.ML) (WebGPU in the browser,
CUDA/OpenCL on desktop) instead of Python/PyTorch. Extensibility is C# (nodes and scripting) instead of Python.

The target is feature parity with ComfyUI where it is meaningful, at ComfyUI polish. This file is the
source of truth for the design; keep it current.

## Layers

```
SpawnGen.Engine          pure .NET, no ML/UI deps: graph, node schema/registry, validator, cached executor
SpawnGen.Nodes.Core      primitives, math, string, image/latent utility nodes (host-side math only)
SpawnGen.Nodes.ML        nodes backed by SpawnDev.ILGPU.ML (checkpoint loader, CLIP encode, KSampler, VAE, ...)
SpawnGen.Scripting       Roslyn C# scripting / user node assemblies (desktop/server hosts)
SpawnGen.Server          Kestrel host: ComfyUI-shaped HTTP + WebSocket API, native CUDA/OpenCL engine
SpawnGen.Studio          Blazor WASM UI: canvas node editor, queue, history, gallery, settings
SpawnGen.Studio.Desktop  Photino shell around Server + Studio
```

The Studio talks to an `IEngineClient`. Two implementations: **in-process** (browser: the engine runs inside
the WASM app on WebGPU) and **remote** (HTTP/WS to SpawnGen.Server). Same UI either way.

## Graph model

The canonical executable graph is ComfyUI's **API-format prompt**:

```json
{ "3": { "class_type": "KSampler", "inputs": { "seed": 5, "steps": 20, "model": ["4", 0] } } }
```

- Workflows authored for ComfyUI load directly (models are resolved by name; see Model management).
- The editor's *UI graph* (positions, groups, colors, widget state, links) is a separate document that
  compiles to a prompt. A ComfyUI UI-format importer is a converter, not a second execution path.

## Nodes

A node is a class with an `Execute` method. The schema is derived by reflection:

```csharp
[Node(Category = "latent")]
public sealed class EmptyLatentImage
{
    public LatentBatch Execute(
        [Input(Min = 16, Max = 8192, Step = 8)] int width = 512,
        [Input(Min = 16, Max = 8192, Step = 8)] int height = 512,
        [Input(Min = 1, Max = 64)] int batchSize = 1) => ...;
}
```

- Parameters are inputs. `int/float/string/bool/enum` become widgets that can also be driven by a link;
  every other type is a socket. Parameter names map to snake_case (`batchSize` -> `batch_size`) so
  ComfyUI workflow JSON round-trips.
- Return value (or flat value-tuple, element names become output names) are the outputs. A `UiOutput`
  element carries preview data for the editor and is not a socket.
- `NodeContext`, `CancellationToken` and `IServiceProvider` parameters are injected.
- Socket type comes from the C# type; custom types are declared with `[Socket("MODEL")]`.
- The same attributes work in user assemblies and in C# scripts, so there is one extension mechanism.

## Execution

1. **Validate**: only the subgraph feeding the output nodes. Unknown types, missing required inputs, bad
   links, type mismatches, out-of-range/unknown-combo values, cycles. Errors are per node/input, for the
   editor to highlight.
2. **Plan**: dependency order of the needed subgraph.
3. **Cache**: each node has a signature (class + coerced literals + upstream signatures). Same signature =
   cached outputs reused, so changing a seed only re-runs what is downstream. `AlwaysExecute` opts out.
4. **Run**: sequential, cancellable, with progress events (executing / progress / executed / cached / error)
   shaped like ComfyUI's websocket messages.

Not yet: list-valued execution (`INPUT_IS_LIST`/`OUTPUT_IS_LIST`), lazy inputs, dynamic subgraph expansion
(loops). Planned once the core node set exists.

## ML node decomposition

`ImageGenerator` in ILGPU.ML is a one-call pipeline. ComfyUI parity needs the stages as separate nodes, so
`SpawnGen.Nodes.ML` builds them from public primitives (`InferenceSession`, `BPETokenizer`,
`DiffusionScheduler`, `ElementWiseKernels`, `Tensor`):

`CheckpointLoader` -> (`MODEL`, `CLIP`, `VAE`), `CLIPTextEncode` -> `CONDITIONING`, `EmptyLatentImage`,
`KSampler` (euler / ddim / ...; CFG), `VAEDecode`, `VAEEncode`, `LoadImage`, `SaveImage`, `PreviewImage`.
Monolithic pipelines the library only exposes whole (FLUX.2/3, TRELLIS.2, ...) are wrapped as single nodes first
and decomposed if/when the library exposes the stages.

Values crossing node boundaries are host-side (`ImageBatch` NHWC, `LatentBatch` NCHW; latents are small).
Large weights are `MODEL/CLIP/VAE` handles whose GPU residency is owned by a model manager in the ML
layer. The engine never disposes node outputs.

## Model management

Models are addressed by name (as in ComfyUI). A model store resolves names to a source: the SpawnDev hub /
R2 host, a local folder (desktop), or OPFS (browser). Browser and desktop differ only in the store.

## Roadmap

1. Engine core + tests (this commit series).
2. Core + ML nodes: txt2img end-to-end with decomposed SD-Turbo on desktop (CUDA/OpenCL) and WebGPU.
3. Studio: canvas node editor (pan/zoom, multi-select, groups, reroutes, undo/redo, search, context menus),
   queue/history, image viewer, workflow save/load incl. ComfyUI import, dark theme.
4. Server + remote client; Photino desktop shell.
5. More nodes: img2img, inpaint, upscale, background removal, depth, LoRA/ControlNet where the library
   allows, FLUX, video, audio (TTS/STT), LLM prompt nodes, 3D.
6. C# scripting node + user node assemblies + node packs.

## Conventions

- net10.0, nullable on. Release for any real ML work (Debug and Release produce different ILGPU IR).
- Everything comes from NuGet; no ProjectReferences into the main dev machine's repos.
- Integration issues found in the libraries are tracked in `docs/library-bug-reports.md`.
