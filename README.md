# SpawnGen

A ComfyUI-class node-graph studio for generative ML, written in C#. ML runs on
[SpawnDev.ILGPU.ML](https://www.nuget.org/packages/SpawnDev.ILGPU.ML) (WebGPU in the browser, CUDA/OpenCL on
desktop) instead of Python; extensibility is C# nodes and scripting instead of Python.

Design and roadmap: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

| Layer | State |
|-------|-------|
| `SpawnGen.Engine`: node schema/registry, ComfyUI API-format prompts, validator, cached executor | done, 50 tests |
| `SpawnGen.Nodes.Core` / `SpawnGen.Nodes.ML`: empty latent, save image, checkpoint loader, CLIP encode, KSampler (euler, euler_ancestral, dpmpp_2m, CFG, denoise), VAE encode/decode, load/save/preview image (PNG); SD-Turbo txt2img and img2img verified on desktop OpenCL | working; schedulers, LoRA/ControlNet next |
| `SpawnGen.Editor`: UI-independent node editor core (document, undo/redo, layout, hit testing, pointer/keyboard interaction, ComfyUI workflow import/export, graph compiler) | working, 100 tests; Blazor canvas next |
| `SpawnGen.Studio`: Blazor WebAssembly canvas editor drawn from C# (no per-move re-render): pan/zoom, drag, link, node search, in-place widget edit, context menu, open/save ComfyUI workflows; browser-tested (`tools/studio-browser-test`) | working; queues to the server (Ctrl+Enter), shows the running node, progress and result pictures inside the nodes; gallery and history next |
| `SpawnGen.Server`: ComfyUI-shaped HTTP + WebSocket API (`/prompt`, `/queue`, `/history`, `/object_info`, `/view`, `/upload/image`, `/ws`), one-at-a-time queue with interrupt, native CUDA/OpenCL; also serves the Studio. SD-Turbo txt2img verified end to end through it on an RX 590 | working; desktop shell next |

## Run it

```powershell
dotnet run --project src/SpawnGen.Server -c Release --urls http://localhost:8188
```

Open http://localhost:8188 for the Studio; the same address is the API. Models, inputs and results live under
`%LOCALAPPDATA%\SpawnGen` (`models\sd_turbo\...`, `input`, `output`, `temp`); change them with `SpawnGen__ModelsPath`, `SpawnGen__OutputPath`,
`SpawnGen__InputPath`, `SpawnGen__TempPath` (environment variables or `--SpawnGen:ModelsPath=...`).
The Studio alone (no server, editing only) is `dotnet run --project src/SpawnGen.Studio -c Release`.

## Build and test

```powershell
dotnet test SpawnGen.slnx -c Release
```

Use Release for any real ML work: Debug and Release produce different ILGPU IR.

The end-to-end txt2img test needs the SD-Turbo files on disk (`SPAWNGEN_MODELS` = a folder with `tokenizer/`, `text_encoder/`,
`unet/`, `vae_decoder/`, the `schmuell/sd-turbo-ort-web` layout) and a GPU. Note: SpawnDev.ILGPU 5.4.1 corrupts streamed weight
uploads on some AMD OpenCL drivers (Polaris), which turns the output into noise; use 5.4.2 or later there.

## Writing a node

```csharp
[Node(Category = "latent")]
public sealed class EmptyLatentImage
{
    public LatentBatch Execute(
        [Input(Min = 16, Max = 8192, Step = 8)] int width = 512,
        [Input(Min = 16, Max = 8192, Step = 8)] int height = 512,
        int batchSize = 1) => new(batchSize, 4, height / 8, width / 8);
}
```

The schema (widgets, ranges, socket types, output names) is derived from the signature. Register with
`new NodeRegistry().Add<EmptyLatentImage>()` or `.AddAssembly(assembly)`, then run a `PromptGraph` (or
`PromptJson.Parse(comfyApiJson)`) through a `PromptExecutor`.
