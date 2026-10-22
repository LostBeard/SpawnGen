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
| Studio UI (canvas node editor) | planned |
| Server / desktop shell | planned |

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
