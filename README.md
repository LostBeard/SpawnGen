# SpawnGen

A ComfyUI-class node-graph studio for generative ML, written in C#. ML runs on
[SpawnDev.ILGPU.ML](https://www.nuget.org/packages/SpawnDev.ILGPU.ML): **WebGPU in the browser, CUDA/OpenCL on the desktop**, no Python.
Extensibility is C# node packs instead of Python custom nodes.

**Try it in your browser (WebGPU): https://lostbeard.github.io/SpawnGen/**
The page runs the whole engine locally in your browser; models download from Hugging Face on first use and are cached by the browser.

One editor, three ways to run it, the same ComfyUI workflow files on all of them:

| Where | How |
|---|---|
| Browser, nothing to install | the GitHub Pages build above (or `src/SpawnGen.Studio` served by any static host) |
| Desktop app | `src/SpawnGen.Desktop`: a Photino window over an in-process server, native CUDA/OpenCL |
| Server | `src/SpawnGen.Server`: ComfyUI-shaped HTTP + WebSocket API (`/prompt`, `/queue`, `/history`, `/object_info`, `/view`, `/upload/image`, `/ws`); also serves the Studio |

How close to ComfyUI it is, feature by feature, is in [docs/PARITY.md](docs/PARITY.md). Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What it can do today

- **Image generation**: SD-Turbo, Stable Diffusion 1.5 (any Optimum-style ONNX export, listed in `checkpoints.json`), FLUX.2 Klein 4B and FLUX 3 Klein 5.6B (text to image and editing from reference pictures). 11 samplers, 6 schedulers,
  KSampler (Advanced), SamplerCustom and the sigma nodes, conditioning nodes (area, mask, combine ...), inpainting, latent upscale, prompt emphasis `(word:1.3)`, latent preview while sampling.
- **Vision**: upscale, background removal, depth (Depth Anything V2), style transfer, pose (MoveNet), object detection (YOLOv8n), image classification (MobileNetV2), CLIP label matching.
- **3D**: Image to 3D with TRELLIS.2 (a picture becomes a textured .glb; desktop engines, about 10 GB of weights the first time), Save / Preview 3D Model, and a built-in 3D viewer.
- **Audio and text**: Kokoro text to speech, Parakeet speech to text, local LLM text generation (small GGUF chat models) whose output can feed a prompt.
- **Over 100 nodes in all, most of them plain host code**: image ops, masks, latents, strings, math, primitives, Reroute / Note / groups, Preview Any.
- **Editor**: node library sidebar, search, undo/redo, copy/paste, groups, colours, workflow tabs with autosave, minimap, settings (link style, snap to grid), history panel with sound and picture players,
  mask editor for Load Image, ComfyUI workflow import/export (PNGs carry their workflow, drop one on the canvas to open it).
- **Examples menu** with 14 ready workflows to start from.
- **Custom nodes in C#**: [docs/CUSTOM_NODES.md](docs/CUSTOM_NODES.md), example pack in `samples/SpawnGen.SampleNodes`.

## Run it

```powershell
dotnet run --project src/SpawnGen.Server -c Release --urls http://localhost:8188
```

Open http://localhost:8188 for the Studio; the same address is the API. Models, inputs and results live under
`%LOCALAPPDATA%\SpawnGen` (`models`, `input`, `output`, `temp`, `custom_nodes`); change them with `SpawnGen__ModelsPath`, `SpawnGen__OutputPath`, `SpawnGen__InputPath`, `SpawnGen__TempPath`,
`SpawnGen__CustomNodesPath` (environment variables or `--SpawnGen:ModelsPath=...`). Models that are not on disk download from Hugging Face on first use into `models\hub`.
The Studio alone (the browser engine, no server) is `dotnet run --project src/SpawnGen.Studio -c Release`.

To add another Stable Diffusion checkpoint, put a `checkpoints.json` in the models folder:

```json
{ "my_sd15": { "repo": "owner/some-sd15-onnx-export", "textWidth": 768, "unetData": "unet/model.onnx_data" } }
```

## Build and test

```powershell
dotnet test SpawnGen.slnx -c Release
```

Use Release for any real ML work: Debug and Release produce different ILGPU IR. Tests that need a GPU and model files are marked `Explicit` (run them by name with
`--filter "Name=..."` and set `SPAWNGEN_MODELS` to the models folder); they save pictures and sounds that are meant to be looked at.
`tools/studio-browser-test` drives the real page in headless Chrome (`npm install`, then `npm run test:...`; see its README).

The browser build is compiled to native WebAssembly (AOT) for the hosted version: `dotnet publish src/SpawnGen.Studio -c Release -p:StudioAot=true` (needs `dotnet workload install wasm-tools`,
takes about an hour; the GitHub Pages workflow does it). An ordinary build interprets the C#, which is fine for development and far slower for big graphs.

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

## License

MIT. Model weights keep their own licences (FLUX.2 Klein is Apache-2.0; check each model's page).
