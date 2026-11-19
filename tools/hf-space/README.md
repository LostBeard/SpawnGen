---
title: SpawnGen
emoji: 🧩
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: ComfyUI-style node graphs, run on your GPU in the browser
tags:
  - webgpu
  - comfyui
  - node-graph
  - text-to-image
  - text-to-video
  - image-to-video
  - image-to-3d
  - blazor
models:
  - stabilityai/sd-turbo
  - stabilityai/sdxl-turbo
  - LostBeard/FLUX.2-klein-4B-q4-ILGPU
  - LostBeard/flux-3-image-klein-q4-ILGPU
  - Lightricks/LTX-2
  - Comfy-Org/Z-Image-Turbo
  - Comfy-Org/ltx-2
  - onnx-community/depth-anything-v2-small
  - onnx-community/Kokoro-82M-v1.0-ONNX
  - istupakov/parakeet-tdt-0.6b-v3-onnx
  - briaai/RMBG-1.4
  - Xenova/movenet-singlepose-lightning
  - salim4n/yolov8n-detect-onnx
  - Qwen/Qwen3-4B
---

# SpawnGen

A ComfyUI-class node-graph studio written in C#, running **entirely in your browser on your GPU** (WebGPU): no server does the inference.
ComfyUI workflows and templates open and run as they do in ComfyUI; models download on first use (from Hugging Face) and stay cached in
the browser.

- **Examples**: text to image (SD-Turbo, SD 1.5, FLUX.2 / FLUX 3 Klein), image to image, inpainting, text to video and picture to video
  (LTX-Video 2B), image to 3D (TRELLIS.2), depth, pose and object detection, background removal, speech.
- **Templates**: ComfyUI's template catalog, each marked with what this browser can run.
- Needs a browser with WebGPU (Chrome or Edge). Large models need a GPU with enough memory; the first run downloads the weights.

Source: [github.com/LostBeard/SpawnGen](https://github.com/LostBeard/SpawnGen) (MIT). Built on SpawnDev.ILGPU and SpawnDev.ILGPU.ML.
