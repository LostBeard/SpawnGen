"""Runs a same-start graph through the ComfyUI server and saves its result latent and picture.

    python run_comfy.py <ckpt> <steps> <cfg> <sampler> <prefix> [start latent name, default start.latent] [lora file] [lora strength]
                        [--patch NodeType '{"input": value, ...}'] ... [--inpaint <unet file> <picture> <mask picture>]
                        [--merge <unet file> <clip file> <lora file> <ratio>]

The start latent (in ComfyUI's input folder) is the noised start; add_noise is disabled so the only randomness is the start itself. The graph matches
ComfyUIRenderParityTests.RunAsync: prompt "a photo of a red apple on a wooden table", empty negative, KSampler (Advanced), scheduler normal.
Each --patch puts a model patch node (FreeU, PerturbedAttentionGuidance, ...) between the model and the sampler, in order. --inpaint samples with an
inpainting UNet (Load Diffusion Model; the checkpoint still gives the CLIP and VAE) and InpaintModelConditioning of the picture and the mask
(LoadImageMask, red channel; noise_mask off, so the whole latent is sampled and the model alone keeps the unmasked area).
--merge loads the UNet and CLIP files alone (a second copy of the checkpoint's parts), puts the LoRA on them, and merges them into the checkpoint's model
and CLIP with ModelMergeSimple and CLIPMergeSimple at the ratio (the checkpoint's share).
"""
import json, sys, time, urllib.request

SERVER = "http://127.0.0.1:8199"


def graph(ckpt, steps, cfg, sampler, prefix, latent="start.latent", lora=None, strength=1.0, patches=(), inpaint=None, merge=None):
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a photo of a red apple on a wooden table", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "LoadLatent", "inputs": {"latent": latent}},
        "5": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["1", 0], "add_noise": "disable", "noise_seed": 0, "steps": steps, "cfg": cfg,
                                                          "sampler_name": sampler, "scheduler": "normal", "positive": ["2", 0], "negative": ["3", 0],
                                                          "latent_image": ["4", 0], "start_at_step": 0, "end_at_step": 10000,
                                                          "return_with_leftover_noise": "disable"}},
        "6": {"class_type": "SaveLatent", "inputs": {"samples": ["5", 0], "filename_prefix": f"latents/{prefix}"}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }
    if lora:
        # Load LoRA (Model and CLIP) between the checkpoint and everything that used it
        g["9"] = {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": lora, "strength_model": strength, "strength_clip": strength}}
        g["2"]["inputs"]["clip"] = ["9", 1]
        g["3"]["inputs"]["clip"] = ["9", 1]
        g["5"]["inputs"]["model"] = ["9", 0]
    if inpaint:
        unet, picture, mask = inpaint
        g["10"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
        g["11"] = {"class_type": "LoadImage", "inputs": {"image": picture}}
        g["12"] = {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}}
        g["13"] = {"class_type": "InpaintModelConditioning", "inputs": {"positive": ["2", 0], "negative": ["3", 0], "pixels": ["11", 0], "vae": ["1", 2],
                                                                        "mask": ["12", 0], "noise_mask": False}}
        g["5"]["inputs"].update({"model": ["10", 0], "positive": ["13", 0], "negative": ["13", 1]})
    if merge:
        unet, clip, merge_lora, ratio = merge
        g["14"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
        g["15"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "stable_diffusion"}}
        g["16"] = {"class_type": "LoraLoader", "inputs": {"model": ["14", 0], "clip": ["15", 0], "lora_name": merge_lora, "strength_model": 1.0, "strength_clip": 1.0}}
        g["17"] = {"class_type": "ModelMergeSimple", "inputs": {"model1": g["5"]["inputs"]["model"], "model2": ["16", 0], "ratio": float(ratio)}}
        g["18"] = {"class_type": "CLIPMergeSimple", "inputs": {"clip1": g["2"]["inputs"]["clip"], "clip2": ["16", 1], "ratio": float(ratio)}}
        g["5"]["inputs"]["model"] = ["17", 0]
        g["2"]["inputs"]["clip"] = ["18", 0]
        g["3"]["inputs"]["clip"] = ["18", 0]
    for i, (node, inputs) in enumerate(patches):
        key = str(30 + i)
        g[key] = {"class_type": node, "inputs": {"model": g["5"]["inputs"]["model"], **inputs}}
        g["5"]["inputs"]["model"] = [key, 0]
    return g


if __name__ == "__main__":
    argv, patches, inpaint, merge = [], [], None, None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--patch":
            patches.append((sys.argv[i + 1], json.loads(sys.argv[i + 2])))
            i += 3
        elif sys.argv[i] == "--merge":
            merge = sys.argv[i + 1:i + 5]
            i += 5
        elif sys.argv[i] == "--inpaint":
            inpaint = sys.argv[i + 1:i + 4]
            i += 4
        else:
            argv.append(sys.argv[i])
            i += 1
    ckpt, steps, cfg, sampler, prefix = argv[0], int(argv[1]), float(argv[2]), argv[3], argv[4]
    latent = argv[5] if len(argv) > 5 else "start.latent"
    lora = argv[6] if len(argv) > 6 and argv[6] else None
    strength = float(argv[7]) if len(argv) > 7 else 1.0
    req = urllib.request.Request(SERVER + "/prompt", data=json.dumps({"prompt": graph(ckpt, steps, cfg, sampler, prefix, latent, lora, strength, patches, inpaint, merge)}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    t0 = time.time()
    while True:
        h = json.load(urllib.request.urlopen(f"{SERVER}/history/{pid}"))
        if pid in h:
            st = h[pid].get("status", {})
            print(prefix, st.get("status_str"), [m[0] for m in st.get("messages", [])][-2:], f"{time.time() - t0:.0f} s")
            break
        time.sleep(2)
