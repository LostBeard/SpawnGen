# Anima run 5 (Lore, 2026-11-28): the CONTROL for run 4: the same graph and seed WITHOUT AnimaLLLiteApply (the hooks take the bare base model).
# compositor-a.png (512 x 512) with a Painter rectangle mask (x 128..383, y 160..351), cfg 1, 2 steps, euler / simple, seed 7; dumps of
# the model calls, block 0 / 27 taps and any module named *lllite*.
import json, urllib.request, os
D = os.path.abspath('run5-control').replace(os.sep, '/')
p = {
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "ModelPatchLoader", "inputs": {"name": "anima-lllite-inpainting-v2.safetensors"}},
 "3": {"class_type": "LoadImage", "inputs": {"image": "compositor-a.png"}},
 "4": {"class_type": "Painter", "inputs": {"mask": "painter-lllite-rect.png [input]", "width": 512, "height": 512, "bg_color": "#000000", "image": ["3", 0]}},
 "5": {"class_type": "AnimaLLLiteApply", "inputs": {"model": ["1", 0], "model_patch": ["2", 0], "image": ["3", 0], "mask": ["4", 1], "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}},
 "6": {"class_type": "SpawnGenHookModelModulesPre", "inputs": {"model": ["1", 0], "name": "tap", "folder": D, "pattern": "(?i)lllite|^diffusion_model\.blocks\.(0|27)$", "max_calls": 1}},
 "7": {"class_type": "SpawnGenDumpModelCall", "inputs": {"model": ["6", 0], "name": "call", "folder": D, "calls": 2}},
 "8": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}},
 "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["8", 0], "text": "girl with red eyes"}},
 "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["8", 0], "text": "worst quality, low quality, score_1, score_2, score_3, blurry"}},
 "11": {"class_type": "SpawnGenDumpConditioning", "inputs": {"conditioning": ["9", 0], "name": "pos", "folder": D}},
 "12": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
 "13": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["12", 0],
        "seed": 7, "steps": 2, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "14": {"class_type": "SpawnGenDumpLatent", "inputs": {"samples": ["13", 0], "name": "out", "folder": D}},
 "15": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
 "16": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["15", 0]}},
 "17": {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": "anima_run5"}},
 "18": {"class_type": "MaskToImage", "inputs": {"mask": ["4", 1]}},
 "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": "anima_run5_mask"}},
}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8199/prompt", data=json.dumps({"prompt": p}).encode(), headers={"Content-Type": "application/json"}))
print(r.read().decode())
