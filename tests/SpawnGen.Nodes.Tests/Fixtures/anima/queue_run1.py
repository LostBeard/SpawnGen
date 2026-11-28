# Anima black-box reference run 1 (Lore, 2026-11-27): the preview template's graph at 512 x 512, 4 steps, with dumps.
import json, urllib.request, os
D = os.path.abspath('run1').replace(os.sep, '/')
POS = "masterpiece, best quality, score_7, safe, anime, a girl with a red umbrella standing in the rain"
NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts"
p = {
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-preview3-base.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "SpawnGenDumpModelCall", "inputs": {"model": ["1", 0], "name": "call", "folder": D, "calls": 2}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}},
 "4": {"class_type": "SpawnGenHookTokenIds", "inputs": {"clip": ["3", 0], "name": "ids", "folder": D}},
 "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": POS}},
 "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": NEG}},
 "7": {"class_type": "SpawnGenDumpConditioning", "inputs": {"conditioning": ["5", 0], "name": "pos", "folder": D}},
 "8": {"class_type": "SpawnGenDumpConditioning", "inputs": {"conditioning": ["6", 0], "name": "neg", "folder": D}},
 "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
 "10": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["9", 0],
        "seed": 42, "steps": 4, "cfg": 4.0, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}},
 "11": {"class_type": "SpawnGenDumpLatent", "inputs": {"samples": ["10", 0], "name": "out", "folder": D}},
 "12": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
 "13": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["12", 0]}},
 "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "anima_run1"}},
}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8199/prompt", data=json.dumps({"prompt": p}).encode(), headers={"Content-Type": "application/json"}))
print(r.read().decode())
