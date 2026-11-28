# Anima run 10 (Lore, 2026-11-28): the turbo LoRA (anima-turbo-lora-v0.2) on anima-base-v1.0 for Tuvok's ML-side LoRA gate: LoraLoaderModelOnly
# at strength 1.0 and 0.5, the adapter's output under it (pos prompt) and two DiT calls each; 512 x 512, cfg 1, euler / simple, 8 steps (only the
# first two calls are dumped), seed 42; the base model WITHOUT the LoRA as the control (same seed).
import json, urllib.request, os
D = os.path.abspath('run10-turbo-lora').replace(os.sep, '/')
POS = "masterpiece, best quality, score_7, safe, anime, a girl with a red umbrella standing in the rain"
p = {
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}},
 "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": POS}},
 "c4": {"class_type": "SpawnGenDumpConditioning", "inputs": {"conditioning": ["4", 0], "name": "pos", "folder": D}},
 "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
}
for tag, strength in (("lora100", 1.0), ("lora050", 0.5), ("nolora", None)):
    model = ["1", 0]
    if strength is not None:
        p[f"l_{tag}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "anima-turbo-lora-v0.2.safetensors", "strength_model": strength}}
        model = [f"l_{tag}", 0]
    p[f"t_{tag}"] = {"class_type": "SpawnGenHookModelModulesPre", "inputs": {"model": model, "name": f"{tag}_tap", "folder": D, "pattern": "^diffusion_model\.llm_adapter$", "max_calls": 1}}
    p[f"d_{tag}"] = {"class_type": "SpawnGenDumpModelCall", "inputs": {"model": [f"t_{tag}", 0], "name": f"{tag}_call", "folder": D, "calls": 2}}
    p[f"k_{tag}"] = {"class_type": "KSampler", "inputs": {"model": [f"d_{tag}", 0], "positive": ["4", 0], "negative": ["4", 0], "latent_image": ["5", 0],
        "seed": 42, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    p[f"o_{tag}"] = {"class_type": "SpawnGenDumpLatent", "inputs": {"samples": [f"k_{tag}", 0], "name": f"{tag}_out", "folder": D}}
print(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8199/prompt", data=json.dumps({"prompt": p}).encode(), headers={"Content-Type": "application/json"})).read().decode()[:120])
