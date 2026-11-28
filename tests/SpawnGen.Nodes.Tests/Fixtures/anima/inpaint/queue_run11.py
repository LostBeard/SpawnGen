# Anima run 11 (Lore, 2026-11-28): the image_anima_lllite_image_inpainting template graph as the template wires it (Load Image -> Painter ->
# Preview Mask -> the subgraph; Compare Images and Save Image at the root), with its turbo switch OFF (the base branch: the PrimitiveInt that
# feeds steps, 30 cut to 4, cfg 4) so the lazy If/Else switches never run Load LoRA, at 512 x 512 (the template's 1024) on compositor-a.png
# with painter-lllite-rect.png as the painting; KSampler seed 42 (the template's randomize), euler / simple. Dumps: the conditioning, the first
# two model calls (a pass-through wrapper after the switch), the final latent; the picture is Save Image's.
import json, urllib.request, os
D = os.path.abspath('run11-inpaint-template').replace(os.sep, '/')
p = {
 # root
 "91": {"class_type": "LoadImage", "inputs": {"image": "compositor-a.png"}},
 "694": {"class_type": "Painter", "inputs": {"mask": "painter-lllite-rect.png [input]", "width": 512, "height": 512, "bg_color": "#000000", "image": ["91", 0]}},
 "695": {"class_type": "MaskPreview", "inputs": {"mask": ["694", 1]}},
 # the subgraph, flattened as the frontend sends it
 "78": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}},
 "79": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
 "81": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
 "82": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["78", 0], "text": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"}},
 "84": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["78", 0], "text": "girl with red eyes"}},
 "85": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
 "86": {"class_type": "ModelPatchLoader", "inputs": {"name": "anima-lllite-inpainting-v2.safetensors"}},
 "87": {"class_type": "AnimaLLLiteApply", "inputs": {"model": ["85", 0], "model_patch": ["86", 0], "image": ["694", 0], "mask": ["695", 0], "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}},
 "109": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["87", 0], "lora_name": "anima-turbo-lora-v0.2.safetensors", "strength_model": 1.0}},
 "115": {"class_type": "PrimitiveBoolean", "inputs": {"value": False}},
 "107": {"class_type": "PrimitiveInt", "inputs": {"value": 4}},
 "108": {"class_type": "PrimitiveInt", "inputs": {"value": 8}},
 "112": {"class_type": "PrimitiveFloat", "inputs": {"value": 4.0}},
 "114": {"class_type": "PrimitiveFloat", "inputs": {"value": 1.0}},
 "110": {"class_type": "ComfySwitchNode", "inputs": {"switch": ["115", 0], "on_false": ["87", 0], "on_true": ["109", 0]}},
 "111": {"class_type": "ComfySwitchNode", "inputs": {"switch": ["115", 0], "on_false": ["107", 0], "on_true": ["108", 0]}},
 "113": {"class_type": "ComfySwitchNode", "inputs": {"switch": ["115", 0], "on_false": ["112", 0], "on_true": ["114", 0]}},
 "900": {"class_type": "SpawnGenDumpModelCall", "inputs": {"model": ["110", 0], "name": "call", "folder": D, "calls": 2}},
 "901": {"class_type": "SpawnGenDumpConditioning", "inputs": {"conditioning": ["84", 0], "name": "pos", "folder": D}},
 "106": {"class_type": "KSampler", "inputs": {"model": ["900", 0], "positive": ["84", 0], "negative": ["82", 0], "latent_image": ["81", 0],
         "seed": 42, "steps": ["111", 0], "cfg": ["113", 0], "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "902": {"class_type": "SpawnGenDumpLatent", "inputs": {"samples": ["106", 0], "name": "out", "folder": D}},
 "105": {"class_type": "VAEDecode", "inputs": {"samples": ["106", 0], "vae": ["79", 0]}},
 # root again
 "46": {"class_type": "SaveImage", "inputs": {"images": ["105", 0], "filename_prefix": "anima_run11"}},
 "77": {"class_type": "ImageCompare", "inputs": {"compare_view": None, "image_a": ["91", 0], "image_b": ["46", 0]}},
}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8199/prompt", data=json.dumps({"prompt": p}).encode(), headers={"Content-Type": "application/json"}))
print(r.read().decode())
