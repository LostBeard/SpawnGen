"""Runs the same-start graph through the ComfyUI server for each sampler: python run_comfy.py prefix sampler [sampler ...]"""
import json, sys, time, urllib.request

SERVER = "http://127.0.0.1:8199"


def graph(sampler, prefix, ckpt="sd_turbo.safetensors", latent="start.latent"):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a photo of a red apple on a wooden table", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "LoadLatent", "inputs": {"latent": latent}},
        "5": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["1", 0], "add_noise": "disable", "noise_seed": 0, "steps": 4, "cfg": 1.0,
                                                          "sampler_name": sampler, "scheduler": "normal", "positive": ["2", 0], "negative": ["3", 0],
                                                          "latent_image": ["4", 0], "start_at_step": 0, "end_at_step": 10000,
                                                          "return_with_leftover_noise": "disable"}},
        "6": {"class_type": "SaveLatent", "inputs": {"samples": ["5", 0], "filename_prefix": f"latents/{prefix}_{sampler}"}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"{prefix}_{sampler}"}},
    }


if __name__ == "__main__":
    prefix = sys.argv[1]
    for s in sys.argv[2:]:
        g = graph(s, prefix)
        if prefix.endswith("_spawngen"):
            json.dump(g, open(f"{prefix}_{s}.json", "w"))
            continue
        req = urllib.request.Request(SERVER + "/prompt", data=json.dumps({"prompt": g}).encode(), headers={"Content-Type": "application/json"})
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
        while True:
            h = json.load(urllib.request.urlopen(f"{SERVER}/history/{pid}"))
            if pid in h:
                st = h[pid].get("status", {})
                print(s, st.get("status_str"), [m[0] for m in st.get("messages", [])][-2:])
                break
            time.sleep(1)
