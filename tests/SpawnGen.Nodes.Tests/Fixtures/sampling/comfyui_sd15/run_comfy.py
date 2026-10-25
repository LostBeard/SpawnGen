"""Runs a same-start graph through the ComfyUI server and saves its result latent and picture.

    python run_comfy.py <ckpt> <steps> <cfg> <sampler> <prefix> [start latent name, default start.latent]

The start latent (in ComfyUI's input folder) is the noised start; add_noise is disabled so the only randomness is the start itself. The graph matches
ComfyUIRenderParityTests.RunAsync: prompt "a photo of a red apple on a wooden table", empty negative, KSampler (Advanced), scheduler normal.
"""
import json, sys, time, urllib.request

SERVER = "http://127.0.0.1:8199"


def graph(ckpt, steps, cfg, sampler, prefix, latent="start.latent"):
    return {
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


if __name__ == "__main__":
    ckpt, steps, cfg, sampler, prefix = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
    latent = sys.argv[6] if len(sys.argv) > 6 else "start.latent"
    req = urllib.request.Request(SERVER + "/prompt", data=json.dumps({"prompt": graph(ckpt, steps, cfg, sampler, prefix, latent)}).encode(),
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
