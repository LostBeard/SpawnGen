# Anima run 9 (Lore, 2026-11-28): which noise stream each noise-adding sampler draws from (run 1's er_sde drew seed + 1's first draw). Seed 7,
# sigmas 1, 0.5, 0, cfg 1, 128 x 128; the first two model calls (input, output) dumped per sampler.
import json, urllib.request, os
D = os.path.abspath('run9-noise-seed').replace(os.sep, '/')
S = ["euler_ancestral", "euler_ancestral_cfg_pp", "dpm_2_ancestral", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
     "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_3m_sde", "er_sde", "sa_solver", "sa_solver_pece", "seeds_2", "seeds_3",
     "exp_heun_2_x0_sde", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp", "ddpm", "lcm"]
p = {
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-preview3-base.safetensors", "weight_dtype": "default"}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}},
 "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "a cat"}},
 "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 128, "height": 128, "batch_size": 1}},
 "6": {"class_type": "ManualSigmas", "inputs": {"sigmas": "1.0, 0.5, 0"}},
 "pc": {"class_type": "SpawnGenDumpConditioning", "inputs": {"conditioning": ["4", 0], "name": "cond", "folder": D}},
}
for i, s in enumerate(S):
    p[f"d{i}"] = {"class_type": "SpawnGenDumpModelCall", "inputs": {"model": ["1", 0], "name": f"s_{s}", "folder": D, "calls": 2}}
    p[f"k{i}"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": s}}
    p[f"c{i}"] = {"class_type": "SamplerCustom", "inputs": {"model": [f"d{i}", 0], "add_noise": True, "noise_seed": 7, "cfg": 1.0, "positive": ["4", 0], "negative": ["4", 0],
        "sampler": [f"k{i}", 0], "sigmas": ["6", 0], "latent_image": ["5", 0]}}
    p[f"o{i}"] = {"class_type": "SpawnGenDumpLatent", "inputs": {"samples": [f"c{i}", 0], "name": f"out_{s}", "folder": D}}
print(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8199/prompt", data=json.dumps({"prompt": p}).encode(), headers={"Content-Type": "application/json"})).read().decode()[:120])
