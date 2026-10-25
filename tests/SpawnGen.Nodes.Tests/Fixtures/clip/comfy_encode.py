"""Dumps what ComfyUI's CLIP makes of prompts: the token chunks (ids and weights per tower) and the conditioning (and SDXL's pooled output).

    python comfy_encode.py <ComfyUI folder> <checkpoint.safetensors> <out prefix>

Writes <prefix>.json (tokens, shapes) and <prefix>_<n>.bin (float32 conditioning of prompt n, then the pooled output when there is one).
Run with ComfyUI's own Python environment; the prompts are PROMPTS below (a long weighted one crosses the 75-token chunk boundary).
"""
import json, sys
import numpy as np

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True  # before model_management is imported
import comfy.sd  # noqa: E402

PROMPTS = [
    "a highly detailed photograph of a (red apple:1.3) on a rustic wooden table next to a window, soft morning light, shallow depth of field, bokeh, "
    "85mm lens, award winning food photography, (ultra realistic, intricate texture, water droplets on the skin, reflections, warm color grading, "
    "cinematic composition:1.2), kitchen background with copper pots and green plants, dust particles in the light beam, film grain, sharp focus",
    "a cat",
]

_, clip, _, _ = comfy.sd.load_checkpoint_guess_config(sys.argv[2], output_vae=False, output_clipvision=False, output_model=False)
out = {"prompts": []}
for n, text in enumerate(PROMPTS):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    towers = {k: [[[int(t[0]), float(t[1])] for t in chunk] for chunk in v] for k, v in tokens.items()}
    blob = cond.float().cpu().numpy().astype('<f4').tobytes()
    if pooled is not None and clip.cond_stage_model.__class__.__name__.startswith("SDXL"):
        blob += pooled.float().cpu().numpy().astype('<f4').tobytes()
        pooled_len = int(pooled.shape[-1])
    else:
        pooled_len = 0
    open(f"{sys.argv[3]}_{n}.bin", "wb").write(blob)
    out["prompts"].append({"text": text, "tokens": towers, "shape": list(cond.shape), "pooled": pooled_len})
json.dump(out, open(f"{sys.argv[3]}.json", "w"), indent=1)
print(json.dumps([(p["shape"], p["pooled"], {k: len(v) for k, v in p["tokens"].items()}) for p in out["prompts"]]))
