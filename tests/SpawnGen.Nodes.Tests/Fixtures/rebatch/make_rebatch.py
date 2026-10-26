"""Runs ComfyUI's own Rebatch Latents / Rebatch Images (comfy_extras/nodes_rebatch.py) on a fixed list and writes what came out to rebatch.json.

    python make_rebatch.py <ComfyUI folder>

Every latent / picture value encodes where it came from (list item * 1000 + batch item * 100 + position), so the output says which item went where.
"""
import json, sys
import torch

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
from comfy_extras.nodes_rebatch import LatentRebatch, ImageRebatch  # noqa: E402


def latent(item, batch, h, w, mask=None, batch_index=None):
    s = torch.zeros(batch, 4, h, w)
    for b in range(batch):
        s[b] = item * 1000 + b * 100 + torch.arange(4 * h * w, dtype=torch.float32).reshape(4, h, w) % 100
    d = {"samples": s}
    if mask is not None:
        d["noise_mask"] = mask
    if batch_index is not None:
        d["batch_index"] = batch_index
    return d


half = torch.ones(1, 1, 64, 64)
half[..., :32] = 0.0
latents = [
    latent(0, 3, 8, 8),                          # no mask
    latent(1, 2, 8, 8, mask=half),               # a mask for both (batch 1, repeated)
    latent(2, 1, 16, 16),                        # a new size closes the batch being filled
    latent(3, 2, 16, 16, batch_index=[7, 9]),    # its own batch indexes
]
out = LatentRebatch.execute(latents, [2]).result[0]
lat = []
for o in out:
    s = o["samples"]
    lat.append({"shape": list(s.shape), "firsts": [float(s[b, 0, 0, 0]) for b in range(s.shape[0])], "batch_index": o["batch_index"],
                "mask": None if "noise_mask" not in o else {"shape": list(o["noise_mask"].shape), "sum": float(o["noise_mask"].sum())}})

images = [torch.full((3, 4, 4, 3), 0.0), torch.full((2, 4, 4, 3), 0.0)]
for i, img in enumerate(images):
    for b in range(img.shape[0]):
        img[b] = (i * 10 + b) / 100.0
outi = ImageRebatch.execute(images, [2]).result[0]
imgs = [{"shape": list(o.shape), "firsts": [round(float(o[b, 0, 0, 0]) * 100) for b in range(o.shape[0])]} for o in outi]

json.dump({"latents": lat, "images": imgs}, open("rebatch.json", "w"), indent=1)
print(json.dumps({"latents": lat, "images": imgs}))
