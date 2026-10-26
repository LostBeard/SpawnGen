"""Runs ComfyUI's own VAE Encode, VAE Encode (for Inpainting) and InpaintModelConditioning (nodes.py) with a stand-in VAE and writes what reaches
the VAE, and the masks they return, to inpaint.json.gz.

    python make_inpaint.py <ComfyUI folder>

The stand-in VAE crops like ComfyUI's VAE.encode (comfy.sd.VAE.vae_encode_crop_pixels, an 8x downscale) and returns the cropped pixels instead of
encoding them, so the fixture holds exactly the pictures the real VAE would be given.
"""
import gzip, json, sys
import torch

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import comfy.sd  # noqa: E402
import nodes  # noqa: E402


class StandInVae:
    crop_input = True
    output_channels = 3

    def spacial_compression_encode(self):
        return 8

    def encode(self, pixels):
        return comfy.sd.VAE.vae_encode_crop_pixels(self, pixels).clone()


def tensor(t):
    return {"shape": list(t.shape), "data": [round(float(v), 7) for v in t.flatten()]}


def picture(batch, h, w, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, h, w, 3, generator=g)


def soft_mask(batch, h, w, seed):
    """A mask with hard 0 / 1 areas and soft edges, so rounding and growing both matter."""
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(batch, h, w)
    m[:, h // 4: h // 4 + max(1, h // 3), w // 3: w // 3 + max(1, w // 3)] = 1.0
    m += torch.rand(batch, h, w, generator=g) * 0.6
    return m.clamp(0, 1)


cases = []
vae = StandInVae()
for (b, h, w, mb, mh, mw, grows) in [
    (1, 16, 24, 1, 16, 24, [0, 1, 6]),     # already a multiple of 8, mask at the picture's size
    (1, 21, 19, 1, 21, 19, [0, 2, 5, 6]),  # crop with an offset in both directions
    (2, 27, 34, 1, 9, 12, [3, 6]),         # mask smaller than the picture (bilinear), one mask for two pictures
    (2, 17, 40, 2, 34, 20, [1, 7]),        # one mask per picture, different aspect
]:
    pixels = picture(b, h, w, h * 100 + w)
    mask = soft_mask(mb, mh, mw, mh * 10 + mw)
    case = {"pixels": tensor(pixels), "mask": tensor(mask)}
    case["vae_encode"] = tensor(nodes.VAEEncode().encode(vae, pixels)[0]["samples"])
    case["for_inpaint"] = []
    for grow in grows:
        out = nodes.VAEEncodeForInpaint().encode(vae, pixels.clone(), mask, grow)[0]
        case["for_inpaint"].append({"grow": grow, "encoded": tensor(out["samples"]), "noise_mask": tensor(out["noise_mask"])})
    cond = [[torch.zeros(1, 1, 1), {}]]
    for noise_mask in (True, False):
        pos, neg, latent = nodes.InpaintModelConditioning().encode(cond, cond, pixels.clone(), vae, mask, noise_mask)
        entry = {"noise_mask_on": noise_mask, "encoded": tensor(latent["samples"]), "concat": tensor(pos[0][1]["concat_latent_image"]),
                 "concat_mask": tensor(pos[0][1]["concat_mask"])}
        if noise_mask:
            entry["noise_mask"] = tensor(latent["noise_mask"])
        assert "noise_mask" not in latent or noise_mask
        case.setdefault("model_conditioning", []).append(entry)
    cases.append(case)

with gzip.open("inpaint.json.gz", "wt") as f:
    json.dump(cases, f, separators=(",", ":"))
print(len(cases), "cases")
