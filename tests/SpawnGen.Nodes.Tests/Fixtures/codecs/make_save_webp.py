"""Runs ComfyUI's own WebP saving (comfy_api ImageSaveHelper: _create_webp_metadata and save_animated_webp, i.e. Pillow's save_all) on a small batch and
writes the frames, the prompt / extra_pnginfo strings and Pillow's files to save_webp/ for SpawnGen's Save Animated WEBP tests.

    python make_save_webp.py <ComfyUI folder>

Cases: a 3-frame lossy animation (quality 80, method 4: libwebp's WebPAnimEncoder, which SpawnDev.Codecs ports), a 3-frame lossless one (the default) and
1-frame lossy and lossless saves (Pillow's still path), plus the EXIF bytes alone for several metadata sets.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import folder_paths  # noqa: E402
from comfy_api.latest._ui import ImageSaveHelper, FolderType  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_webp")
os.makedirs(HERE, exist_ok=True)
W, H = 48, 32


class Hidden:
    def __init__(self, prompt, extra):
        self.prompt, self.extra_pnginfo = prompt, extra


class Cls:
    hidden = None


def frames(n):
    y, x = np.mgrid[0:H, 0:W]
    out = []
    for k in range(n):
        rgb = np.stack([(x * 5 + k * 30) % 256, (y * 7 + k * 11) % 256, ((x * y) // 3 + k * 50) % 256], -1).astype(np.float32) / 255.0
        rgb[6:14, 4 + 10 * k:16 + 10 * k] = [0.95, 0.12, 0.10]
        out.append(rgb)
    return torch.from_numpy(np.stack(out))


PROMPT = {"3": {"class_type": "KSampler", "inputs": {"seed": 7, "text": "café \"quoted\""}}}
EXTRA = {"workflow": {"nodes": [1, 2, 3], "version": 0.4}, "other": "x"}

if __name__ == "__main__":
    folder_paths.set_output_directory(HERE)
    Cls.hidden = Hidden(PROMPT, EXTRA)
    cases = []
    for name, n, lossless in [("anim_lossy", 3, False), ("anim_lossless", 3, True), ("one_lossy", 1, False), ("one_lossless", 1, True)]:
        imgs = frames(n)
        r = ImageSaveHelper.save_animated_webp(imgs, name, FolderType.output, Cls, fps=6.0, lossless=lossless, quality=80, method=4)
        os.replace(os.path.join(HERE, r.filename), os.path.join(HERE, name + ".webp"))
        cases.append({"name": name, "frames": n, "lossless": lossless})
    frames(3).numpy().astype("<f4").tofile(os.path.join(HERE, "frames.f32"))  # 3 x H x W x 3, little-endian float32
    exif = []
    for prompt, extra in [(PROMPT, EXTRA), (PROMPT, None), ({"1": {}}, {"a": 1}), ({"1": {"inputs": {"t": "odd"}}}, {"k1": "v", "k2": [1], "k3": {}})]:
        from PIL import Image
        Cls.hidden = Hidden(prompt, extra)
        b = ImageSaveHelper._create_webp_metadata(Image.new("RGB", (2, 2)), Cls).tobytes()
        meta = [["prompt", json.dumps(prompt)]] + [[k, json.dumps(v)] for k, v in (extra or {}).items()]
        exif.append({"metadata": meta, "hex": b.hex()})
    Cls.hidden = Hidden(PROMPT, EXTRA)
    with open(os.path.join(HERE, "cases.json"), "w") as fh:
        json.dump({"width": W, "height": H, "fps": 6.0, "prompt": json.dumps(PROMPT),
                   "extra": [[k, json.dumps(v)] for k, v in EXTRA.items()], "cases": cases, "exif": exif}, fh, indent=1)
    for f in sorted(os.listdir(HERE)):
        print(f, os.path.getsize(os.path.join(HERE, f)))
