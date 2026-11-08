"""Runs ComfyUI's folder_paths.get_save_image_path on folders holding the given names (files, and folders for names ending in "/") and records the counter
it picks, for SpawnGen's SavePrefix.NextCounter tests (save_counter.json). Made on Windows: os.path.normcase there ignores case.

    python make_save_counter.py <ComfyUI folder>
"""
import json, os, sys, tempfile

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import folder_paths  # noqa: E402

CASES = [
    ("ComfyUI", []),
    ("ComfyUI", ["ComfyUI_00001_.png", "ComfyUI_00002.flac"]),
    ("ComfyUI", ["ComfyUI_00007_.png", "comfyui_00009_.png"]),
    ("ComfyUI", ["ComfyUI_abc_.png"]),
    ("ComfyUI", ["ComfyUI_00003/"]),
    ("ComfyUI", ["ComfyUIx_00005_.png", "Comfy_00008_.png"]),
    ("ComfyUI", ["ComfyUI_00004_00009.png"]),
    ("ComfyUI", ["ComfyUI_.png"]),
    ("ComfyUI", ["ComfyUI_ 12 .png"]),
    ("ComfyUI", ["ComfyUI_-3_.png"]),
    ("ComfyUI", ["ComfyUI_00002_.png", "ComfyUI_00010.latent", "ComfyUI_00006_.webm", "other.txt"]),
    ("café", ["café_00002_.png"]),
    ("clip", ["clip_00001.wav", "clip_00001_.wav", "clip_00002.flac"]),
]

if __name__ == "__main__":
    out = []
    for prefix, names in CASES:
        with tempfile.TemporaryDirectory() as root:
            for n in names:
                p = os.path.join(root, n.rstrip("/"))
                if n.endswith("/"):
                    os.makedirs(p)
                else:
                    open(p, "wb").close()
            counter = folder_paths.get_save_image_path(prefix, root)[2]
        out.append({"prefix": prefix, "names": [n.rstrip("/") for n in names], "counter": counter})
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump({"platform": sys.platform, "cases": out}, open(os.path.join(here, "save_counter.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for c in out:
        print(c["counter"], c["prefix"], c["names"])
