"""Writes tests/SpawnGen.Nodes.Tests/Fixtures/comfy/comfy_resize_nodes.json.gz: black-box measurements of ComfyUI's Resize Image/Mask, Upscale Latent
(bislerp), Resize And Pad Image and Image Stitch, made by calling the node classes of a ComfyUI checkout on chosen tensors.

    python -I -W ignore make_resize_fixtures.py <ComfyUI checkout> <output .json.gz>

Only the nodes' public execute entry points are called. Exceptions are recorded as their type and message (never a traceback).
"""
import contextlib, gzip, importlib, io, json, math, os, random, sys

src, out_path = sys.argv[1], os.path.abspath(sys.argv[2])
sys.path.insert(0, src)
os.chdir(src)
sys.argv = [sys.argv[0], "--cpu"]
quiet = io.StringIO()
with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
    import numpy as np
    import torch
    import comfy.options
    comfy.options.enable_args_parsing()
    post = importlib.import_module("comfy_extras.nodes_post_processing")
    images = importlib.import_module("comfy_extras.nodes_images")
    core = importlib.import_module("nodes")


def call(fn):
    try:
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            res = fn()
        val = getattr(res, "result", None) or getattr(res, "args", None) or res
        return val[0] if isinstance(val, (tuple, list)) else val
    except BaseException as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def tensor_json(t, with_data=True):
    if isinstance(t, dict):
        t = t["samples"]
    if isinstance(t, str):
        return {"error": t}
    d = {"shape": list(t.shape)}
    if with_data:
        d["data"] = [float(v) for v in t.flatten().tolist()]
    return d


def grad(h, w, c=3):
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([x / max(w - 1, 1), y / max(h - 1, 1), (x * 7 + y * 3) % 11 / 10.0, ((x + 2 * y) % 5) / 4.0], -1)[..., :c]
    return torch.from_numpy(a.astype(np.float32))


cases = []
rng = random.Random(7)

# Resize Image/Mask: output sizes over random inputs and every resize type.
for _ in range(40):
    w, h = rng.randint(3, 60), rng.randint(3, 60)
    for rt in [
        {"resize_type": "scale dimensions", "width": rng.randint(1, 50), "height": 0, "crop": "disabled"},
        {"resize_type": "scale dimensions", "width": 0, "height": rng.randint(1, 50), "crop": "disabled"},
        {"resize_type": "scale by multiplier", "multiplier": round(rng.uniform(0.05, 3), 2)},
        {"resize_type": "scale longer dimension", "longer_size": rng.randint(1, 50)},
        {"resize_type": "scale shorter dimension", "shorter_size": rng.randint(1, 50)},
        {"resize_type": "scale width", "width": rng.randint(1, 50)},
        {"resize_type": "scale height", "height": rng.randint(1, 50)},
        {"resize_type": "scale total pixels", "megapixels": round(rng.uniform(0.01, 0.05), 3)},
        {"resize_type": "scale to multiple", "multiple": rng.randint(1, 16)},
    ]:
        x = torch.full((1, h, w, 3), 0.5)
        r = call(lambda: post.ResizeImageMaskNode.execute(input=x, resize_type=dict(rt), scale_method="nearest-exact"))
        cases.append({"node": "ResizeImageMaskNode", "input": {"shape": [1, h, w, 3]}, "args": rt, "method": "nearest-exact", "out": tensor_json(r, False)})

# Resize Image/Mask: pixels, matching, masks, zero sizes.
pixel_cases = [
    (grad(20, 30), {"resize_type": "scale to multiple", "multiple": 8}, "bilinear"),
    (grad(23, 37), {"resize_type": "scale to multiple", "multiple": 5}, "bilinear"),
    (grad(23, 37), {"resize_type": "scale to multiple", "multiple": 5}, "lanczos"),
    (grad(20, 30), {"resize_type": "scale dimensions", "width": 12, "height": 12, "crop": "center"}, "bicubic"),
    (grad(20, 30), {"resize_type": "match size", "match": [1, 9, 14, 3], "crop": "center"}, "area"),
    (grad(20, 30), {"resize_type": "match size", "match": [1, 9, 14], "crop": "disabled"}, "bilinear"),
    (grad(9, 50), {"resize_type": "scale width", "width": 2}, "bilinear"),
    (grad(9, 50), {"resize_type": "scale dimensions", "width": 2, "height": 0, "crop": "disabled"}, "bilinear"),
    (grad(9, 50), {"resize_type": "scale by multiplier", "multiplier": 0.05}, "bilinear"),
    (grad(9, 50), {"resize_type": "scale longer dimension", "longer_size": 2}, "bilinear"),
    (grad(20, 20), {"resize_type": "scale shorter dimension", "shorter_size": 7}, "bilinear"),
    (grad(20, 30), {"resize_type": "scale dimensions", "width": 0, "height": 0, "crop": "center"}, "bilinear"),
    (grad(20, 30)[..., 0], {"resize_type": "scale by multiplier", "multiplier": 0.5}, "bicubic"),
    (grad(17, 26)[..., 0], {"resize_type": "scale to multiple", "multiple": 4}, "bilinear"),
]
for x, rt, method in pixel_cases:
    x = x[None]
    rt2 = dict(rt)
    if "match" in rt2:
        rt2["match"] = torch.zeros(rt2["match"])
    r = call(lambda: post.ResizeImageMaskNode.execute(input=x, resize_type=rt2, scale_method=method))
    cases.append({"node": "ResizeImageMaskNode", "input": tensor_json(x), "args": rt, "method": method, "out": tensor_json(r)})

# Upscale Latent with bislerp: random latents up and down, and zero / parallel / opposite channel vectors.
nprng = np.random.default_rng(3)
lat = []
for (h, w, H, W) in [(4, 5, 8, 10), (6, 6, 9, 12), (5, 7, 9, 8), (3, 3, 8, 8), (16, 12, 8, 9)]:
    lat.append(torch.from_numpy(nprng.normal(size=(1, 4, h, w)).astype(np.float32)))
    lat[-1].target = (H, W)
row = np.array([[0, 0], [1, 0], [1, 0], [2, 0], [-1, 0], [0, 3], [0, 0], [0, 0]], np.float32)
edge = torch.from_numpy(row.T[:, None, :][None].copy()); edge.target = (8, 16)
lat.append(edge)
for d in [0.999991, 0.9999895, -0.999991, -0.9999895, 0.5]:
    th = math.acos(d)
    pair = np.stack([np.array([1.0, 0.0]), 2.0 * np.array([math.cos(th), math.sin(th)])]).astype(np.float32)
    t = torch.from_numpy(pair.T[:, None, :][None].copy()); t.target = (8, 8)
    lat.append(t)
for x in lat:
    H, W = x.target
    r = call(lambda: getattr(core.LatentUpscale(), core.LatentUpscale.FUNCTION)(samples={"samples": x}, upscale_method="bislerp", width=W * 8, height=H * 8, crop="disabled"))
    cases.append({"node": "LatentUpscale", "input": tensor_json(x), "args": {"width": W * 8, "height": H * 8}, "method": "bislerp", "out": tensor_json(r)})

# Resize And Pad Image on RGBA, and sizes that round to 0.
rgba = torch.tensor([0.2, 0.4, 0.6, 0.5]).repeat(1, 2, 4, 1)
for color in ("black", "white"):
    r = call(lambda: images.ResizeAndPadImage.execute(image=rgba, target_width=4, target_height=4, padding_color=color, interpolation="bilinear"))
    cases.append({"node": "ResizeAndPadImage", "input": tensor_json(rgba), "args": {"target_width": 4, "target_height": 4, "padding_color": color}, "method": "bilinear", "out": tensor_json(r)})
thin = torch.full((1, 1, 100, 3), 0.5)
for method in ("bilinear", "area"):
    r = call(lambda: images.ResizeAndPadImage.execute(image=thin, target_width=10, target_height=10, padding_color="black", interpolation=method))
    cases.append({"node": "ResizeAndPadImage", "input": tensor_json(thin), "args": {"target_width": 10, "target_height": 10, "padding_color": "black"}, "method": method, "out": tensor_json(r)})

meta = {"torch": torch.__version__, "numpy": np.__version__}
with gzip.open(out_path, "wt", encoding="utf-8") as f:
    json.dump({"meta": meta, "cases": cases}, f)
print(len(cases), "cases ->", out_path)
