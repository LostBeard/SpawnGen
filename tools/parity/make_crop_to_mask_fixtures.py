"""Writes tests/SpawnGen.Nodes.Tests/fixtures/comfy/comfy_image_nodes_crop_to_mask.json.gz: black-box measurements of ComfyUI's Crop Image to Mask
(ImageCropToMask), made by calling the node class of a ComfyUI checkout on chosen tensors.

    python -I -W ignore make_crop_to_mask_fixtures.py <ComfyUI checkout> <output .json.gz>

Only the node's public execute entry point is called. Exceptions are recorded as their type and message (never a traceback). Picture values are
multiples of 1/1024 so the file stays small while the float (not 8-bit) path is still exercised; outputs are whole 8-bit levels and stored as bytes.
"""
import contextlib, gzip, importlib, io, json, os, sys

src, out_path = sys.argv[1], os.path.abspath(sys.argv[2])
sys.path.insert(0, os.path.abspath(src))
os.chdir(src)
sys.argv = [sys.argv[0], "--cpu"]
quiet = io.StringIO()
with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
    import numpy as np
    import torch
    import comfy.options
    comfy.options.enable_args_parsing()
    images = importlib.import_module("comfy_extras.nodes_images")

rng = np.random.default_rng(20261111)
tensors, cases, seen = [], [], {}


def add(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    key = (a.shape, a.tobytes())
    if key not in seen:
        tensors.append({"shape": list(a.shape), "data": [float(v) for v in a.ravel().tolist()]})
        seen[key] = len(tensors) - 1
    return seen[key]


def picture(b, h, w, c=3):
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([x / max(w - 1, 1), y / max(h - 1, 1), ((x * 7 + y * 3) % 13) / 12.0, ((x + 2 * y) % 5) / 4.0], -1)[..., :c]
    p = np.clip(base[None] * 0.7 + rng.random((b, h, w, c)) * 0.3, 0, 1)
    return np.round(p * 1024) / 1024


def blob(h, w, cy, cx, ry, rx, peak=1.6):
    y, x = np.mgrid[0:h, 0:w]
    return np.clip(peak - np.sqrt(((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2), 0, 1)


def q(m):
    return np.round(np.asarray(m) * 1024) / 1024


def run(name, img, mask, width=64, height=64, pad_factor=1.0, grow_mask=0, background="#000000"):
    ii, mi = add(img), add(mask)
    args = {"width": width, "height": height, "pad_factor": pad_factor, "grow_mask": grow_mask, "background": background, "case": name}
    try:
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            res = images.ImageCropToMask.execute(images=torch.from_numpy(np.asarray(img, np.float32)), masks=torch.from_numpy(np.asarray(mask, np.float32)),
                                                 width=width, height=height, pad_factor=pad_factor, grow_mask=grow_mask, background=background)
        val = getattr(res, "result", None) or getattr(res, "args", None) or res
        t = val[0]
        b = np.round(t.numpy().astype(np.float64) * 255)
        assert np.abs(t.numpy() * 255 - b).max() < 1e-3, "output is not whole 8-bit levels"
        out = {"shape": list(t.shape), "bytes": [int(v) for v in b.ravel().tolist()]}
    except BaseException as e:  # noqa: BLE001
        out = {"error": f"{type(e).__name__}: {e}"}
    cases.append({"node": "ImageCropToMask", "args": args, "inputs": {"images": ii, "masks": mi}, "out": [out]})


H, W = 48, 40
p = picture(1, H, W)
m = q(blob(H, W, 22, 19, 9, 7))
run("blob", p, m[None])
run("blob-wide-pad", p, m[None], 96, 64, 1.37, 0, "#ff8000")
run("blob-tall-pad", p, m[None], 64, 104, 2.0, 0, "#102030")
run("grow+3", p, m[None], 64, 64, 1.1, 3, "#ffffff")
run("grow-2", p, m[None], 72, 64, 1.0, -2, "#336699")
run("grow+32", p, m[None], 64, 64, 1.0, 32)
run("loadimage-style inverted", p, (1 - m)[None], 64, 64, 1.2, 0, "#808080")
run("inverted grow-3", p, (1 - m)[None], 64, 64, 1.0, -3)
run("width not multiple of 8", p, m[None], 70, 66, 1.05)

# Batches: a mask per picture, one mask for all, and the mismatch errors.
p2 = picture(2, H, W)
m2 = np.stack([m, q(blob(H, W, 30, 12, 6, 5))])
run("two", p2, m2, 64, 64, 1.5)
run("two broadcast mask", p2, m[None], 80, 64, 1.0, 1, "#00ff00")
run("batch mismatch", p2, np.concatenate([m2, m[None]]), 64, 64)
run("batch mismatch 1 picture", p, m2, 64, 64)

# Mask of another size (bilinear to the picture's size).
run("small mask", p, q(blob(20, 17, 9, 8, 4, 3))[None], 64, 64, 1.2)
run("large mask", p, q(blob(90, 70, 40, 33, 15, 12))[None], 64, 64, 1.0)

# Pre-scale to max(width, height): shrinking pictures, including sides that truncate to one short (98 -> 63).
pb = picture(1, 98, 50)
run("prescale 98x50 -> 63", pb, q(blob(98, 50, 50, 24, 20, 12))[None], 64, 64)
pc = picture(1, 120, 150)
mc = q(blob(120, 150, 60, 80, 25, 30))
run("prescale 150x120", pc, mc[None], 96, 64, 1.25, 0, "#c0ffee")
run("prescale grow-4", pc, mc[None], 64, 72, 1.0, -4)
run("prescale inverted", pc, (1 - mc)[None], 64, 64, 1.1)
dot = np.zeros((120, 150)); dot[60, 75] = 1
run("prescale single dot", pc, dot[None], 64, 64)
run("prescale constant 0.5 (stretch)", pc, np.full((1, 120, 150), 0.5), 64, 96, 1.3, 2, "#ff0000")
run("prescale constant 0.15 (fallback inverts)", pc, np.full((1, 120, 150), 0.15), 64, 64)

# Empty and constant masks, the 0.05 cut-off.
run("empty", p, np.zeros((1, H, W)))
run("all ones", p, np.ones((1, H, W)), 64, 64, 1.0, 0, "#123456")
run("constant 0.03 (cut to zero)", p, np.full((1, H, W), 0.03), 64, 64)
run("constant 0.5 (stretch)", p, np.full((1, H, W), 0.5), 72, 64, 1.0, 0, "#3070b0")
cut = m.copy(); cut[5:10, 5:15] = np.float32([0.0499, 0.04999, 0.05, 0.05001, 0.0501] * 2)[None, :].repeat(5, 0)
run("values around the cut-off", p, cut[None], 64, 64, 1.6)
thr = np.zeros((H, W)); thr[20:28, 18:26] = 1; thr[24, 5:18] = 0.80001; thr[10:20, 22] = 205 / 255
run("values around the 0.8 threshold", p, thr[None], 64, 64)

# The inversion rule: edges mean (corners counted twice) > 0.5 and centre mean < 0.5, on the raw mask.
N = 60
pn = picture(1, N, N)
def ring(top=True, left=True, bottom=False, right=False, centre_hole=0):
    r = np.zeros((N, N))
    if top: r[0, :] = 1
    if left: r[:, 0] = 1
    if bottom: r[-1, :] = 1
    if right: r[:, -1] = 1
    return r
c1 = np.zeros((N, N)); c1[0, :] = 1; c1[1:59, 0] = 1
c2 = np.zeros((N, N)); c2[0, :] = 1; c2[1:60, 0] = 1
run("edges exactly 0.5 (keep)", pn, c1[None])
run("edges just above 0.5 (invert)", pn, c2[None])
for h in (14, 15, 16):
    hole = np.ones((N, N)); hole[15:15 + h, 15:45] = 0
    run(f"centre hole {h} rows", pn, hole[None], 64, 64, 1.0, 0, "#204060")
raw = np.zeros((N, N)); raw[-1, :] = 0.04; raw[:, -1] = 0.04; raw[0, :] = 1; raw[0:59, 0] = 1; raw[59, 0] = 0.04
run("rule sees the raw mask", pn, raw[None])
odd = picture(1, 61, 63)
oh = np.full((61, 63), 0.5); oh[0, :] = 1; oh[-1, :] = 1; oh[:, 0] = 1; oh[:, -1] = 1; oh[45, 1:62] = 0
run("centre region bound 61x63", odd, oh[None])
noise = np.clip(rng.random((N, N)), 0, 1); noise[0, :] = 1; noise[-1, :] = 1; noise[:, 0] = 1; noise[:, -1] = 1
run("noisy mask, opaque border", pn, q(noise)[None], 64, 64, 1.0, 0, "#7591d5")

# Picture channels.
run("one-channel picture", picture(1, H, W, 1), m[None], 64, 64, 1.2)
run("four-channel picture", picture(1, H, W, 4), m[None], 64, 64, 1.2, 0, "#ff00ff")
run("two-channel picture", picture(1, H, W, 2), m[None])

# Background strings.
for bg in ["ff8000", "##FF8000", "#fff", "#ff8000aa", "red", "", " #ff8000", "#ff 800", "#+f+800", "#-10000", "#gg0000"]:
    run(f"background {bg!r}", p, m[None], 64, 64, 1.8, 0, bg)

with gzip.open(out_path, "wt", encoding="utf-8") as f:
    json.dump({"tensors": tensors, "cases": cases}, f, separators=(",", ":"))
print(f"{len(cases)} cases, {sum(1 for c in cases if 'error' in c['out'][0])} errors -> {out_path}")
