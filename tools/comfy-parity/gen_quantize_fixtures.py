# Writes tests/SpawnGen.Nodes.Tests/Fixtures/comfy/comfy_image_quantize.json.gz: ComfyUI's own ImageQuantize (comfy_extras/nodes_post_processing.py,
# class Quantize) run unchanged on torch + numpy + Pillow 13, for SpawnGen's ImageQuantize to match byte for byte.
#
#   python -I tools/comfy-parity/gen_quantize_fixtures.py <ComfyUI source folder>
#
# Only the Quantize class is taken from ComfyUI's file (by AST, so kornia and the server parts are not needed); nothing of ComfyUI or Pillow is copied here.
# Inputs are float32 pictures k / 65535 (k integers), stored as k, so the C# test feeds bit-identical floats. The big pictures (more than 65536 colours,
# which makes Pillow coarsen its colour hash) come from a xorshift32 stream the test regenerates; their outputs are stored as a SHA-256 of the bytes.
import ast, gzip, hashlib, json, math, os, sys
import numpy as np
import torch
import PIL
from PIL import Image

SRC = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "SpawnGen.Nodes.Tests", "Fixtures", "comfy", "comfy_image_quantize.json.gz")


class _io:
    class ComfyNode:
        pass

    @staticmethod
    def NodeOutput(*a, **k):
        return a


ns = {"np": np, "torch": torch, "Image": Image, "math": math, "io": _io}
path = os.path.join(SRC, "comfy_extras", "nodes_post_processing.py")
tree = ast.parse(open(path, encoding="utf-8").read())
body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Quantize"]
assert len(body) == 1
exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
Quantize = ns["Quantize"]
DITHERS = ["none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"]


def xorshift_u16(seed, n):
    # xorshift32 (13, 17, 5); each step gives one 16-bit value (the top half).
    out = np.empty(n, dtype=np.int64)
    s = seed & 0xFFFFFFFF
    for i in range(n):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        out[i] = s >> 16
    return out


def procedural(kind, b, h, w, c, seed):
    r = xorshift_u16(seed, b * h * w * c).reshape(b, h, w, c)
    if kind == "random":
        return r
    # "gradnoise": a diagonal gradient per channel plus +-1024 of noise, clamped; integer arithmetic only.
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    k = np.empty((b, h, w, c), dtype=np.int64)
    for ch in range(c):
        base = [x * 65535 // (w - 1), y * 65535 // (h - 1), (x + y) * 65535 // (w + h - 2), 65535 - x * 65535 // (w - 1)][ch % 4]
        k[..., ch] = base[None]
    return np.clip(k + (r % 2049) - 1024, 0, 65535)


rng = np.random.RandomState(1234)


def smooth(h, w, cells, c=3):
    coarse = rng.rand(cells + 1, cells + 1, c)
    ys = np.linspace(0, cells, h)
    xs = np.linspace(0, cells, w)
    y0 = np.minimum(ys.astype(int), cells - 1)
    x0 = np.minimum(xs.astype(int), cells - 1)
    fy = (ys - y0)[:, None, None]
    fx = (xs - x0)[None, :, None]
    a = coarse[y0][:, x0]
    bb = coarse[y0][:, x0 + 1]
    cc = coarse[y0 + 1][:, x0]
    d = coarse[y0 + 1][:, x0 + 1]
    return a * (1 - fy) * (1 - fx) + bb * (1 - fy) * fx + cc * fy * (1 - fx) + d * fy * fx


def to_k(v):
    return np.clip(np.round(v * 65535), 0, 65535).astype(np.int64)


pictures = {}
# A smooth three-channel gradient.
h, w = 48, 64
y, x = np.meshgrid(np.arange(h) / (h - 1), np.arange(w) / (w - 1), indexing="ij")
pictures["gradient"] = to_k(np.stack([x, y, (x + y) / 2], -1))[None]
# Photo-like: smooth blobs plus fine noise (thousands of colours).
pictures["photo"] = to_k(np.clip(smooth(72, 96, 5) * 0.9 + rng.rand(72, 96, 3) * 0.1, 0, 1))[None]
# Few colours: blocks of 5 colours (fewer than most colour counts asked for), sizes not a multiple of the bayer tiles.
blocks = np.zeros((37, 41, 3))
pal5 = np.array([[0.1, 0.2, 0.9], [0.95, 0.9, 0.1], [0.5, 0.5, 0.5], [0.02, 0.6, 0.3], [1.0, 1.0, 1.0]])
for yy in range(37):
    for xx in range(41):
        blocks[yy, xx] = pal5[(yy // 9 + xx // 7) % 5]
pictures["blocks"] = to_k(blocks)[None]
# Pure noise, many more than 256 colours on a small picture.
pictures["noise"] = to_k(rng.rand(33, 29, 3))[None]
# One colour.
pictures["flat"] = to_k(np.full((5, 7, 3), [0.3, 0.6, 0.9]))[None]
# Grey ramp: ties along the colour axes.
g = np.tile(np.linspace(0, 1, 50)[None, :, None], (6, 1, 3))
pictures["grey"] = to_k(g)[None]
# A batch of two different pictures (each gets its own palette) with an alpha channel that passes through.
bat = np.stack([np.clip(smooth(20, 24, 3, 4), 0, 1), np.clip(rng.rand(20, 24, 4), 0, 1)])
pictures["rgba_batch"] = to_k(bat)
# Values outside 0..1 (wrap like torch's float to uint8 cast).
pictures["outside"] = to_k(np.clip(smooth(16, 18, 2) * 1.2 - 0.1, 0, 1))[None]

tensors = []


def add_bytes(arr):
    tensors.append({"shape": list(arr.shape), "bytes": arr.reshape(-1).astype(int).tolist()})
    return len(tensors) - 1


def add_k(arr):
    tensors.append({"shape": list(arr.shape), "k": arr.reshape(-1).astype(int).tolist()})
    return len(tensors) - 1


def as_float(k):
    return torch.from_numpy(k.astype(np.float32) / np.float32(65535))


def palettes(img, colors):
    pals = []
    rgb = img[..., :3]
    for bi in range(rgb.shape[0]):
        im = Image.fromarray((rgb[bi] * 255).to(torch.uint8).numpy(), mode="RGB")
        pals.append(im.quantize(colors=colors).getpalette())
    return pals


cases = []
for name, k in pictures.items():
    idx = add_k(k)
    img = as_float(k)
    if name == "outside":
        img = img * 1.5 - 0.25  # float32, like the C# test does
    for colors in [1, 2, 3, 16, 256]:
        for dither in DITHERS:
            if name in ("flat", "outside", "grey") and colors in (3, 16):
                continue
            (out,) = Quantize.execute(img, colors, dither)
            out = out.numpy()
            rgb_bytes = np.round(out[..., :3] * 255).astype(np.int64)
            assert np.array_equal(rgb_bytes.astype(np.float32) / np.float32(255), out[..., :3])
            if out.shape[-1] == 4:
                assert np.array_equal(out[..., 3], img.numpy()[..., 3])
            cases.append({"picture": name, "input": idx, "colors": colors, "dither": dither, "out": add_bytes(rgb_bytes), "palettes": palettes(img, colors)})

big = [("random", 1, 240, 320, 3, 2463534242), ("gradnoise", 1, 260, 300, 3, 88172645)]
for kind, b, h, w, c, seed in big:
    k = procedural(kind, b, h, w, c, seed)
    img = as_float(k)
    for colors in [2, 16, 256]:
        for dither in DITHERS:
            if dither in ("bayer-2", "bayer-16") and colors != 16:
                continue
            (out,) = Quantize.execute(img, colors, dither)
            rgb_bytes = np.round(out.numpy()[..., :3] * 255).astype(np.uint8)
            cases.append({"picture": kind, "procedural": {"kind": kind, "shape": [b, h, w, c], "seed": seed}, "colors": colors, "dither": dither,
                          "sha256": hashlib.sha256(rgb_bytes.tobytes()).hexdigest(), "palettes": palettes(img, colors)})
            print(kind, colors, dither, len(cases[-1]["palettes"][0]) // 3, flush=True)

doc = {"versions": {"torch": torch.__version__, "numpy": np.__version__, "pillow": PIL.__version__}, "tensors": tensors, "cases": cases}
with gzip.open(OUT, "wt", encoding="utf-8") as f:
    json.dump(doc, f, separators=(",", ":"))
print(len(cases), "cases ->", os.path.abspath(OUT), os.path.getsize(OUT), "bytes")
