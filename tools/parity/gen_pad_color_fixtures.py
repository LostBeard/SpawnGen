# Writes tests/SpawnGen.Nodes.Tests/Fixtures/comfy/comfy_pad_color_nodes.json.gz: inputs and outputs of ComfyUI's PadImageNode, ImageSDRToHDR and
# ColorTransfer, computed by ComfyUI's own (unchanged) node code on real torch / kornia / numpy, with the server parts stubbed out.
#
#   python -I gen_pad_color_fixtures.py <ComfyUI checkout> <output .json.gz>
#
# Needs torch, numpy, kornia and Pillow (the versions the fixture was made with are recorded in it). Tensors are stored as little-endian float32
# bytes in base64, so the expected values are exactly what torch returned.
import ast, base64, gzip, json, math, os, sys, types
from unittest.mock import MagicMock

import numpy as np
import torch
import kornia
import PIL
from PIL import Image

SRC, OUT = sys.argv[1], sys.argv[2]


# ---- load the node modules with the server parts stubbed ----

def stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def extract(path, names, glb):
    """Runs only the named top-level functions of a ComfyUI file (comfy/utils.py pulls in the whole model stack otherwise)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), glb)
            found.add(node.name)
    assert found == set(names), set(names) - found


class ProgressBar:
    def __init__(self, total): pass
    def update(self, n): pass


utils_ns = {"torch": torch, "math": math, "np": np, "Image": Image, "interpolate": torch.nn.functional.interpolate, "ProgressBar": ProgressBar}
extract(os.path.join(SRC, "comfy", "utils.py"), ["common_upscale", "lanczos", "bislerp", "repeat_to_batch_size"], utils_ns)
comfy = stub("comfy")
comfy.__path__ = []
comfy.utils = stub("comfy.utils", **utils_ns)
comfy.model_management = stub("comfy.model_management", get_torch_device=lambda: torch.device("cpu"), intermediate_device=lambda: torch.device("cpu"),
                              intermediate_dtype=lambda: torch.float32)
comfy.cli_args = stub("comfy.cli_args", args=types.SimpleNamespace(disable_metadata=False))


class _IO:
    class ComfyNode:
        hidden = None

    @staticmethod
    def NodeOutput(*a, **k):
        return a

    def __getattr__(self, n):
        return MagicMock()


IO = _IO()
stub("comfy_api")
stub("comfy_api.latest", ComfyExtension=object, IO=IO, io=IO, UI=MagicMock())
stub("nodes", MAX_RESOLUTION=16384)
stub("node_helpers")
for name in ["folder_paths", "av", "av.video", "av.video.reformatter", "server", "comfy_execution", "comfy_execution.preview_tonemap",
             "comfy_extras.nodes_latent"]:
    sys.modules[name] = MagicMock()

sys.path.insert(0, SRC)
import comfy_extras.nodes_images as ni  # noqa: E402
import comfy_extras.nodes_post_processing as npp  # noqa: E402

# ---- fixture writing ----

torch.manual_seed(20261022)
tensors, keys, cases = [], {}, []


def T(t):
    t = t.detach().to(torch.float32).contiguous()
    return {"shape": list(t.shape), "b64": base64.b64encode(t.numpy().astype("<f4").tobytes()).decode("ascii")}


def tid(t):
    if id(t) not in keys:
        keys[id(t)] = len(tensors)
        tensors.append(T(t))
    return keys[id(t)]


def add(node, args, inputs, out):
    cases.append({"node": node, "args": args, "inputs": {k: tid(v) for k, v in inputs.items()}, "out": T(out)})


def rand(*shape, lo=0.0, hi=1.0):
    return torch.rand(*shape) * (hi - lo) + lo


# ---- PadImageNode ----

pad_images = [rand(2, 5, 7, 3), rand(1, 4, 6, 4), rand(1, 3, 5, 1), rand(1, 1, 4, 3), rand(1, 6, 1, 2)]
pad_types = [
    {"padding_type": "pad dimensions", "left": 0, "top": 0, "right": 0, "bottom": 0},
    {"padding_type": "pad dimensions", "left": 2, "top": 1, "right": 3, "bottom": 0},
    {"padding_type": "pad dimensions", "left": 9, "top": 8, "right": 10, "bottom": 13},
    {"padding_type": "pad by multiplier", "multiplier": 1.0},
    {"padding_type": "pad by multiplier", "multiplier": 1.5},
    {"padding_type": "pad by multiplier", "multiplier": 1.37},
    {"padding_type": "pad by multiplier", "multiplier": 2.25},
    {"padding_type": "pad by multiplier", "multiplier": 3.0},
    {"padding_type": "pad to multiple", "multiple": 1},
    {"padding_type": "pad to multiple", "multiple": 4},
    {"padding_type": "pad to multiple", "multiple": 3},
    {"padding_type": "pad to multiple", "multiple": 32},
]
pad_styles = [
    {"padding_style": "mirror image"},
    {"padding_style": "color", "color": "#000000", "feathering": 0},
    {"padding_style": "color", "color": "#ff8000", "feathering": 0},
    {"padding_style": "color", "color": "#12ab34cc", "feathering": 2},
    {"padding_style": "color", "color": "#3366CC", "feathering": 10},
    {"padding_style": "color", "color": "#gg0000", "feathering": 1},
    {"padding_style": "color", "color": "red", "feathering": 0},
]
for im in pad_images:
    # every size rule with a mirror and with a feathered colour; every colour / feathering with two size rules
    for pt, ps in [(pt, ps) for pt in pad_types for ps in pad_styles[:1] + pad_styles[3:4]] + \
                  [(pt, ps) for pt in pad_types[1:3] for ps in pad_styles[1:3] + pad_styles[4:]]:
            out = ni.PadImageNode.execute(im, dict(pt), dict(ps))[0]
            add("PadImageNode", {"padding_type": pt, "padding_style": ps}, {"image": im}, out)

# ---- ImageSDRToHDR ----

sdr = rand(2, 6, 7, 3, lo=-0.1, hi=1.1)
sdr[0, 0, :, :] = torch.tensor([0.0, 0.04045, 1.0])  # the toe edge, black and white
sdr[0, 1, 0, :] = 1.0
sdr4 = rand(1, 4, 5, 4)
sdr1 = rand(1, 3, 3, 1)
for im in [sdr, sdr4, sdr1]:
    for target in [650.0, 0.0, 1000.0, 4000.0, 203.0, 100.0, 10000.0]:
        out = ni.ImageSDRToHDR.execute(im, {"algorithm": "lottes", "target_luminance": target})[0]
        add("ImageSDRToHDR", {"algorithm": {"algorithm": "lottes", "target_luminance": target}}, {"image": im}, out)

# ---- ColorTransfer ----

target = rand(3, 8, 9, 3)
target[2] = target[2] * 0.5 + torch.tensor([0.4, 0.1, 0.0])  # a frame with another cast
ref1 = rand(1, 6, 7, 3) * torch.tensor([0.9, 0.6, 0.3])
ref2 = rand(2, 5, 5, 3, lo=0.2, hi=0.9)
ref4 = rand(4, 6, 6, 3)
wide = rand(2, 5, 6, 3, lo=-0.2, hi=1.2)  # values outside 0..1
flat = torch.full((1, 4, 4, 3), 0.25)
target4 = rand(2, 6, 5, 4)
ref44 = rand(1, 5, 6, 4)
target1 = rand(2, 5, 5, 1)
ref11 = rand(1, 4, 6, 1) ** 2
stats = [{"source_stats": "per_frame"}, {"source_stats": "uniform"}, {"source_stats": "target_frame", "target_index": 1},
         {"source_stats": "target_frame", "target_index": 99}]
for method in ["reinhard_lab", "mkl_lab", "histogram"]:
    for tgt, ref in [(target, ref1), (target, ref2), (target, ref4), (wide, ref1), (ref1, wide)]:
        for st in stats:
            for strength in ([1.0, 0.5, 1.7] if tgt is target and ref is ref1 else [1.0, 0.6] if tgt is wide else [1.0]):
                out = npp.ColorTransfer.execute(tgt, ref, method, dict(st), strength)[0]
                add("ColorTransfer", {"method": method, "source_stats": st, "strength": strength}, {"image_target": tgt, "image_ref": ref}, out)
    for st in stats[:2]:
        add("ColorTransfer", {"method": method, "source_stats": st, "strength": 0.0}, {"image_target": wide, "image_ref": ref1},
            npp.ColorTransfer.execute(wide, ref1, method, dict(st), 0.0)[0])
    # a flat frame: zero spread, the eps clamps decide
    add("ColorTransfer", {"method": method, "source_stats": stats[0], "strength": 1.0}, {"image_target": flat, "image_ref": ref2},
        npp.ColorTransfer.execute(flat, ref2, method, dict(stats[0]), 1.0)[0])
for tgt, ref in [(target4, ref44), (target1, ref11)]:
    for st in stats[:3]:
        for strength in [1.0, 0.3]:
            out = npp.ColorTransfer.execute(tgt, ref, "histogram", dict(st), strength)[0]
            add("ColorTransfer", {"method": "histogram", "source_stats": st, "strength": strength}, {"image_target": tgt, "image_ref": ref}, out)

versions = {"torch": torch.__version__, "numpy": np.__version__, "kornia": kornia.__version__, "pillow": PIL.__version__}
with gzip.open(OUT, "wt", encoding="utf-8") as fp:
    json.dump({"versions": versions, "tensors": tensors, "cases": cases}, fp, separators=(",", ":"))
print(f"{len(cases)} cases, {len(tensors)} input tensors -> {OUT} ({os.path.getsize(OUT)} bytes)")
