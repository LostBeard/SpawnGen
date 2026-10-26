"""Runs ComfyUI's own compositor nodes (comfy_extras/nodes_compositor.py: Add Layer, Layers From Bounding Boxes, Create Layered Image) on chosen inputs
and writes what came out to compositor.json.gz.

    python make_compositor.py <ComfyUI folder>

A LAYERS document is kept as {"version", "canvas", "layers": [item, ...]} with each picture or mask as {"tensor": index}; pictures are base64 ("u8" bytes k
for values k/255, else "f32" little-endian float32). Create Layered Image cases keep the composite, the mask and what the editor gets (layer
fingerprints, boxes, canvas, stale flag). A node that raises keeps the exception type and message.
"""
import base64, copy, gzip, json, math, sys
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import comfy_extras.nodes_compositor as nc  # noqa: E402


class _UI:
    """UI previews would write files into ComfyUI's temp folder; the outputs under test do not depend on them."""
    class PreviewImage:
        def __init__(self, *a, **k):
            self.values = []

        def as_dict(self):
            return {"images": []}


nc.UI = _UI

tensors = []


def tensor(t):
    t = t.detach().cpu().float().contiguous()
    b = np.rint(t.numpy().astype(np.float64) * 255.0)
    if b.size and b.min() >= 0 and b.max() <= 255 and torch.equal(torch.from_numpy(b).float() / 255.0, t):
        tensors.append({"shape": list(t.shape), "u8": base64.b64encode(b.astype(np.uint8).tobytes()).decode()})
    else:
        tensors.append({"shape": list(t.shape), "f32": base64.b64encode(t.numpy().astype("<f4").tobytes()).decode()})
    return len(tensors) - 1


def enc(v):
    if isinstance(v, torch.Tensor):
        return {"tensor": tensor(v)}
    if isinstance(v, dict):
        return {k: enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    return v


rng = np.random.default_rng(4321)


def image(b, h, w, c=3, lo=0, hi=256):
    return torch.from_numpy(rng.integers(lo, hi, size=(b, h, w, c)).astype(np.uint8)).float() / 255.0


def mask(b, h, w):
    return torch.from_numpy(rng.integers(0, 256, size=(b, h, w)).astype(np.uint8)).float() / 255.0


cases = []


def run(node, fn, args):
    case = {"node": node, "args": enc(args)}
    try:
        out = fn()
        res = out.result
        if node == "ImageCompositor":
            case["out"] = [{"tensor": tensor(res[0])}, {"tensor": tensor(res[1])}]
            ui = out.ui
            case["ui"] = {k: ui[k] for k in ("compositor_inputs", "compositor_bboxes", "compositor_canvas", "compositor_state_stale") if k in ui}
        else:
            case["out"] = [enc(res[0])]
    except Exception as e:  # noqa: BLE001 - the error itself is what the case checks
        case["error"] = type(e).__name__
        case["message"] = str(e)
    cases.append(case)
    return out.result[0] if "error" not in case else None


def add(image, layers=None, **kw):
    args = {"image": image, "layers": layers, **kw}
    return run("AddLayer", lambda: nc.AddLayer.execute(image, layers=layers, **kw), args)


def from_boxes(image, bboxes, **kw):
    args = {"image": image, "bboxes": bboxes, **kw}
    return run("LayersFromBoundingBoxes", lambda: nc.LayersFromBoundingBoxes.execute(image, bboxes, **kw), args)


def comp(layers, compositor=None):
    args = {"layers": layers, "compositor": compositor}
    run("ImageCompositor", lambda: nc.ImageCompositor.execute(layers, compositor), args)


def fingerprints(layers):
    out = nc.ImageCompositor.execute(layers, None)
    return out.ui["compositor_inputs"]


# ---- Add Layer ---------------------------------------------------------------------------------------------------------------------------------------

base = image(1, 14, 18)
top = image(2, 8, 10, 4)
doc1 = add(base)
doc2 = add(top, layers=doc1, mask=mask(1, 8, 10), name="top", x=-3, y=5, opacity=0.5, blend_mode="screen", rotation=30.0, width=12, height=0,
           z_index=-2, flip_h=True)
doc3 = add(image(1, 6, 7), layers=doc2, x=9, y=7, opacity=0.8, blend_mode="multiply", flip_v=True, z_index=3)

# ---- Layers From Bounding Boxes ----------------------------------------------------------------------------------------------------------------------

frames = image(3, 10, 12)
boxes = [{"x": 2, "y": 3, "width": 8, "height": 6, "metadata": {"name": "a", "z_index": 5, "content_rect": [1, 2, 6, 5]}},
         {"x": 10.7, "y": "4", "width": 0, "height": 9, "metadata": {"desc": "b", "content_rect": [8, 1, 20, 20]}}]
fb1 = from_boxes(frames, boxes, mask=mask(3, 10, 12), layers=doc1)
fb2 = from_boxes(frames, boxes, mask=mask(1, 6, 9), crop_to_content=False, canvas_width=30, canvas_height=24)
fb3 = from_boxes(image(2, 6, 6), [[{"x": 1, "y": 1, "width": 4, "height": 4}], [{"x": 9, "y": 9, "width": 1, "height": 1}]], layers=fb2)
from_boxes(image(1, 6, 6), [{"type": "obj", "bbox": [100, 200, 600, 900]}])
fb4 = from_boxes(image(2, 6, 6), [{"type": "obj", "bbox": [100, 200, 600, 900], "desc": "el"}], canvas_width=20, canvas_height=16)
fb5 = from_boxes(image(1, 6, 6), json.dumps({"x": 3, "y": 2, "width": 5, "height": 5, "metadata": {"content_rect": [0, 0, 0, 3]}}))
from_boxes(image(1, 6, 6), "{oops")
from_boxes(image(1, 6, 6), "  ")
from_boxes(image(1, 6, 6), [5])

# ---- Create Layered Image ----------------------------------------------------------------------------------------------------------------------------

comp(doc1)
comp(doc2)
comp(doc3)
comp(fb1)
comp(fb2)
comp(fb3)
comp(fb4)
comp(fb5)
comp({"version": 1, "layers": []})
comp(None)

# Every blend mode: a half-transparent layer over a partly transparent base.
bm_base = add(image(1, 6, 6, 4, 0, 256))
bm_top = image(1, 5, 5, 4, 0, 256)
for mode in list(nc._LAYER_MODES):
    comp(nc.AddLayer.execute(bm_top, layers=bm_base, x=1, y=1, opacity=0.7, blend_mode=mode).result[0])

# Turns, flips and sizes (bicubic rotation, Lanczos resize, the transposes for quarter and half turns).
for rot, w, h in [(30.0, 0, 0), (45.0, 9, 5), (90.0, 0, 0), (-90.0, 0, 0), (180.0, 7, 0), (-135.5, 0, 0), (360.0, 0, 0), (0.25, 0, 0)]:
    comp(nc.AddLayer.execute(image(1, 7, 9, 4), layers=add(image(1, 16, 16)), x=3, y=2, rotation=rot, width=w, height=h, flip_v=True).result[0])

# Masks: the picture's size, another size (bilinear), one mask for a batch, a shorter mask batch.
masked = nc.AddLayer.execute(image(1, 8, 8), layers=None, mask=mask(1, 8, 8)).result[0]
comp(masked)
comp(nc.AddLayer.execute(image(1, 8, 8, 4), layers=None, mask=mask(1, 5, 11)).result[0])
comp(nc.AddLayer.execute(image(3, 6, 6), layers=None, mask=mask(1, 6, 6), x=2).result[0])
comp(nc.AddLayer.execute(image(3, 6, 6), layers=None, mask=mask(2, 6, 6), y=1).result[0])

# Saved compositions: replayed when their inputs match the layers, stale otherwise.
replay_doc = nc.AddLayer.execute(image(1, 6, 8, 4), layers=nc.AddLayer.execute(image(1, 9, 9)).result[0], x=2, y=1, opacity=0.6, blend_mode="overlay").result[0]
fp = fingerprints(replay_doc)
state = {"version": 1, "canvas": {"w": 14.4, "h": "12"}, "inputs": fp,
         "background": {"color": "#3A7", "opacity": 0.8, "visible": True}, "order": [1, 0],
         "layers": [{"visible": True, "opacity": 2, "blend": "multiply", "transform": {"x": "a", "y": 2, "w": -1, "h": 7.6, "rotation": 0.3}},
                    {"visible": 1, "opacity": 0.5, "blend": "difference", "flipH": True, "transform": {"x": 1.5, "y": -2, "w": 10, "h": 6}}]}
comp(replay_doc, state)
comp(replay_doc, json.dumps(state))
comp(replay_doc, {**state, "order": [0, 0], "background": {"color": "bad", "visible": True}})
comp(replay_doc, {**state, "layers": state["layers"][:1] + [{"visible": False}]})
comp(replay_doc, {**state, "inputs": ["0" * 16, fp[1]]})
comp(replay_doc, {**state, "canvas": {"w": 0, "h": 5}})
comp(replay_doc, {**state, "canvas": {"w": "nan", "h": 5}})
comp(replay_doc, {**state, "version": 2})
comp(replay_doc, "{}")
comp(replay_doc, "not json")
comp(replay_doc, {**state, "canvas": {"w": 20000, "h": 5}})
comp(replay_doc, {**state, "canvas": {"w": "13.5", "h": 12.5}})  # round() halves to even

# Errors from the document.
comp({"version": 2, "layers": []})
comp({"version": 1, "layers": [{"image": image(1, 2, 2), "type": "raster", "z_index": 0, "blend_mode": "nope"}]})
comp(nc.AddLayer.execute(image(51, 1, 1)).result[0])

# ---- compositor_blend itself, on edge values: every mode's formula and threshold, the sRGB curves, the two composites ------------------------------

import comfy_extras.compositor_blend as cb  # noqa: E402

edges = [0.0, 1e-7, 1e-6, 2e-6, 0.0031308, 0.0031309, 0.04045, 0.0404, 0.25, 0.4999999, 0.5, 0.5000001, 0.75, 0.999999, 1.0, 1.2, -0.1]
pairs = [(a, b) for a in edges for b in edges]
I = np.array([[a, b, 0.3] for a, b in pairs] + [[0.5, 0.5, 0.5], [0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5000001], [0.9, 0.1, 0.2], [0.2, 0.2, 0.2]], dtype=np.float32)
Lr = np.array([[b, a, 0.6] for a, b in pairs] + [[0.2, 0.7, 0.1], [0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [1, 0, 0], [0, 0, 0], [0.4, 0.6, 0.6]], dtype=np.float32)
ramp = np.array(edges + list(np.linspace(-0.1, 1.1, 121)), dtype=np.float32)
blend = {"i": tensor(torch.from_numpy(I)), "l": tensor(torch.from_numpy(Lr)), "ramp": tensor(torch.from_numpy(ramp)),
         "to_linear": tensor(torch.from_numpy(cb.srgb_to_linear(ramp))), "to_srgb": tensor(torch.from_numpy(cb.linear_to_srgb(ramp))), "modes": {}}
for mode in cb._LAYER_MODES:
    blend["modes"][mode] = tensor(torch.from_numpy(cb.blend_pixel(mode, I, Lr)))
# blend_composite over every alpha pair, for a union mode and a clip-to-backdrop mode, at three opacities.
alphas = [0.0, 0.3, 1.0]
back = np.array([[0.123, 0.5, 0.77, a] for a in alphas for _ in alphas], dtype=np.float32)
lay = np.array([[0.7, 0.1, 0.456, b] for _ in alphas for b in alphas], dtype=np.float32)
blend["back"], blend["layer"], blend["composites"] = tensor(torch.from_numpy(back)), tensor(torch.from_numpy(lay)), []
for mode in ("normal", "multiply", "screen"):
    for opacity in (0.0, 0.35, 1.0):
        out = cb.blend_composite(cb.resolve_mode(mode), back, lay, opacity)
        blend["composites"].append({"mode": mode, "opacity": opacity, "out": tensor(torch.from_numpy(out))})

with gzip.open("compositor.json.gz", "wt", encoding="utf-8") as f:
    json.dump({"tensors": tensors, "cases": cases, "blend": blend}, f, separators=(",", ":"))
print(len(cases), "cases,", len(tensors), "tensors")
