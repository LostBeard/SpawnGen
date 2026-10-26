"""Runs ComfyUI's own bounding-box nodes on chosen inputs and writes what came out to bbox.json.gz.

    python make_bbox.py <ComfyUI folder>

Nodes: Create Bounding Boxes (comfy_extras/nodes_bounding_boxes.py), Crop Image (ImageCropV2, nodes_images.py), Draw BBoxes (nodes_rtdetr.py),
Crop By Bounding Boxes (nodes_sdpose.py), Split Image into List of Tiles and Merge List of Tiles to Image (nodes_images.py). Each case keeps the
arguments (JSON values as they are), the pictures (base64: "u8" bytes k for values k/255, else "f32" little-endian float32) and the outputs; a
node that raises keeps the
exception type and message. Previews also keep the rectangles where ComfyUI writes text (number tags, descriptions, labels): SpawnGen has no font
engine yet, so the tests compare every other pixel.
"""
import base64, gzip, json, math, sys
import numpy as np
import torch
from PIL import ImageDraw

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import comfy_extras.nodes_bounding_boxes as nb  # noqa: E402
import comfy_extras.nodes_images as ni  # noqa: E402
import comfy_extras.nodes_rtdetr as nr  # noqa: E402
import comfy_extras.nodes_sdpose as ns  # noqa: E402
import nodes  # noqa: E402


class _NoPreview:
    """UI previews would write files into ComfyUI's temp folder; the outputs under test do not depend on them."""
    @staticmethod
    def PreviewImage(*a, **k):
        return None


ni.UI = _NoPreview
nodes.PreviewImage.save_images = lambda self, images, prefix="", *a, **k: {"ui": {"images": [{"filename": "bg.png"}]}}

# Record where text is written (and the tag rectangles drawn under it), the only pixels that need a font.
_calls = []
_orig_text, _orig_rect = ImageDraw.ImageDraw.text, ImageDraw.ImageDraw.rectangle


def _text(self, xy, text, fill=None, font=None, *a, **k):
    _calls.append(("text", self.textbbox(xy, text, font=font), text))
    return _orig_text(self, xy, text, fill, font, *a, **k)


def _rect(self, xy, fill=None, outline=None, width=1):
    _calls.append(("rect", [float(v) for v in (xy[0], xy[1], xy[2], xy[3])] if len(xy) == 4 else [float(v) for p in xy for v in p], None))
    return _orig_rect(self, xy, fill, outline, width)


ImageDraw.ImageDraw.text = _text
ImageDraw.ImageDraw.rectangle = _rect


def text_rects():
    """Inclusive pixel rectangles to leave out: every text's box (one pixel more each side) and a filled rectangle drawn right before a text (a tag)."""
    rects = []
    for i, (kind, box, text) in enumerate(_calls):
        if kind == "text":
            l, t, r, b = box
            rects.append([int(math.floor(l)) - 1, int(math.floor(t)) - 1, int(math.ceil(r)) + 1, int(math.ceil(b)) + 1])
            if i > 0 and _calls[i - 1][0] == "rect" and str(text).isdigit():
                x0, y0, x1, y1 = _calls[i - 1][1]
                rects.append([int(x0), int(y0), int(x1), int(y1)])
    _calls.clear()
    return rects


tensors = []


def tensor(t):
    t = t.detach().cpu().float().contiguous()
    b = np.rint(t.numpy().astype(np.float64) * 255.0)
    if b.size and b.min() >= 0 and b.max() <= 255 and torch.equal(torch.from_numpy(b).float() / 255.0, t):
        tensors.append({"shape": list(t.shape), "u8": base64.b64encode(b.astype(np.uint8).tobytes()).decode()})  # values k/255
    else:
        tensors.append({"shape": list(t.shape), "f32": base64.b64encode(t.numpy().astype("<f4").tobytes()).decode()})
    return len(tensors) - 1


rng = np.random.default_rng(1234)


def image(b, h, w, c=3, lo=0, hi=256):
    """A picture of values k/255 (k random), as ComfyUI pictures from 8-bit files are."""
    return torch.from_numpy(rng.integers(lo, hi, size=(b, h, w, c)).astype(np.uint8)).float() / 255.0


cases = []


def run(node, fn, args, inputs, text=False):
    _calls.clear()
    case = {"node": node, "args": args, "inputs": {k: tensor(v) for k, v in inputs.items() if v is not None}}
    try:
        out = fn()
        res = out.result if hasattr(out, "result") else out
        enc = []
        for o in res:
            if isinstance(o, torch.Tensor):
                enc.append({"tensor": tensor(o)})
            elif isinstance(o, list) and o and all(isinstance(x, torch.Tensor) for x in o):
                enc.append({"tensors": [tensor(x) for x in o]})
            else:
                enc.append({"value": o})
        case["out"] = enc
        if text:
            case["text_rects"] = text_rects()
    except Exception as e:  # noqa: BLE001 - the error itself is what the case checks
        case["error"] = type(e).__name__
        case["message"] = str(e)
    _calls.clear()
    cases.append(case)


# ---- Create Bounding Boxes ---------------------------------------------------------------------------------------------------------------------------

def cbb(width, height, editor_state=None, last_incoming=None, background=None, bboxes=None):
    args = {"width": width, "height": height, "editor_state": editor_state, "last_incoming": last_incoming, "bboxes": bboxes}
    run("CreateBoundingBoxes", lambda: nb.CreateBoundingBoxes.execute(width, height, editor_state=editor_state, last_incoming=last_incoming,
                                                                         background=background, bboxes=bboxes),
        args, {"background": background}, text=True)


state = [
    {"x": 10, "y": 8, "width": 60, "height": 40, "metadata": {"type": "obj", "desc": "a red barn", "palette": ["#aa2211", "#33cc55", "#123456"]}},
    {"x": 50, "y": 30, "width": 70, "height": 36, "metadata": {"type": "text", "text": "HELLO", "desc": "sign", "palette": {"a": "#ffee00", "b": ""}}},
    {"x": -15, "y": 60, "width": 40.5, "height": 50, "metadata": {"type": "obj", "palette": ["#0000ff", "bad", "#00ff00", "#ff00ff", "#00ffff", "#ffffff"]}},
    {"x": 130, "y": 70, "width": -30, "height": -20},
    {"x": 3, "y": 3, "width": 1, "height": 1, "metadata": {"palette": ["#ff0000"]}},
]
cbb(144, 96, editor_state=state)
cbb(144, 96, editor_state=[])
cbb(144, 96)
cbb(200, 64, editor_state=[{"x": 0, "y": 0, "width": 200, "height": 64, "metadata": {"type": "obj", "palette": ["#+1+2+3", "#ab", "###00ff00"]}},
                           {"x": 20, "y": 10, "width": 60, "height": 30, "metadata": {"palette": ["#1234567", "#12345g"]}}])
elements = [
    {"type": "obj", "bbox": [100, 50, 600, 400], "desc": "a cat", "color_palette": ["#112233", "#445566"]},
    {"type": "text", "bbox": [700, 900, 650, 300], "text": "Sale", "desc": "big letters"},
    {"type": "obj", "bbox": ["120", 5.5, 999, True], "color_palette": []},
]
cbb(160, 120, bboxes=json.dumps(elements))
cbb(160, 120, bboxes=elements, editor_state=state[:1])
cbb(160, 120, bboxes=elements[0])
normalized = nb.boxes_from_input(elements, 160, 120)
cbb(160, 120, bboxes=elements, last_incoming=normalized, editor_state=state[:2])
frames = [[{"x": 5, "y": 6, "width": 40, "height": 30, "label": "cat", "score": 0.5}, {"x": 60, "y": 10, "width": 20, "height": 20}], [{"x": 1, "y": 1, "width": 2, "height": 2}]]
cbb(120, 90, bboxes=frames)
cbb(120, 90, bboxes={"x": 4.5, "y": 2, "width": 30, "height": 30.25, "metadata": {"type": "text", "text": "", "desc": 7}})
cbb(120, 90, bboxes="   ")
cbb(120, 90, bboxes="[]")
cbb(120, 90, bboxes="{not json")
# json.loads errors carry Python's position text (line, column, char).
for bad in ['[1, 2', '[1,]', '{"a" 1}', '{"a": 1,}', '"abc', r'"a\x"', r'"\u12"', '"a\tb"', '1 2', '\n\n  [1, }', '-', '[-]', 'tru', '{"a": [1, 2}',
            '[{"x": 1}] x', '[1, "ab', r'"a\ud800"', '"\x01"', '{"a": 1 "b"}']:
    cbb(120, 90, bboxes=bad)
cbb(120, 90, bboxes={"foo": 1})
cbb(120, 90, bboxes=[{"foo": 1}])
cbb(120, 90, bboxes=[5])
cbb(120, 90, bboxes=5)
cbb(120, 90, bboxes=[{"bbox": [1, 2, 3]}, {"bbox": [1, 2, 3, 4]}])
cbb(120, 90, bboxes=[{"bbox": [1, 2, 3, 4]}, {"bbox": [1, None, 3, 4]}])
cbb(120, 90, bboxes=[{"bbox": [1, 2, 3, 4]}, {"bbox": [1, "x", 3, 4]}])
cbb(120, 90, editor_state=[{"x": 10, "y": 10, "width": 50, "height": 40, "metadata": {"type": "text", "text": "T", "desc": 5}}])
cbb(120, 90, editor_state=[{"x": 10, "y": 10, "width": 50, "height": 40, "metadata": {"desc": 5}}])
cbb(120, 90, editor_state=[{"x": 10, "y": 10, "width": 5, "height": 40, "metadata": {"desc": 5}}])
cbb(120, 90, editor_state=[{"x": 10, "y": 10, "width": 50, "height": 40, "metadata": {"palette": [3]}}])
cbb(120, 90, editor_state={"x": 1, "y": 1, "width": 5, "height": 5})
cbb(2048, 300, editor_state=state[:3])
cbb(100, 60, editor_state=state[:3], background=image(2, 40, 52, 3))
cbb(100, 60, editor_state=state[:2], background=image(1, 30, 30, 4))
cbb(100, 60, editor_state=state[:2], background=image(1, 30, 33, 2))
cbb(100, 60, editor_state=state[:2], background=image(1, 30, 30, 1))
cbb(100, 60, editor_state=state[:2], background=image(1, 12, 1100, 3))

# ---- Crop Image (ImageCropV2) ------------------------------------------------------------------------------------------------------------------------

crop_src = image(2, 20, 30, 3)
for region in [{"x": 3, "y": 4, "width": 10, "height": 7}, {"x": 25, "y": 15, "width": 10, "height": 10}, {"x": 100, "y": 100, "width": 5, "height": 5},
               {"x": -5, "y": 2, "width": 10, "height": 10}, {"x": -10, "y": -8, "width": 12, "height": 12}, {}, {"x": 2, "y": 2, "width": 0, "height": 5},
               {"x": 2, "y": 2, "width": -3, "height": 5}, {"x": 2.0, "y": 2, "width": 5, "height": 5}, {"x": 50.5, "y": 2, "width": 5, "height": 5},
               {"x": True, "y": 0, "width": 3, "height": 2}, {"x": 1, "y": 1, "width": 4, "height": 3, "metadata": {"z": 1}}]:
    run("ImageCropV2", lambda r=region: ni.ImageCropV2.execute(crop_src, r), {"crop_region": region}, {"image": crop_src})
run("ImageCropV2", lambda: ni.ImageCropV2.execute(crop_src, [[{"x": 1, "y": 1, "width": 4, "height": 3}]]),
    {"crop_region": [[{"x": 1, "y": 1, "width": 4, "height": 3}]]}, {"image": crop_src})

# ---- Draw BBoxes ---------------------------------------------------------------------------------------------------------------------------------------

def draw(bboxes, img=None):
    run("DrawBBoxes", lambda: nr.DrawBBoxes.execute(bboxes, image=img), {"bboxes": bboxes}, {"image": img}, text=True)


draw_src = image(2, 40, 50, 3)
draw([{"x": 5, "y": 5, "width": 20, "height": 15}, {"x": 30, "y": 20, "width": 30, "height": 30, "score": 0.2}], draw_src)
draw([[{"x": 5, "y": 5, "width": 20, "height": 15, "label": "dog", "score": 0.9}, {"x": 0, "y": 0, "width": 49, "height": 39, "label": "cat", "score": 0.3}],
      [{"x": -5, "y": -5, "width": 12, "height": 12, "label": "unknown"}, {"x": 10.7, "y": 3.2, "width": 9.9, "height": 0, "score": 1}]], draw_src)
draw([[{"x": 5, "y": 5, "width": 2, "height": 1}, {"x": 20, "y": 20, "width": 0, "height": 0}, {"x": 40, "y": 30, "width": 1, "height": 3}]], draw_src)
draw({"x": 1, "y": 2, "width": 10, "height": 10, "label": "person", "score": 0.75}, draw_src)
draw([], draw_src)
draw([[], [{"x": 2, "y": 2, "width": 5, "height": 5}]], draw_src)
draw([[{"x": 2, "y": 2, "width": 5, "height": 5}], [], [{"x": 3, "y": 3, "width": 5, "height": 5}]], draw_src)
draw([{"x": 5, "y": 5, "width": 20, "height": 15}, {"x": 7, "y": 9, "width": 25, "height": 11}])
draw([[{"x": 5.5, "y": 5, "width": 20.9, "height": 15}], [{"x": 70, "y": 1, "width": 3, "height": 3}]])
draw([])
draw([{"x": 5, "y": 5, "width": -3, "height": 4}], draw_src)
draw([{"x": 5, "y": 5, "width": 4}], draw_src)
draw([{"x": 1, "y": 1, "width": 30, "height": 20}], image(1, 24, 36, 4))
draw([], image(1, 6, 7, 1))
draw([{"x": 1, "y": 1, "width": 3, "height": 3}], image(1, 6, 7, 1))
draw([], image(1, 6, 7, 2))
draw([{"x": 2, "y": 2, "width": 9, "height": 9, "score": 0.5}, {"x": 4, "y": 4, "width": 9, "height": 9, "score": 0.5, "label": "car"},
      {"x": 6, "y": 6, "width": 9, "height": 9, "score": 0.1, "label": "bus"}], image(1, 24, 24, 3))
odd = torch.tensor([[[[-0.2, 0.5, 1.3], [1.0, 0.999, 0.0001]]]], dtype=torch.float32)
draw([], odd)

# ---- Crop By Bounding Boxes ------------------------------------------------------------------------------------------------------------------------------

def crop_by(img, bboxes, ow, oh, padding, keep="stretch"):
    run("CropByBBoxes", lambda: ns.CropByBBoxes.execute(img, bboxes, ow, oh, padding, keep),
        {"bboxes": bboxes, "output_width": ow, "output_height": oh, "padding": padding, "keep_aspect": keep}, {"image": img})


cb_src = image(3, 36, 44, 3)
crop_by(cb_src, [[{"x": 4, "y": 6, "width": 20, "height": 12}, {"x": 30, "y": 20, "width": 6, "height": 10}]], 16, 12, 0)
crop_by(cb_src, [[{"x": 4, "y": 6, "width": 20, "height": 12}], [], [{"x": -6, "y": 30, "width": 20.7, "height": 20}]], 10, 14, 3, "pad")
crop_by(cb_src, [[{"x": 4, "y": 6, "width": 20, "height": 12}], [{"x": 10, "y": 10, "width": 5, "height": 9}]], 12, 12, 2, "pad")
crop_by(cb_src, {"x": 40, "y": 30, "width": 30, "height": 30}, 8, 8, 0)
crop_by(cb_src, [[{"x": 50, "y": 50, "width": 10, "height": 10}]], 9, 7, 0)
crop_by(image(1, 2, 2, 3), [[{"x": 50, "y": 50, "width": 10, "height": 10}]], 9, 7, 0)
crop_by(cb_src, [], 9, 7, 0)
crop_by(cb_src, [[], [], []], 9, 7, 0)
crop_by(cb_src, [{"x": 4, "y": 6, "width": 20, "height": 12}], 9, 7, 0)
crop_by(cb_src, [[{"x": "4", "y": 6, "width": 20, "height": 12}]], 9, 7, 0)
crop_by(image(2, 30, 40, 4), [[{"x": 4, "y": 6, "width": 33, "height": 7}]], 20, 20, 1, "pad")

# ---- Split Image into List of Tiles / Merge List of Tiles to Image ---------------------------------------------------------------------------------------

def split(img, tw, th, overlap):
    run("SplitImageToTileList", lambda: (ni.SplitImageToTileList.execute(img, tw, th, overlap).result[0],),
        {"tile_width": tw, "tile_height": th, "overlap": overlap}, {"image": img})


tile_src = image(2, 46, 58, 3)
split(tile_src, 24, 20, 6)
split(tile_src, 32, 32, 30)
split(tile_src, 64, 64, 0)
split(tile_src, 58, 23, 23)
split(image(1, 12, 30, 3), 10, 10, 9)  # stride round(2.5): the half goes to even
split(image(1, 33, 17, 4), 10, 10, 3)


def merge(tiles, fw, fh, overlap):
    case = {"node": "ImageMergeTileList", "args": {"final_width": fw, "final_height": fh, "overlap": overlap},
            "inputs": {}, "tiles": [tensor(t) for t in tiles]}
    try:
        out = ni.ImageMergeTileList.execute(tiles, [fw], [fh], [overlap])
        case["out"] = [{"tensor": tensor(out.result[0])}]
    except Exception as e:  # noqa: BLE001
        case["error"] = type(e).__name__
        case["message"] = str(e)
    cases.append(case)


merge_src = image(1, 46, 58, 3)
for tw, th, ov in [(24, 20, 6), (32, 32, 30), (64, 64, 0), (58, 23, 23)]:
    tiles = ni.SplitImageToTileList.execute(merge_src, tw, th, ov).result[0]
    merge(tiles, 58, 46, ov)
merge(ni.SplitImageToTileList.execute(tile_src, 24, 20, 6).result[0], 58, 46, 6)
tiles = [image(1, 20, 24, 3) for _ in range(12)]
merge(tiles, 58, 46, 6)
merge(tiles[:5], 58, 46, 6)
merge(tiles, 61, 40, 9)
merge([image(1, 20, 24, 3)] + [image(2, 20, 24, 3)] * 3, 40, 20, 8)
merge([image(2, 20, 24, 3)] + [image(1, 20, 24, 3)] * 3, 40, 20, 8)
merge(ni.SplitImageToTileList.execute(image(1, 33, 17, 4), 10, 10, 3).result[0], 17, 33, 3)

with gzip.open("bbox.json.gz", "wt", encoding="utf-8") as f:
    json.dump({"tensors": tensors, "cases": cases}, f, separators=(",", ":"))
print(len(cases), "cases,", len(tensors), "tensors")
