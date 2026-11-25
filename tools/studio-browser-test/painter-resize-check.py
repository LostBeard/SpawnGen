"""Which resize does ComfyUI's Painter use? Compares the measured outputs (out/painter-fixtures, probe-comfy-painter-run.js) with
candidate resizes of the uploaded painting: Pillow's filters on the RGBA bytes and torch's interpolate modes on the float picture,
each composited over the background / the base picture with the painting's alpha. Prints the largest difference per candidate.
Usage: python painter-resize-check.py [tag]   (tag: white or mixed)"""
import sys, os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

tag = sys.argv[1] if len(sys.argv) > 1 else 'white'
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'painter-fixtures')
load = lambda n: np.asarray(Image.open(os.path.join(d, n))).astype(np.float64)
up = Image.open(os.path.join(d, f'{tag}-uploaded.png'))
rgba = np.asarray(up).astype(np.float32) / 255.0

def report(name, mask, want_mask, image=None, want_image=None):
    line = f'  {name:34s} mask max diff {np.abs(np.round(mask * 255) - want_mask).max():6.1f}'
    if image is not None:
        line += f' | image max diff {np.abs(np.clip(image * 255, 0, 255).astype(np.uint8).astype(np.float64) - want_image).max():6.1f}'
    print(line)

def candidates(w, h):
    out = {}
    for fname, f in [('nearest', Image.NEAREST), ('bilinear', Image.BILINEAR), ('bicubic', Image.BICUBIC), ('lanczos', Image.LANCZOS), ('box', Image.BOX), ('hamming', Image.HAMMING)]:
        out['pillow ' + fname] = np.asarray(up.resize((w, h), f)).astype(np.float32) / 255.0
    t = torch.from_numpy(rgba).permute(2, 0, 1)[None]
    for mode, kw in [('nearest', {}), ('nearest-exact', {}), ('bilinear', {'align_corners': False}), ('bicubic', {'align_corners': False}), ('area', {}),
                     ('bilinear', {'align_corners': False, 'antialias': True}), ('bicubic', {'align_corners': False, 'antialias': True})]:
        r = F.interpolate(t, size=(h, w), mode=mode, **kw)[0].permute(1, 2, 0).numpy()
        out['torch ' + mode + (' antialias' if kw.get('antialias') else '')] = r
    return out

for variant, bg in [('size', (0, 0, 0))]:
    want_image, want_mask = load(f'{tag}-{variant}-image.png'), load(f'{tag}-{variant}-mask.png')[:, :, 0]
    h, w = want_mask.shape
    print(f'{variant}: output {w}x{h} from a {up.size[0]}x{up.size[1]} painting')
    for name, r in candidates(w, h).items():
        a = np.clip(r[:, :, 3], 0, 1)
        image = r[:, :, :3] * a[:, :, None] + np.array(bg, np.float32) / 255.0 * (1 - a[:, :, None])
        report(name, a, want_mask, image, want_image)
for variant in ['imageother', 'imagebatch']:
    f = os.path.join(d, f'{tag}-{variant}-mask.png')
    if not os.path.exists(f):
        continue
    want_mask = load(f'{tag}-{variant}-mask.png')[:, :, 0]
    h, w = want_mask.shape
    print(f'{variant}: output {w}x{h} (the base picture is 300x200)')
    for name, r in candidates(w, h).items():
        report(name, np.clip(r[:, :, 3], 0, 1), want_mask)
