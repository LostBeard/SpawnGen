"""Reads the paintings probe-comfy-painter-brush.js saved (out/painter-brush/*.png): for each, the painted area's bounding box, the alpha
along the horizontal line through the stamp's centre, and the colour. Usage: python painter-brush-read.py [name substring]"""
import sys, os, glob
import numpy as np
from PIL import Image

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'painter-brush')
want = sys.argv[1] if len(sys.argv) > 1 else ''
for f in sorted(glob.glob(os.path.join(d, '*.png'))):
    name = os.path.basename(f)[:-4]
    if name == 'last' or want not in name:
        continue
    a = np.asarray(Image.open(f).convert('RGBA')).astype(np.int32)
    alpha = a[:, :, 3]
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        print(f'{name}: {a.shape[1]}x{a.shape[0]}, nothing painted'); continue
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cy = (y0 + y1) // 2
    row = alpha[cy, x0:x1 + 1]
    col = alpha[y0:y1 + 1, (x0 + x1) // 2]
    solid = a[alpha == alpha.max()][0]
    line = f'{name}: {a.shape[1]}x{a.shape[0]} | box x {x0}..{x1} ({x1 - x0 + 1}) y {y0}..{y1} ({y1 - y0 + 1}) | alpha max {alpha.max()} | painted px {len(xs)} | colour at max {tuple(int(v) for v in solid[:3])}'
    print(line)
    def show(v, label):
        if len(v) <= 44: print(f'    {label}: ' + ' '.join(str(int(x)) for x in v))
        else: print(f'    {label} (first 22 / middle / last 22): ' + ' '.join(str(int(x)) for x in v[:22]) + ' ... ' + str(int(v[len(v) // 2])) + ' ... ' + ' '.join(str(int(x)) for x in v[-22:]))
    show(row, f'row y={cy}')
    show(col, f'col x={(x0 + x1) // 2}')
