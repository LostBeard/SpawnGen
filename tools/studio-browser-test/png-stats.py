"""Prints mode, size and a few pixel facts of PNG files: a reading aid for black-box measurements. Usage: python png-stats.py file.png [...]"""
import sys
from PIL import Image
import numpy as np

for f in sys.argv[1:]:
    im = Image.open(f)
    a = np.asarray(im)
    line = f'{f.split("/")[-1].split(chr(92))[-1]}: mode {im.mode} size {im.size[0]}x{im.size[1]}'
    if a.ndim == 2:
        a = a[:, :, None]
    for c in range(a.shape[2]):
        ch = a[:, :, c]
        line += f' | ch{c} min {ch.min()} max {ch.max()} mean {ch.mean():.3f}'
    h, w = a.shape[:2]
    pts = [(0, 0), (w // 2, h // 2), (w // 4, h // 4), (int(w * 0.375), int(h * 0.4)), (w // 2, int(h * 0.8))]
    line += ' | px ' + ' '.join(f'({x},{y})={tuple(int(v) for v in a[y, x])}' for x, y in pts)
    vals = np.unique(a.reshape(-1, a.shape[2]), axis=0)
    line += f' | {len(vals)} distinct colours'
    print(line)
