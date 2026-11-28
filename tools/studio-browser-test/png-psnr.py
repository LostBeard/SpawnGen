# PSNR between two pictures of the same size (8-bit RGB, values compared as 0..1): python -I png-psnr.py ours.png theirs.png
import sys
import numpy as np
from PIL import Image

a = np.asarray(Image.open(sys.argv[1]).convert('RGB'), dtype=np.float64) / 255.0
b = np.asarray(Image.open(sys.argv[2]).convert('RGB'), dtype=np.float64) / 255.0
if a.shape != b.shape:
    print(f'sizes differ: {a.shape} vs {b.shape}')
    sys.exit(2)
mse = float(np.mean((a - b) ** 2))
psnr = float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)
print(f'{a.shape[1]} x {a.shape[0]}: PSNR {psnr:.1f} dB, largest difference {np.max(np.abs(a - b)) * 255:.0f}/255')
