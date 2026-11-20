"""Writes the spec-correct IMAGE / MASK Load Image should give for every file of the decode corpus (TJ 2026-11-20: picture decoding is held
to correctness, not to ComfyUI's FFmpeg-based Load Image; its measured differences are listed in docs/PARITY.md).

    <python with numpy + Pillow> make_reference.py <corpus folder>

PNG: decoded here from the PNG specification (ISO/IEC 15948: zlib, the five filters, Adam7), not by Pillow: a sample of bit depth b is
s / (2^b - 1) (16-bit kept at 16-bit), a palette entry / 255, grey repeated to RGB; transparency from the alpha channel, the palette's tRNS,
or a grey / RGB tRNS colour (those pixels fully transparent); the eXIf orientation applied (TIFF/EXIF tag 274). JPEG: libjpeg-turbo through
Pillow, with the EXIF orientation applied in full (mirrors too) and CMYK converted by Pillow; Load Image's JPEG decoder is held to it
within the JPEG standard's freedom (inverse-DCT rounding, chroma upsampling), as tests/SpawnGen.Nodes.Tests/JpegTests does.
The MASK follows Load Image's (ComfyUI's) convention: 1 - alpha at full size when the picture has transparency (a palette picture always
counts), else an empty 64 x 64 mask.

Output: <corpus>/reference/<name>.image and .mask (uint8, k/255) or .image.f32 / .mask.f32 (float32) when a value is not a whole 8-bit
step, and reference.json with the shapes and which files are exact (PNG) or held within a tolerance (JPEG).
"""
import io, json, os, struct, sys, zlib
import numpy as np
from PIL import Image, ImageOps

CORPUS = sys.argv[1]
OUT = os.path.join(CORPUS, 'reference')
os.makedirs(OUT, exist_ok=True)
ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def unfilter(data, pos, width, height, bpp_bits):
    """Rows of one (sub)image starting at data[pos]: returns (rows as bytes, next pos)."""
    stride = (width * bpp_bits + 7) // 8
    bpp = max(1, bpp_bits // 8)
    prev = bytearray(stride); rows = []
    for _ in range(height):
        ft = data[pos]; line = bytearray(data[pos + 1:pos + 1 + stride]); pos += 1 + stride
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0; b = prev[i]; c = prev[i - bpp] if i >= bpp else 0
            line[i] = (line[i] + (0, a, b, (a + b) // 2, paeth(a, b, c))[ft]) & 255
        rows.append(bytes(line)); prev = line
    return rows, pos


def samples(row, width, depth, channels):
    if depth == 8: return np.frombuffer(row, np.uint8)[:width * channels].reshape(width, channels).astype(np.int64)
    if depth == 16: return np.frombuffer(row, '>u2')[:width * channels].reshape(width, channels).astype(np.int64)
    bits = np.unpackbits(np.frombuffer(row, np.uint8))
    vals = bits[:width * depth].reshape(width, depth)
    return (vals * (1 << np.arange(depth - 1, -1, -1))).sum(1).reshape(width, 1).astype(np.int64)


def png(path):
    d = open(path, 'rb').read(); p = 8; idat = b''; plte = trns = exif = None
    while p < len(d):
        n, t = struct.unpack('>I4s', d[p:p + 8]); body = d[p + 8:p + 8 + n]
        if t == b'IHDR': w, h, depth, ct, _, _, il = struct.unpack('>IIBBBBB', body)
        elif t == b'PLTE': plte = body
        elif t == b'tRNS': trns = body
        elif t == b'IDAT': idat += body
        elif t == b'eXIf': exif = body
        p += 12 + n
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ct]
    raw = zlib.decompress(idat)
    grid = np.zeros((h, w, channels), np.int64)
    if not il:
        rows, _ = unfilter(raw, 0, w, h, depth * channels)
        for y, r in enumerate(rows): grid[y] = samples(r, w, depth, channels)
    else:
        pos = 0
        for x0, y0, dx, dy in ADAM7:
            pw, ph = len(range(x0, w, dx)), len(range(y0, h, dy))
            if pw == 0 or ph == 0: continue
            rows, pos = unfilter(raw, pos, pw, ph, depth * channels)
            for j, r in enumerate(rows): grid[y0 + j * dy, x0::dx] = samples(r, pw, depth, channels)
    top = (1 << depth) - 1
    alpha = None
    if ct == 3:
        pal = np.frombuffer(plte, np.uint8).reshape(-1, 3).astype(np.float64) / 255
        idx = grid[..., 0]
        rgb = pal[idx]
        pa = np.ones(256);
        if trns: pa[:len(trns)] = np.frombuffer(trns, np.uint8) / 255
        alpha = pa[idx]  # a palette picture always has transparency (fully opaque without tRNS)
    else:
        colour = grid[..., :3] if ct in (2, 6) else np.repeat(grid[..., :1], 3, axis=2)
        rgb = colour.astype(np.float64) / top
        if ct in (4, 6): alpha = grid[..., -1].astype(np.float64) / top
        elif trns:
            key = np.array(struct.unpack('>' + 'H' * (len(trns) // 2), trns), np.int64)
            match = np.all(grid[..., :len(key)] == key, axis=2)
            alpha = np.where(match, 0.0, 1.0)
    orientation = 1
    if exif:
        im = Image.new('L', (1, 1)); ex = Image.Exif(); ex.load(exif); orientation = ex.get(0x0112, 1)
    return rgb, alpha, orientation


def orient(a, o):
    """EXIF orientation 1..8 (TIFF tag 274) applied to an H x W (x C) array."""
    if a is None: return None
    return {1: lambda x: x, 2: lambda x: x[:, ::-1], 3: lambda x: x[::-1, ::-1], 4: lambda x: x[::-1],
            5: lambda x: np.swapaxes(x, 0, 1), 6: lambda x: np.swapaxes(x, 0, 1)[:, ::-1],
            7: lambda x: np.swapaxes(x, 0, 1)[::-1, ::-1], 8: lambda x: np.swapaxes(x, 0, 1)[::-1]}[o](a)


def write(name, what, a):
    k = np.rint(a * 255)
    if np.abs(a * 255 - k).max() < 1e-6:
        k.astype(np.uint8).tofile(os.path.join(OUT, f'{name}.{what}')); return False
    a.astype('<f4').tofile(os.path.join(OUT, f'{name}.{what}.f32')); return True


manifest = {}
for name in sorted(n for n in os.listdir(CORPUS) if n.lower().endswith(('.png', '.jpg', '.jpeg'))):
    path = os.path.join(CORPUS, name)
    if name.endswith('.png'):
        rgb, alpha, o = png(path); exact = True
        rgb, alpha = orient(rgb, o), orient(alpha, o)
    else:
        im = ImageOps.exif_transpose(Image.open(path)); exact = False
        rgb = np.array(im.convert('RGB')).astype(np.float64) / 255; alpha = None
    mask = (1 - alpha) if alpha is not None else np.zeros((64, 64))
    entry = {'height': rgb.shape[0], 'width': rgb.shape[1], 'mask_height': mask.shape[0], 'mask_width': mask.shape[1], 'exact': exact}
    if write(name, 'image', rgb): entry['image_float'] = True
    if write(name, 'mask', mask): entry['mask_float'] = True
    manifest[name] = entry
json.dump(manifest, open(os.path.join(OUT, 'reference.json'), 'w'), indent=1, sort_keys=True)
print(len(manifest), 'references in', OUT)
