"""Writes the Load Image decode corpus: one small picture per PNG / JPEG case Load Image must read as ComfyUI does.

    <comfyui-venv>/python make_corpus.py <out folder>

The pictures are crops of the shipped examples/example.png (made with FLUX.2 Klein 4B, Apache-2.0: docs/EXAMPLE-PICTURE.md) at an odd size, so
row padding, partial MCUs and partial Adam7 passes are all exercised. Pillow (MIT-CMU licence) only WRITES the files here; the expected values
come from ComfyUI's Load Image, measured black box by run_comfy.py. Pillow cannot write interlaced PNGs, so those are written by png_raw()
below from the PNG specification (ISO/IEC 15948, Adam7 in section 8.2).
"""
import io, os, struct, sys, zlib
from PIL import Image

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, '..', '..', 'src', 'SpawnGen.Studio', 'wwwroot', 'examples', 'example.png')
W, H = 97, 61
base = Image.open(SOURCE).convert('RGB').crop((200, 140, 200 + W, 140 + H))  # the coat and the face: colour and detail
alpha = Image.linear_gradient('L').resize((W, H)).rotate(90, expand=False)  # a smooth ramp: every alpha level


def save(img, name, **kw):
    img.save(os.path.join(OUT, name), **kw)


def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)


def png_raw(name, width, height, depth, color_type, rows, interlace, plte=None, trns=None):
    """A PNG from sample rows (bytes, packed as the PNG spec packs them), filter type 0, Adam7 when interlace."""
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    bits = depth * channels

    def unpack(row):  # row bytes -> list of per-pixel sample tuples (as ints)
        if bits >= 8:
            step = bits // 8
            return [row[i * step:(i + 1) * step] for i in range(width)]
        per = 8 // bits
        return [(row[i // per] >> (8 - bits * (i % per + 1))) & ((1 << bits) - 1) for i in range(width)]

    def pack(pixels):
        if bits >= 8:
            return b''.join(pixels)
        out = bytearray((len(pixels) * bits + 7) // 8)
        for i, v in enumerate(pixels):
            out[i * bits // 8] |= v << (8 - bits * (i % (8 // bits) + 1))
        return bytes(out)

    grid = [unpack(r) for r in rows]
    raw = bytearray()
    if not interlace:
        for r in rows:
            raw += b'\x00' + r
    else:
        for x0, y0, dx, dy in ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)):
            for y in range(y0, height, dy):
                px = [grid[y][x] for x in range(x0, width, dx)]
                if px:
                    raw += b'\x00' + pack(px)
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, depth, color_type, 0, 0, 1 if interlace else 0))
    if plte:
        data += chunk(b'PLTE', plte)
    if trns:
        data += chunk(b'tRNS', trns)
    data += chunk(b'IDAT', zlib.compress(bytes(raw), 9)) + chunk(b'IEND', b'')
    open(os.path.join(OUT, name), 'wb').write(data)


# ---- PNG through Pillow ----
save(base, 'png_rgb8.png')
rgba = base.copy(); rgba.putalpha(alpha); save(rgba, 'png_rgba8.png')
save(base.convert('L'), 'png_gray8.png')
la = base.convert('L'); la = Image.merge('LA', (la, alpha)); save(la, 'png_gray_alpha8.png')
pal = base.quantize(colors=256); save(pal, 'png_palette8.png')
pal_t = base.quantize(colors=256); save(pal_t, 'png_palette8_trns.png', transparency=bytes([0, 64, 128, 255] * 64))
for colours, bits in ((2, 1), (4, 2), (16, 4)):
    save(base.quantize(colors=colours), f'png_palette{bits}.png', bits=bits)
save(base.convert('1'), 'png_gray1.png')
save(base, 'png_rgb8_trns.png', transparency=tuple(base.getpixel((10, 10))))
save(base.convert('L'), 'png_gray8_trns.png', transparency=base.convert('L').getpixel((10, 10)))
g16 = Image.new('I;16', (W, H)); g16.putdata([v * 257 for v in base.convert('L').getdata()]); save(g16, 'png_gray16.png')
g16big = Image.new('I;16', (W, H)); g16big.putdata([min(v * 9, 65535) for v in base.convert('L').getdata()]); save(g16big, 'png_gray16_low_values.png')

# ---- PNG written here: 16-bit colour, low-depth grey, and interlaced (Adam7) ----
def rows_of(img, mode_bytes):
    return [b''.join(mode_bytes(img.getpixel((x, y))) for x in range(W)) for y in range(H)]

rgb_rows = rows_of(base, lambda p: bytes(p))
png_raw('png_rgb16.png', W, H, 16, 2, [b''.join(struct.pack('>HHH', *(c * 257 + (x % 7) for c in base.getpixel((x, y)))) for x in range(W)) for y in range(H)], False)
rgba_img = rgba
png_raw('png_rgba16.png', W, H, 16, 6, [b''.join(struct.pack('>HHHH', *(c * 257 + 3 for c in rgba_img.getpixel((x, y)))) for x in range(W)) for y in range(H)], False)
grey = base.convert('L')
for bits in (2, 4):
    levels = (1 << bits) - 1
    vals = [[grey.getpixel((x, y)) * levels // 255 for x in range(W)] for y in range(H)]
    per = 8 // bits
    rows = []
    for y in range(H):
        row = bytearray((W * bits + 7) // 8)
        for x, v in enumerate(vals[y]):
            row[x // per] |= v << (8 - bits * (x % per + 1))
        rows.append(bytes(row))
    png_raw(f'png_gray{bits}.png', W, H, bits, 0, rows, False)
png_raw('png_rgb8_interlaced.png', W, H, 8, 2, rgb_rows, True)
png_raw('png_rgba8_interlaced.png', W, H, 8, 6, rows_of(rgba, lambda p: bytes(p)), True)
png_raw('png_gray8_interlaced.png', W, H, 8, 0, rows_of(grey, lambda p: bytes([p])), True)
pal_img = base.quantize(colors=16)
plte = bytes(pal_img.getpalette()[:16 * 3])
idx = [[pal_img.getpixel((x, y)) for x in range(W)] for y in range(H)]
pal_rows = []
for y in range(H):
    row = bytearray((W * 4 + 7) // 8)
    for x, v in enumerate(idx[y]):
        row[x // 2] |= v << (4 if x % 2 == 0 else 0)
    pal_rows.append(bytes(row))
png_raw('png_palette4_interlaced.png', W, H, 4, 3, pal_rows, True, plte=plte)
png_raw('png_rgb16_interlaced.png', W, H, 16, 2, [b''.join(struct.pack('>HHH', *(c * 257 for c in base.getpixel((x, y)))) for x in range(W)) for y in range(H)], True)

# ---- JPEG ----
save(base, 'jpeg_420.jpg', quality=90, subsampling=2)
save(base, 'jpeg_422.jpg', quality=90, subsampling=1)
save(base, 'jpeg_444.jpg', quality=90, subsampling=0)
save(base.convert('L'), 'jpeg_gray.jpg', quality=90)
save(base, 'jpeg_progressive_420.jpg', quality=90, subsampling=2, progressive=True)
save(base, 'jpeg_progressive_444.jpg', quality=90, subsampling=0, progressive=True)
save(base, 'jpeg_q50_420.jpg', quality=50, subsampling=2)
save(base, 'jpeg_restart_420.jpg', quality=90, subsampling=2, restart_marker_blocks=3)
save(base.convert('CMYK'), 'jpeg_cmyk.jpg', quality=90)
r, g, b = base.split()
save(Image.merge('CMYK', (r, g, b, base.convert('L').point(lambda v: 255 - v))), 'jpeg_cmyk_k.jpg', quality=90)  # a real K channel (convert('CMYK') leaves K at 0)
for orientation in range(1, 9):
    exif = Image.Exif(); exif[0x0112] = orientation
    save(base, f'jpeg_orientation{orientation}.jpg', quality=90, subsampling=2, exif=exif.tobytes())
exif = Image.Exif(); exif[0x0112] = 6
save(rgba, 'png_rgba8_orientation6.png', exif=exif.tobytes())

print(len(os.listdir(OUT)), 'files in', OUT)
