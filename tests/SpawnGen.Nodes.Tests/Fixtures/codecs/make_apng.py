"""Makes small animated PNGs with Pillow (blend over / source, dispose none / background / previous, RGBA and RGB, semi-transparent overlays, changed
sub-rectangles) and runs ComfyUI's own Load Image on each (it decodes through PyAV: ffmpeg's APNG decoder composites the frames); writes the files and
Load Image's IMAGE / MASK to apng/load_apng.json.gz.

    python make_apng.py <ComfyUI folder>
"""
import gzip, json, os, struct, sys, zlib
import numpy as np
from PIL import Image

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import folder_paths  # noqa: E402
import nodes  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apng")
os.makedirs(HERE, exist_ok=True)
W, H = 24, 16


def rgba_frames():
    y, x = np.mgrid[0:H, 0:W]
    base = np.zeros((H, W, 4), np.uint8)
    base[..., 0] = (x * 10) % 256
    base[..., 1] = (y * 15) % 256
    base[..., 2] = 90
    base[..., 3] = 255
    base[:, :4, 3] = 0             # a transparent strip
    base[:4, :, 3] = 160           # a half-transparent band
    out = [base]
    for k in range(1, 4):
        f = out[-1].copy()
        f[5:11, 3 + 5 * k:9 + 5 * k] = [240, 40 * k, 20, 120 + 40 * (k % 2)]  # a semi-transparent square that moves
        f[12:14, 2:6] = [10, 200, 30, 70 * k]
        out.append(f)
    return [Image.fromarray(a, "RGBA") for a in out]


def rgb_frames():
    return [Image.fromarray(np.asarray(f)[..., :3].copy(), "RGB") for f in rgba_frames()]


def probe(d0, d1):
    """A hand-built 4x1 RGBA APNG (Pillow chooses its own rectangles): three full frames, the first two disposed with d0 / d1, frames 2 and 3 blended over;
    each frame has a transparent, an opaque, a half-transparent and a transparent pixel, so every disposal pair shows how ffmpeg composites."""
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    def frame(rgb):
        px = np.zeros((1, 4, 4), np.uint8); px[..., :3] = rgb; px[..., 3] = [0, 255, 128, 0]
        return px
    frames = [(frame((10, 20, 30)), d0, 0), (frame((200, 100, 50)), d1, 1), (frame((90, 90, 90)), 0, 1)]
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 1, 8, 6, 0, 0, 0)) + chunk(b"acTL", struct.pack(">II", 3, 0))
    seq = 0
    for i, (px, dispose, blend) in enumerate(frames):
        out += chunk(b"fcTL", struct.pack(">IIIIIHHBB", seq, 4, 1, 0, 0, 1, 10, dispose, blend)); seq += 1
        z = zlib.compress(b"\x00" + px[0].tobytes())
        if i == 0: out += chunk(b"IDAT", z)
        else: out += chunk(b"fdAT", struct.pack(">I", seq) + z); seq += 1
    return out + chunk(b"IEND", b"")


PROBES = [f"probe_{d0}_{d1}.png" for d0 in range(3) for d1 in range(3)]

CASES = [
    ("over_none.png", rgba_frames, dict(blend=1, disposal=0)),
    ("over_background.png", rgba_frames, dict(blend=1, disposal=1)),
    ("over_previous.png", rgba_frames, dict(blend=1, disposal=2)),
    ("source_none.png", rgba_frames, dict(blend=0, disposal=0)),
    ("rgb.png", rgb_frames, dict(blend=0, disposal=0)),
]


def tensor(t):
    t = t.float().cpu()
    return {"shape": list(t.shape), "data": [round(float(v), 7) for v in t.flatten()]}


if __name__ == "__main__":
    for name, frames, opts in CASES:
        f = frames()
        f[0].save(os.path.join(HERE, name), save_all=True, append_images=f[1:], duration=100, loop=0, **opts)
    for name in PROBES:
        with open(os.path.join(HERE, name), "wb") as fh:
            fh.write(probe(int(name[6]), int(name[8])))
    folder_paths.set_input_directory(HERE)
    out = {}
    for name in [c[0] for c in CASES] + PROBES:
        image, mask = nodes.LoadImage().load_image(name)
        out[name] = {"image": tensor(image), "mask": tensor(mask)}
        print(name, tuple(image.shape), tuple(mask.shape))
    with gzip.open(os.path.join(HERE, "load_apng.json.gz"), "wt") as fh:
        json.dump(out, fh)
