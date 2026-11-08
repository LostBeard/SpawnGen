"""Makes small WebP, GIF and BMP pictures with Pillow and runs ComfyUI's own Load Image (nodes.LoadImage) on each; writes the pictures, what
Load Image returned (IMAGE and MASK) and Pillow's RGBA of every frame (libwebp's / Pillow's own decode: the exact pixels) to load_image.json.gz.

    python make_load_image.py <ComfyUI folder>

ComfyUI's Load Image decodes through PyAV (ffmpeg) first and falls back to Pillow for what PyAV cannot open (animated WebP); every frame of the
first frame's size becomes one picture of the batch. PyAV converts most pixel formats through swscale to float, 1 to 2.4 / 255 away from the
decoded values, so SpawnGen takes the batch and mask layout from ComfyUI and the pixels from the reference decoders.
"""
import gzip, json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import folder_paths  # noqa: E402
import nodes  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 40, 24


def frame(k, alpha):
    """A gradient with a moving square; with alpha, a transparent band and a half-transparent one (not for GIF, which has one transparent index)."""
    y, x = np.mgrid[0:H, 0:W]
    rgb = np.stack([(x * 6 + k * 40) % 256, (y * 10) % 256, ((x + y) * 4 + k * 70) % 256], -1).astype(np.uint8)
    rgb[4:12, 4 + 8 * k:12 + 8 * k] = [250, 30, 30]
    if not alpha:
        return Image.fromarray(rgb, "RGB")
    a = np.full((H, W), 255, np.uint8)
    a[:, W - 6:] = 0
    a[H - 5:, :] = 128
    return Image.fromarray(np.dstack([rgb, a]), "RGBA")


def gif_frames():
    out = []
    for k in range(3):
        p = frame(k, False).quantize(colors=63, dither=Image.Dither.NONE)
        px = np.array(p)
        px[:, W - 6:] = 63  # the transparent index
        q = Image.fromarray(px, "P")
        q.putpalette(p.getpalette()[:63 * 3] + [0, 0, 0])
        out.append(q)
    return out


def make():
    files = {}
    frame(0, True).save(os.path.join(HERE, "still_lossless.webp"), lossless=True)
    frame(0, False).save(os.path.join(HERE, "still_lossy.webp"), quality=80, method=4)
    frame(0, True).save(os.path.join(HERE, "still_lossy_alpha.webp"), quality=80, method=4)
    f = [frame(k, True) for k in range(3)]
    f[0].save(os.path.join(HERE, "anim_lossless.webp"), save_all=True, append_images=f[1:], duration=100, lossless=True)
    g = gif_frames()
    g[0].save(os.path.join(HERE, "anim.gif"), save_all=True, append_images=g[1:], duration=100, loop=0, transparency=63, disposal=2)
    frame(1, False).save(os.path.join(HERE, "still.bmp"))
    return ["still_lossless.webp", "still_lossy.webp", "still_lossy_alpha.webp", "anim_lossless.webp", "anim.gif", "still.bmp"]


def tensor(t):
    t = t.float().cpu()
    return {"shape": list(t.shape), "data": [round(float(v), 7) for v in t.flatten()]}


if __name__ == "__main__":
    names = make()
    folder_paths.set_input_directory(HERE)
    out = {}
    for n in names:
        image, mask = nodes.LoadImage().load_image(n)
        frames = []
        with Image.open(os.path.join(HERE, n)) as im:
            for k in range(getattr(im, "n_frames", 1)):
                im.seek(k)
                frames.append(list(np.array(im.convert("RGBA")).flatten().tolist()))
        out[n] = {"image": tensor(image), "mask": tensor(mask), "pillow_rgba": frames}
        print(n, tuple(image.shape), tuple(mask.shape))
    with gzip.open(os.path.join(HERE, "load_image.json.gz"), "wt") as fh:
        json.dump(out, fh)
