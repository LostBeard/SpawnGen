"""Writes the small input folder the dataset node tests load (Fixtures/dataset/set), the same files ComfyUI loads for the references.

    python make_inputs.py

Pictures of odd sizes (so every resize and crop rounds), RGB, an RGBA one, a near twin of one picture (for Deduplicate Images), an upper-case
extension, a 16-bit grey PNG (Pillow opens it as I;16 and clips it to 255 when converting to RGB), captions with a CRLF, a BOM and Unicode case
pairs, a file that is not a picture, a kohya-style "3_kohya" folder (loaded 3 times) and a plain sub folder.
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "set")


def picture(w, h, channels, seed):
    r = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 7) % 256] + ([((x * 13 + y * 5) % 256)] if channels == 4 else []), -1)
    noise = r.integers(-40, 41, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def save(rel, arr):
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


def text(rel, data: bytes):
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(data)


a = picture(37, 23, 3, 1)
save("a.png", a)
twin = a.copy()
twin[5, 7] = 255 - twin[5, 7]
twin[11, 30] = 0
save("a_twin.png", twin)
save("B.PNG", picture(20, 31, 3, 2))
save("c.png", picture(24, 18, 4, 3))
g16 = (np.arange(9 * 7, dtype=np.uint16).reshape(7, 9) * 1000 + 3)
Image.fromarray(g16).save(os.path.join(OUT, "g16.png"))
save("3_kohya/k.png", picture(12, 14, 3, 4))
save("plain/p.png", picture(10, 10, 3, 5))

text("a.txt", "  A Photo of a Stra\u00dfe\r\nwith \u03a3\u0391\u03a3 and \u0130stanbul  \n".encode("utf-8"))
text("c.txt", "\ufeffcat \ufb01sh\t ".encode("utf-8"))
text("3_kohya/k.txt", "kohya caption".encode("utf-8"))
text("notes.md", b"not a picture")
print(sorted(os.listdir(OUT)))
