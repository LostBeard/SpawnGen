# Black-box fixture for SpawnGen's Load Image and Load Image (as Mask): small pictures in several PNG and JPEG modes, and what ComfyUI's nodes make of
# them (IMAGE, MASK and each channel mask), or the error type and message when ComfyUI fails on one, with Pillow's RGBA of each file. Written for the clean-room rewrite
# (docs/CLEANROOM.md): ComfyUI is run, not read; tracebacks are suppressed.
#
#   cd _scratch; comfyui-venv/Scripts/python.exe -I -W ignore probe/harness3.py comfyui-ref ../tools/parity/make_image_node_fixtures.py
#
# The harness puts ComfyUI on sys.path and gives this script np, torch, contextlib, quiet, importlib and exp (this file's path). Output:
# tests/SpawnGen.Nodes.Tests/Fixtures/image_nodes/load_image_modes.json.gz (the files themselves are stored in it, base64).
import base64, gzip, json, struct, tempfile, zlib
from PIL import Image

N = importlib.import_module("nodes"); FP = importlib.import_module("folder_paths")
d = tempfile.mkdtemp(); FP.set_input_directory(d)
rng = np.random.default_rng(3)
px = rng.integers(0, 256, (5, 7, 4), dtype=np.uint8)

def save(name, img, **kw): img.save(os.path.join(d, name), **kw)
save("rgba.png", Image.fromarray(px, "RGBA"))
save("rgb.png", Image.fromarray(px[..., :3].copy(), "RGB"))
save("l.png", Image.fromarray(px[..., 0].copy(), "L"))
save("la.png", Image.merge("LA", [Image.fromarray(px[..., 0].copy()), Image.fromarray(px[..., 3].copy())]))
pal = [255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 9, 9]
p = Image.fromarray(px[..., 0].copy() % 4, "P"); p.putpalette(pal); save("palette_trns.png", p, transparency=bytes([255, 0, 128, 255]))
p = Image.fromarray(px[..., 0].copy() % 4, "P"); p.putpalette(pal); save("palette.png", p)
c16 = rng.integers(0, 65536, (5, 7, 4)).astype(np.uint16)
def png16(name, arr, ct):
    h, w, c = arr.shape
    raw = b"".join(b"\x00" + arr[y].astype(">u2").tobytes() for y in range(h))
    ch = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xffffffff)
    open(os.path.join(d, name), "wb").write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, ct, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b""))
png16("rgb16.png", c16[..., :3], 2); png16("rgba16.png", c16, 6); png16("grey16.png", c16[..., :1], 0)
Image.fromarray(px[..., 0].copy(), "L").save(os.path.join(d, "grey.jpg"), quality=100)
stripes = np.repeat(np.arange(6, dtype=np.uint8)[None, :, None] * 40, 4, 0).repeat(3, 2)
ex = Image.Exif(); ex[0x0112] = 6
Image.fromarray(stripes, "RGB").save(os.path.join(d, "rotated6.jpg"), exif=ex, quality=100, subsampling=0)

def tensor(t): return {"shape": list(t.shape), "data": [float(v) for v in t.flatten()]}
def run(f):
    try:
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet): return f()
    except BaseException as e: return f"{type(e).__name__}: {e}"

LM = N.LoadImageMask
out = {}
for name in sorted(os.listdir(d)):
    case = {"file": base64.b64encode(open(os.path.join(d, name), "rb").read()).decode()}
    r = run(lambda: N.LoadImage().load_image(name))
    if isinstance(r, str): case["error"] = r
    else: case["image"], case["mask"] = tensor(r[0]), tensor(r[1])
    for ch in ["alpha", "red", "green", "blue"]:
        m = run(lambda: getattr(LM(), LM.FUNCTION)(name, ch))
        case["mask_" + ch] = m if isinstance(m, str) else tensor(m[0])
    # Pillow's view of the file, for the cases ComfyUI fails on: RGBA8, and a 16-bit grey picture's high bytes (what SpawnGen loads).
    with Image.open(os.path.join(d, name)) as im:
        case["pillow_rgba"] = np.asarray(im.convert("RGBA")).flatten().tolist() if im.mode not in ("I;16", "I;16B", "I") else None
    if name == "grey16.png": case["high_byte"] = (c16[..., 0] >> 8).flatten().tolist()
    out[name] = case
    print(name, case.get("error") or (case["image"]["shape"], case["mask"]["shape"]))

target = os.path.join(os.path.dirname(exp), "..", "..", "tests", "SpawnGen.Nodes.Tests", "Fixtures", "image_nodes")
os.makedirs(target, exist_ok=True)
with gzip.open(os.path.join(target, "load_image_modes.json.gz"), "wt", encoding="utf-8") as f: json.dump(out, f)
