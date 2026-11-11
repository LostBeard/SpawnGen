"""Writes src/SpawnGen.Nodes.Core/Resources/sdr_to_hdr_gain.bin: ComfyUI's Convert SDR to HDR (ImageSDRToHDR) measured black-box on grey ramps.

    python -I -W ignore make_sdr_to_hdr_table.py <ComfyUI checkout> <output .bin>

Measured (2026-11-11): the node maps each pixel to HLG(M (lin rgb) * gain(Y, target)), where lin is the sRGB decoding of the input clamped to 0..1,
Y = 0.2126 R + 0.7152 G + 0.0722 B of lin, M is BT.2087's Rec.709 to Rec.2020 matrix and HLG the BT.2100 OETF. Only gain is ComfyUI's own calibrated
curve, and no description of it was found, so it is tabulated here: for each target luminance knot and each 8-bit sRGB level k of a grey, gain =
HLG^-1(output) / Y(k / 255) (level 0 repeats level 1, where the curve is linear). SpawnGen interpolates between levels and knots.

Only the node's public execute entry point is called. Layout (little-endian): int32 rows, int32 columns (256), float32 knots[rows] (target luminance
in nits; the first is 0, the calibrated rendering), float32 gain[rows][columns].
"""
import contextlib, importlib, io, os, struct, sys

src, out_path = sys.argv[1], os.path.abspath(sys.argv[2])
sys.path.insert(0, os.path.abspath(src))
os.chdir(src)
sys.argv = [sys.argv[0], "--cpu"]
quiet = io.StringIO()
with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
    import numpy as np
    import torch
    import comfy.options
    comfy.options.enable_args_parsing()
    images = importlib.import_module("comfy_extras.nodes_images")

A = 0.17883277
B = 1 - 4 * A
C = 0.5 - A * np.log(4 * A)


def hlg_inverse(e):
    return np.where(e <= 0.5, e * e / 3, (np.exp((e - C) / A) + B) / 12)


def srgb_decode(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


levels = np.arange(256) / 255.0
grey = torch.from_numpy(levels.astype(np.float32)).view(1, 1, 256, 1).expand(1, 1, 256, 3).contiguous()
luminance = srgb_decode(levels)

# 0 (calibrated), 84 knots from 0.001 to 1000 nits and 16 from 1000 to 10000 equally spaced in log, plus the default 650. 1000 is a knot on both sides
# because the curve clips at the HLG reference peak there.
knots = [0.0] + sorted(set([round(float(x), 6) for x in np.exp(np.linspace(np.log(0.001), np.log(1000.0), 84))]
                           + [round(float(x), 6) for x in np.exp(np.linspace(np.log(1000.0), np.log(10000.0), 16))] + [650.0]))

rows = []
for t in knots:
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        res = images.ImageSDRToHDR.execute(image=grey, algorithm={"algorithm": "lottes", "target_luminance": t})
    val = getattr(res, "result", None) or getattr(res, "args", None) or res
    e = hlg_inverse(val[0][0, 0, :, 0].double().numpy())
    gain = np.empty(256)
    gain[1:] = e[1:] / luminance[1:]
    gain[0] = gain[1]
    rows.append(gain)

with open(out_path, "wb") as f:
    f.write(struct.pack("<ii", len(knots), 256))
    f.write(np.asarray(knots, dtype="<f4").tobytes())
    f.write(np.asarray(rows, dtype="<f4").tobytes())
print(f"{len(knots)} knots x 256 levels -> {out_path}")
