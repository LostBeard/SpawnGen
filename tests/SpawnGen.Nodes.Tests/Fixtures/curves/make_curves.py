"""Evaluates ComfyUI's CurveInput (comfy_api/latest/_input/curve_types.py: MonotoneCubicCurve, LinearCurve) on test point sets and writes each
to_lut(256) to curves.json, for SpawnGen's Curve.

    python make_curves.py <ComfyUI folder>
"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import nodes  # noqa: E402,F401  (initialises comfy_api in the order ComfyUI does)
from comfy_api.input import CurveInput  # noqa: E402

CASES = [
    {"points": [[0.0, 0.0], [1.0, 1.0]], "interpolation": "monotone_cubic"},
    {"points": [[0.0, 0.1], [0.25, 0.6], [0.5, 0.4], [0.75, 0.9], [1.0, 0.8]]},
    {"points": [[0.8, 0.9], [0.1, 0.05], [0.5, 0.5], [0.3, 0.45]], "interpolation": "monotone_cubic"},
    {"points": [[0.0, 0.0], [0.3, 0.0], [0.6, 1.0], [1.0, 1.0]]},
    {"points": [[0.0, 0.0], [0.05, 0.9], [1.0, 1.0]]},
    {"points": [[0.2, 0.2], [0.5, 0.3], [0.5, 0.7], [0.8, 0.8]]},
    {"points": [[0.4, 0.6]]},
    {"points": [[0.1, 0.2], [0.4, 0.9], [0.9, 0.3]], "interpolation": "linear"},
    {"points": [[0.0, 1.0], [0.5, 0.5], [0.5, 0.2], [1.0, 0.0]], "interpolation": "linear"},
    [[0.0, 0.2], [1.0, 0.7]],
]
out = []
for c in CASES:
    out.append({"curve": c, "lut": [float(v) for v in CurveInput.from_raw(c).to_lut(256)]})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves.json"), "w") as fh:
    json.dump(out, fh)
print(len(out), "curves")
