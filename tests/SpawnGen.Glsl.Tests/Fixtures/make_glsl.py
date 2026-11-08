"""Renders ComfyUI's GLSL blueprint shaders (and SpawnGen's own test shaders in this folder, *.frag) with ComfyUI's own renderer
(comfy_extras/nodes_glsl.py _render_shader_batch: OpenGL ES 3.0 through EGL / ANGLE) and writes the inputs and outputs to glsl.json.gz.

    python make_glsl.py <ComfyUI folder>

The blueprint shaders are ComfyUI's (GPL): they are read from <ComfyUI folder>/blueprints at run time and are not stored here; the fixture holds only
the test picture, the uniforms and ComfyUI's output pixels (float32, rows top to bottom, RGBA, base64). The tests read the same blueprints.
"""
import base64, glob, gzip, json, os, sys
import numpy as np

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
from comfy_extras.nodes_glsl import _render_shader_batch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 48, 32


def picture():
    """A test picture with gradients, a hard edge and saturated patches (RGB, float32, rows top to bottom)."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.stack([x / (W - 1), y / (H - 1), 0.5 + 0.5 * np.sin(x * 0.4) * np.cos(y * 0.3)], -1)
    img[8:16, 10:22] = [0.9, 0.1, 0.15]
    img[18:26, 28:40] = [0.1, 0.8, 0.3]
    img[:, 44:] = [0.05, 0.05, 0.9]
    return img.astype(np.float32)


def curves():
    t = np.linspace(0, 1, 256, dtype=np.float32)
    return [t ** 0.7, np.clip(t * 1.2 - 0.1, 0, 1), t ** 1.5, 0.5 - 0.5 * np.cos(t * np.pi)]


def blueprint_shader(name):
    with open(os.path.join(sys.argv[1], "blueprints", name + ".json"), encoding="utf-8") as fh:
        d = json.load(fh)
    found = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("type") == "GLSLShader" and o.get("widgets_values"):
                found.append(o["widgets_values"][0])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    return found[0]


# (case name, blueprint name or own shader file, floats, ints, bools, use curves)
CASES = [
    ("brightness_contrast", "Brightness and Contrast", [0.3, 0.4], [], [], False),
    ("chromatic_aberration", "Chromatic Aberration", [6.0], [1], [], False),
    ("color_adjustment", "Color Adjustment", [0.2, 0.3, -0.1, 0.5], [], [], False),
    ("color_balance", "Color Balance", [0.1, -0.2, 0.3, 0.2, 0.1, -0.1, -0.3, 0.2, 0.1], [], [True], False),
    ("color_balance_raw", "Color Balance", [0.1, -0.2, 0.3, 0.2, 0.1, -0.1, -0.3, 0.2, 0.1], [], [False], False),
    ("color_curves", "Color Curves", [], [], [], True),
    ("edge_preserving_blur", "Edge-Preserving Blur", [3.0, 0.2], [1], [], False),
    ("film_grain_smooth", "Film Grain", [0.6, 1.3, 0.5, 0.3], [0], [], False),
    ("film_grain_grainy", "Film Grain", [0.6, 1.0, 0.5, 0.3], [1], [], False),
    ("glow", "Glow", [0.6, 0.5, 0.3], [0, 1], [], False),
    ("glow_screen", "Glow", [0.8, 0.4, 0.2], [2, 0], [], False),
    ("hue_saturation", "Hue and Saturation", [30.0, 20.0, 10.0, 0.0], [0, 0], [], False),
    ("hue_saturation_reds", "Hue and Saturation", [-40.0, 30.0, -10.0, 0.0], [1, 0], [], False),
    ("hue_saturation_colorize", "Hue and Saturation", [200.0, 50.0, 10.0, 0.0], [7, 0], [], False),
    ("blur_gaussian", "Image Blur", [4.0], [0], [], False),
    ("blur_box", "Image Blur", [3.0], [1], [], False),
    ("blur_radial", "Image Blur", [60.0], [2], [], False),
    ("image_channels", "Image Channels", [], [], [], False),
    ("image_levels", "Image Levels", [0.1, 0.9, 1.3, 0.05, 0.95], [0], [], False),
    ("sharpen", "Sharpen", [0.8], [], [], False),
    ("unsharp_mask", "Unsharp Mask", [1.0, 2.0, 0.05], [], [], False),
]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


if __name__ == "__main__":
    img = picture()
    lut = curves()
    out = {"width": W, "height": H, "picture": b64(img), "curves": [b64(c) for c in lut], "cases": []}
    own = sorted(glob.glob(os.path.join(HERE, "*.frag")))
    cases = list(CASES) + [("own_" + os.path.basename(f)[:-5], os.path.basename(f), [0.25, 0.5, 0.75], [3, -2], [True], False) for f in own]
    for name, source, floats, ints, bools, use_curves in cases:
        code = open(os.path.join(HERE, source), encoding="utf-8").read() if source.endswith(".frag") else blueprint_shader(source)
        outputs = _render_shader_batch(code, W, H, [[img]], floats, ints, bools, lut if use_curves else [])[0]
        out["cases"].append({"name": name, "shader": source, "floats": floats, "ints": ints, "bools": bools, "curves": use_curves,
                             "outputs": [b64(o) for o in outputs]})
        print(name, [round(float(o.mean()), 4) for o in outputs])
    with gzip.open(os.path.join(HERE, "glsl.json.gz"), "wt") as fh:
        json.dump(out, fh)
