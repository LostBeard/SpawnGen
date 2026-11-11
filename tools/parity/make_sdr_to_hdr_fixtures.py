"""Writes tests/SpawnGen.Nodes.Tests/Fixtures/comfy/comfy_image_nodes_sdr_hdr.json.gz: ComfyUI's Convert SDR to HDR (ImageSDRToHDR) on chosen
pictures, for the edges the main fixture does not reach: values outside 0..1, saturated colours at high targets, one and four channels, targets
between the gain table's knots, above 1000 nits and below 0.1.

    python -I -W ignore make_sdr_to_hdr_fixtures.py <ComfyUI checkout> <output .json.gz>

Only the node's public execute entry point is called; exceptions are recorded as their type and message.
"""
import contextlib, gzip, importlib, io, json, os, sys

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

rng = np.random.default_rng(20261111)
tensors, cases = [], []


def add(a):
    a = np.asarray(a, dtype=np.float32)
    tensors.append({"shape": list(a.shape), "data": [float(v) for v in a.ravel().tolist()]})
    return len(tensors) - 1


def run(name, img, target):
    ii = add(img)
    try:
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            res = images.ImageSDRToHDR.execute(image=torch.from_numpy(np.asarray(img, np.float32)), algorithm={"algorithm": "lottes", "target_luminance": target})
        val = getattr(res, "result", None) or getattr(res, "args", None) or res
        out = {"shape": list(val[0].shape), "data": [float(v) for v in val[0].numpy().ravel().tolist()]}
    except BaseException as e:  # noqa: BLE001
        out = {"error": f"{type(e).__name__}: {e}"}
    cases.append({"node": "ImageSDRToHDR", "args": {"target_luminance": target, "case": name}, "inputs": {"image": ii}, "out": [out]})


q = lambda a: np.round(np.asarray(a) * 1024) / 1024
pic = q(rng.random((1, 12, 12, 3)))
sat = np.zeros((1, 2, 6, 3)); sat[0, 0, 0] = [1, 0, 0]; sat[0, 0, 1] = [0, 1, 0]; sat[0, 0, 2] = [0, 0, 1]; sat[0, 0, 3] = [1, 1, 0]
sat[0, 0, 4] = [0, 1, 1]; sat[0, 0, 5] = [1, 0, 1]; sat[0, 1] = q(rng.random((6, 3)))
for t in [0.0, 0.05, 0.1, 3.7, 47.0, 123.4, 650.0, 999.0, 1000.0, 1001.0, 1700.0, 6543.0, 10000.0]:
    run(f"random t={t}", pic, t)
for t in [650.0, 1000.0, 2500.0, 10000.0]:
    run(f"saturated t={t}", sat, t)
run("out of range", q(rng.uniform(-0.5, 1.5, (1, 6, 6, 3))), 650.0)
run("one channel", q(rng.random((2, 5, 5, 1))), 400.0)
run("four channels", q(rng.random((1, 5, 5, 4))), 800.0)
run("two channels", q(rng.random((1, 5, 5, 2))), 650.0)

with gzip.open(out_path, "wt", encoding="utf-8") as f:
    json.dump({"tensors": tensors, "cases": cases}, f, separators=(",", ":"))
print(f"{len(cases)} cases -> {out_path}")
