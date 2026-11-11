# Black-box: the sigmas ComfyUI's GITS Scheduler puts out for every coeff and 2..40 steps (outputs only). Usage (from _scratch):
#   comfyui-venv/Scripts/python.exe -W ignore measure_gits.py comfyui-ref out.json
import contextlib, importlib, io, json, os, sys
src = os.path.abspath(sys.argv[1]); out = sys.argv[2]; sys.path.insert(0, src); os.chdir(src); sys.argv = [sys.argv[0], "--cpu"]
quiet = io.StringIO()
with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
    import torch
    import comfy.options
    comfy.options.enable_args_parsing()
    mod = importlib.import_module("comfy_extras.nodes_gits")
cls = getattr(mod, "NODE_CLASS_MAPPINGS", {}).get("GITSScheduler")
if cls is None:
    for v in vars(mod).values():
        try:
            if isinstance(v, type) and v.define_schema().node_id == "GITSScheduler": cls = v
        except Exception: pass
def run(coeff, steps, denoise=1.0):
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        if hasattr(cls, "FUNCTION"): r = getattr(cls(), cls.FUNCTION)(coeff=coeff, steps=steps, denoise=denoise)
        else: r = cls.execute(coeff=coeff, steps=steps, denoise=denoise)
    r = getattr(r, "result", None) or getattr(r, "args", None) or r
    return [float(v) for v in r[0].tolist()]
res = {}
try:
    for k in range(15):
        coeff = round(0.80 + 0.05 * k, 2)
        res[str(coeff)] = {str(n): run(coeff, n) for n in range(2, 41)}
except BaseException as e:
    print(type(e).__name__, str(e)[:120])
json.dump(res, open(out, "w"))
c = res.get("1.2", {})
for n in ("2", "3", "20", "21", "25"): print(n, len(c.get(n, [])), [round(v, 4) for v in c.get(n, [])][:6])
