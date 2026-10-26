"""Runs ComfyUI's own Video Linear / Video Triangle CFG Guidance cfg functions (comfy_extras/nodes_video_model.py) on fixed batches and writes
the results to video_cfg.json.

    python make_video_cfg.py <ComfyUI folder>
"""
import json, sys
import torch

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
from comfy_extras.nodes_video_model import VideoLinearCFGGuidance, VideoTriangleCFGGuidance  # noqa: E402


class StandInModel:
    """Holds the cfg function the node sets, as ModelPatcher does."""
    def clone(self):
        return StandInModel()

    def set_model_sampler_cfg_function(self, fn):
        self.fn = fn


def values(t):
    return [round(float(v), 8) for v in t.flatten()]


cases = []
g = torch.Generator().manual_seed(7)
for node in (VideoLinearCFGGuidance, VideoTriangleCFGGuidance):
    for batch in (1, 2, 5, 8):
        for (cond_scale, min_cfg) in ((7.5, 1.0), (3.3, 2.5), (1.0, 4.0)):
            cond = torch.randn(batch, 4, 3, 2, generator=g)
            uncond = torch.randn(batch, 4, 3, 2, generator=g)
            fn = node().patch(StandInModel(), min_cfg)[0].fn
            out = fn({"cond": cond, "uncond": uncond, "cond_scale": cond_scale})
            cases.append({"node": node.__name__, "batch": batch, "cond_scale": cond_scale, "min_cfg": min_cfg,
                          "cond": values(cond), "uncond": values(uncond), "out": values(out)})

with open("video_cfg.json", "w") as f:
    json.dump(cases, f, separators=(",", ":"))
print(len(cases), "cases")
