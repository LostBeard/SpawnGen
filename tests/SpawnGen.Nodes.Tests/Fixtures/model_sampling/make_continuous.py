"""Dumps ComfyUI's continuous model samplings (ModelSamplingContinuousEDM / ModelSamplingContinuousV as its nodes build them) to continuous.json:
the sigma table, every scheduler's sigmas for 10 steps, timestep / sigma / percent_to_sigma, and calculate_input / calculate_denoised / noise_scaling.

    python make_continuous.py <ComfyUI folder>
"""
import json, sys
import torch

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import comfy.model_sampling as ms  # noqa: E402
import comfy.samplers  # noqa: E402

configs = {
    "edm_v": (ms.ModelSamplingContinuousEDM, ms.V_PREDICTION, 120.0, 0.002, 1.0),
    "edm_edm": (ms.ModelSamplingContinuousEDM, ms.EDM, 120.0, 0.002, 0.5),
    "edm_eps": (ms.ModelSamplingContinuousEDM, ms.EPS, 80.0, 0.03, 1.0),
    "cont_v": (ms.ModelSamplingContinuousV, ms.V_PREDICTION, 500.0, 0.03, 1.0),
}
x = torch.linspace(-1.5, 1.5, 12)
out = {}
for name, (base, kind, smax, smin, sdata) in configs.items():
    class Advanced(base, kind):
        pass
    s = Advanced(None)
    s.set_parameters(smin, smax, sdata)
    d = {"sigma_max": smax, "sigma_min": smin, "sigma_data": sdata,
         "table": [float(v) for v in s.sigmas[[0, 1, 2, 499, 500, 501, 997, 998, 999]]],
         "schedules": {}, "timestep": {}, "sigma": {}, "percent": {}}
    for sch in comfy.samplers.SCHEDULER_NAMES:
        d["schedules"][sch] = [float(v) for v in comfy.samplers.calculate_sigmas(s, sch, 10)]
    for sig in [0.002, 0.03, 0.5, 1.0, 7.25, 80.0, 120.0]:
        d["timestep"][repr(sig)] = float(s.timestep(torch.tensor(sig)))
    for t in [-1.5, -0.25, 0.0, 0.3, 0.9, 1.19]:
        d["sigma"][repr(t)] = float(s.sigma(torch.tensor(t)))
    for p in [0.0, 0.1, 0.25, 0.5, 0.9, 1.0]:
        d["percent"][repr(p)] = float(s.percent_to_sigma(p))
    sig = torch.tensor([3.5])
    out_model = torch.linspace(0.7, -0.4, 12)
    d["input"] = [float(v) for v in s.calculate_input(sig, x.clone())]
    d["denoised"] = [float(v) for v in s.calculate_denoised(sig, out_model.clone(), x.clone())]
    d["noise_scaling"] = [float(v) for v in s.noise_scaling(sig, out_model.clone(), x.clone(), max_denoise=True)]
    out[name] = d
json.dump(out, open("continuous.json", "w"), indent=1)
print(json.dumps({k: v["schedules"]["karras"][:3] for k, v in out.items()}))
