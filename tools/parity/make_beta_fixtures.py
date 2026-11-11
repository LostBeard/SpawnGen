# Reference values for SpawnGen's BetaMath (the beta CDF and quantile) from scipy.stats.beta (BSD-3), run in _scratch/comfyui-venv:
#
#   _scratch/comfyui-venv/Scripts/python.exe -I tools/parity/make_beta_fixtures.py
#
# Output: tests/SpawnGen.Nodes.Tests/Fixtures/beta/beta_scipy.json. The quantiles include the beta scheduler's (1 - linspace(0, 1, steps)) for
# alpha = beta = 0.6, the default of ComfyUI's "beta" scheduler, and a spread of other shapes.
import json, os, sys
import numpy as np
from scipy import stats
cases = []
shapes = [(0.6, 0.6), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (2.0, 5.0), (5.0, 1.5), (0.3, 3.0), (10.0, 10.0), (0.1, 0.1), (50.0, 2.0)]
for a, b in shapes:
    qs = sorted(set([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999] + list(1 - np.linspace(0, 1, 21))))
    for q in qs:
        cases.append({"a": a, "b": b, "q": float(q), "ppf": float(stats.beta.ppf(q, a, b))})
    for x in [1e-6, 0.001, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 0.999]:
        cases.append({"a": a, "b": b, "x": x, "cdf": float(stats.beta.cdf(x, a, b))})
out = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "..", "..", "tests", "SpawnGen.Nodes.Tests", "Fixtures", "beta", "beta_scipy.json")
json.dump({"scipy": __import__("scipy").__version__, "cases": cases}, open(out, "w"), indent=0)
print(len(cases), "cases")
