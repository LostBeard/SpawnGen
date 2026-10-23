"""Reference values for SpawnGen's sampler, guidance and scheduler tests, computed by ComfyUI's own code.

    python -I make_reference.py <ComfyUI checkout> reference.json      (needs torch, numpy and scipy, nothing else)

ComfyUI's modules (comfy/k_diffusion/*.py, comfy/extra_samplers/uni_pc.py, comfy/model_sampling.py, the scheduler and cfg functions of comfy/samplers.py
and the comfy_extras node files) are loaded from the checkout with small stand-ins for what they import (tqdm, torchsde, the model patcher, the node API).
They are then driven by a stand-in denoiser and a recorded noise sequence, so every number in the output is what ComfyUI computes; the C# tests replay
the same denoiser and noise and compare.
"""
import ast
import importlib.util
import json
import math
import os
import sys
import types

import numpy
import scipy.stats
import torch

COMFY = sys.argv[1]
OUT = sys.argv[2]
torch.set_grad_enabled(False)


def path(*parts):
    return os.path.join(COMFY, *parts)


def module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    parent, _, child = name.rpartition('.')
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, m)
    return m


def package(name):
    m = module(name)
    m.__path__ = []
    return m


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    parent, _, child = name.rpartition('.')
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, m)
    spec.loader.exec_module(m)
    return m


def extract(file, names, namespace):
    """Runs the named top-level functions / classes of a source file in the namespace (the real code, without the file's imports)."""
    tree = ast.parse(open(file, encoding='utf-8').read())
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
            name = node.name if not isinstance(node, ast.Assign) else getattr(node.targets[0], 'id', None)
            if name in names:
                exec(compile(ast.Module([node], []), file, 'exec'), namespace)
                found.add(name)
    missing = set(names) - found
    if missing:
        raise RuntimeError(f'{file}: {missing} not found')
    return namespace


# --- stand-ins for what the ComfyUI files import ------------------------------------------------------------------------------------------

class _Tqdm:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def update(self, n=1): pass


package('tqdm')
module('tqdm.auto', trange=lambda *a, **k: range(*a), tqdm=_Tqdm)
module('torchsde', BrownianTree=None)
comfy = package('comfy')
for p in ['comfy.ldm', 'comfy.ldm.modules', 'comfy.ldm.modules.diffusionmodules', 'comfy.k_diffusion', 'comfy.extra_samplers']:
    package(p)
ns = {'torch': torch, 'np': numpy, 'math': math}
extract(path('comfy', 'ldm', 'modules', 'diffusionmodules', 'util.py'), ['make_beta_schedule'], ns)
module('comfy.ldm.modules.diffusionmodules.util', make_beta_schedule=ns['make_beta_schedule'])
module('comfy.memory_management', aimdo_enabled=False)
module('comfy.utils', model_trange=lambda *a, **k: range(*a))
mp = extract(path('comfy', 'model_patcher.py'), ['set_model_options_post_cfg_function', 'set_model_options_pre_cfg_function'], {})
module('comfy.model_patcher', **{k: v for k, v in mp.items() if not k.startswith('__')})
ku = extract(path('comfy', 'k_diffusion', 'utils.py'), ['append_dims'], {'torch': torch})
module('comfy.k_diffusion.utils', append_dims=ku['append_dims'])
model_sampling = load('comfy.model_sampling', path('comfy', 'model_sampling.py'))
load('comfy.k_diffusion.deis', path('comfy', 'k_diffusion', 'deis.py'))
sa_solver = load('comfy.k_diffusion.sa_solver', path('comfy', 'k_diffusion', 'sa_solver.py'))
sampling = load('comfy.k_diffusion.sampling', path('comfy', 'k_diffusion', 'sampling.py'))
uni_pc = load('comfy.extra_samplers.uni_pc', path('comfy', 'extra_samplers', 'uni_pc.py'))

# The parts of comfy/samplers.py used here: schedulers, cfg_function / sampling_function and the KSAMPLER wrappers.
SAMP = {'torch': torch, 'math': math, 'numpy': numpy, 'scipy': scipy, 'k_diffusion_sampling': sampling, 'uni_pc': uni_pc, 'comfy': comfy, 'partial': __import__('functools').partial}
extract(path('comfy', 'samplers.py'), ['simple_scheduler', 'ddim_scheduler', 'normal_scheduler', 'beta_scheduler', 'linear_quadratic_schedule',
                                       'kl_optimal_scheduler', 'cfg_function', 'sampling_function', 'Sampler', 'KSAMPLER', 'ksampler', 'KSAMPLER_NAMES'], SAMP)


class CFGGuiderStub:
    def __init__(self, model_patcher):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.cfg = 1.0


samplers_mod = module('comfy.samplers', CFGGuider=CFGGuiderStub, cfg_function=SAMP['cfg_function'], ksampler=SAMP['ksampler'],
                      KSAMPLER=SAMP['KSAMPLER'], encode_model_conds=lambda fn, conds, x, device, kind: conds)
module('comfy.sampler_helpers', convert_cond=lambda c: c)
module('comfy.model_management')
module('node_helpers', conditioning_set_values=lambda c, v: c)


class _Anything:
    def __init__(self, *a, **k): pass
    def __getattr__(self, name): return _Anything()
    def __call__(self, *a, **k): return _Anything()
    def __class_getitem__(cls, item): return cls


io = types.ModuleType('io')
io.ComfyNode = object
io.NodeOutput = lambda *a, **k: a[0] if len(a) == 1 else a
io.__getattr__ = lambda name: _Anything
package('comfy_api')
module('comfy_api.latest', ComfyExtension=object, io=io)


def node_module(name):
    return load('comfy_extras.' + name, path('comfy_extras', name + '.py'))


package('comfy_extras')

# --- the stand-in model -------------------------------------------------------------------------------------------------------------------

C, H, W = 2, 3, 4
SHAPE = (1, C, H, W)
N = C * H * W
idx = torch.arange(N, dtype=torch.float64)
MU = {'C': (0.3 * torch.sin(1.7 * idx + 0.2)).float().reshape(SHAPE),
      'U': (-0.2 * torch.cos(0.9 * idx)).float().reshape(SHAPE),
      'M': (0.1 * torch.sin(0.5 * idx + 1.0)).float().reshape(SHAPE),
      'E': (0.05 * torch.cos(1.3 * idx)).float().reshape(SHAPE)}
AMP = {'C': 0.05, 'U': 0.03, 'M': 0.04, 'E': 0.02}


def denoiser(key, x, sigma):
    """A Gaussian-data denoiser (data N(mu, 1)) plus a small nonlinear term, elementwise; the C# tests compute the same."""
    s = sigma.reshape(-1, 1, 1, 1).to(torch.float64)
    xd = x.to(torch.float64)
    mu = MU[key].to(torch.float64)
    k = 1.0 / (1.0 + s * s)
    f = torch.tanh(xd) if key in ('C', 'M') else torch.sin(xd)
    return (mu + k * (xd - mu) + AMP[key] * f * s / (1.0 + s)).float()


def fake_calc_cond_batch(model, conds, x, timestep, model_options):
    return [torch.zeros_like(x) if c is None else denoiser(c, x, timestep) for c in conds]


SAMP['calc_cond_batch'] = fake_calc_cond_batch
samplers_mod.calc_cond_batch = fake_calc_cond_batch


def make_sampling(kind='eps', zsnr=False):
    base = model_sampling.ModelSamplingDiscrete
    types_ = {'eps': model_sampling.EPS, 'v_prediction': model_sampling.V_PREDICTION, 'x0': model_sampling.X0,
              'img_to_img': model_sampling.IMG_TO_IMG, 'img_to_img_flow': model_sampling.IMG_TO_IMG_FLOW}
    cls = type('MS', (base, types_[kind]), {})
    return cls(None, zsnr=zsnr)


class InnerModel:
    """What sampling_function / cfg_function get as `model` (TSR reads current_patcher, PerpNeg extra_conds)."""
    def __init__(self, ms):
        self.current_patcher = types.SimpleNamespace(get_model_object=lambda n: ms)
        self.extra_conds = None
        self.model_sampling = ms


class FakeModel:
    """The KSamplerX0Inpaint-shaped model a k-diffusion sampler is given: each call runs ComfyUI's sampling_function over the stand-in denoiser."""
    def __init__(self, ms, cfg, cond='C', uncond='U', guider=None):
        self.ms = ms
        self.cfg = cfg
        self.cond, self.uncond = cond, uncond
        self.guider = guider
        self.calls = 0
        inner = InnerModel(ms)
        self.inner = inner
        self.inner_model = types.SimpleNamespace(inner_model=types.SimpleNamespace(model_sampling=ms),
                                                 model_patcher=types.SimpleNamespace(get_model_object=lambda n: ms))

    def __call__(self, x, sigma, model_options={}, seed=None, **kw):
        self.calls += 1
        if self.guider is not None:
            return self.guider(x, sigma, model_options)
        return SAMP['sampling_function'](self.inner, x, sigma, self.uncond, self.cond, self.cfg, model_options=model_options, seed=seed)


class Draws:
    """The noise draws in order (torch.randn from a fixed generator), recorded so the C# side can replay them."""
    def __init__(self, seed):
        self.g = torch.Generator().manual_seed(seed)
        self.list = []

    def draw(self):
        t = torch.randn(SHAPE, generator=self.g)
        self.list.append(t)
        return t

    def sampler(self, sigma, sigma_next):
        return self.draw()


class Brownian:
    """SpawnGen's Brownian path over sigma (the stand-in for torchsde's tree), from the same draws: see Samplers.BrownianNoise."""
    def __init__(self, draws, smin, smax):
        self.draws = draws
        first = draws.draw()
        self.points = {float(smin): torch.zeros(SHAPE)}
        if smax > smin:
            self.points[float(smax)] = first * math.sqrt(float(smax) - float(smin))

    def at(self, tau):
        tau = float(tau)
        if tau in self.points:
            return self.points[tau]
        keys = sorted(self.points)
        z = self.draws.draw()
        hi = 0
        while hi < len(keys) and keys[hi] < tau:
            hi += 1
        if hi == 0:
            w = self.points[keys[0]] + z * math.sqrt(keys[0] - tau)
        elif hi == len(keys):
            w = self.points[keys[-1]] + z * math.sqrt(tau - keys[-1])
        else:
            a, b = keys[hi - 1], keys[hi]
            frac, std = (tau - a) / (b - a), math.sqrt((tau - a) * (b - tau) / (b - a))
            w = self.points[a] + frac * (self.points[b] - self.points[a]) + std * z
        self.points[tau] = w
        return w

    def __call__(self, sigma, sigma_next):
        w0 = self.at(sigma)
        w1 = self.at(sigma_next)
        return (w1 - w0) / math.sqrt(abs(float(sigma_next) - float(sigma)))


def flat(t):
    return [float(v) for v in t.reshape(-1)]


def sampler_function(name, extra):
    if name == 'ddim':
        return SAMP['ksampler']('euler').sampler_function, {}
    if name == 'uni_pc':
        return uni_pc.sample_unipc, {}
    if name == 'uni_pc_bh2':
        return uni_pc.sample_unipc_bh2, {}
    ks = SAMP['ksampler'](name, extra)
    return ks.sampler_function, ks.extra_options


CASES = []


def run_case(case_name, sampler_name, sigmas, cfg=2.0, extra=None, kind='eps', seed=1, func=None, model_options=None, guider=None):
    ms = make_sampling(kind)
    extra = dict(extra or {})
    draws = Draws(seed)
    gx = torch.Generator().manual_seed(1000 + seed)
    x0 = (torch.randn(SHAPE, generator=gx) * float(sigmas[0])).float()
    model = FakeModel(ms, cfg, guider=guider)
    if func is None:
        func, extra = sampler_function(sampler_name, extra)
    extra_args = {'model_options': dict(model_options or {}), 'seed': 0}
    callbacks = []
    sig = torch.tensor(sigmas, dtype=torch.float32)
    kwargs = dict(extra)
    if sampler_name in ('dpmpp_sde', 'dpmpp_sde_gpu'):
        positive = sig[sig > 0]
        kwargs['noise_sampler'] = Brownian(draws, positive.min(), sig.max())
    elif sampler_name not in ('uni_pc', 'uni_pc_bh2', 'dpm_fast', 'dpm_adaptive') and 'noise_sampler' in __import__('inspect').signature(func).parameters:
        kwargs['noise_sampler'] = draws.sampler
    elif sampler_name in ('dpm_adaptive',):
        kwargs['noise_sampler'] = draws.sampler
    x = func(model, x0.clone(), sig.clone(), extra_args=extra_args, callback=lambda d: callbacks.append(d['denoised']), disable=True, **kwargs)
    CASES.append({'name': case_name, 'sampler': sampler_name, 'kind': kind, 'cfg': cfg, 'sigmas': [float(v) for v in sig], 'x': flat(x0),
                  'draws': [flat(d) for d in draws.list], 'expected': flat(x), 'calls': model.calls,
                  'last_denoised': flat(callbacks[-1]) if callbacks else None, 'options': {k: v for k, v in (extra or {}).items() if isinstance(v, (int, float, str, bool))}})


karras8 = [float(v) for v in sampling.get_sigmas_karras(8, 0.0291675, 14.614642)]
karras4 = [float(v) for v in sampling.get_sigmas_karras(4, 0.0291675, 14.614642)]
partial = [14.614642, 6.0, 2.5, 1.1, 0.4]  # a schedule that stops above 0 (a split)

for name in SAMP['KSAMPLER_NAMES'] + ['ddim', 'uni_pc', 'uni_pc_bh2']:
    run_case(name, name, karras8)
    if name not in ('dpm_fast', 'dpm_adaptive'):
        run_case(name + '/partial', name, partial, seed=2)

# cfg 1: the CFG++ samplers still ask for the negative (disable_cfg1_optimization); the others skip it.
for name in ['euler_cfg_pp', 'euler_ancestral_cfg_pp', 'dpmpp_2s_ancestral_cfg_pp', 'dpmpp_2m_cfg_pp', 'res_multistep_cfg_pp', 'res_multistep_ancestral_cfg_pp',
             'gradient_estimation_cfg_pp', 'cfgpp_ud10_ab', 'euler', 'dpmpp_2m']:
    run_case(name + '/cfg1', name, karras8, cfg=1.0, seed=3)

# Option variants (what the sampler nodes pass).
run_case('euler_ancestral/eta', 'euler_ancestral', karras8, extra={'eta': 0.5, 's_noise': 0.8})
run_case('dpm_2_ancestral/eta', 'dpm_2_ancestral', karras8, extra={'eta': 0.7, 's_noise': 1.1})
run_case('dpmpp_2s_ancestral/eta', 'dpmpp_2s_ancestral', karras8, extra={'eta': 0.6, 's_noise': 0.9})
run_case('dpmpp_2m_sde/heun', 'dpmpp_2m_sde', karras8, extra={'eta': 0.8, 's_noise': 1.0, 'solver_type': 'heun'})
run_case('dpmpp_3m_sde/eta', 'dpmpp_3m_sde', karras8, extra={'eta': 0.5, 's_noise': 0.7})
run_case('dpmpp_sde/r', 'dpmpp_sde', karras8, extra={'eta': 0.7, 's_noise': 0.9, 'r': 0.3})
run_case('lms/order2', 'lms', karras8, extra={'order': 2})
run_case('lms/order6', 'lms', karras8, extra={'order': 6})
run_case('dpm_adaptive/order2', 'dpm_adaptive', karras8, extra={'order': 2, 'rtol': 0.05, 'atol': 0.0078, 'h_init': 0.05})
run_case('dpm_adaptive/eta', 'dpm_adaptive', karras8, extra={'order': 3, 'eta': 0.5, 's_noise': 1.0, 'pcoeff': 0.1, 'dcoeff': 0.05})
run_case('seeds_2/phi_2', 'seeds_2', karras8, extra={'eta': 0.6, 's_noise': 0.9, 'r': 0.7, 'solver_type': 'phi_2'})
run_case('lcm/ramp', 'lcm', karras8, extra={'s_noise': 1.3, 's_noise_end': 0.4, 'noise_clip_std': 1.5})
run_case('res_multistep_ancestral/eta', 'res_multistep_ancestral', karras8, extra={'eta': 0.4, 's_noise': 1.2})

# The sampler nodes themselves (their options and noise scalers), run through ComfyUI's node code.
module('comfy.sample')
module('comfy.patcher_extension')
module('latent_preview')
nodes_cs = node_module('nodes_custom_sampler')


def run_node(case_name, node_sampler, sigmas, cfg=2.0, seed=4, widgets=None):
    run_case(case_name, 'node', sigmas, cfg=cfg, seed=seed, func=node_sampler.sampler_function, extra=node_sampler.extra_options)
    CASES[-1]['widgets'] = widgets


ms_eps = make_sampling('eps')
run_node('SamplerER_SDE/er', nodes_cs.SamplerER_SDE.execute('ER-SDE', 3, 0.7, 1.0), karras8, widgets=['ER-SDE', 3, 0.7, 1.0])
run_node('SamplerER_SDE/reverse', nodes_cs.SamplerER_SDE.execute('Reverse-time SDE', 2, 0.5, 0.8), karras8, widgets=['Reverse-time SDE', 2, 0.5, 0.8])
run_node('SamplerER_SDE/ode', nodes_cs.SamplerER_SDE.execute('ODE', 3, 1.0, 1.0), karras8, widgets=['ODE', 3, 1.0, 1.0])


class MSHolder:
    def __init__(self, ms):
        self.ms = ms
        self.model_options = {}

    def get_model_object(self, name):
        return self.ms


run_node('SamplerSASolver/interval', nodes_cs.SamplerSASolver.execute(MSHolder(ms_eps), 0.8, 0.1, 0.7, 1.0, 3, 4, False, False), karras8,
         widgets=[0.8, 0.1, 0.7, 1.0, 3, 4, False, False])
run_node('SamplerSASolver/pece_simple', nodes_cs.SamplerSASolver.execute(MSHolder(ms_eps), 1.0, 0.2, 0.8, 0.9, 2, 2, True, True), karras8,
         widgets=[1.0, 0.2, 0.8, 0.9, 2, 2, True, True])
run_node('SamplerSASolver/orders', nodes_cs.SamplerSASolver.execute(MSHolder(ms_eps), 1.0, 0.0, 1.0, 1.0, 4, 1, False, False), karras8,
         widgets=[1.0, 0.0, 1.0, 1.0, 4, 1, False, False])
nas = node_module('nodes_advanced_samplers')
run_node('SamplerEulerCFGpp/alternative', nas.SamplerEulerCFGpp.execute('alternative'), karras8, widgets=['alternative'])

# Flow (CONST) sampling: the RF forms of the ancestral samplers and the logit half-log-SNR.
flow = [0.95, 0.8, 0.6, 0.35, 0.1, 0.0]
for name in ['euler_ancestral', 'dpm_2_ancestral', 'dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_2m_sde', 'dpmpp_3m_sde', 'er_sde', 'seeds_2', 'seeds_3',
             'euler_ancestral_cfg_pp', 'sa_solver', 'res_multistep_ancestral', 'exp_heun_2_x0_sde']:
    run_case(name + '/flow', name, flow, kind='img_to_img_flow', seed=5)

# --- guidance: ComfyUI's patch nodes over a 4-step euler run, and the guiders ------------------------------------------------------------


class FakePatcher:
    """Enough of ModelPatcher for the patch nodes: clone, the model_options setters, get_model_object, wrappers."""
    def __init__(self, ms, options=None):
        self.ms = ms
        self.model_options = options if options is not None else {}
        self.model = types.SimpleNamespace(diffusion_model=types.SimpleNamespace(in_channels=C), model_sampling=ms)

    def clone(self):
        return FakePatcher(self.ms, {k: (list(v) if isinstance(v, list) else v) for k, v in self.model_options.items()})

    def get_model_object(self, name):
        return self.ms

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options = mp['set_model_options_post_cfg_function'](self.model_options, fn, disable_cfg1_optimization)

    def set_model_sampler_pre_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options = mp['set_model_options_pre_cfg_function'](self.model_options, fn, disable_cfg1_optimization)

    def set_model_sampler_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options['sampler_cfg_function'] = fn
        if disable_cfg1_optimization:
            self.model_options['disable_cfg1_optimization'] = True

    def set_model_denoise_mask_function(self, fn):
        self.model_options['denoise_mask_function'] = fn


GUIDE = []


def run_patch(case_name, patched, cfg=3.0, sigmas=karras4, sampler='euler', seed=6):
    run_case(case_name, sampler, sigmas, cfg=cfg, seed=seed, model_options=patched.model_options)
    CASES[-1]['patch'] = case_name


base = FakePatcher(ms_eps)
nm_cfg = node_module('nodes_cfg')
run_patch('CFGZeroStar', nm_cfg.CFGZeroStar.execute(base))
run_patch('CFGNorm', nm_cfg.CFGNorm.execute(base, 0.8, False))
run_patch('CFGNorm/pre', nm_cfg.CFGNorm.execute(base, 0.7, True))
nm_apg = node_module('nodes_apg')
run_patch('APG', nm_apg.APG.execute(base, 0.5, 3.0, 0.0))
run_patch('APG/momentum', nm_apg.APG.execute(base, 1.5, 0.5, -0.5))
run_patch('APG/cfg1', nm_apg.APG.execute(base, 0.5, 3.0, 0.3), cfg=1.0)
nm_mahiro = node_module('nodes_mahiro')
run_patch('Mahiro', nm_mahiro.Mahiro.execute(base))
nm_tcfg = node_module('nodes_tcfg')
run_patch('TCFG', nm_tcfg.TCFG.execute(base))
run_patch('TCFG/cfg1', nm_tcfg.TCFG.execute(base), cfg=1.0)
nm_fresca = node_module('nodes_fresca')
run_patch('FreSca', nm_fresca.FreSca.execute(base, 0.8, 1.4, 1))
run_patch('FreSca/wide', nm_fresca.FreSca.execute(base, 1.2, 0.9, 20))
nm_eps = node_module('nodes_eps')
run_patch('Epsilon Scaling', nm_eps.EpsilonScaling.execute(base, 1.02))
run_patch('TemporalScoreRescaling', nm_eps.TemporalScoreRescaling.execute(base, 0.9, 0.7))
nm_lumina = node_module('nodes_lumina2')
run_patch('RenormCFG', nm_lumina.RenormCFG.execute(base, 100.0, 1.0))
run_patch('RenormCFG/trunc', nm_lumina.RenormCFG.execute(base, 3.0, 0.5))
module('comfy.sd')
module('comfy.latent_formats')
module('comfy.ldm.modules.attention', COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE=False)
module('nodes', MAX_RESOLUTION=16384)
nm_adv = node_module('nodes_model_advanced')
run_patch('RescaleCFG', nm_adv.RescaleCFG().patch(base, 0.7)[0])
run_patch('RescaleCFG/full', nm_adv.RescaleCFG().patch(base, 1.0)[0], cfg=7.0)
nm_perp = node_module('nodes_perpneg')
run_patch('PerpNeg', nm_perp.PerpNeg.execute(base, 'E', 0.8))
# A chain: two post-cfg patches and a pre-cfg one, in order.
chain = nm_eps.EpsilonScaling.execute(nm_cfg.CFGZeroStar.execute(nm_tcfg.TCFG.execute(base)), 1.01)
run_patch('chain', chain)
run_patch('CFGZeroStar/dpmpp_2m_cfg_pp', nm_cfg.CFGZeroStar.execute(base), sampler='dpmpp_2m_cfg_pp')

# Guiders: Dual CFG (regular / nested) and Perp-Neg, through their real predict_noise.


def guider_case(case_name, guider, cfg_for_model=1.0, sampler='euler', seed=7, model_options=None):
    def call(x, sigma, model_options_):
        return guider.predict_noise(x, sigma, model_options_)
    run_case(case_name, sampler, karras4, cfg=cfg_for_model, seed=seed, guider=call, model_options=model_options)
    CASES[-1]['guider'] = case_name


for style, c1, c2 in [('regular', 3.0, 2.0), ('nested', 3.0, 2.0), ('regular', 1.0, 1.0), ('regular', 2.5, 1.0)]:
    g = nodes_cs.Guider_DualCFG(FakePatcher(ms_eps))
    g.set_cfg(c1, c2, nested=(style == 'nested'))
    g.conds = {'positive': 'C', 'middle': 'M', 'negative': 'U'}
    g.inner_model = InnerModel(ms_eps)
    guider_case(f'DualCFGGuider/{style}/{c1}/{c2}', g)
for cfg, neg_scale in [(4.0, 0.8), (1.0, 0.0), (3.0, 0.0)]:
    g = nm_perp.Guider_PerpNeg(FakePatcher(ms_eps))
    g.set_cfg(cfg, neg_scale)
    g.conds = {'positive': 'C', 'negative': 'U', 'empty_negative_prompt': 'E'}
    g.inner_model = InnerModel(ms_eps)
    guider_case(f'PerpNegGuider/{cfg}/{neg_scale}', g)
# A guider with a CFG++ sampler and a post-cfg patch: the patch sees the guider's estimates.
g = nm_perp.Guider_PerpNeg(FakePatcher(ms_eps))
g.set_cfg(3.0, 0.5)
g.conds = {'positive': 'C', 'negative': 'U', 'empty_negative_prompt': 'E'}
g.inner_model = InnerModel(ms_eps)
guider_case('PerpNegGuider/cfg_pp_zerostar', g, sampler='euler_cfg_pp', model_options=nm_cfg.CFGZeroStar.execute(base).model_options)

# Inpainting: ComfyUI's KSamplerX0Inpaint around the model (the kept area re-noised from the run's noise on the way in, pinned on the way out),
# optionally with Differential Diffusion's mask function, and KSAMPLER's noise scaling at the start and inverse at the end.
SAMP['comfy'] = comfy
extract(path('comfy', 'samplers.py'), ['KSamplerX0Inpaint'], SAMP)
dd = node_module('nodes_differential_diffusion')
INPAINT = []


def run_inpaint(case_name, sampler_name, sigmas, dd_strength=None, cfg=2.0, seed=8):
    ms = ms_eps
    sig = torch.tensor(sigmas, dtype=torch.float32)
    g = torch.Generator().manual_seed(2000 + seed)
    latent = torch.randn(SHAPE, generator=g) * 0.7
    noise = torch.randn(SHAPE, generator=g)
    plane = torch.tensor([1.0, 1.0, 0.0, 0.5, 0.25, 1.0, 0.0, 0.75, 1.0, 0.1, 0.9, 0.0]).reshape(1, 1, H, W)
    mask = plane.repeat(1, C, 1, 1)
    model = FakeModel(ms, cfg)
    inner = types.SimpleNamespace(inner_model=types.SimpleNamespace(
        model_sampling=ms, scale_latent_inpaint=lambda x, sigma, noise, latent_image, denoise_mask: ms.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (noise.ndim - 1)), noise, latent_image)),
        model_patcher=types.SimpleNamespace(get_model_object=lambda n: ms))

    class Inner:
        inner_model = inner.inner_model
        model_patcher = inner.model_patcher

        def __call__(self, x, sigma, model_options={}, seed=None):
            return model(x, sigma, model_options=model_options, seed=seed)

    model_k = SAMP['KSamplerX0Inpaint'](Inner(), sig)
    model_k.latent_image = latent
    model_k.noise = noise
    max_denoise = math.isclose(float(ms.sigma_max), float(sig[0]), rel_tol=1e-05) or float(sig[0]) > float(ms.sigma_max)
    x0 = ms.noise_scaling(sig[0], noise, latent, max_denoise)
    options = {}
    if dd_strength is not None:
        options['denoise_mask_function'] = lambda *a, **k: dd.DifferentialDiffusion.forward(*a, **k, strength=dd_strength)
    func, extra = sampler_function(sampler_name, {})
    draws = Draws(seed)
    kwargs = dict(extra)
    if 'noise_sampler' in __import__('inspect').signature(func).parameters:
        kwargs['noise_sampler'] = draws.sampler
    callbacks = []
    x = func(model_k, x0.clone(), sig.clone(), extra_args={'denoise_mask': mask, 'model_options': options, 'seed': 0},
             callback=lambda d: callbacks.append(d['denoised']), disable=True, **kwargs)
    x = ms.inverse_noise_scaling(sig[-1], x)
    INPAINT.append({'name': case_name, 'sampler': sampler_name, 'sigmas': [float(v) for v in sig], 'cfg': cfg, 'latent': flat(latent), 'noise': flat(noise),
                    'mask': flat(plane), 'dd_strength': dd_strength, 'draws': [flat(d) for d in draws.list], 'expected': flat(x),
                    'last_denoised': flat(callbacks[-1]), 'calls': model.calls})


run_inpaint('inpaint/euler', 'euler', karras8)
run_inpaint('inpaint/euler_ancestral', 'euler_ancestral', karras8)
run_inpaint('inpaint/dpmpp_2m/partial', 'dpmpp_2m', partial)
run_inpaint('inpaint/differential', 'euler', karras8, dd_strength=1.0)
run_inpaint('inpaint/differential_soft', 'dpmpp_2m', karras8, dd_strength=0.6)
run_inpaint('inpaint/differential_partial', 'heun', partial, dd_strength=1.0)

# --- schedulers --------------------------------------------------------------------------------------------------------------------------

SCHED = []
for kind, zsnr in [('eps', False), ('eps', True)]:
    ms = make_sampling(kind, zsnr)
    for steps in [1, 2, 4, 10, 25]:
        out = {
            'simple': SAMP['simple_scheduler'](ms, steps),
            'sgm_uniform': SAMP['normal_scheduler'](ms, steps, sgm=True),
            'karras': sampling.get_sigmas_karras(n=steps, sigma_min=float(ms.sigma_min), sigma_max=float(ms.sigma_max)),
            'exponential': sampling.get_sigmas_exponential(n=steps, sigma_min=float(ms.sigma_min), sigma_max=float(ms.sigma_max)),
            'ddim_uniform': SAMP['ddim_scheduler'](ms, steps),
            'beta': SAMP['beta_scheduler'](ms, steps),
            'normal': SAMP['normal_scheduler'](ms, steps),
            'linear_quadratic': SAMP['linear_quadratic_schedule'](ms, steps),
            'kl_optimal': SAMP['kl_optimal_scheduler'](n=steps, sigma_min=float(ms.sigma_min), sigma_max=float(ms.sigma_max)),
        }
        for name, s in out.items():
            SCHED.append({'zsnr': zsnr, 'lcm': False, 'scheduler': name, 'steps': steps, 'sigmas': [float(v) for v in s]})
# LCM's distilled table (ModelSamplingDiscreteDistilled, as ModelSamplingDiscrete builds it).
nm_lcm = types.SimpleNamespace(**{k: getattr(nm_adv, k) for k in ['ModelSamplingDiscreteDistilled', 'LCM']})
ms_lcm = type('MSL', (nm_lcm.ModelSamplingDiscreteDistilled, nm_lcm.LCM), {})(None, zsnr=False)
for steps in [1, 4, 10]:
    for name, s in {'simple': SAMP['simple_scheduler'](ms_lcm, steps), 'normal': SAMP['normal_scheduler'](ms_lcm, steps),
                    'sgm_uniform': SAMP['normal_scheduler'](ms_lcm, steps, sgm=True), 'ddim_uniform': SAMP['ddim_scheduler'](ms_lcm, steps),
                    'beta': SAMP['beta_scheduler'](ms_lcm, steps)}.items():
        SCHED.append({'zsnr': False, 'lcm': True, 'scheduler': name, 'steps': steps, 'sigmas': [float(v) for v in s]})

MS_FACTS = {
    'sigma_min': float(ms_eps.sigma_min), 'sigma_max': float(ms_eps.sigma_max),
    'timesteps': [[s, float(ms_eps.timestep(torch.tensor(s)))] for s in [0.0292, 0.5, 1.0, 3.3333, 7.7, 14.6]],
    'percent': [[p, ms_eps.percent_to_sigma(p)] for p in [0.0, 0.1, 0.2, 0.5, 0.8, 0.999, 1.0]],
    'lcm_timesteps': [[s, float(ms_lcm.timestep(torch.tensor(s)))] for s in [0.0292, 0.5, 1.0, 3.3333, 14.6]],
    'lcm_sigma': [[t, float(ms_lcm.sigma(torch.tensor(float(t))))] for t in [19, 100, 500.5, 999]],
    'zsnr_sigma_max': float(make_sampling('eps', True).sigma_max),
}
# calculate_denoised of each prediction type on a sample.
xs = torch.linspace(-1.5, 1.7, N).reshape(SHAPE)
outs = torch.cos(torch.arange(N, dtype=torch.float32)).reshape(SHAPE)
PRED = []
for kind, ms in [('eps', make_sampling('eps')), ('v_prediction', make_sampling('v_prediction')), ('x0', make_sampling('x0')),
                 ('img_to_img', make_sampling('img_to_img')), ('img_to_img_flow', make_sampling('img_to_img_flow')), ('lcm', ms_lcm)]:
    for sigma in [0.3, 2.0, 9.0]:
        st = torch.tensor([sigma])
        PRED.append({'kind': kind, 'sigma': sigma, 'input': flat(ms.calculate_input(st, xs)), 'denoised': flat(ms.calculate_denoised(st, outs, xs)),
                     'noise_scaling': flat(ms.noise_scaling(st, outs, xs, False)), 'noise_scaling_max': flat(ms.noise_scaling(st, outs, xs, True))})

nm_ays = node_module('nodes_align_your_steps')
nm_os = node_module('nodes_optimalsteps')
nm_gits = load('comfy_extras.nodes_g' + 'its', path('comfy_extras', 'nodes_g' + 'its.py'))
TABLES = []
for mt in ['SD1', 'SDXL', 'SVD']:
    for steps, den in [(10, 1.0), (4, 1.0), (25, 1.0), (10, 0.5), (7, 0.35), (1, 1.0)]:
        TABLES.append({'node': 'AlignYourStepsScheduler', 'args': [mt, steps, den], 'sigmas': [float(v) for v in nm_ays.AlignYourStepsScheduler.execute(mt, steps, den)]})
for mt in ['FLUX', 'Wan', 'Chroma']:
    for steps, den in [(20, 1.0), (10, 1.0), (40, 0.5), (3, 1.0)]:
        TABLES.append({'node': 'OptimalStepsScheduler', 'args': [mt, steps, den], 'sigmas': [float(v) for v in nm_os.OptimalStepsScheduler.execute(mt, steps, den)]})
for coeff in [0.8, 1.2, 1.5, 1.05]:
    for steps, den in [(10, 1.0), (2, 1.0), (20, 1.0), (30, 1.0), (12, 0.6)]:
        TABLES.append({'node': 'GITSScheduler', 'args': [coeff, steps, den], 'sigmas': [float(v) for v in nm_gits.GITSScheduler.execute(coeff, steps, den)]})
for args in [(20, 14.614642, 0.0291675, 0.0, 0.5), (7, 10.0, 0.1, 0.5, 1.2), (1, 14.6, 0.03, 0.0, 0.5)]:
    TABLES.append({'node': 'LaplaceScheduler', 'args': list(args), 'sigmas': [float(v) for v in nodes_cs.LaplaceScheduler.execute(*args)]})
for args in [(20, 14.614642, 0.0291675, 7.0), (5, 3.0, 0.5, 2.5), (1, 14.6, 0.03, 7.0)]:
    TABLES.append({'node': 'KarrasScheduler', 'args': list(args), 'sigmas': [float(v) for v in nodes_cs.KarrasScheduler.execute(*args)]})
for args in [(20, 14.614642, 0.0291675), (3, 5.0, 0.2)]:
    TABLES.append({'node': 'ExponentialScheduler', 'args': list(args), 'sigmas': [float(v) for v in nodes_cs.ExponentialScheduler.execute(*args)]})
for args in [(20, 14.614642, 0.0291675, 1.0), (6, 9.0, 0.1, 2.0)]:
    TABLES.append({'node': 'PolyexponentialScheduler', 'args': list(args), 'sigmas': [float(v) for v in nodes_cs.PolyexponentialScheduler.execute(*args)]})
for args in [(20, 19.9, 0.1, 0.001), (4, 10.0, 0.5, 0.01)]:
    TABLES.append({'node': 'VPScheduler', 'args': list(args), 'sigmas': [float(v) for v in nodes_cs.VPScheduler.execute(*args)]})
for args in [(4, 1.0), (2, 0.5), (3, 0.7), (10, 1.0)]:
    TABLES.append({'node': 'SDTurboScheduler', 'args': list(args), 'sigmas': [float(v) for v in nodes_cs.SDTurboScheduler.execute(MSHolder(ms_eps), *args)]})
src = torch.tensor([14.6, 8.0, 4.0, 2.0, 1.0, 0.0])
for args in [(2, -1.0, 12.0, 'linear'), (3, -1.0, 3.0, 'cosine'), (4, 6.0, 1.5, 'sine'), (1, -1.0, 0.0, 'linear')]:
    TABLES.append({'node': 'ExtendIntermediateSigmas', 'args': list(args), 'input': flat(src),
                   'sigmas': [float(v) for v in nodes_cs.ExtendIntermediateSigmas.execute(src, *args)]})
for text in ['1, 0.5', '14.6 7.2 3 .5 0', '-2, +3.25,4', 'nothing', '1e-3, 2.5.5']:
    try:
        TABLES.append({'node': 'ManualSigmas', 'args': [text], 'sigmas': [float(v) for v in nodes_cs.ManualSigmas.execute(text)]})
    except ValueError:
        TABLES.append({'node': 'ManualSigmas', 'args': [text], 'error': True})
for args in [(0.0, False), (0.0, True), (1.0, False), (1.0, True), (0.25, False), (0.6, True)]:
    TABLES.append({'node': 'SamplingPercentToSigma', 'args': list(args), 'sigmas': [float(nodes_cs.SamplingPercentToSigma.execute(MSHolder(ms_eps), *args))]})
full = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 0.0])
for d in [1.0, 0.5, 0.0, 0.3, 0.1]:
    hi, lo = nodes_cs.SplitSigmasDenoise.execute(full, d)
    TABLES.append({'node': 'SplitSigmasDenoise', 'args': [d], 'input': flat(full), 'high': flat(hi), 'low': flat(lo)})
for st in [0, 2, 5, 9]:
    hi, lo = nodes_cs.SplitSigmas.execute(full, st)
    TABLES.append({'node': 'SplitSigmas', 'args': [st], 'input': flat(full), 'high': flat(hi), 'low': flat(lo)})

# KSampler's schedule (discarding the penultimate sigma for dpm_2 / dpm_2_ancestral / uni_pc) and its denoise rule.
KS = []
for sampler_name in ['euler', 'dpm_2', 'uni_pc_bh2']:
    for scheduler in ['normal', 'karras', 'beta', 'simple']:
        for steps, denoise in [(10, 1.0), (6, 0.6), (3, 0.3), (20, 0.99995)]:
            discard = sampler_name in ('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2')

            def calc(n):
                n2 = n + 1 if discard else n
                handlers = {'normal': lambda: SAMP['normal_scheduler'](ms_eps, n2), 'simple': lambda: SAMP['simple_scheduler'](ms_eps, n2),
                            'beta': lambda: SAMP['beta_scheduler'](ms_eps, n2),
                            'karras': lambda: sampling.get_sigmas_karras(n=n2, sigma_min=float(ms_eps.sigma_min), sigma_max=float(ms_eps.sigma_max))}
                s = handlers[scheduler]()
                return torch.cat([s[:-2], s[-1:]]) if discard else s
            if denoise > 0.9999:
                s = calc(steps)
            else:
                s = calc(int(steps / denoise))[-(steps + 1):]
            KS.append({'sampler': sampler_name, 'scheduler': scheduler, 'steps': steps, 'denoise': denoise, 'sigmas': flat(s)})

# Differential Diffusion's mask function on its own.
DD = []
mask = torch.linspace(0, 1, H * W).reshape(1, 1, H, W)
fake_model = types.SimpleNamespace(inner_model=types.SimpleNamespace(model_sampling=ms_eps))
for strength in [1.0, 0.6, 0.0]:
    for sig_list, sigma in [(karras8, karras8[0]), (karras8, karras8[3]), (karras8, karras8[6]), (partial, partial[2])]:
        m = dd.DifferentialDiffusion.forward(torch.tensor([sigma]), mask.clone(), {'model': fake_model, 'sigmas': torch.tensor(sig_list)}, strength)
        DD.append({'strength': strength, 'sigmas': sig_list, 'sigma': sigma, 'mask': flat(mask), 'result': flat(m)})

# FreSca's Fourier filter alone on a bigger plane (odd and even sides).
FF = []
for (hh, ww, cut, lo, hi) in [(8, 8, 2, 0.5, 1.5), (5, 6, 1, 1.0, 1.25), (7, 4, 20, 2.0, 0.3)]:
    g = torch.Generator().manual_seed(hh * 31 + ww)
    v = torch.randn((1, 2, hh, ww), generator=g)
    FF.append({'h': hh, 'w': ww, 'cutoff': cut, 'low': lo, 'high': hi, 'input': flat(v),
               'result': flat(nm_fresca.Fourier_filter(v, scale_low=lo, scale_high=hi, freq_cutoff=cut))})

def compact(v):
    """Floats to 9 significant digits (enough for float32); NaN and infinities as strings, which JSON has no numbers for."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return str(v)
        return float(f'{v:.9g}')
    if isinstance(v, dict):
        return {k: compact(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [compact(x) for x in v]
    return v


json.dump(compact({'shape': [C, H, W], 'mu': {k: flat(v) for k, v in MU.items()}, 'amp': AMP, 'cases': CASES, 'schedulers': SCHED, 'tables': TABLES,
           'ksampler': KS, 'inpaint': INPAINT, 'model_sampling': MS_FACTS, 'predictions': PRED, 'differential_diffusion': DD, 'fourier': FF}), open(OUT, 'w'), separators=(',', ':'))
print(f'{len(CASES)} sampler cases, {len(SCHED)} schedules, {len(TABLES)} table / sigma node results -> {OUT}')
