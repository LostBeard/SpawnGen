"""Writes tests/SpawnGen.Nodes.Tests/Fixtures/comfy/comfy_audio_nodes.json.gz: black-box measurements of ComfyUI's model-free audio nodes (Trim Audio
Duration, Split / Join Audio Channels, Concatenate Audio, Merge Audio, Adjust Audio Volume, Empty Audio, Audio Equalizer (3-Band)), made by calling
the node classes of a ComfyUI checkout on chosen waveforms.

    python -I -W ignore make_audio_fixtures.py <ComfyUI checkout> <output .json.gz>

Only the nodes' public execute entry points are called. Exceptions are recorded as their type and message (never a traceback). Input samples are
multiples of 1/4096; tensors are stored as little-endian float32 in base64. An AUDIO value is {"waveform": [batch, channels, samples], "sample_rate"}.
"""
import base64, contextlib, gzip, importlib, io, json, os, sys

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
    nodes = importlib.import_module("comfy_extras.nodes_audio")

rng = np.random.default_rng(20261111)
tensors, cases, seen = [], [], {}


def add(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    key = (a.shape, a.tobytes())
    if key not in seen:
        tensors.append({"shape": list(a.shape), "f32": base64.b64encode(a.astype("<f4").tobytes()).decode()})
        seen[key] = len(tensors) - 1
    return seen[key]


def wave(batch, channels, n, peak=0.9, tone=None, rate=8000):
    """Noise plus an optional tone, multiples of 1/4096."""
    w = rng.uniform(-peak, peak, (batch, channels, n)) * 0.5
    if tone is not None:
        t = np.arange(n) / rate
        w += 0.5 * peak * np.sin(2 * np.pi * tone * t)[None, None, :]
    return np.round(np.clip(w, -1, 1) * 4096) / 4096


def ramp(batch, channels, n):
    return np.broadcast_to(np.arange(n, dtype=np.float32) / 4096, (batch, channels, n)).copy()


def audio(w, rate):
    return {"waveform": torch.from_numpy(np.asarray(w, np.float32)), "sample_rate": rate}


def audio_arg(w, rate):
    return {"tensor": add(w), "sample_rate": rate}


def result_audio(v):
    return {"tensor": add(v["waveform"].detach().cpu().numpy()), "sample_rate": int(v["sample_rate"])}


def run(node, inputs, args, name):
    """inputs: name -> (waveform, rate); args: plain widget values."""
    case = {"node": node, "case": name, "inputs": {k: audio_arg(*v) for k, v in inputs.items()}, "args": args}
    try:
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            res = getattr(nodes, node).execute(**{k: audio(*v) for k, v in inputs.items()}, **args)
        val = getattr(res, "result", None) or getattr(res, "args", None) or res
        case["out"] = [result_audio(v) for v in val]
    except Exception as e:
        case["error"] = type(e).__name__
        case["message"] = str(e)
    cases.append(case)


# Trim Audio Duration ---------------------------------------------------------------------------------------------------
r20 = ramp(1, 1, 20)
for start, dur in [(0.5, 1.0), (-0.5, 60.0), (0.25, 0.35), (-100.0, 0.3), (5.0, 1.0), (0.0, 0.0), (1.95, 0.01), (0.05, 0.05), (0.15, 0.25),
                   (-0.25, 0.1), (1.0, 1.0), (0.0, 2.0), (1.9, 5.0), (-2.0, 0.5), (-1.95, 0.15)]:
    run("TrimAudioDuration", {"audio": (r20, 10)}, {"start_index": start, "duration": dur}, f"ramp {start} {dur}")
st = wave(2, 2, 300)
for start, dur in [(0.0, 0.01), (0.0123, 0.0071), (-0.01, 0.005), (0.037, 1.0), (0.0375, 0.0005)]:
    run("TrimAudioDuration", {"audio": (st, 8000)}, {"start_index": start, "duration": dur}, f"stereo batch {start} {dur}")
run("TrimAudioDuration", {"audio": (np.zeros((1, 1, 0), np.float32), 10)}, {"start_index": 1.0, "duration": 1.0}, "empty")

# Split / Join ----------------------------------------------------------------------------------------------------------
run("SplitAudioChannels", {"audio": (wave(1, 2, 7), 8000)}, {}, "stereo")
run("SplitAudioChannels", {"audio": (wave(2, 2, 5), 16000)}, {}, "stereo batch")
run("SplitAudioChannels", {"audio": (wave(1, 1, 5), 8000)}, {}, "mono")
run("SplitAudioChannels", {"audio": (wave(1, 3, 5), 8000)}, {}, "three channels")
m5, m3, m40 = wave(1, 1, 5), wave(1, 1, 3), wave(1, 1, 40, tone=440)
run("JoinAudioChannels", {"audio_left": (m5, 8000), "audio_right": (wave(1, 1, 5), 8000)}, {}, "equal")
run("JoinAudioChannels", {"audio_left": (m5, 8000), "audio_right": (m3, 8000)}, {}, "right shorter")
run("JoinAudioChannels", {"audio_left": (m3, 8000), "audio_right": (m40, 12000)}, {}, "rates 8k 12k")
run("JoinAudioChannels", {"audio_left": (m40, 16000), "audio_right": (m40, 8000)}, {}, "rates 16k 8k")
run("JoinAudioChannels", {"audio_left": (m40, 44100), "audio_right": (wave(1, 1, 50), 48000)}, {}, "rates 44.1k 48k")
run("JoinAudioChannels", {"audio_left": (wave(1, 2, 5), 8000), "audio_right": (m5, 8000)}, {}, "stereo left")
run("JoinAudioChannels", {"audio_left": (wave(2, 1, 5), 8000), "audio_right": (wave(2, 1, 6), 8000)}, {}, "batch 2")
run("JoinAudioChannels", {"audio_left": (wave(2, 1, 5), 8000), "audio_right": (wave(1, 1, 5), 8000)}, {}, "batch 2 and 1")

# Concatenate -----------------------------------------------------------------------------------------------------------
mono, stereo = wave(1, 1, 6), wave(1, 2, 4)
for d in ("after", "before"):
    run("AudioConcat", {"audio1": (mono, 8000), "audio2": (stereo, 8000)}, {"direction": d}, f"mono stereo {d}")
    run("AudioConcat", {"audio1": (stereo, 8000), "audio2": (mono, 8000)}, {"direction": d}, f"stereo mono {d}")
    run("AudioConcat", {"audio1": (m40, 8000), "audio2": (wave(1, 1, 3), 16000)}, {"direction": d}, f"rates {d}")
    run("AudioConcat", {"audio1": (m40, 22050), "audio2": (wave(1, 2, 9), 11025)}, {"direction": d}, f"rates down {d}")
run("AudioConcat", {"audio1": (wave(1, 3, 4), 8000), "audio2": (stereo, 8000)}, {"direction": "after"}, "three channels")
run("AudioConcat", {"audio1": (wave(2, 1, 4), 8000), "audio2": (wave(2, 1, 3), 8000)}, {"direction": "after"}, "batch 2")

# Merge -----------------------------------------------------------------------------------------------------------------
a3 = np.array([[[0.5, 0.25, -0.5]]], np.float32)
for method in ("add", "mean", "subtract", "multiply"):
    run("AudioMerge", {"audio1": (a3, 8000), "audio2": (np.array([[[0.25]]], np.float32), 8000)}, {"merge_method": method}, f"shorter {method}")
    run("AudioMerge", {"audio1": (a3, 8000), "audio2": (np.array([[[0.25, 0.25, 0.25, 9.0]]], np.float32), 8000)}, {"merge_method": method}, f"longer {method}")
    run("AudioMerge", {"audio1": (wave(1, 2, 30), 8000), "audio2": (wave(1, 2, 30, peak=1.0), 8000)}, {"merge_method": method}, f"stereo {method}")
    run("AudioMerge", {"audio1": (wave(1, 1, 8), 8000), "audio2": (wave(1, 2, 8), 8000)}, {"merge_method": method}, f"mono stereo {method}")
    run("AudioMerge", {"audio1": (wave(1, 2, 8), 8000), "audio2": (wave(1, 1, 8), 8000)}, {"merge_method": method}, f"stereo mono {method}")
run("AudioMerge", {"audio1": (a3, 8000), "audio2": (np.array([[[1.0, 0.5, -0.25]]], np.float32), 8000)}, {"merge_method": "add"}, "peak above one")
run("AudioMerge", {"audio1": (a3, 8000), "audio2": (np.array([[[2.0, 2.0]]], np.float32), 8000)}, {"merge_method": "multiply"}, "peak exactly one")
run("AudioMerge", {"audio1": (np.zeros((1, 1, 4), np.float32), 8000), "audio2": (np.zeros((1, 1, 4), np.float32), 8000)}, {"merge_method": "add"}, "silence")
run("AudioMerge", {"audio1": (m40, 8000), "audio2": (wave(1, 1, 3), 12000)}, {"merge_method": "add"}, "audio1 lower rate")
run("AudioMerge", {"audio1": (m40, 12000), "audio2": (wave(1, 1, 30), 8000)}, {"merge_method": "add"}, "audio2 lower rate")
run("AudioMerge", {"audio1": (np.zeros((1, 1, 0), np.float32), 8000), "audio2": (a3, 8000)}, {"merge_method": "add"}, "empty audio1")
run("AudioMerge", {"audio1": (a3, 8000), "audio2": (a3, 8000)}, {"merge_method": "divide"}, "unknown method")
run("AudioMerge", {"audio1": (wave(2, 1, 5), 8000), "audio2": (wave(1, 1, 5), 8000)}, {"merge_method": "add"}, "batch 2 and 1")

# Volume / Empty --------------------------------------------------------------------------------------------------------
vol = np.array([[[0.5, -0.1, 0.9, 0.0]]], np.float32)
for v in (0, -20, 1, 6, 100, -100, 3):
    run("AudioAdjustVolume", {"audio": (vol, 8000)}, {"volume": v}, f"volume {v}")
for d, sr, ch in [(0.5, 44100, 1), (2.5, 1, 2), (3.5, 1, 1), (0.0, 8000, 2), (0.01, 8000, 2), (1.0 / 3, 3, 1), (0.123456, 1000, 1)]:
    run("EmptyAudio", {}, {"duration": d, "sample_rate": sr, "channels": ch}, f"empty {d} {sr} {ch}")

# Equalizer ---------------------------------------------------------------------------------------------------------------
eight = np.array([[[0.5, -0.25, 0.125, 0.0, 0.25, -0.5, 0.0, 0.0]]], np.float32)
eight2 = np.concatenate([eight, -eight], 1)
tone = wave(1, 2, 400, tone=1000, rate=8000)
music = wave(1, 2, 2000, peak=0.6, tone=220, rate=44100)
eq = [
    (eight2, 8000, dict(low_gain_dB=6.0, low_freq=100, mid_gain_dB=-3.0, mid_freq=1000, mid_q=2.0, high_gain_dB=12.0, high_freq=2000)),
    (eight2, 8000, dict(low_gain_dB=0.0, low_freq=100, mid_gain_dB=0.0, mid_freq=1000, mid_q=0.707, high_gain_dB=0.0, high_freq=5000)),
    (eight2, 8000, dict(low_gain_dB=0.0, low_freq=100, mid_gain_dB=-3.0, mid_freq=1000, mid_q=2.0, high_gain_dB=0.0, high_freq=2000)),
    (eight2, 8000, dict(low_gain_dB=6.0, low_freq=100, mid_gain_dB=0.0, mid_freq=1000, mid_q=0.707, high_gain_dB=0.0, high_freq=5000)),
    (eight2, 8000, dict(low_gain_dB=0.0, low_freq=100, mid_gain_dB=0.0, mid_freq=1000, mid_q=0.707, high_gain_dB=12.0, high_freq=2000)),
    (tone, 8000, dict(low_gain_dB=-24.0, low_freq=500, mid_gain_dB=-6.0, mid_freq=1000, mid_q=0.707, high_gain_dB=3.5, high_freq=3000)),
    (music, 44100, dict(low_gain_dB=4.5, low_freq=120, mid_gain_dB=-2.2, mid_freq=2500, mid_q=1.3, high_gain_dB=-7.0, high_freq=9000)),
    (music, 44100, dict(low_gain_dB=24.0, low_freq=20, mid_gain_dB=24.0, mid_freq=4000, mid_q=10.0, high_gain_dB=24.0, high_freq=15000)),
    (music, 44100, dict(low_gain_dB=-24.0, low_freq=500, mid_gain_dB=0.1, mid_freq=200, mid_q=0.1, high_gain_dB=-0.1, high_freq=1000)),
    (wave(2, 1, 64, tone=300, rate=16000), 16000, dict(low_gain_dB=3.0, low_freq=300, mid_gain_dB=3.0, mid_freq=2000, mid_q=0.5, high_gain_dB=3.0, high_freq=6000)),
]
for i, (w, sr, kw) in enumerate(eq):
    run("AudioEqualizer3Band", {"audio": (w, sr)}, kw, f"eq {i}")

with gzip.open(out_path, "wt", encoding="utf-8") as f:
    json.dump({"tensors": tensors, "cases": cases}, f, separators=(",", ":"))
print(f"{len(cases)} cases, {len(tensors)} tensors, {sum(1 for c in cases if 'error' in c)} errors -> {out_path}")
