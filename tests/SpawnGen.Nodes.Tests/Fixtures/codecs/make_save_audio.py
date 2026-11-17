"""Runs ComfyUI's own audio saving (comfy_api AudioSaveHelper.save_audio: PyAV, flt frame -> the codec's first sample format) on small clips and writes
the samples, the prompt / extra_pnginfo strings and ComfyUI's FLAC files to save_audio/ for SpawnGen's Save Audio tests.

    python make_save_audio.py <ComfyUI folder>

Cases: stereo 44.1 kHz (2 s: several FLAC blocks, a partial last one), mono 22.05 kHz with an odd length, a loud clip that clips at +-1, and stereo
without metadata. PyAV picks s16 for FLAC, so these are 16-bit files from clip(rint(x * 32768)).
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import folder_paths  # noqa: E402
import av  # noqa: E402
from comfy_api.latest._ui import AudioSaveHelper, FolderType  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_audio")
os.makedirs(HERE, exist_ok=True)


class Hidden:
    def __init__(self, prompt, extra):
        self.prompt, self.extra_pnginfo = prompt, extra


class Cls:
    hidden = None


def clip(channels, rate, seconds, gain, seed):
    n = int(round(rate * seconds)) if isinstance(seconds, float) else seconds
    t = np.arange(n) / rate
    rng = np.random.default_rng(seed)
    out = []
    for c in range(channels):
        f = 220.0 * (c + 1.5)
        x = 0.45 * np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 3.7 * f * t + c) * np.exp(-t) + 0.05 * rng.standard_normal(n)
        x[n // 3:n // 3 + 200] = 0.0  # a silent stretch (FLAC's constant subframes)
        out.append(x * gain)
    return torch.from_numpy(np.stack(out).astype(np.float32))[None]  # [batch 1, channels, samples]


PROMPT = {"3": {"class_type": "SaveAudio", "inputs": {"filename_prefix": "audio/ComfyUI", "text": "café \"quoted\""}}}
EXTRA = {"workflow": {"nodes": [1, 2, 3], "version": 0.4}}

if __name__ == "__main__":
    folder_paths.set_output_directory(HERE)
    cases = []
    for name, ch, rate, seconds, gain, meta in [("stereo_44100", 2, 44100, 2.0, 1.0, True), ("mono_22050_odd", 1, 22050, 33333, 1.0, True),
                                                ("loud_clipping", 2, 48000, 1.0, 2.5, True), ("stereo_no_metadata", 2, 32000, 1.0, 1.0, False)]:
        Cls.hidden = Hidden(PROMPT, EXTRA) if meta else Hidden(None, None)
        wave = clip(ch, rate, seconds, gain, len(cases))
        r = AudioSaveHelper.save_audio({"waveform": wave, "sample_rate": rate}, name, FolderType.output, Cls, format="flac")
        os.replace(os.path.join(HERE, r[0].filename), os.path.join(HERE, name + ".flac"))
        wave[0].numpy().astype("<f4").tofile(os.path.join(HERE, name + ".f32"))  # channels x samples, little-endian float32
        cases.append({"name": name, "channels": ch, "sample_rate": rate, "samples": wave.shape[-1], "metadata": meta})
    # Opus (ComfyUI resamples to an Opus rate first, then libopus at the quality's bit rate): stereo 44.1 kHz -> 48 kHz at 128k, mono 22.05 kHz -> 24 kHz
    # at 64k. The packets come from libopus' float build; SpawnGen's tests compare the Ogg pages around them and the decoded sound.
    opus = []
    for name, quality in [("stereo_44100", "128k"), ("mono_22050_odd", "64k")]:
        c = next(c for c in cases if c["name"] == name)
        Cls.hidden = Hidden(PROMPT, EXTRA)
        wave = torch.from_numpy(np.fromfile(os.path.join(HERE, name + ".f32"), "<f4").reshape(c["channels"], -1))[None]
        r = AudioSaveHelper.save_audio({"waveform": wave, "sample_rate": c["sample_rate"]}, name + "_" + quality, FolderType.output, Cls, format="opus", quality=quality)
        os.replace(os.path.join(HERE, r[0].filename), os.path.join(HERE, f"{name}_{quality}.opus"))
        import comfy.audio
        target = {44100: 48000, 22050: 24000}[c["sample_rate"]]
        comfy.audio.resample(wave[0], c["sample_rate"], target).numpy().astype("<f4").tofile(os.path.join(HERE, f"{name}_{target}.resampled.f32"))
        opus.append({"name": name, "quality": quality, "file": f"{name}_{quality}.opus", "rate": target, "resampled": f"{name}_{target}.resampled.f32"})
    # MP3 (libmp3lame through PyAV, no resampling): stereo 44.1 kHz at V0 (variable) and 128k, mono 22.05 kHz at V0 and 320k (libmp3lame clamps it to
    # 160 kbps, the MPEG-2 maximum). The frames come from LAME; SpawnGen's tests compare the tag, the Info / Xing frame, the frame headers and the sound.
    mp3 = []
    for name, quality in [("stereo_44100", "V0"), ("stereo_44100", "128k"), ("mono_22050_odd", "V0"), ("mono_22050_odd", "320k")]:
        c = next(c for c in cases if c["name"] == name)
        Cls.hidden = Hidden(PROMPT, EXTRA)
        wave = torch.from_numpy(np.fromfile(os.path.join(HERE, name + ".f32"), "<f4").reshape(c["channels"], -1))[None]
        r = AudioSaveHelper.save_audio({"waveform": wave, "sample_rate": c["sample_rate"]}, name + "_" + quality, FolderType.output, Cls, format="mp3", quality=quality)
        os.replace(os.path.join(HERE, r[0].filename), os.path.join(HERE, f"{name}_{quality}.mp3"))
        mp3.append({"name": name, "quality": quality, "file": f"{name}_{quality}.mp3"})
    json.dump({"pyav": av.__version__, "libavcodec": ".".join(map(str, av.library_versions["libavcodec"])),
               "prompt": json.dumps(PROMPT), "extra": {k: json.dumps(v) for k, v in EXTRA.items()}, "cases": cases, "opus": opus, "mp3": mp3},
              open(os.path.join(HERE, "cases.json"), "w"), indent=1)
    print("wrote", [c["name"] for c in cases])
