"""Writes FLAC files with PyAV (FFmpeg's encoder: 16-bit from s16, 24-bit from s32) and records what ComfyUI's own Load Audio reading gives for each
(comfy_extras.nodes_audio.load: PyAV decode, then f32_pcm), for SpawnGen's Load Audio tests (load_audio/).

    python make_load_audio.py <ComfyUI folder>
"""
import io, json, os, sys
import numpy as np

sys.path.insert(0, sys.argv[1])
from comfy.cli_args import args  # noqa: E402
args.cpu = True
import av  # noqa: E402
from comfy_extras.nodes_audio import load  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "load_audio")
os.makedirs(HERE, exist_ok=True)


def write_flac(path, channels, rate, n, fmt, layout, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    wave = np.stack([0.6 * np.sin(2 * np.pi * (180 + 70 * c) * t) + 0.02 * rng.standard_normal(n) for c in range(channels)]).astype(np.float32)
    wave[:, n // 2:n // 2 + 50] = 0.999  # near full scale
    buf = io.BytesIO()
    oc = av.open(buf, mode="w", format="flac")
    st = oc.add_stream("flac", rate=rate, layout=layout)
    st.codec_context.format = fmt
    fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(wave.T).reshape(1, -1), format="flt", layout=layout)
    fr.sample_rate = rate
    fr.pts = 0
    oc.mux(st.encode(fr))
    oc.mux(st.encode(None))
    oc.close()
    open(path, "wb").write(buf.getvalue())


if __name__ == "__main__":
    cases = []
    for name, ch, rate, n, fmt, layout in [("stereo16_44100", 2, 44100, 30000, "s16", "stereo"), ("mono16_22050", 1, 22050, 12345, "s16", "mono"),
                                           ("stereo24_48000", 2, 48000, 20000, "s32", "stereo"), ("surround24_48000", 6, 48000, 9000, "s32", "5.1")]:
        path = os.path.join(HERE, name + ".flac")
        write_flac(path, ch, rate, n, fmt, layout, len(cases))
        wav, sr = load(path)
        wav.numpy().astype("<f4").tofile(os.path.join(HERE, name + ".f32"))  # channels x samples, ComfyUI's floats
        cases.append({"name": name, "channels": wav.shape[0], "samples": wav.shape[1], "sample_rate": sr})
    json.dump({"pyav": av.__version__, "cases": cases}, open(os.path.join(HERE, "cases.json"), "w"), indent=1)
    print(cases)
