"""Writes the noised start latent both ComfyUI and SpawnGen load (VAE space, as LoadLatent with latent_format_version_0 reads it in both)."""
import json, struct, sys
import numpy as np


def write(path, arr):
    data = arr.astype('<f4').tobytes()
    header = {"latent_tensor": {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [0, len(data)]},
              "latent_format_version_0": {"dtype": "F32", "shape": [0], "data_offsets": [len(data), len(data)]}}
    h = json.dumps(header).encode()
    h += b' ' * ((8 - len(h) % 8) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h)))
        f.write(h)
        f.write(data)


rng = np.random.default_rng(7)
sigma_max = 14.614641189575195
x = (rng.standard_normal((1, 4, 64, 64)) * np.sqrt(1 + sigma_max ** 2)).astype(np.float32)  # max_denoise start
write(sys.argv[1], x / np.float32(0.18215))   # the model multiplies by 0.18215 (ComfyUI's process_latent_in, SpawnGen's LoadLatent)
