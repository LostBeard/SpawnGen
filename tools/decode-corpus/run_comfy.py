"""Measures ComfyUI's Load Image on every file of the decode corpus (black box, over its HTTP API) and writes what it gave as 8-bit values.

    <comfyui-venv>/python run_comfy.py <corpus folder> <comfyui input folder> [url=http://127.0.0.1:8199]

Per file: LoadImage -> SpawnGenDumpTensors (IMAGE) and LoadImage -> MaskToImage -> SpawnGenDumpTensors (MASK); the dump node is
_scratch/comfyui-ref/custom_nodes/spawngen_dump.py (raw float32). Every value must be an exact k/255 (checked); the expected files are
<name>.image (uint8, frames x height x width x 3) and <name>.mask (uint8, frames x mask height x mask width), and expected.json lists the
shapes. Values that are not whole k/255 are kept exactly instead, as <name>.image.f32 / .mask.f32 (float32), and
expected.json says so. A file ComfyUI refuses is listed with its error instead.
"""
import json, os, shutil, sys, time, urllib.request
import numpy as np

CORPUS, COMFY_INPUT = sys.argv[1], sys.argv[2]
URL = sys.argv[3] if len(sys.argv) > 3 else 'http://127.0.0.1:8199'
OUT = os.path.abspath(os.path.join(CORPUS, '..', '..', '..', '..', '_scratch', 'decode-corpus-comfy', 'expected'))  # kept out of the repo: the test's oracle is reference/ (make_reference.py)
DUMP = os.path.abspath(os.path.join(CORPUS, '..', '..', '..', '..', '_scratch', 'decode-corpus-dumps'))
os.makedirs(OUT, exist_ok=True); os.makedirs(DUMP, exist_ok=True)


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req))


def get(path):
    return json.load(urllib.request.urlopen(URL + path))


def read(prefix):
    f = [n for n in os.listdir(DUMP) if n.startswith(prefix + '_images_f32_')][0]
    shape = [int(s) for s in f[len(prefix + '_images_f32_'):-4].split('x')]
    return np.fromfile(os.path.join(DUMP, f), '<f4').reshape(shape)


def to_u8(a, what):
    k = np.rint(a * 255)
    worst = float(np.abs(a * 255 - k).max())
    if worst > 1e-3:
        raise ValueError(f'{what}: values are not k/255 (worst {worst})')
    return k.astype(np.uint8)


manifest = {}
for name in sorted(n for n in os.listdir(CORPUS) if n.lower().endswith(('.png', '.jpg', '.jpeg'))):
    tag = 'dc_' + name.replace('.', '_')
    shutil.copy(os.path.join(CORPUS, name), os.path.join(COMFY_INPUT, 'dc_' + name))
    for f in os.listdir(DUMP):
        if f.startswith(tag + '_'):
            os.remove(os.path.join(DUMP, f))
    prompt = {
        '1': {'class_type': 'LoadImage', 'inputs': {'image': 'dc_' + name}},
        '2': {'class_type': 'PrimitiveFloat', 'inputs': {'value': time.time() % 1000}},  # a new value each run: never served from the cache
        '3': {'class_type': 'SpawnGenDumpTensors', 'inputs': {'images': ['1', 0], 'fps': ['2', 0], 'name': tag + '_image', 'folder': DUMP}},
        '4': {'class_type': 'MaskToImage', 'inputs': {'mask': ['1', 1]}},
        '5': {'class_type': 'SpawnGenDumpTensors', 'inputs': {'images': ['4', 0], 'fps': ['2', 0], 'name': tag + '_mask', 'folder': DUMP}},
    }
    try:
        pid = post('/prompt', {'prompt': prompt})['prompt_id']
    except urllib.error.HTTPError as e:
        manifest[name] = {'error': e.read().decode()[:400]}; print(name, 'REFUSED'); continue
    for _ in range(600):
        h = get('/history/' + pid)
        if pid in h and h[pid].get('status', {}).get('completed') is not None:
            break
        time.sleep(0.2)
    status = h[pid]['status']
    if status.get('status_str') != 'success':
        msgs = [m for m in status.get('messages', []) if m[0] == 'execution_error']
        manifest[name] = {'error': (msgs[0][1].get('exception_message', '') if msgs else str(status))[:400]}
        print(name, 'ERROR', manifest[name]['error'][:120]); continue
    if any(m[0] == 'execution_cached' and m[1].get('nodes') for m in status.get('messages', [])):
        raise RuntimeError(f'{name}: served from the cache')
    image, mask = read(tag + '_image'), read(tag + '_mask')
    entry = {'frames': image.shape[0], 'height': image.shape[1], 'width': image.shape[2],
             'mask_height': mask.shape[1], 'mask_width': mask.shape[2], 'mask_any': bool(mask.max() > 0)}
    for what, a, ext in (('image', image, '.image'), ('mask', mask[..., 0], '.mask')):
        try:
            to_u8(a, f'{name} {what}').tofile(os.path.join(OUT, name + ext))
        except ValueError as e:  # not whole 8-bit steps: kept exactly, as float32
            a.astype('<f4').tofile(os.path.join(OUT, name + ext + '.f32'))
            entry[what + '_float'] = True
            print('  ', e)
    manifest[name] = entry
    print(name, manifest[name])
json.dump(manifest, open(os.path.join(OUT, 'expected.json'), 'w'), indent=1, sort_keys=True)
