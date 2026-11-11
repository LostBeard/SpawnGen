"""Black-box measurement of how ComfyUI's Load Image-Text (from Folder) reads a subfolder's repeat count from its name (server as in
measure_folders.py; run after its setup): python measure_repeats.py <root>
"""
import json, os, sys, time, urllib.request, uuid
import numpy as np
from PIL import Image
ROOT = sys.argv[1]; INP = os.path.join(ROOT, 'input', 'probe_k'); OUT = os.path.join(ROOT, 'output'); URL = 'http://127.0.0.1:8199'
dirs = ['3', '-1_neg', '+2_plus', '007_x', '3_', '1_a_b', '2 _sp', '2_']
for i, d in enumerate(dirs):
    os.makedirs(os.path.join(INP, d), exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), (10 * (i + 1), 0, 0), np.uint8)).save(os.path.join(INP, d, 'x.png'))
print(os.listdir(INP))
body = json.dumps({'prompt': {'1': {'class_type': 'LoadImageTextDataSetFromFolder', 'inputs': {'folder': 'probe_k'}},
                              '2': {'class_type': 'SaveImageDataSetToFolder', 'inputs': {'images': ['1', 0], 'folder_name': 'k', 'filename_prefix': 'i', 'mode': 'overwrite'}}},
                   'client_id': uuid.uuid4().hex}).encode()
try: pid = json.load(urllib.request.urlopen(urllib.request.Request(URL + '/prompt', body, {'Content-Type': 'application/json'})))['prompt_id']
except urllib.error.HTTPError as e: print('rejected', json.loads(e.read()).get('error', {}).get('message')); sys.exit()
for _ in range(200):
    h = json.load(urllib.request.urlopen(URL + '/history/' + pid))
    if pid in h:
        for m in h[pid]['status'].get('messages', []):
            if m[0] == 'execution_error': print('error:', m[1].get('exception_type'), m[1].get('exception_message', '').strip())
        break
    time.sleep(0.5)
d = os.path.join(OUT, 'k')
seq = [int(np.asarray(Image.open(os.path.join(d, f)))[0, 0, 0]) // 10 - 1 for f in sorted(os.listdir(d))] if os.path.isdir(d) else []
from collections import Counter
c = Counter(seq); print([(dirs[k], c.get(k, 0)) for k in range(len(dirs))], 'order', [dirs[k] for k in dict.fromkeys(seq)])
