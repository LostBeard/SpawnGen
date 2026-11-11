"""Black-box measurement of ComfyUI's dataset nodes over its HTTP API: folder options, the folder walk, captions, extensions and the contrast pivot.
No ComfyUI code is imported or read; only outputs and error type/message are printed. Start ComfyUI with --input-directory <root>/input
--output-directory <root>/output --temp-directory <root>/temp --port 8199, then: python measure_folders.py <root> setup; python measure_folders.py <root> run
"""
import json, os, sys, time, urllib.request, uuid
import numpy as np
from PIL import Image
ROOT = sys.argv[1]; INP = os.path.join(ROOT, 'input'); OUT = os.path.join(ROOT, 'output'); URL = 'http://127.0.0.1:8199'
def color(i): return (i * 15, 255 - i * 15, 7 * i)
def pic(rel, i, fmt=None):
    p = os.path.join(INP, 'probe_ds', rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), color(i), np.uint8)).save(p, fmt) if fmt else Image.fromarray(np.full((4, 4, 3), color(i), np.uint8)).save(p)
def txt(rel, s):
    p = os.path.join(INP, 'probe_ds', rel); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, 'wb').write(s.encode())
if sys.argv[2] == 'setup':
    for d in (INP, OUT, os.path.join(ROOT, 'temp')): os.makedirs(d, exist_ok=True)
    pic('z.png', 1); pic('aa_dir/q.png', 2); txt('aa_dir/q.txt', 'q cap'); pic('aa_dir/deep/d.png', 3); txt('aa_dir/deep/d.txt', 'd cap')
    pic('2_rep/r.png', 4); txt('2_rep/r.txt', 'r cap'); pic('0_zero/o.png', 5); pic('x_y/s.png', 6); pic('10_ten/t.png', 7)
    pic('e.bmp', 8); pic('f.gif', 9); pic('h.jpeg', 10); pic('k.webp', 11, 'WEBP'); pic('b.png', 12); txt('b.txt', '  b cap \n'); pic('m.JPG', 13, 'JPEG')
    os.makedirs(os.path.join(INP, 'probe_ds', 'empty_dir'), exist_ok=True)
    print(os.listdir(os.path.join(INP, 'probe_ds'))); sys.exit()
def get(path): return json.load(urllib.request.urlopen(URL + path))
def run(prompt):
    body = json.dumps({'prompt': prompt, 'client_id': uuid.uuid4().hex}).encode()
    try: pid = json.load(urllib.request.urlopen(urllib.request.Request(URL + '/prompt', body, {'Content-Type': 'application/json'})))['prompt_id']
    except urllib.error.HTTPError as e: return 'rejected: ' + json.loads(e.read()).get('error', {}).get('message', '?')
    for _ in range(600):
        h = get('/history/' + pid)
        if pid in h:
            st = h[pid]['status']
            for m in st.get('messages', []):
                if m[0] == 'execution_error': return 'error: %s: %s' % (m[1].get('exception_type'), m[1].get('exception_message', '').strip())
            return st.get('status_str')
        time.sleep(0.5)
    return 'timeout'
info = get('/object_info/LoadImageDataSetFromFolder')['LoadImageDataSetFromFolder']['input']['required']['folder']
print('folder options:', info[1].get('options') if len(info) > 1 else info[0])
ids = {color(i): i for i in range(1, 18)}
def ident(path):
    a = np.asarray(Image.open(path).convert('RGB'), np.int16)[1, 1]
    best = min(ids, key=lambda c: sum(abs(int(x) - y) for x, y in zip(a, c))); return ids[best]
L = lambda cls, **i: {'class_type': cls, 'inputs': i}
print('P1', run({'1': L('LoadImageTextDataSetFromFolder', folder='probe_ds'), '2': L('SaveImageTextDataSetToFolder', images=['1', 0], texts=['1', 1], folder_name='t', filename_prefix='i', mode='overwrite')}))
print('P2', run({'1': L('LoadImageDataSetFromFolder', folder='probe_ds'), '2': L('SaveImageDataSetToFolder', images=['1', 0], folder_name='p', filename_prefix='i', mode='overwrite')}))
for sub in ('t', 'p'):
    d = os.path.join(OUT, sub)
    if not os.path.isdir(d): print(sub, 'no folder'); continue
    pngs = sorted(f for f in os.listdir(d) if f.endswith('.png'))
    print(sub, [(f[2:7], ident(os.path.join(d, f)), open(os.path.join(d, f[:-4] + '.txt'), 'rb').read() if os.path.exists(os.path.join(d, f[:-4] + '.txt')) else None) for f in pngs])
E = lambda c: L('EmptyImage', width=4, height=4, batch_size=1, color=c)
print('P3', run({'1': E(0x202020), '2': E(0xE0E0E0), '3': L('ImageBatch', image1=['1', 0], image2=['2', 0]), '4': L('AdjustContrast', images=['3', 0], factor=2.0), '5': L('SaveImage', images=['4', 0], filename_prefix='c_batch'),
                 '6': E(0x404040), '7': L('SolidMask', value=0.0, width=4, height=4), '8': L('JoinImageWithAlpha', image=['6', 0], alpha=['7', 0]), '9': L('AdjustContrast', images=['8', 0], factor=2.0), '10': L('SaveImage', images=['9', 0], filename_prefix='c_rgba'),
                 '11': E(0x400000), '12': L('AdjustContrast', images=['11', 0], factor=2.0), '13': L('SaveImage', images=['12', 0], filename_prefix='c_chan')}))
for f in sorted(os.listdir(OUT)):
    if f.startswith('c_'): im = Image.open(os.path.join(OUT, f)); print(f, im.mode, np.asarray(im)[0, 0])
