"""Black-box measurement over ComfyUI's HTTP API: repeat order in a subfolder, captions with fewer texts than pictures, batches in Deduplicate
Images, Make Image Grid with fewer pictures than columns, Shuffle Pairs with lists of different lengths (server as in measure_folders.py):
python measure_lists.py <root> setup; python measure_lists.py <root> run
"""
import json, os, sys, time, urllib.request, uuid
import numpy as np
from PIL import Image
ROOT = sys.argv[1]; INP = os.path.join(ROOT, 'input'); OUT = os.path.join(ROOT, 'output'); URL = 'http://127.0.0.1:8199'
def pic(rel, v):
    p = os.path.join(INP, rel); os.makedirs(os.path.dirname(p), exist_ok=True); Image.fromarray(np.full((4, 4, 3), (v, 0, 0), np.uint8)).save(p)
if sys.argv[2] == 'setup':
    pic('pa/2_two/a.png', 10); pic('pa/2_two/b.png', 20); pic('pa/z.png', 30)
    pic('one/x.png', 40); open(os.path.join(INP, 'one', 'x.txt'), 'w').write('only')
    pic('three/a.png', 50); pic('three/b.png', 60); pic('three/c.png', 70); sys.exit()
def run(name, prompt):
    body = json.dumps({'prompt': prompt, 'client_id': uuid.uuid4().hex}).encode()
    try: pid = json.load(urllib.request.urlopen(urllib.request.Request(URL + '/prompt', body, {'Content-Type': 'application/json'})))['prompt_id']
    except urllib.error.HTTPError as e: print(name, 'rejected', json.loads(e.read()).get('error', {}).get('message')); return
    for _ in range(300):
        h = json.load(urllib.request.urlopen(URL + '/history/' + pid))
        if pid in h:
            msg = [m[1] for m in h[pid]['status'].get('messages', []) if m[0] == 'execution_error']
            print(name, 'error: %s: %s' % (msg[0].get('exception_type'), msg[0].get('exception_message', '').strip()) if msg else h[pid]['status'].get('status_str')); return
        time.sleep(0.5)
def files(sub):
    d = os.path.join(OUT, sub)
    if not os.path.isdir(d): return 'none'
    return [(f, int(np.asarray(Image.open(os.path.join(d, f)))[0, 0, 0]) if f.endswith('.png') else open(os.path.join(d, f), 'rb').read()) for f in sorted(os.listdir(d))]
L = lambda cls, **i: {'class_type': cls, 'inputs': i}
run('A', {'1': L('LoadImageTextDataSetFromFolder', folder='pa'), '2': L('SaveImageDataSetToFolder', images=['1', 0], folder_name='A', filename_prefix='i', mode='overwrite')}); print(' A', files('A'))
run('B', {'1': L('LoadImageTextDataSetFromFolder', folder='three'), '2': L('LoadImageTextDataSetFromFolder', folder='one'),
          '3': L('SaveImageTextDataSetToFolder', images=['1', 0], texts=['2', 1], folder_name='B', filename_prefix='i', mode='overwrite')}); print(' B', files('B'))
run('C', {'1': L('EmptyImage', width=4, height=4, batch_size=2, color=0x800000), '2': L('ImageDeduplication', images=['1', 0], similarity_threshold=0.95),
          '3': L('SaveImageDataSetToFolder', images=['2', 0], folder_name='C', filename_prefix='i', mode='overwrite')}); print(' C', files('C'))
run('D', {'1': L('EmptyImage', width=4, height=4, batch_size=2, color=0x800000), '2': L('ImageGrid', images=['1', 0], columns=4, cell_width=32, cell_height=32, padding=2),
          '3': L('SaveImage', images=['2', 0], filename_prefix='D')})
print(' D', [(f, Image.open(os.path.join(OUT, f)).size) for f in os.listdir(OUT) if f.startswith('D_')])
run('F', {'1': L('LoadImageTextDataSetFromFolder', folder='three'), '2': L('LoadImageTextDataSetFromFolder', folder='one'),
          '3': L('ShuffleImageTextDataset', images=['1', 0], texts=['2', 1], seed=0), '4': L('SaveImageTextDataSetToFolder', images=['3', 0], texts=['3', 1], folder_name='F', filename_prefix='i', mode='overwrite')}); print(' F', files('F'))
