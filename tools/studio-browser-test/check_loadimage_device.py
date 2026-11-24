"""Checks the pictures drive-loadimage-device.js saved (Load Image's IMAGE and MASK as 16-bit PNGs, Save Image (Advanced): value x 65535
truncated) against the decode corpus's spec-correct reference (tools/decode-corpus/make_reference.py). Exit 0 when every file matches:
PNG to within one 16-bit step; JPEG within one 16-bit step too (the device decoder equals libjpeg; the host decoder is 2-3/255 off it, so a
JPEG that passes also shows the GPU path ran).

    python check_loadimage_device.py
"""
import importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'loadimage-device')
CORPUS = os.path.join(HERE, '..', '..', 'tests', 'SpawnGen.Nodes.Tests', 'Fixtures', 'decode_corpus')
spec = importlib.util.spec_from_file_location('ref', os.path.join(HERE, '..', 'decode-corpus', 'make_reference.py'))
sys.argv = [sys.argv[0], CORPUS + os.sep + '__no_output__']  # make_reference only needs its png() decoder here: point its OUT nowhere useful
src = open(spec.origin, encoding='utf-8').read().split('\nmanifest = {}')[0]  # the definitions, not the run
ref_mod = {}
exec(compile(src.replace("os.makedirs(OUT, exist_ok=True)", ""), spec.origin, 'exec'), ref_mod)
png = ref_mod['png']
reference = json.load(open(os.path.join(CORPUS, 'reference', 'reference.json')))


def expected(name, what, shape):
    e = reference[name]
    p = os.path.join(CORPUS, 'reference', f'{name}.{what}')
    if e.get(what + '_float'):
        return np.fromfile(p + '.f32', '<f4').reshape(shape).astype(np.float64)
    return np.fromfile(p, np.uint8).reshape(shape).astype(np.float64) / 255


results = json.load(open(os.path.join(OUT, 'results.json')))
failed = 0
for name, r in results['results'].items():
    e = reference[name]
    try:
        img, _, _ = png(os.path.join(OUT, name + '.image.png'))  # 16-bit samples as s / 65535
        msk, _, _ = png(os.path.join(OUT, name + '.mask.png'))
    except Exception as ex:
        print(f'FAIL {name}: no saved picture ({ex}); {r}'); failed += 1; continue
    ei = expected(name, 'image', (e['height'], e['width'], 3))
    em = expected(name, 'mask', (e['mask_height'], e['mask_width']))
    ok_shape = img.shape == ei.shape and msk.shape[:2] == em.shape
    di = np.abs(img - ei).max() * 65535 if img.shape == ei.shape else float('inf')
    dm = np.abs(msk[..., 0] - em).max() * 65535 if msk.shape[:2] == em.shape else float('inf')
    ok = ok_shape and di <= 1.0001 and dm <= 1.0001
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}: IMAGE {img.shape} largest {di:.2f}/65535 ({di / 257:.3f}/255), MASK {msk.shape[:2]} largest {dm:.2f}/65535")
print('browser errors:', results['problems'] or 'none')
sys.exit(1 if failed or results['problems'] else 0)
