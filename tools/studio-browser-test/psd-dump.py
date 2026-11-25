"""Prints the structure of a PSD file (header, layer records, additional layer info keys, composite): a reading aid for black-box measurements.
Usage: python psd-dump.py file.psd"""
import struct, sys

d = open(sys.argv[1], 'rb').read()
p = 0
def take(n):
    global p
    b = d[p:p + n]; p += n; return b
def u16(): return struct.unpack('>H', take(2))[0]
def i16(): return struct.unpack('>h', take(2))[0]
def u32(): return struct.unpack('>I', take(4))[0]
def i32(): return struct.unpack('>i', take(4))[0]

sig, ver = take(4), u16(); take(6)
ch, h, w, depth, mode = u16(), u32(), u32(), u16(), u16()
print(f'header: {sig} version {ver} channels {ch} {w}x{h} depth {depth} mode {mode} (3 = RGB)')
n = u32(); print(f'color mode data: {n} bytes'); take(n)
n = u32(); end = p + n; print(f'image resources: {n} bytes')
while p < end:
    s = take(4); rid = u16(); ln = d[p]; nm = take(1 + ln + (0 if (1 + ln) % 2 == 0 else 1)); sz = u32(); take(sz + (sz & 1))
    print(f'  resource {rid} ({sz} bytes)')
p = end
lm = u32(); lm_end = p + lm; print(f'layer and mask info: {lm} bytes')
if lm:
    li = u32(); li_end = p + li; cnt = i16(); print(f'  layer info: {li} bytes, layer count {cnt} (negative: first alpha is the merged transparency)')
    layers = []
    for k in range(abs(cnt)):
        top, left, bottom, right = i32(), i32(), i32(), i32(); nc = u16()
        chans = [(i16(), u32()) for _ in range(nc)]
        bsig, bkey = take(4), take(4); opacity, clipping, flags, _ = d[p], d[p + 1], d[p + 2], d[p + 3]; p += 4
        extra = u32(); ex_end = p + extra
        mask = u32(); take(mask); br = u32(); take(br)
        ln = d[p]; name = d[p + 1:p + 1 + ln]; p += 1 + ln; p += (-(1 + ln)) % 4
        keys = []
        while p < ex_end:
            take(4); key = take(4); sz = u32(); val = take(sz + (sz & 1)); keys.append((key.decode('latin1'), sz, val))
        p = ex_end
        layers.append(chans)
        print(f'  layer {k}: "{name.decode("latin1")}" rect left {left} top {top} right {right} bottom {bottom} ({right - left}x{bottom - top}) channels {[(c, s) for c, s in chans]}')
        print(f'    blend {bkey} opacity {opacity} clipping {clipping} flags {flags:#04x} (bit 1 set = hidden) mask {mask} ranges {br}')
        for key, sz, val in keys:
            shown = val.decode('utf-16-be', 'replace')[2:] if key == 'luni' else val[:24].hex()
            print(f'    info {key} ({sz} bytes): {shown}')
    for k, chans in enumerate(layers):
        comps = []
        for cid, size in chans:
            comps.append(struct.unpack('>H', d[p:p + 2])[0]); p += size
        print(f'  layer {k} channel compressions: {comps} (0 raw, 1 RLE)')
    p = li_end
    if p < lm_end:
        g = u32(); print(f'  global layer mask info: {g} bytes'); take(g)
    while p + 12 <= lm_end:
        s = take(4); key = take(4); sz = u32(); take(sz + ((-sz) % 4)); print(f'  tagged block {key} ({sz} bytes)')
p = lm_end
comp = u16(); print(f'composite image: compression {comp}, {len(d) - p} bytes of data; file {len(d)} bytes')
