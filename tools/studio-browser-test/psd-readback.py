"""Reads a PSD with psd-tools (an independent reader) and saves every layer and the flattened picture as PNG files next to a summary.
Usage: python psd-readback.py file.psd out_prefix      Exit code 3: psd-tools is not installed."""
import sys
try:
    from psd_tools import PSDImage
except ImportError:
    sys.exit(3)

psd = PSDImage.open(sys.argv[1])
prefix = sys.argv[2]
print(f'canvas {psd.width}x{psd.height} mode {psd.color_mode} channels {psd.channels} layers {len(psd)}')
count = 0
for i, layer in enumerate(psd):
    image = layer.topil()
    size = None if image is None else image.size
    mode = None if image is None else image.mode
    print(f'layer {i}: "{layer.name}" left {layer.left} top {layer.top} {layer.width}x{layer.height} blend {layer.blend_mode} opacity {layer.opacity} visible {layer.visible} picture {mode} {size}')
    if image is None or size != (layer.width, layer.height):
        print(f'FAIL layer {i} has no readable picture of its own size')
        sys.exit(1)
    image.save(f'{prefix}-{i}.png')
    count += 1
flat = psd.topil()
if flat is None or flat.size != (psd.width, psd.height):
    print('FAIL the flattened picture is not readable')
    sys.exit(1)
flat.save(f'{prefix}-flat.png')
print(f'OK {count} layers and the flattened picture read ({flat.mode})')
