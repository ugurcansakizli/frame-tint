# ab_snap.py: saves pictures from the running Frame Tint's preview with different live settings, for side-by-side
# comparison. Run ON THE HEADSET while Frame Tint runs and the headset is worn:
#   python3 tools/ab_snap.py <output folder> "sharp=0&solid=0&vivid=1.2" "sharp=1&solid=1&vivid=1.8" ...
# Each setting is applied with /set (only answers on the headset itself), then after a short wait one picture is saved.
# The pictures show your room: keep them to yourself.
import os, sys, time, urllib.request

def frames(n, timeout=8):
    """the next n JPEG pictures from the preview stream"""
    out, buf = [], b''
    with urllib.request.urlopen('http://127.0.0.1:8000/stream', timeout=timeout) as r:
        while len(out) < n:
            chunk = r.read(65536)
            if not chunk: break
            buf += chunk
            while True:
                a = buf.find(b'\xff\xd8'); b = buf.find(b'\xff\xd9', a + 2)
                if a < 0 or b < 0: break
                out.append(buf[a:b + 2]); buf = buf[b + 2:]
    return out

folder = sys.argv[1]; os.makedirs(folder, exist_ok=True)
for i, setting in enumerate(sys.argv[2:]):
    print(urllib.request.urlopen('http://127.0.0.1:8000/set?' + setting).read().decode())
    time.sleep(1.5)
    pics = frames(4)                               # skip the first ones, they may predate the change
    name = os.path.join(folder, '%d_%s.jpg' % (i, setting.replace('&', '_').replace('=', '')))
    open(name, 'wb').write(pics[-1]); print('saved', name)
