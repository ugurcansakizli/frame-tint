# flicker.py: measures how much the colors flicker in the running Frame Tint, for different live settings.
# Run ON THE HEADSET while Frame Tint runs, the headset is worn and the head is kept STILL:
#   python3 tools/flicker.py "fovea=1&steady=1" "fovea=0&steady=1" "fovea=1&steady=0.3" ...
# For each setting it reads about 4 seconds of the preview and reports the average color change from one picture to the
# next, counted only where the brightness stayed the same (so head or hand movement is not counted as flicker).
import sys, time, urllib.request
import numpy as np, cv2

def frames(n, timeout=10):
    out, buf = [], b''
    with urllib.request.urlopen('http://127.0.0.1:8000/stream', timeout=timeout) as r:
        while len(out) < n:
            chunk = r.read(65536)
            if not chunk: break
            buf += chunk
            while True:
                a = buf.find(b'\xff\xd8'); b = buf.find(b'\xff\xd9', a + 2)
                if a < 0 or b < 0: break
                out.append(cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)); buf = buf[b + 2:]
    return out

for setting in sys.argv[1:]:
    urllib.request.urlopen('http://127.0.0.1:8000/set?' + setting).read()
    time.sleep(2.0)                                     # let the change settle
    pics = [cv2.cvtColor(cv2.resize(p, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2Lab).astype(np.float32) for p in frames(50)]
    d, still = [], []
    for p, q in zip(pics, pics[1:]):
        ok = np.abs(p[..., 0] - q[..., 0]) < 4           # brightness unchanged: nothing moved there
        if ok.mean() < 0.5: continue                    # the head moved: skip this pair
        d.append(np.hypot(p[..., 1] - q[..., 1], p[..., 2] - q[..., 2])[ok].mean()); still.append(ok.mean())
    print('%-34s flicker %.2f (color change per picture, %d still pairs of %d)' % (setting, np.mean(d) if d else float('nan'), len(d), len(pics) - 1))
