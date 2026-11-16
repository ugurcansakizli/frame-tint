# cam_check.py: watches both front cameras for 8 seconds and prints what each one really delivers.
#   ~/.local/share/frametint/venv/bin/python cam_check.py        (wear the headset and keep turning your head while it runs)
import sys, os, time, struct, threading, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('FT_HOME', os.path.expanduser('~/.local/share/frametint'))
GRAB = os.environ.get('FT_GRAB_CMD', os.path.join(ROOT, 'bin', 'ftgrab')).split()
SECS = float(os.environ.get('FT_SECS', '8'))
rows = {0: [], 1: []}; procs = []
def reader(i):
    pr = subprocess.Popen(GRAB + [str(i), '0', '.', '1'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=HERE); procs.append(pr)
    prev = None
    while True:
        hd = pr.stdout.read(24)
        if len(hd) < 24: return
        magic, w, h, seq, ts = struct.unpack('<4sIIId', hd)
        buf = pr.stdout.read(w * h)
        if len(buf) < w * h: return
        a = np.frombuffer(buf, np.uint8).reshape(h, w)[::8, ::8].astype(np.int16)
        change = float(np.abs(a - prev).mean()) if prev is not None else -1.0
        prev = a
        rows[i].append((time.time(), seq, ts, float(a.mean()), change))
for i in (0, 1): threading.Thread(target=reader, args=(i,), daemon=True).start()
print('watching both cameras for %.0f seconds: keep turning your head...' % SECS, flush=True)
time.sleep(2); t_start = time.time(); time.sleep(SECS); t_end = time.time()
for pr in procs:
    try: pr.terminate()
    except Exception: pass
def summary(i, name):
    r = [x for x in rows[i] if t_start <= x[0] <= t_end]
    if len(r) < 3: print('%s camera (%d): only %d frames arrived' % (name, i, len(r))); return None
    t = np.array([x[0] for x in r]); seq = np.array([x[1] for x in r]); ts = np.array([x[2] for x in r]); br = np.array([x[3] for x in r]); ch = np.array([x[4] for x in r])
    dt = np.diff(t) * 1000; ds = np.diff(seq); dts = np.diff(ts) * 1000
    steps = {int(k): int((ds == k).sum()) for k in np.unique(ds)}
    print('%s camera (%d): %.1f frames/s' % (name, i, len(r) / (t_end - t_start)))
    print('   time between frames: usual %.0f ms, longest %.0f ms; gaps over 50 ms: %d' % (np.median(dt), dt.max(), int((dt > 50).sum())))
    print('   grabber clock between frames: usual %.0f ms, longest %.0f ms' % (np.median(dts), dts.max()))
    print('   frame number steps (step: how often): %s' % steps)
    print('   brightness: lowest %.0f, usual %.0f, highest %.0f; dark frames (under half the usual): %d' % (br.min(), np.median(br), br.max(), int((br < 0.5 * np.median(br)).sum())))
    print('   picture change from one frame to the next: usual %.2f; frames with almost no change (under 0.3): %d of %d' % (np.median(ch), int((ch < 0.3).sum()), len(ch)))
    return t
t0 = summary(0, 'right'); t1 = summary(1, 'left')
if t0 is not None and t1 is not None:
    gap = np.array([np.abs(t1 - x).min() for x in t0]) * 1000
    print('pairing: for each right frame, the nearest left frame arrived %.0f ms away (usual), %.0f ms (worst); pairs more than 12 ms apart: %d of %d' % (np.median(gap), gap.max(), int((gap > 12).sum()), len(gap)))
