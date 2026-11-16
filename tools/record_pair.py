# record_pair.py: records about 9 seconds of BOTH front cameras (matched frame by frame) plus head tracking,
# into frametint-pair.npz, so the two views can be compared while the head is moving. Shows nothing in the headset.
#   ~/.local/share/frametint/venv/bin/python record_pair.py
import sys, os, json, time, struct, threading, subprocess
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('FT_HOME', os.path.expanduser('~/.local/share/frametint'))
W, H, VIEW_DEG = 640, 480, 115
SECS = float(os.environ.get('FT_SECS', '9'))
GRAB = os.environ.get('FT_GRAB_CMD', os.path.join(ROOT, 'bin', 'ftgrab')).split()
OVERLAY = os.environ.get('FT_OVERLAY_CMD', os.path.join(ROOT, 'bin', 'ftoverlay')).split()
cal = None
for p in ('/persist/xrservice.json', os.path.join(HERE, 'device_config.json')):
    try: cal = json.load(open(p)); break
    except Exception: pass
if cal is None: sys.exit('PROBLEM: no lens calibration found')
def axes(o):
    x = np.array(o['plus_x'], np.float64); z = np.array(o['plus_z'], np.float64)
    x /= np.linalg.norm(x); z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], axis=1)
C0 = axes(cal['cad']['cameras'][0]); C1 = C0 @ axes(cal['cameras'][1]['extrinsics']); V = np.diag([-1.0, -1.0, 1.0])
def make_maps(cam, C):
    kb = [i for i in cal['cameras'][cam]['intrinsics'] if i.get('cameraModel') == 'kb'][0]
    K = np.array([[kb['fx'], 0, kb['cx']], [0, kb['fy'], kb['cy']], [0, 0, 1]], np.float64)
    D = np.array([kb.get('k1', 0), kb.get('k2', 0), kb.get('k3', 0), kb.get('k4', 0)], np.float64)
    f = W / 2 / np.tan(np.radians(VIEW_DEG / 2))
    return cv2.fisheye.initUndistortRectifyMap(K, D, V @ C, np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]), (W, H), cv2.CV_16SC2)
maps = [make_maps(0, C0), make_maps(1, C1)]
q = [[], []]; procs = []; poses = []
def reader(i):
    pr = subprocess.Popen(GRAB + [str(i), '0', '.', '1'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=HERE); procs.append(pr)
    while True:
        hd = pr.stdout.read(24)
        if len(hd) < 24: return
        magic, w, h, seq, ts = struct.unpack('<4sIIId', hd)
        buf = pr.stdout.read(w * h)
        if len(buf) < w * h: return
        q[i].append((ts, seq, time.time(), np.frombuffer(buf, np.uint8).reshape(h, w)))
        if len(q[i]) > 8: del q[i][:len(q[i]) - 5]
def pose_reader(ovl):
    while True:
        rec = ovl.stdout.read(52)
        if len(rec) < 52: return
        if rec[:4] == b'FTP1': poses.append((time.time(), time.monotonic()) + tuple(np.frombuffer(rec, np.float32, 12, 4)))
for i in (0, 1): threading.Thread(target=reader, args=(i,), daemon=True).start()
ovl = subprocess.Popen(OVERLAY + ['1', '1.5', '1', 'poses'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=HERE)
threading.Thread(target=pose_reader, args=(ovl,), daemon=True).start()
print('getting ready... (wear the headset)', flush=True)
t_wait = time.time()
while not (q[0] and q[1] and poses):
    if time.time() - t_wait > 20: sys.exit('PROBLEM: cameras or head tracking did not start (is the Frame Tint driver loaded and the headset on your head?)')
    time.sleep(0.05)
print('RECORDING %.0f seconds: first nod your head up and down a few times, then turn it left and right a few times' % SECS, flush=True)
jl, jr, info = [], [], []; last = -1; t0 = time.time()
while time.time() - t0 < SECS:
    r = q[0][-1] if q[0] else None
    if r is None or r[1] == last: time.sleep(0.001); continue
    last = r[1]; give_up = time.time() + 0.014; best = None
    while True:
        c = list(q[1]); best = min(c, key=lambda x: abs(x[0] - r[0])) if c else None
        if (best is not None and abs(best[0] - r[0]) < 0.012) or (c and c[-1][0] > r[0] + 0.012) or time.time() > give_up: break
        time.sleep(0.001)
    if best is None: continue
    jr.append(cv2.imencode('.jpg', cv2.remap(r[3], maps[0][0], maps[0][1], cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
    jl.append(cv2.imencode('.jpg', cv2.remap(best[3], maps[1][0], maps[1][1], cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
    info.append((r[0], best[0], r[1], best[1], r[2], best[2]))
for pr in procs:
    try: pr.terminate()
    except Exception: pass
try: ovl.stdin.close()
except Exception: pass
out = os.path.join(HERE, 'frametint-pair.npz')
a = np.empty(len(jr), object); a[:] = jr; b = np.empty(len(jl), object); b[:] = jl
np.savez(out, right=a, left=b, info=np.array(info, np.float64), poses=np.array([p for p in poses if p[0] >= t0 - 0.5], np.float64))
print('DONE: %d matched pairs, %d head positions, saved %s (%.0f MB)' % (len(jr), len(poses), out, os.path.getsize(out) / 1e6), flush=True)
