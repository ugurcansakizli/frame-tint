# record.py: saves 30 seconds of camera pictures together with the headset's head tracking, into frametint-rec.npz,
# so the color memory can be checked and tuned against real data.
#   ~/.local/share/frametint/bin/ftgrab 0 0 . 1 | ~/.local/share/frametint/venv/bin/python tools/record.py
import sys, os, json, time, struct, threading, subprocess
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('FT_HOME', os.path.expanduser('~/.local/share/frametint'))
SECONDS, EVERY, W, H, VIEW_DEG, CAMERA = 30, 3, 640, 480, 115, 0
OVERLAY = os.environ.get('FT_OVERLAY_CMD', os.path.join(ROOT, 'bin', 'ftoverlay')).split()

cal = None
for p in ('/persist/xrservice.json', os.path.join(HERE, 'device_config.json')):
    try: cal = json.load(open(p)); break
    except Exception: pass
if cal is None: sys.exit('PROBLEM: no lens calibration found')
cam = cal['cameras'][CAMERA]; kb = [i for i in cam['intrinsics'] if i.get('cameraModel') == 'kb'][0]
K = np.array([[kb['fx'], 0, kb['cx']], [0, kb['fy'], kb['cy']], [0, 0, 1]], np.float64)
D = np.array([kb.get(k, 0) for k in ('k1', 'k2', 'k3', 'k4')], np.float64)
cad = cal['cad']['cameras'][CAMERA]
x = np.array(cad['plus_x'], np.float64); z = np.array(cad['plus_z'], np.float64); x /= np.linalg.norm(x); z /= np.linalg.norm(z)
C = np.stack([x, np.cross(z, x), z], axis=1); V = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64).T
f = W / 2 / np.tan(np.radians(VIEW_DEG / 2))
maps = cv2.fisheye.initUndistortRectifyMap(K, D, (C.T @ V).T, np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]), (W, H), cv2.CV_16SC2)

poses = []
def pose_reader(o):
    while True:
        rec = o.stdout.read(52)
        if len(rec) < 52: return
        if rec[:4] == b'FTP1': poses.append((time.time(),) + struct.unpack('<12f', rec[4:]))
try:
    ovl = subprocess.Popen(OVERLAY + ['1', '1.5', '1', 'poses'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=HERE)
except Exception as e:
    sys.exit('PROBLEM: could not start ftoverlay: %s' % e)
threading.Thread(target=pose_reader, args=(ovl,), daemon=True).start()

src = sys.stdin.buffer
jpgs, info, n, t0, said = [], [], 0, None, -1
print('Keep the headset on. Waiting for the camera and head tracking...', flush=True)
while True:
    hd = src.read(24)
    if len(hd) < 24: sys.exit('PROBLEM: the frame grabber stopped')
    magic, w, h, seq, ts = struct.unpack('<4sIIId', hd)
    buf = src.read(w * h)
    if len(buf) < w * h: sys.exit('PROBLEM: the frame grabber stopped')
    now = time.time()
    if not poses:
        if n == 0: print('...no head tracking yet', flush=True)
        n += 1
        if n > 300: sys.exit('PROBLEM: no head tracking arrived from ftoverlay (see its lines above)')
        continue
    if t0 is None:
        t0 = now; n = 0
        print('RECORDING. First 15 s: stand still, turn your head slowly left, right, up, down.', flush=True)
    el = now - t0
    if int(el) // 5 != said:
        said = int(el) // 5; print('  %2d s' % (said * 5) + ('   Now hold your hands in front of you and move them slowly.' if said == 3 else ''), flush=True)
    if el >= SECONDS: break
    if n % EVERY == 0:
        g = cv2.remap(np.frombuffer(buf, np.uint8).reshape(h, w), maps[0], maps[1], cv2.INTER_LINEAR)
        jpgs.append(cv2.imencode('.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()); info.append((now, ts, seq))
    n += 1
out = os.path.join(HERE, 'frametint-rec.npz')
np.savez(out, jpg=np.array(jpgs, dtype=object), info=np.array(info, np.float64), poses=np.array(poses, np.float64), time_mono=time.monotonic(), time_wall=time.time())
print('RESULT saved %d pictures and %d head positions to %s (%.1f MB)' % (len(jpgs), len(poses), out, os.path.getsize(out) / 1e6), flush=True)
os._exit(0)
