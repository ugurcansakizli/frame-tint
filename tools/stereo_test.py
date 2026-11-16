# stereo_test.py: a black-and-white test of "one view per eye". The left eye sees the left front camera, the right
# eye the right front camera, so near things (hands) should look near and the right size. No colors yet.
#
#   ~/.local/share/frametint/venv/bin/python stereo_test.py
#
# Settings (before ./venv/bin/python):
#   FT_MONO=1    both eyes get the same camera (the old way), to compare
#   FT_RES       width of each eye's picture (default 640)
#   FT_IPD       distance between your eyes in metres (default 0.063)
#   FT_SWAP=1    swap the two eyes (if depth looks inside-out)
#   FT_CAL=cad   use the design drawing instead of the factory measurement for the left camera's direction
#   FT_SCALE     panel size, 1 = life size (default 1)
#   FT_PAIR=0    the old way of matching the two cameras' frames (newest of each), to compare
# Stop with Ctrl+C.
import sys, os, json, time, struct, threading, subprocess
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('FT_HOME', os.path.expanduser('~/.local/share/frametint'))
W = int(os.environ.get('FT_RES', '640')) // 8 * 8; H = W * 3 // 4
VIEW_DEG, DIST = 115, 1.5
SCALE = float(os.environ.get('FT_SCALE', '1')); IPD = float(os.environ.get('FT_IPD', '0.063'))
MONO = os.environ.get('FT_MONO', '0') != '0'; SWAP = os.environ.get('FT_SWAP', '0') != '0'; CAL = os.environ.get('FT_CAL', 'factory')
GRAB = os.environ.get('FT_GRAB_CMD', os.path.join(ROOT, 'bin', 'ftgrab')).split()
OVERLAY = os.environ.get('FT_OVERLAY_CMD', os.path.join(ROOT, 'bin', 'ftoverlay')).split()
POSE_LAG = float(os.environ.get('FT_POSE_LAG', '0.02'))
PANEL_W = 2 * DIST * np.tan(np.radians(VIEW_DEG / 2)) * SCALE

cal = None
for p in ('/persist/xrservice.json', os.path.join(HERE, 'device_config.json')):
    try:
        cal = json.load(open(p)); print('lens calibration from', p, flush=True); break
    except Exception as e:
        print('could not use', p, '-', type(e).__name__, flush=True)
if cal is None: sys.exit('PROBLEM: no lens calibration found')

def axes(o):
    x = np.array(o['plus_x'], np.float64); z = np.array(o['plus_z'], np.float64)
    x /= np.linalg.norm(x); z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], axis=1)

C0 = axes(cal['cad']['cameras'][0])                                   # right front camera, in the headset's frame
C1 = axes(cal['cad']['cameras'][1]) if CAL == 'cad' else C0 @ axes(cal['cameras'][1]['extrinsics'])   # left front camera
V = np.diag([-1.0, -1.0, 1.0])

def make_maps(cam, C, shift_px):
    kb = [i for i in cal['cameras'][cam]['intrinsics'] if i.get('cameraModel') == 'kb'][0]
    K = np.array([[kb['fx'], 0, kb['cx']], [0, kb['fy'], kb['cy']], [0, 0, 1]], np.float64)
    D = np.array([kb.get('k1', 0), kb.get('k2', 0), kb.get('k3', 0), kb.get('k4', 0)], np.float64)
    f = W / 2 / np.tan(np.radians(VIEW_DEG / 2))
    Kn = np.array([[f, 0, W / 2 + shift_px], [0, f, H / 2], [0, 0, 1]])
    return cv2.fisheye.initUndistortRectifyMap(K, D, V @ C, Kn, (W, H), cv2.CV_16SC2)

shift = (IPD / 2) / PANEL_W * W                    # so that far-away things are seen straight ahead by each eye
maps_right = make_maps(0, C0, +shift)
maps_left = make_maps(1, C1, -shift)
print('each eye %dx%d, eye shift %.1f pixels, %s' % (W, H, shift, 'SAME camera for both eyes (old way)' if MONO else 'one camera per eye'), flush=True)

cams = [{'raw': None, 't': 0.0, 'n': 0, 'problem': '', 'q': []}, {'raw': None, 't': 0.0, 'n': 0, 'problem': '', 'q': []}]
PAIR = os.environ.get('FT_PAIR', '1') != '0'      # 1: each right frame waits for the left frame of the same moment; 0: the old way (newest of each)
procs = []
def reader(i):
    try:
        pr = subprocess.Popen(GRAB + [str(i), '0', '.', '1'], stdout=subprocess.PIPE, cwd=HERE); procs.append(pr)
    except Exception as e:
        cams[i]['problem'] = 'could not start the frame grabber: %s' % e; return
    src = pr.stdout
    while True:
        hd = src.read(24)
        if len(hd) < 24: cams[i]['problem'] = 'the frame grabber for camera %d stopped (is the Frame Tint camera driver loaded? start Frame Tint once first)' % i; return
        magic, w, h, seq, ts = struct.unpack('<4sIIId', hd)
        if magic != b'FTG1' or w * h == 0 or w * h > 4096 * 4096: cams[i]['problem'] = 'unexpected data from camera %d' % i; return
        buf = src.read(w * h)
        if len(buf) < w * h: cams[i]['problem'] = 'camera %d stopped mid-frame' % i; return
        fr = np.frombuffer(buf, np.uint8).reshape(h, w); q = cams[i]['q']; q.append((ts, fr))
        if len(q) > 6: del q[:len(q) - 4]
        cams[i]['raw'] = fr; cams[i]['t'] = time.time(); cams[i]['ts'] = ts; cams[i]['n'] += 1

poses = []
def pose_reader(ovl):
    while True:
        rec = ovl.stdout.read(52)
        if len(rec) < 52: return
        if rec[:4] != b'FTP1': continue
        m = np.frombuffer(rec, np.float32, 12, 4).reshape(3, 4).copy()
        poses.append((time.time(), m))
        if len(poses) > 200: del poses[:100]
def pose_at(t):
    best = None
    for q in list(poses):
        if best is None or abs(q[0] - t) < abs(best[0] - t): best = q
    return best[1] if best is not None and abs(best[0] - t) < 0.25 else None

for i in ((0,) if MONO else (0, 1)):
    threading.Thread(target=reader, args=(i,), daemon=True).start()
try:
    ovl = subprocess.Popen(OVERLAY + ['%.3f' % PANEL_W, '%.3f' % DIST, '1', 'poses', 'sbs'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=HERE, env=dict(os.environ, FT_TEXEL=os.environ.get('FT_TEXEL', '1')))
except Exception as e:
    sys.exit('PROBLEM: could not start ftoverlay: %s' % e)
threading.Thread(target=pose_reader, args=(ovl,), daemon=True).start()

pic = np.empty((H, 2 * W, 4), np.uint8); pic[..., 3] = 255
last, t0, n0, gap_sum, out_of_step = 0, time.time(), 0, 0.0, 0
def partner(ts_r):
    give_up = time.time() + 0.014
    while True:
        q = list(cams[1]['q'])
        best = min(q, key=lambda x: abs(x[0] - ts_r)) if q else None
        if best is not None and abs(best[0] - ts_r) < 0.012: return best
        if (q and q[-1][0] > ts_r + 0.012) or time.time() > give_up: return best
        time.sleep(0.001)
print('look in the headset. Ctrl+C here to stop.', flush=True)
try:
    while True:
        for c in cams:
            if c['problem']: sys.exit('PROBLEM: ' + c['problem'])
        if ovl.poll() is not None: sys.exit('PROBLEM: the panel program stopped (see its messages above)')
        r = cams[0]; l = cams[0] if MONO else cams[1]
        if r['raw'] is None or l['raw'] is None or r['n'] == last:
            time.sleep(0.002); continue
        last = r['n']; raw_r, t_r, ts_r = r['raw'], r['t'], r['ts']
        if MONO: raw_l, ts_l = raw_r, ts_r
        elif PAIR:
            pr = partner(ts_r); ts_l, raw_l = pr if pr is not None else (ts_r, raw_r)
        else: raw_l, ts_l = l['raw'], l['ts']
        if abs(ts_l - ts_r) >= 0.012: out_of_step += 1
        right = cv2.remap(raw_r, maps_right[0], maps_right[1], cv2.INTER_LINEAR)
        left = right if MONO else cv2.remap(raw_l, maps_left[0], maps_left[1], cv2.INTER_LINEAR)
        if SWAP: left, right = right, left
        pic[:, :W, 0] = pic[:, :W, 1] = pic[:, :W, 2] = left
        pic[:, W:, 0] = pic[:, W:, 1] = pic[:, W:, 2] = right
        pm = pose_at(t_r - POSE_LAG)
        ovl.stdin.write(struct.pack('<4sIII', b'FTV2', 2 * W, H, int(pm is not None)) + (pm if pm is not None else np.zeros((3, 4), np.float32)).tobytes() + pic.tobytes()); ovl.stdin.flush()
        n0 += 1; gap_sum += abs(ts_r - ts_l)
        if time.time() - t0 >= 3:
            print('%.0f pictures/s; the two eyes are %.1f ms apart on average; pictures out of step in the last 3 s: %d; head tracking %s' % (n0 / (time.time() - t0), gap_sum / n0 * 1000, out_of_step, 'on' if pm is not None else 'OFF'), flush=True)
            t0, n0, gap_sum, out_of_step = time.time(), 0, 0.0, 0
except (KeyboardInterrupt, BrokenPipeError):
    print('stopped')
finally:
    for pr in procs:
        try: pr.terminate()
        except Exception: pass
    try: ovl.stdin.close()
    except Exception: pass
