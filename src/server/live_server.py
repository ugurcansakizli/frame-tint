# live_server.py: a live color preview of the Steam Frame's front camera, viewable in a web browser on your PC.
#
#   ~/.local/share/frametint/venv/bin/python live_server.py            (normally started by the "Frame Tint" launcher)
#   then look in the headset (a panel in front of you), or open  http://<headset address>:8000  in a browser
#
# Everything runs as the normal user: the camera driver inside SteamVR (see src/driver) lends the camera buffers to
# ftgrab, so no sudo is needed. The in-headset panel (ftoverlay) joins SteamVR as a background app.
# Settings (normally set in ~/.config/frametint/frametint.conf, or before the command, e.g.  FT_SCALE=1 FT_ALPHA=0.6 python live_server.py):
#   FT_SCALE   size of the panel: 1 = life size (fills most of your view), 0.5 = half size (default)
#   FT_ALPHA   opacity of the panel, 0..1 (default 1)
#   FT_OVERLAY set to 0 to skip the in-headset panel
#   FT_MEMORY  set to 0 to switch the color memory off
#   FT_CORES   processor cores the color model may use (default 4,5,6,7; fastest is 2,3,4,5,6,7 but the headset may stutter)
#   FT_TINT    0 (default): the panel is our whole colored picture, covering the headset's own view.
#              1: the panel is a see-through wash of color over the headset's own view (does not line up well).
#   FT_STRENGTH  how strong the tint is, 0..1.5 (default 1)
#   FT_VIVID   how much stronger than the model's own answer the colors are shown: 1 = as is, 1.8 (default), up to about 3
#   FT_BALANCE white balance, 0..1: how much of the picture's overall color cast is removed (default 0.5, 0 = off)
#   FT_THINGS_EVERY  seconds between outline answers (default 0.2; larger = cooler headset)
#   FT_DISPLAY_CORES  processor cores for the display work (default 2: one fast core to itself)
#   FT_THINGS_CORES   processor cores for the outline model (default 3; with one view per eye 0,1)
#   FT_RES     width of the picture shown, in pixels (default 640; 960 is sharper and costs some speed)
#   FT_STEREO  one view per eye (default 1): the left eye gets the left front camera, the right eye the right one,
#              so near things look near and the right size. 0 = the old flat picture, same for both eyes.
#   FT_IPD     distance between your eyes in metres (default 0.063)
#   FT_LEFT_CORES  processor cores for the left eye's picture (default 3)
#   FT_THINGS  set to 0 to switch off the outline model (one color per thing); needs fastsam-s.onnx (see models/README.md) and segmenter.py
#   FT_SNAP    set to 0 to switch off edge snapping (colors cut to the outlines of things)
#   FT_PIN     set to 0 to stop pinning each picture to the room (the panel then rides with the head)
#
# Color memory: colors are stored by direction in the room (using the headset's own head tracking), so a thing keeps
# its color when you look away and back, colors stay put when you turn your head, and every new answer for a spot is
# averaged into what is already known about it. Where a spot no longer looks like what was remembered (a hand, a
# moved thing, you leaning), the newest answer's colors are used instead, carried along with whatever moved.
#
# What runs here, on the headset:
#   - ./grab streams the camera's long-exposure frames (about 30 a second).
#   - The color model answers as fast as it can (roughly 2 a second) from the newest frame.
#   - The picture you see is always the NEWEST camera frame for brightness, with the model's latest colors laid on
#     top. Brightness is live; colors come from the color memory (or, without head tracking, from the latest answer).
# Stop with Ctrl+C.
import sys, os, json, time, struct, threading, subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, cv2, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
# Install layout: <ROOT>/bin (ftgrab, ftoverlay), <ROOT>/lib (this file, segmenter.py), <ROOT>/models, <ROOT>/venv.
# Running from a git checkout instead? Point FT_HOME at the installed folder (default ~/.local/share/frametint).
_up = os.path.dirname(HERE)
ROOT = os.environ.get('FT_HOME') or (_up if os.path.isdir(os.path.join(_up, 'bin')) else os.path.expanduser('~/.local/share/frametint'))
def model_file(name):
    """a model by file name: an absolute path as is, else <ROOT>/models/<name>, else next to this file"""
    if os.path.isabs(name): return name
    for d in (os.path.join(ROOT, 'models'), HERE):
        if os.path.exists(os.path.join(d, name)): return os.path.join(d, name)
    return os.path.join(ROOT, 'models', name)
MODEL = next((a for a in sys.argv[1:] if not a.startswith('--')), 'ddcolor-fast2-8bit-256.onnx')
PORT = int(os.environ.get('FT_PORT', '8000'))
GRAB = os.environ.get('FT_GRAB', os.path.join(ROOT, 'bin', 'ftgrab') + ' 0 0 . 1').split()
STDIN = '--stdin' in sys.argv
SCALE = float(os.environ.get('FT_SCALE', '0.5')); ALPHA = float(os.environ.get('FT_ALPHA', '1')); DIST = 1.5
OVERLAY = os.environ.get('FT_OVERLAY_CMD', os.path.join(ROOT, 'bin', 'ftoverlay')).split() if os.environ.get('FT_OVERLAY', '1') != '0' else None
CAMERA, VIEW_DEG = 0, 115
MW, MH = 960, 720            # straightened picture the model looks at
PW, PH = 640, 480            # size the color work is based on
OW = max(PW, int(os.environ.get('FT_RES', str(PW))) // 8 * 8); OH = OW * PH // PW      # size of the picture shown
MEMORY = os.environ.get('FT_MEMORY', '1') != '0'
VIVID = float(os.environ.get('FT_VIVID', '1.8'))
BALANCE = float(os.environ.get('FT_BALANCE', '0.5'))
THINGS_EVERY = float(os.environ.get('FT_THINGS_EVERY', '0.2'))
STEREO = os.environ.get('FT_STEREO', '1') != '0' and os.environ.get('FT_TINT', '0') == '0' and OVERLAY is not None
IPD = float(os.environ.get('FT_IPD', '0.063'))
LEFT_CORES = set(int(c) for c in os.environ.get('FT_LEFT_CORES', '3').split(',') if c.strip() != '') if STEREO else set()
GRAB_LEFT = os.environ.get('FT_GRAB_LEFT', os.path.join(ROOT, 'bin', 'ftgrab') + ' 1 0 . 1').split()
DISPLAY_CORES = set(int(c) for c in os.environ.get('FT_DISPLAY_CORES', '2').split(',') if c.strip() != '')
cv2.setNumThreads(int(os.environ.get('FT_CV_THREADS', '1')))   # picture work stays on the core of the thread that asks for it
_g = np.arange(256, dtype=np.uint8).reshape(1, 256)
LUT_L = cv2.cvtColor(cv2.cvtColor(_g, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[:, :, 0].copy()   # gray value -> Lab brightness
LUT_L8 = np.clip(np.round(LUT_L * 2.55), 0, 255).astype(np.uint8)
TINT = os.environ.get('FT_TINT', '0') != '0'; STRENGTH = float(os.environ.get('FT_STRENGTH', '1')); PIN = os.environ.get('FT_PIN', '1') != '0'
TW, TH_ = PW // 2, PH // 2    # the tint is soft, so a small picture is enough
CORES = sorted(int(c) for c in os.environ.get('FT_CORES', '4,5,6,7').split(',') if c.strip() != '')
POSE_LAG = float(os.environ.get('FT_POSE_LAG', '0.02'))   # a camera frame reaches us about this long after it was taken
SMOOTH = 0.5                 # how much of the previous colors to keep at each new answer (steadier, slower to change)

def load_calibration():
    for p in ('/persist/xrservice.json', os.path.join(HERE, 'device_config.json')):
        try:
            c = json.load(open(p)); print('lens calibration from', p, flush=True); return c
        except Exception as e:
            print('could not use', p, '-', type(e).__name__, flush=True)
    sys.exit('PROBLEM: no lens calibration found')

def make_maps(cal, w, h):
    cam = cal['cameras'][CAMERA]
    kb = [i for i in cam['intrinsics'] if i.get('cameraModel') == 'kb'][0]
    K = np.array([[kb['fx'], 0, kb['cx']], [0, kb['fy'], kb['cy']], [0, 0, 1]], np.float64)
    D = np.array([kb.get('k1', 0), kb.get('k2', 0), kb.get('k3', 0), kb.get('k4', 0)], np.float64)
    cad = cal['cad']['cameras'][CAMERA]
    x = np.array(cad['plus_x'], np.float64); z = np.array(cad['plus_z'], np.float64)
    x /= np.linalg.norm(x); z /= np.linalg.norm(z); y = np.cross(z, x)
    C = np.stack([x, y, z], axis=1)
    V = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64).T
    f = w / 2 / np.tan(np.radians(VIEW_DEG / 2))
    Kn = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    return cv2.fisheye.initUndistortRectifyMap(K, D, (C.T @ V).T, Kn, (w, h), cv2.CV_16SC2)

def make_left_maps(cal, w, h):
    """the left front camera, straightened to the same forward view as the right one (factory-measured angle between them)"""
    def axes(o):
        x = np.array(o['plus_x'], np.float64); z = np.array(o['plus_z'], np.float64)
        x /= np.linalg.norm(x); z /= np.linalg.norm(z)
        return np.stack([x, np.cross(z, x), z], axis=1)
    C1 = axes(cal['cad']['cameras'][CAMERA]) @ axes(cal['cameras'][1]['extrinsics'])
    kb = [i for i in cal['cameras'][1]['intrinsics'] if i.get('cameraModel') == 'kb'][0]
    K = np.array([[kb['fx'], 0, kb['cx']], [0, kb['fy'], kb['cy']], [0, 0, 1]], np.float64)
    D = np.array([kb.get('k1', 0), kb.get('k2', 0), kb.get('k3', 0), kb.get('k4', 0)], np.float64)
    f = w / 2 / np.tan(np.radians(VIEW_DEG / 2))
    Kn = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    return cv2.fisheye.initUndistortRectifyMap(K, D, np.diag([-1.0, -1.0, 1.0]) @ C1, Kn, (w, h), cv2.CV_16SC2)

cal = load_calibration()
model_maps = make_maps(cal, MW, MH)
view_maps = make_maps(cal, PW, PH)
out_maps = view_maps if OW == PW else make_maps(cal, OW, OH)
left_maps = None
if STEREO:
    try:
        left_maps = make_left_maps(cal, OW, OH)
    except Exception as e:
        print('one view per eye is off: could not set up the left camera (%s: %s)' % (type(e).__name__, e), flush=True); STEREO = False; LEFT_CORES = set()
SHM = None; SHM_PATH = '/dev/shm/frametint-view'
if STEREO:
    try:                                           # three picture slots shared with the panel program: no pixels through the pipe
        SHM = np.memmap(SHM_PATH, dtype=np.uint8, mode='w+', shape=(3, OH, 2 * OW, 4)); SHM[..., 3] = 255
    except Exception as e:
        print('no shared memory (%s: %s); pictures go through the pipe' % (type(e).__name__, e), flush=True); SHM = None
EYE_SHIFT = int(round((IPD / 2) / (2 * DIST * np.tan(np.radians(VIEW_DEG / 2)) * SCALE) * OW)) if STEREO else 0   # far things straight ahead of each eye
try:
    _kb = [i for i in cal['cameras'][CAMERA]['intrinsics'] if i.get('cameraModel') == 'kb'][0]
    print('picture shown at %dx%d; the camera itself would fill about %d pixels across this view' % (OW, OH, round(2 * _kb['fx'] * np.tan(np.radians(VIEW_DEG / 2)))), flush=True)
except Exception:
    pass

VF = PW / 2 / np.tan(np.radians(VIEW_DEG / 2))     # focal length of the straightened preview, in pixels

# --- color memory v2 + motion-following; pasted into live_server.py between the markers ---
TH = [10, 28]                                      # how different a spot must look before memory gives way
def _ss(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)

class Memory:
    """Colors remembered by direction in the room: a 360-degree map of color (Lab a,b), how sure we are, and brightness."""
    W, H, CAP = 1024, 512, 12.0
    def __init__(s, PW, PH, VF):
        s.PW, s.PH, s.VF = PW, PH, VF
        lon = (np.arange(s.W, dtype=np.float32) + 0.5) / s.W * 2 * np.pi - np.pi
        lat = np.pi / 2 - (np.arange(s.H, dtype=np.float32) + 0.5) / s.H * np.pi
        lon, lat = np.meshgrid(lon, lat)
        s.dirs = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), -np.cos(lat) * np.cos(lon)], -1).reshape(-1, 3).astype(np.float32)
        s.abl = np.zeros((s.H, s.W, 3), np.float32)    # a, b, brightness
        s.w = np.zeros((s.H, s.W), np.float32); s.lock = threading.Lock()
        gw, gh = PW // 4, PH // 4                      # the view is looked up on a coarse grid; color is soft anyway
        u = (np.arange(gw, dtype=np.float32) + 0.5) * 4; v = (np.arange(gh, dtype=np.float32) + 0.5) * 4
        u, v = np.meshgrid(u, v)
        d = np.stack([(u - PW / 2) / VF, -(v - PH / 2) / VF, -np.ones_like(u)], -1)
        s.view_dirs = (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3).astype(np.float32); s.gshape = (gh, gw)
    def write(s, ab, gray, R):
        """ab: (2,h,w) colors for a camera frame, gray: that frame, R: how the head was turned. Any size, same view."""
        PW, PH, VF = s.PW, s.PH, s.VF
        dh = s.dirs @ R.astype(np.float32)             # room directions as seen from the head
        zf = -dh[:, 2]; ok = zf > 0.3; zs = np.where(ok, zf, 1)
        u = PW / 2 + VF * dh[:, 0] / zs; v = PH / 2 - VF * dh[:, 1] / zs
        ok &= (u > 0) & (u < PW) & (v > 0) & (v < PH)
        ok = ok.reshape(s.H, s.W)
        ys, xs = np.where(ok.any(1))[0], np.where(ok.any(0))[0]
        if len(ys) == 0: return
        x0, x1 = (0, s.W) if xs[-1] - xs[0] > s.W * 0.75 else (xs[0], xs[-1] + 1)   # full width when across the seam
        y0, y1 = ys[0], ys[-1] + 1
        um = u.reshape(s.H, s.W)[y0:y1, x0:x1].astype(np.float32); vm = v.reshape(s.H, s.W)[y0:y1, x0:x1].astype(np.float32)
        okc = ok[y0:y1, x0:x1]
        edge = np.minimum(np.minimum(um, PW - um) / (PW * 0.12), np.minimum(vm, PH - vm) / (PH * 0.12))
        e = np.where(okc, np.clip(edge - 0.25, 0, 1), 0).astype(np.float32)        # trust fades out toward the picture's edge
        sx, sy = ab.shape[2] / PW, ab.shape[1] / PH
        src = np.dstack([ab[0], ab[1], cv2.GaussianBlur(cv2.resize(gray, (ab.shape[2], ab.shape[1]), interpolation=cv2.INTER_AREA), (0, 0), 3).astype(np.float32)])
        new = cv2.remap(src, um * sx - 0.5, vm * sy - 0.5, cv2.INTER_LINEAR)
        with s.lock:
            w = s.w[y0:y1, x0:x1]; A = s.abl[y0:y1, x0:x1]
            changed = _ss(np.abs(new[..., 2] - A[..., 2]), 20, 50) * (w > 0)       # it looks different there now:
            w *= (1 - 0.8 * changed * (e > 0))                                     # let the new answer mostly replace it
            alpha = e / (w + e + 1e-6)
            A += (new - A) * alpha[..., None]
            w += e; np.minimum(w, s.CAP, out=w)
    def read(s, R):
        """for the view of a head turned as R: colors (2,h,w), remembered brightness (h,w), how sure (h,w); quarter size."""
        dw = s.view_dirs @ R.T.astype(np.float32)
        lon = np.arctan2(dw[:, 0], -dw[:, 2]); lat = np.arcsin(np.clip(dw[:, 1], -1, 1))
        px = ((lon / np.pi + 1) / 2 * s.W - 0.5).reshape(s.gshape).astype(np.float32)
        py = ((0.5 - lat / np.pi) * s.H - 0.5).reshape(s.gshape).astype(np.float32)
        with s.lock:
            a = cv2.remap(s.abl, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            c = cv2.remap(np.minimum(s.w, 1.0), px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return a[..., :2].transpose(2, 0, 1), a[..., 2], c
    def filled(s):
        return float((s.w > 0).mean())

class Follower:
    """Carries the newest answer's colors onto the newest camera frame: first undoes the head turn since the answer's
    frame, then follows whatever else moved (hands, things, you leaning) by matching the two frames."""
    def __init__(s, PW, PH, VF, scale=2):
        s.w, s.h = PW // scale, PH // scale; f = VF / scale
        s.K = np.array([[f, 0, s.w / 2], [0, f, s.h / 2], [0, 0, 1]]); s.Ki = np.linalg.inv(s.K); s.S = np.diag([1.0, -1.0, -1.0])
        s.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        gx, gy = np.meshgrid(np.arange(s.w, dtype=np.float32), np.arange(s.h, dtype=np.float32)); s.gx, s.gy = gx, gy
        s.ans = None
    def new_answer(s, ab, gray, R):
        s.ans = (np.ascontiguousarray(np.dstack([cv2.resize(ab[0], (s.w, s.h), interpolation=cv2.INTER_AREA), cv2.resize(ab[1], (s.w, s.h), interpolation=cv2.INTER_AREA)])),
                 cv2.resize(gray, (s.w, s.h), interpolation=cv2.INTER_AREA), None if R is None else R.copy())
    def render(s, gray_small, R):
        """colors (h,w,2) for the newest frame, and how badly the match failed (h,w, 0..255)."""
        ab, g0, R0 = s.ans
        if R is not None and R0 is not None:
            Hm = s.K @ s.S @ R0.T @ R @ s.S @ s.Ki
            ab = cv2.warpPerspective(ab, Hm, (s.w, s.h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            g0 = cv2.warpPerspective(g0, Hm, (s.w, s.h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        fl = s.dis.calc(gray_small, g0, None)
        mx, my = s.gx + fl[..., 0], s.gy + fl[..., 1]
        out = cv2.remap(ab, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        miss = cv2.GaussianBlur(cv2.absdiff(cv2.remap(g0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), gray_small), (0, 0), 3)
        return out, miss

def guided(ab, gray, r=10, eps=0.004):
    """edge-following smoothing: ab (h,w,2) float, gray (h,w) uint8 -> ab that changes where the gray picture has an edge"""
    I = gray.astype(np.float32) / 255.0; k = (2 * r + 1, 2 * r + 1)
    mI = cv2.blur(I, k); vI = cv2.blur(I * I, k) - mI * mI
    out = np.empty_like(ab)
    for c in range(2):
        p = ab[..., c]; mp = cv2.blur(p, k); cov = cv2.blur(I * p, k) - mI * mp
        a = cov / (vI + eps); b = mp - a * mI
        out[..., c] = cv2.blur(a, k) * I + cv2.blur(b, k)
    return out
class Regions:
    """cuts the picture into small patches that follow the edges, and gives every patch one color"""
    def __init__(s, w, h, step=12):
        s.w, s.h = w, h
        ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
        s.seeds = np.zeros((h, w), np.int32); s.seeds[ys, xs] = np.arange(1, ys.size + 1).reshape(ys.shape); s.n = ys.size + 1
    def labels(s, gray):
        g = cv2.GaussianBlur(gray, (0, 0), 1.2)
        m = s.seeds.copy(); cv2.watershed(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), m)
        m[m < 0] = 0
        return m
    def flatten(s, ab, gray):
        lab = s.labels(gray).ravel(); cnt = np.bincount(lab, minlength=s.n).astype(np.float32) + 1e-6
        out = np.empty_like(ab)
        for c in range(2):
            mean = np.bincount(lab, weights=ab[..., c].ravel(), minlength=s.n) / cnt
            out[..., c] = mean[lab].reshape(s.h, s.w)
        return out, lab.reshape(s.h, s.w)

class Things:
    def __init__(s, w, h):
        s.w, s.h = w, h; s.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)); s.gx, s.gy = gx, gy
        s.cur = None
    def new_outlines(s, lab, gray_small, Hfun_R):
        s.cur = (lab, gray_small, Hfun_R)
    def carry(s, gray_small, homography):
        """the outlines moved onto the newest frame (undo head turn, then follow motion)"""
        lab, g0, _ = s.cur
        if homography is not None:
            lab = cv2.warpPerspective(lab.astype(np.float32), homography, (s.w, s.h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP).astype(np.int32)
            g0 = cv2.warpPerspective(g0, homography, (s.w, s.h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        fl = s.dis.calc(gray_small, g0, None)
        return cv2.remap(lab.astype(np.float32), s.gx + fl[..., 0], s.gy + fl[..., 1], cv2.INTER_NEAREST).astype(np.int32)
def paint(ab, patch_lab, n_patch, thing_lab):
    """ab (h,w,2); patch_lab (h,w) edge-following patches of the newest frame; thing_lab (h,w) carried outlines.
    Each patch votes for the thing it mostly lies in; each thing gets the average color of its patches."""
    pl = patch_lab.ravel(); tl = thing_lab.ravel(); nt = int(tl.max()) + 1
    votes = np.bincount(pl.astype(np.int64) * nt + tl, minlength=n_patch * nt).reshape(n_patch, nt)
    owner = votes.argmax(1); share = votes[np.arange(n_patch), owner] / (votes.sum(1) + 1e-6)
    owner[share < 0.6] = 0                                   # a patch split between things stays on its own
    cnt = np.bincount(pl, minlength=n_patch).astype(np.float32) + 1e-6
    out = np.empty_like(ab)
    for c in range(2):
        pm = np.bincount(pl, weights=ab[..., c].ravel(), minlength=n_patch) / cnt           # color per patch
        tsum = np.bincount(owner, weights=pm * cnt, minlength=nt); tcnt = np.bincount(owner, weights=cnt, minlength=nt) + 1e-6
        tm = tsum / tcnt                                                                     # color per thing
        col = np.where(owner > 0, tm[owner], pm)
        out[..., c] = col[pl].reshape(ab.shape[:2])
    return out

SNAP = True
STEREO_SRC = {'src': None}
TIMES = {}                                         # how long each part of the display work takes (ms, smoothed)
def _tick(name, t):
    ms = (time.perf_counter() - t) * 1000; TIMES[name] = ms if name not in TIMES else TIMES[name] * 0.9 + ms * 0.1
    return time.perf_counter()
_regions = {}
things = None                                      # set when the outline model is running

def blend(mem, fol, gray, R, PW, PH):
    """the colors to show for the newest frame: remembered where the room still looks the same, followed elsewhere."""
    t = time.perf_counter()
    gs = cv2.resize(gray, (fol.w, fol.h), interpolation=cv2.INTER_AREA)
    fresh, miss = fol.render(gs, R)
    fresh = fresh * (1 - 0.5 * _ss(miss.astype(np.float32), 30, 80))[..., None]     # no good match: tone the color down
    t = _tick('follow', t)
    if mem is not None and R is not None:
        mab, mL, c = mem.read(R)
        gq = cv2.GaussianBlur(cv2.resize(gs, (mL.shape[1], mL.shape[0]), interpolation=cv2.INTER_AREA), (0, 0), 1.5).astype(np.float32)
        wm = c * (1 - _ss(np.abs(gq - mL), TH[0], TH[1]))
        wm = cv2.resize(cv2.GaussianBlur(wm, (0, 0), 1.0), (fol.w, fol.h), interpolation=cv2.INTER_LINEAR)[..., None]
        mab = cv2.resize(np.ascontiguousarray(mab.transpose(1, 2, 0)), (fol.w, fol.h), interpolation=cv2.INTER_LINEAR)
        fresh = mab * wm + fresh * (1 - wm)
    t = _tick('memory', t)
    if SNAP:                                       # one color per edge-following patch, so colors stop at the edges of things
        if (fol.w, fol.h) not in _regions: _regions[(fol.w, fol.h)] = Regions(fol.w, fol.h)
        reg = _regions[(fol.w, fol.h)]; fresh = np.ascontiguousarray(fresh, dtype=np.float32)
        th = things
        if th is not None and th.cur is not None:  # one color per thing: the outline model says which patches belong together
            pl = reg.labels(gs); R0 = th.cur[2]
            Hm = fol.K @ fol.S @ R0.T @ R @ fol.S @ fol.Ki if (R is not None and R0 is not None) else None
            fresh = guided(paint(fresh, pl, reg.n, th.carry(gs, Hm)), gs, 3, 0.002)
        else:
            flat, pl = reg.flatten(fresh, gs)
            fresh = guided(flat, gs, 4, 0.002)
    t = _tick('edges', t)
    if VIVID != 1:                                 # stronger colors; weak ones are pushed apart, strong ones level off
        fresh = np.ascontiguousarray(fresh, dtype=np.float32)
        c = cv2.magnitude(fresh[..., 0], fresh[..., 1]) + 1e-3
        fresh = fresh * (75.0 * np.tanh(VIVID * c / 75.0) / c)[..., None]
    t = _tick('vivid', t)
    STEREO_SRC['src'] = (np.ascontiguousarray(fresh, dtype=np.float32), gs)      # the left eye borrows these colors
    out = cv2.resize(fresh, (PW, PH), interpolation=cv2.INTER_LINEAR)
    return out.transpose(2, 0, 1)
# --- end of color memory v2 ---

SNAP = os.environ.get('FT_SNAP', '1') != '0'
memory = Memory(PW, PH, VF) if MEMORY else None
follower = Follower(PW, PH, VF)
poses = []                                         # recent (time, R, position) from the headset's head tracking

def pose_at(t):
    best = None
    for q in list(poses):
        if best is None or abs(q[0] - t) < abs(best[0] - t): best = q
    return best if best is not None and abs(best[0] - t) < 0.25 else None

acks = {'n': 0, 'w': 0, 't': time.time(), 'pw': 0, 'pt': time.time()}; ack_cv = threading.Condition()
def pose_reader(ovl):
    while True:
        rec = ovl.stdout.read(52)
        if len(rec) < 52: return
        if rec[:4] == b'FTA1':                    # the panel program has handed a picture to SteamVR
            with ack_cv:
                acks['n'] += 1; ack_cv.notify_all()
            acks['w'] += 1
            if time.time() - acks['t'] >= 2:
                state['panel_fps'] = acks['w'] / (time.time() - acks['t']); acks['t'], acks['w'] = time.time(), 0
            continue
        if rec[:4] == b'FTQ1':                    # Quit was pressed in the dashboard panel
            print('Quit pressed in the dashboard: stopping Frame Tint', flush=True); os._exit(0)
        if rec[:4] == b'FTT1':                    # the user switched Frame Tint on or off with the controller
            state['paused'] = struct.unpack('<f', rec[4:8])[0] == 0
            print('Frame Tint is now', 'PAUSED (hidden)' if state['paused'] else 'ON', flush=True)
            continue
        if rec[:4] != b'FTP1': continue
        m = np.frombuffer(rec, np.float32, 12, 4).reshape(3, 4)
        poses.append((time.time(), m[:, :3].astype(np.float64), m[:, 3].astype(np.float64)))
        acks['pw'] += 1
        if time.time() - acks['pt'] >= 2:
            state['pose_fps'] = acks['pw'] / (time.time() - acks['pt']); acks['pt'], acks['pw'] = time.time(), 0
        if len(poses) > 200: del poses[:100]

state = {'paused': False, 'raw': None, 'pose': None, 'mem': False, 'seq': 0, 'frames': 0, 'ab': None, 'answers': 0, 'model_ms': 0.0, 'jpeg': None, 'jpeg_n': 0,
         'cam_fps': 0.0, 'ans_fps': 0.0, 'problem': '', 'last': 0.0, 'shown': 0, 'viewers': 0, 'things': -1, 'things_ms': 0.0, 'disp_fps': 0.0, 'raw_l': None, 'last_l': 0.0, 'panel_fps': -1.0, 'ts': 0.0, 'mispairs': 0, 'pair_ms': 0.0, 'pose_fps': 0.0, 'pose_age': 0.0}
lock = threading.Lock()
new_jpeg = threading.Condition()

def reader():                                     # keeps only the newest camera frame
    if STDIN:
        src = sys.stdin.buffer
    else:
        try:
            src = subprocess.Popen(GRAB, stdout=subprocess.PIPE, cwd=HERE).stdout
        except Exception as e:
            state['problem'] = 'could not start the frame grabber: %s' % e; return
    t0, n0 = time.time(), 0
    while True:
        hd = src.read(24)
        if len(hd) < 24:
            state['problem'] = 'the frame grabber stopped (see its PROBLEM line above)'; return
        magic, w, h, seq, ts = struct.unpack('<4sIIId', hd)
        if magic != b'FTG1' or w * h == 0 or w * h > 4096 * 4096:
            state['problem'] = 'unexpected data from the frame grabber'; return
        buf = src.read(w * h)
        if len(buf) < w * h:
            state['problem'] = 'the frame grabber stopped mid-frame'; return
        with lock:
            state['raw'] = np.frombuffer(buf, np.uint8).reshape(h, w); state['ts'] = ts; state['pose'] = pose_at(time.time() - POSE_LAG); state['pose_age'] = state['pose_age'] * 0.9 + 0.1 * ((time.time() - POSE_LAG - state['pose'][0]) * 1000 if state['pose'] is not None else 250); state['seq'] = state['frames'] + 1; state['frames'] += 1; state['last'] = time.time()
        n0 += 1
        if time.time() - t0 >= 2:
            state['cam_fps'] = n0 / (time.time() - t0); t0, n0 = time.time(), 0

left_q = []                                        # the last few left-camera frames, each with the moment its grabber saw it
def left_partner(ts_r):
    """the left-camera frame taken at the same moment as the right-camera frame seen at ts_r (both grabbers use the same
    clock). Showing the two eyes pictures from different moments makes the view jump whenever the head moves."""
    give_up = time.time() + 0.014
    while True:
        q = list(left_q)
        best = min(q, key=lambda x: abs(x[0] - ts_r)) if q else None
        if best is not None and abs(best[0] - ts_r) < 0.012: return best
        if (q and q[-1][0] > ts_r + 0.012) or time.time() > give_up: return best     # its partner is not coming
        time.sleep(0.001)

def reader_left():                                # the left front camera, newest frame only
    while True:
        try:
            src = subprocess.Popen(GRAB_LEFT, stdout=subprocess.PIPE, cwd=HERE).stdout
        except Exception as e:
            print('left camera: could not start its frame grabber (%s); both eyes get the right camera' % e, flush=True); return
        while True:
            hd = src.read(24)
            if len(hd) < 24: break
            magic, w, h, seq, ts = struct.unpack('<4sIIId', hd)
            if magic != b'FTG1' or w * h == 0 or w * h > 4096 * 4096: break
            buf = src.read(w * h)
            if len(buf) < w * h: break
            left_q.append((ts, np.frombuffer(buf, np.uint8).reshape(h, w)))
            if len(left_q) > 6: del left_q[:len(left_q) - 4]
            state['raw_l'] = left_q[-1][1]; state['last_l'] = time.time()
        print('left camera: its frame grabber stopped (is the Frame Tint camera driver loaded?); both eyes get the right camera. Trying again in 5 s', flush=True)
        state['raw_l'] = None; time.sleep(5)

class LeftEye:
    """Makes the left eye's picture on its own core while the main display work makes the right eye's.
    Its colors are the right eye's colors from one picture ago, moved to where things are in the left camera's view."""
    def __init__(s):
        s.cv = threading.Condition(); s.job = None; s.done = threading.Event(); s.result = None
        s.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST); s.grid = None
    def ask(s, raw_l):
        s.done.clear()
        with s.cv:
            s.job = raw_l; s.cv.notify()
    def run(s):
        try:
            mine = LEFT_CORES & ALL_CORES
            if mine: os.sched_setaffinity(0, mine)
            print('left eye on cores', sorted(os.sched_getaffinity(0)), flush=True)
        except Exception as e:
            print('could not choose cores:', e, flush=True)
        while True:
            with s.cv:
                s.cv.wait_for(lambda: s.job is not None)
                raw_l, s.job = s.job, None
            try:
                s.result = s.make(raw_l)
            except Exception as e:
                print('left eye problem: %s: %s' % (type(e).__name__, e), flush=True); s.result = None
            s.done.set()
    def make(s, raw_l):
        t = time.perf_counter()
        lo = cv2.remap(raw_l, left_maps[0], left_maps[1], cv2.INTER_LINEAR)
        src = STEREO_SRC['src']
        if src is None:
            out = cv2.cvtColor(lo, cv2.COLOR_GRAY2BGR)
        else:
            ab, gs = src; fh, fw = gs.shape
            if s.grid is None or s.grid[0].shape != gs.shape:
                s.grid = np.meshgrid(np.arange(fw, dtype=np.float32), np.arange(fh, dtype=np.float32))
            ls = cv2.resize(lo, (fw, fh), interpolation=cv2.INTER_AREA)
            fl = s.dis.calc(ls, gs, None)                                   # where each spot of the left view is in the right view
            ab = cv2.remap(ab, s.grid[0] + fl[..., 0], s.grid[1] + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ab = guided(ab, ls, 3, 0.002)
            ab = cv2.resize(ab, (OW, OH), interpolation=cv2.INTER_LINEAR)
            lab8 = np.empty((OH, OW, 3), np.uint8)
            lab8[..., 0] = cv2.LUT(lo, LUT_L8); lab8[..., 1] = np.clip(ab[..., 0] + 128.5, 0, 255); lab8[..., 2] = np.clip(ab[..., 1] + 128.5, 0, 255)
            out = cv2.cvtColor(lab8, cv2.COLOR_Lab2BGR)
        _tick('left eye', t)
        return out

def colorer():                                    # the slow part: asks the model for colors, newest frame each time
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    try:                                          # the model gets the fast cores to itself; everything else keeps off them
        mine = set(CORES) & ALL_CORES
        if mine: os.sched_setaffinity(0, mine); so.intra_op_num_threads = len(mine)
        print('color model on cores', sorted(os.sched_getaffinity(0)), 'with', so.intra_op_num_threads, 'threads', flush=True)
    except Exception as e:
        print('could not choose cores:', e, flush=True)
    sess = ort.InferenceSession(model_file(MODEL), so, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name; S = sess.get_inputs()[0].shape[2]
    print('model ready:', MODEL, flush=True)
    last, t0, n0 = 0, time.time(), 0
    cast = None                                   # the overall color cast, smoothed over the last few answers
    while True:
        if state['paused']: time.sleep(0.1); continue
        with lock:
            raw, seq, pose = state['raw'], state['seq'], state['pose']
        if raw is None or seq == last:
            time.sleep(0.005); continue
        last = seq
        g = cv2.remap(raw, model_maps[0], model_maps[1], cv2.INTER_LINEAR)
        x = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        t = time.time()
        ab = sess.run(None, {name: np.repeat(x[None, None], 3, axis=1)})[0][0]
        ms = (time.time() - t) * 1000
        if BALANCE > 0:                           # white balance: take out part of the most common color (walls, ceiling)
            ok = x[::4, ::4] > 0.03               # not the black corners
            if ok.sum() > 100:
                med = np.array([np.median(ab[0][::4, ::4][ok]), np.median(ab[1][::4, ::4][ok])], np.float32)
                cast = med if cast is None else cast * 0.7 + med * 0.3
            if cast is not None:
                ab = ab - (BALANCE * cast)[:, None, None]
        small = np.stack([cv2.resize(ab[0], (PW, PH), interpolation=cv2.INTER_AREA), cv2.resize(ab[1], (PW, PH), interpolation=cv2.INTER_AREA)])
        gs = cv2.resize(g, (PW, PH), interpolation=cv2.INTER_AREA)
        if memory is not None and pose is not None:
            memory.write(small, gs, pose[1])
        follower.new_answer(small, gs, None if pose is None else pose[1])
        with lock:
            state['mem'] = memory is not None and pose is not None
            state['ab'] = small if state['ab'] is None else state['ab'] * SMOOTH + small * (1 - SMOOTH)
            state['answers'] += 1; state['model_ms'] = ms
        n0 += 1
        if time.time() - t0 >= 4:
            state['ans_fps'] = n0 / (time.time() - t0); t0, n0 = time.time(), 0

def outliner():                                   # finds the outlines of things a few times a second
    global things
    path = model_file(os.environ.get('FT_THINGS_MODEL', 'fastsam-s.onnx'))
    if os.environ.get('FT_THINGS', '1') == '0' or not SNAP: return
    try:
        import segmenter
        mine = set(int(c) for c in os.environ.get('FT_THINGS_CORES', '0,1' if STEREO else '3').split(',')) & ALL_CORES
        if mine: os.sched_setaffinity(0, mine)
        seg = segmenter.Segmenter(path, threads=max(1, len(mine)))
    except Exception as e:
        print('no outline model (%s: %s); colors follow edges only' % (type(e).__name__, e), flush=True); return
    th = Things(follower.w, follower.h); things = th
    print('outline model ready on cores', sorted(os.sched_getaffinity(0)), flush=True)
    last = 0
    while True:
        if state['paused']: time.sleep(0.1); continue
        with lock:
            raw, seq, pose = state['raw'], state['seq'], state['pose']
        if raw is None or seq == last:
            time.sleep(0.01); continue
        last = seq; t = time.time()
        g = cv2.remap(raw, view_maps[0], view_maps[1], cv2.INTER_LINEAR)
        mk = seg.masks(g)
        th.new_outlines(segmenter.label_map(mk, th.w, th.h), cv2.resize(g, (th.w, th.h), interpolation=cv2.INTER_AREA), None if pose is None else pose[1])
        dt = time.time() - t; state['things'] = len(mk); state['things_ms'] = dt * 1000
        time.sleep(max(0.0, THINGS_EVERY - dt))

def _panel_cores():                               # the panel program may use every core the color model and the right eye do not
    try: os.sched_setaffinity(0, (ALL_CORES - set(CORES) - DISPLAY_CORES) or ALL_CORES)
    except Exception: pass

def start_overlay():                              # the in-headset panel, if ./ftoverlay has been built
    import shutil
    if not OVERLAY or not (os.path.exists(OVERLAY[0]) or shutil.which(OVERLAY[0])):
        print('no in-headset panel (ftoverlay not built or switched off); browser preview only', flush=True); return None
    width = 2 * DIST * np.tan(np.radians(VIEW_DEG / 2)) * SCALE          # life size at SCALE = 1
    try:
        ovl = subprocess.Popen(OVERLAY + ['%.3f' % width, '%.3f' % DIST, '%.3f' % ALPHA, 'poses'] + (['sbs'] if STEREO else []), stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=HERE,
                               env=dict(os.environ, FT_TEXEL=os.environ.get('FT_TEXEL', '1'), **({'FT_SHM': SHM_PATH} if SHM is not None else {})),
                               preexec_fn=_panel_cores)
        threading.Thread(target=pose_reader, args=(ovl,), daemon=True).start()
        return ovl
    except Exception as e:
        print('could not start the in-headset panel:', e, flush=True); return None

_ex, _ey = np.meshgrid(np.arange(TW, dtype=np.float32), np.arange(TH_, dtype=np.float32))
EDGE = np.clip(np.minimum(np.minimum(_ex, TW - 1 - _ex) / (TW * 0.08), np.minimum(_ey, TH_ - 1 - _ey) / (TH_ * 0.08)), 0, 1)

def vivid(ab):
    """Stronger colors: weak and medium colors are pushed apart, strong ones level off instead of going garish."""
    if VIVID == 1: return ab
    c = np.sqrt(ab[0] * ab[0] + ab[1] * ab[1]) + 1e-3
    return ab * (75.0 * np.tanh(VIVID * c / 75.0) / c)[None]

def tint_picture(g, ab):
    """A see-through wash (RGBA) that, laid over the headset's own gray view, shifts it toward the colors in ab.
    Shown result = (1-alpha)*gray + alpha*C. We want gray + shift, so C = gray + shift/alpha, with alpha per pixel as
    small as it can be while C stays a displayable color: where there is little color the headset's own view is untouched."""
    gs = cv2.GaussianBlur(cv2.resize(g, (TW, TH_), interpolation=cv2.INTER_AREA), (0, 0), 2)
    L = cv2.cvtColor(cv2.cvtColor(gs, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)[:, :, 0]
    a = cv2.resize(ab[0], (TW, TH_), interpolation=cv2.INTER_AREA) * STRENGTH; b = cv2.resize(ab[1], (TW, TH_), interpolation=cv2.INTER_AREA) * STRENGTH
    rgb = cv2.cvtColor(np.dstack([L, a, b]).astype(np.float32), cv2.COLOR_Lab2RGB)
    g0 = np.clip(gs.astype(np.float32) / 255.0, 0.02, 0.98)[..., None]
    d = rgb - g0                                                    # the color shift we want, per channel
    need = np.maximum(d / (1 - g0), -d / g0).max(axis=2)            # smallest alpha that keeps C displayable
    alpha = np.clip(need * 1.15, 0, 0.85)
    C = np.clip(g0 + d / np.maximum(alpha, 1e-3)[..., None], 0, 1)
    out = np.empty((TH_, TW, 4), np.uint8)
    out[..., :3] = C * 255; out[..., 3] = alpha * EDGE * 255
    return out

def panel_sender(box, cv):                        # hands pictures to the headset panel, so the display work never waits for it
    while True:
        with cv:
            cv.wait_for(lambda: box['msg'] is not None or box['ovl'] is None)
            if box['ovl'] is None: return
            msg, box['msg'] = box['msg'], None
            k, box['pend'] = box['pend'], -1
            if k >= 0: box['flight'] = k
        try:
            n = acks['n']
            box['ovl'].stdin.write(msg); box['ovl'].stdin.flush()
            if k >= 0:                            # a shared picture: wait until the panel program has taken it before sending the next
                with ack_cv:
                    ack_cv.wait_for(lambda: acks['n'] != n, timeout=0.5)
        except (BrokenPipeError, OSError):
            print('the in-headset panel stopped; carrying on with the browser preview', flush=True); box['ovl'] = None; return

def composer():                                   # the fast part: newest brightness + latest colors -> one JPEG
    last = 0
    ovl = start_overlay()
    box = {'ovl': ovl, 'msg': None, 'pend': -1, 'flight': -1}; box_cv = threading.Condition()
    if ovl is not None:
        try:
            import fcntl; fcntl.fcntl(ovl.stdin.fileno(), 1031, 1 << 20)      # a roomier pipe to the panel
        except Exception as e:
            print('pipe size not changed:', e, flush=True)
        threading.Thread(target=panel_sender, args=(box, box_cv), daemon=True).start()   # started before this thread takes its own core
    left = None
    if STEREO and ovl is not None:
        left = LeftEye(); threading.Thread(target=left.run, daemon=True).start()
        threading.Thread(target=reader_left, daemon=True).start()
        sbs = np.zeros((OH, 2 * OW, 4), np.uint8); sbs[..., 3] = 255
        print('one view per eye: each eye %dx%d, eye shift %d pixels' % (OW, OH, EYE_SHIFT), flush=True)
    f0, fn = time.time(), 0
    try:
        mine = DISPLAY_CORES & ALL_CORES
        if mine: os.sched_setaffinity(0, mine)
        print('display work on cores', sorted(os.sched_getaffinity(0)), flush=True)
    except Exception as e:
        print('could not choose cores:', e, flush=True)
    while True:
        if state['paused']: time.sleep(0.1); continue
        with lock:
            raw, seq, ab, pose, arrived, ts_r = state['raw'], state['seq'], state['ab'], state['pose'], state['last'], state['ts']
        if raw is None or seq == last:
            time.sleep(0.003); continue
        last = seq
        asked = False
        if left is not None:                      # the left camera's frame of the same moment (they arrive a few ms apart)
            pr = left_partner(ts_r)
            if pr is not None:
                gap = abs(pr[0] - ts_r) * 1000; state['pair_ms'] = state['pair_ms'] * 0.9 + gap * 0.1
                if gap >= 12: state['mispairs'] += 1
                left.ask(pr[1]); asked = True
        t = t_start = time.perf_counter()
        g = cv2.remap(raw, view_maps[0], view_maps[1], cv2.INTER_LINEAR)
        g_out = g if OW == PW else cv2.remap(raw, out_maps[0], out_maps[1], cv2.INTER_LINEAR)
        t = _tick('straighten', t)
        if follower.ans is not None:              # remembered colors where the room is unchanged, followed colors elsewhere
            ab = blend(memory if state['mem'] else None, follower, g, None if pose is None else pose[1], OW, OH)
        elif ab is not None and OW != PW:
            ab = np.stack([cv2.resize(ab[0], (OW, OH), interpolation=cv2.INTER_LINEAR), cv2.resize(ab[1], (OW, OH), interpolation=cv2.INTER_LINEAR)])
        t = _tick('colors', t)
        ovl = box['ovl']
        need_full = (not TINT) or ovl is None or (state['viewers'] > 0 and (state['shown'] + 1) % 2 == 0)
        if not need_full:
            out = None
        elif ab is None:
            out = cv2.cvtColor(g_out, cv2.COLOR_GRAY2BGR)
        else:
            lab8 = np.empty((OH, OW, 3), np.uint8)   # whole-number Lab: faster to turn into a picture
            lab8[..., 0] = cv2.LUT(g_out, LUT_L8); lab8[..., 1] = np.clip(ab[0] + 128.5, 0, 255); lab8[..., 2] = np.clip(ab[1] + 128.5, 0, 255)
            out = cv2.cvtColor(lab8, cv2.COLOR_Lab2BGR)
        t = _tick('picture', t)
        if ovl is not None:                       # the headset panel: a wash of color, or the whole picture
            try:
                if left is not None:              # two views side by side, each moved sideways so far things are straight ahead of its eye
                    lpic = left.result if (asked and left.done.wait(0.15)) else None
                    if lpic is None: lpic = out   # no left camera right now: the right camera for both eyes
                    e = EYE_SHIFT; slot = -1
                    if SHM is not None:           # a slot the panel program is not reading and that is not waiting to be sent
                        with box_cv:
                            slot = [i for i in (0, 1, 2) if i != box['flight'] and i != box['pend']][0]
                    buf = SHM[slot] if slot >= 0 else sbs
                    buf[:, :OW - e] = cv2.cvtColor(lpic, cv2.COLOR_BGR2RGBA)[:, e:]
                    buf[:, OW + e:] = cv2.cvtColor(out, cv2.COLOR_BGR2RGBA)[:, :OW - e]
                    pic = buf
                else:
                    pic = tint_picture(g, ab) if (TINT and ab is not None) else (np.zeros((TH_, TW, 4), np.uint8) if TINT else cv2.cvtColor(out, cv2.COLOR_BGR2RGBA))
                pm = np.hstack([pose[1], pose[2][:, None]]).astype(np.float32) if (PIN and pose is not None) else np.zeros((3, 4), np.float32)
                if left is not None and slot >= 0:
                    msg = struct.pack('<4sIII', b'FTS1', pic.shape[1], pic.shape[0], int(PIN and pose is not None)) + pm.tobytes() + struct.pack('<I', slot)
                else:
                    slot = -1; msg = struct.pack('<4sIII', b'FTV2', pic.shape[1], pic.shape[0], int(PIN and pose is not None)) + pm.tobytes() + pic.tobytes()
                with box_cv:
                    box['msg'] = msg; box['pend'] = slot; box_cv.notify()
            except Exception as e:
                print('could not prepare the panel picture:', e, flush=True)
        t = _tick('panel', t); _tick('total', t_start)
        late = (time.time() - arrived) * 1000; TIMES['behind'] = late if 'behind' not in TIMES else TIMES['behind'] * 0.9 + late * 0.1
        fn += 1
        if time.time() - f0 >= 2:
            state['disp_fps'] = fn / (time.time() - f0); f0, fn = time.time(), 0
        state['shown'] += 1
        if TINT: time.sleep(0.012)                # leave the processor some air; about 20 pictures a second is plenty for color
        if out is not None and state['viewers'] > 0 and state['shown'] % 2 == 0:      # the browser preview: only when someone is watching, every other picture
            ok, jpg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with new_jpeg:
                    state['jpeg'] = jpg.tobytes(); state['jpeg_n'] += 1; new_jpeg.notify_all()

PAGE = b'''<!doctype html><meta charset="utf-8"><title>Frame Tint live</title>
<body style="margin:0;background:#0b0d10;color:#e8ecf1;font:15px system-ui,sans-serif">
<div style="padding:12px 16px"><b>Frame Tint live preview</b> &nbsp; <span id="s" style="color:#8b95a3"></span></div>
<img src="/stream" style="display:block;width:100%;max-width:1280px;margin:0 auto">
<script>setInterval(async()=>{try{const r=await (await fetch('/stats')).json();
document.getElementById('s').textContent=r.problem?('PROBLEM: '+r.problem):(r.age<0?'waiting for the camera...':r.age>1.5?('camera paused for '+r.age.toFixed(0)+' s: the headset thinks nobody is wearing it'):
('camera '+r.cam_fps.toFixed(0)+' frames/s | colors '+r.ans_fps.toFixed(1)+' answers/s, '+r.model_ms.toFixed(0)+' ms each | '+(r.memory<0?'color memory off (no head tracking)':'color memory on, '+r.memory+'% of the room seen')+' | '+(r.things<0?'no outline model':r.things+' things outlined, '+r.things_ms.toFixed(0)+' ms')+' | '+r.disp+' | '+(r.answers?'':'waiting for the first colors...')));}catch(e){}},1000);</script>'''

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers(); self.wfile.write(PAGE)
        elif self.path == '/stats':
            d = {k: state[k] for k in ('cam_fps', 'ans_fps', 'model_ms', 'answers', 'frames', 'problem', 'things', 'things_ms')}
            d['memory'] = round(memory.filled() * 100) if (memory is not None and state['mem']) else -1
            d['age'] = time.time() - state['last'] if state['last'] else -1
            T = dict(TIMES); d['disp'] = 'display %.0f pictures/s made, %s reach the headset, %.0f ms each (%s), %.0f ms behind the camera; head tracking %.0f/s, off by %.0f ms%s' % (state['disp_fps'], ('%.0f/s' % state['panel_fps']) if state['panel_fps'] >= 0 else 'unknown how many', T.get('total', 0), ', '.join('%s %.0f' % (k, T[k]) for k in ('straighten', 'follow', 'memory', 'edges', 'vivid', 'picture', 'panel', 'left eye') if k in T), T.get('behind', 0), state['pose_fps'], abs(state['pose_age']), ('; two eyes %.1f ms apart, %d pictures out of step so far' % (state['pair_ms'], state['mispairs'])) if STEREO else '')
            body = json.dumps(d).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(body)
        elif self.path == '/stream':
            self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=ft'); self.end_headers()
            n = 0; state['viewers'] += 1
            try:
                while True:
                    with new_jpeg:
                        new_jpeg.wait_for(lambda: state['jpeg_n'] != n, timeout=5)
                        jpg, n = state['jpeg'], state['jpeg_n']
                    if jpg is None: continue
                    self.wfile.write(b'--ft\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                state['viewers'] -= 1
        else:
            self.send_response(404); self.end_headers()

ALL_CORES = set(os.sched_getaffinity(0))
try:
    rest = ALL_CORES - set(CORES)                 # the light helpers (camera readers, head tracking, panel notes) may run anywhere but the color model's cores
    if rest: os.sched_setaffinity(0, rest)        # threads started from here inherit this; the model thread overrides it
except Exception as e:
    print('could not choose cores:', e, flush=True)
for fn in (reader, colorer, composer, outliner):
    threading.Thread(target=fn, daemon=True).start()
print('status page: http://<this headset\'s address>:%d  (Ctrl+C here to stop)' % PORT, flush=True)
try:
    ThreadingHTTPServer(('0.0.0.0', PORT), H).serve_forever()
except KeyboardInterrupt:
    print('stopped')
