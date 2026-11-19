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
#   FT_MODEL   the color model's file name in the models folder (default ddcolor-fast2-8bit-256.onnx)
#   FT_REMOTE  <PC address>:<port> of the optional PC helper (pc/colorize_server.py): the large color model runs on the
#              PC's graphics card; when the PC does not answer, the headset's own model is used (default: off)
#   FT_FOVEA   set to 1 for more detailed colors where you look (uses the headset's eye tracking): every other color
#              answer looks only at the area around your gaze, so the model sees it bigger; the rest of the view keeps
#              its colors from the whole-view answers and the color memory
#   FT_FOVEA_DEG    width of that area in degrees (default 60; smaller = more detail but less context for the model)
#   FT_FOVEA_EVERY  one answer in this many is a gaze-area answer (default 2)
#   FT_CONTRAST  0..1: photo-like brightness shown: black blacks, white whites, local contrast (default 1)
#   FT_MODEL_CONTRAST  set to 0 to give the color model the camera's flat picture instead of a contrast-corrected one
#   FT_STEADY  0..1: how much of each new picture's colors is taken; lower = steadier, less flicker (default 0.2)
#   FT_SOLID   set to 0 to let averaged colors go muddy (by default they keep the strength of the colors averaged)
#   FT_DEGRAIN set to 1 to remove the camera's grain (a median filter: also halves fine detail; default off)
#   FT_SHARP   set to 1 to fit the colors to the full-size picture's edges (costs about 6 ms; default off)
#   Most of these can also be changed while running, from the headset itself: curl "localhost:8000/set?steady=0.3&vivid=2"
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
MODEL = next((a for a in sys.argv[1:] if not a.startswith('--')), os.environ.get('FT_MODEL', 'ddcolor-fast2-8bit-256.onnx'))
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
TUNE = {'vivid': VIVID,                                         # can be changed while running: /set?vivid=1.5&sharp=0&solid=0 (from the headset only)
        'sharp': os.environ.get('FT_SHARP', '0') != '0',        # colors sharpened to the full-size picture's edges (costs 6 ms, little gain)
        'degrain': os.environ.get('FT_DEGRAIN', '0') != '0',    # remove the camera's grain from the brightness shown (a median filter: also halves fine detail)
        'model_contrast': os.environ.get('FT_MODEL_CONTRAST', '1') != '0',   # the color model sees a contrast-corrected picture
        'solid': os.environ.get('FT_SOLID', '1') != '0',        # averaged colors keep their strength instead of going muddy
        'balance': float(os.environ.get('FT_BALANCE', '0.5')),  # white balance (see BALANCE)
        'contrast': float(os.environ.get('FT_CONTRAST', '1')),  # 0..1: photo-like brightness shown: black blacks, white whites, local contrast
        'fovea': True,                                          # gaze-area answers (with FT_FOVEA=1)
        'remote': True,                                         # use the PC helper when FT_REMOTE is set
        'steady': float(os.environ.get('FT_STEADY', '0.2'))}    # 0..1: how much of each new picture's colors is taken (1 = no smoothing over time)
STEADY = {'ab': None, 'g': None, 'R': None}                   # the colors last shown, for smoothing over time
LEVELS = {'lo': None, 'hi': None}                             # the picture's darkest and brightest levels, smoothed
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
GAZE = os.environ.get('FT_FOVEA', '0') == '1' and OVERLAY is not None     # read the eye tracker (through ftoverlay)
FOVEA_DEG = min(100.0, max(20.0, float(os.environ.get('FT_FOVEA_DEG', '60'))))
FOVEA_EVERY = max(2, int(os.environ.get('FT_FOVEA_EVERY', '2')))

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
    def write(s, ab, gray, R, weight=None):
        """ab: (2,h,w) colors for a camera frame, gray: that frame, R: how the head was turned. Any size, same view.
        weight: optional (PH,PW) trust per spot of the view (1 = a normal answer)."""
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
        if weight is not None:                                                     # some spots of this answer count more
            e *= cv2.remap(weight.astype(np.float32), um - 0.5, vm - 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
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
        mean = np.stack([np.bincount(lab, weights=ab[..., c].ravel(), minlength=s.n) / cnt for c in range(2)], 1)
        if TUNE['solid']: mean = keep_chroma(mean, np.bincount(lab, weights=cv2.magnitude(ab[..., 0], ab[..., 1]).ravel(), minlength=s.n) / cnt)
        return mean[lab].reshape(s.h, s.w, 2).astype(np.float32), lab.reshape(s.h, s.w)

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
def keep_chroma(mean, strength):
    """mean (n,2) average a,b of some spots; strength (n,) their average color strength. An average of different colors
    is weaker than the colors themselves (muddy, grey-brown); give it back the average strength, at most doubled."""
    m = np.sqrt((mean * mean).sum(1)) + 1e-3
    return mean * np.minimum(strength / m, 2.0)[:, None]

def guided_up(ab, gray, guide, r=3, eps=0.004):
    """sharp colors at full size: the edge-following fit of ab (h,w,2) to gray (h,w) is worked out small, then applied
    with the full-size picture guide (H,W uint8), so colors stop at the edges of the full picture instead of bleeding."""
    I = gray.astype(np.float32) / 255.0; k = (2 * r + 1, 2 * r + 1); H, W = guide.shape
    mI = cv2.blur(I, k); vI = cv2.blur(I * I, k) - mI * mI
    A = np.empty_like(ab); B = np.empty_like(ab)
    for c in range(2):
        p = ab[..., c]; mp = cv2.blur(p, k); cov = cv2.blur(I * p, k) - mI * mp
        a = cov / (vI + eps); A[..., c] = cv2.blur(a, k); B[..., c] = cv2.blur(mp - a * mI, k)
    Ig = (guide.astype(np.float32) / 255.0)[..., None]
    return cv2.resize(A, (W, H), interpolation=cv2.INTER_LINEAR) * Ig + cv2.resize(B, (W, H), interpolation=cv2.INTER_LINEAR)

def paint(ab, patch_lab, n_patch, thing_lab):
    """ab (h,w,2); patch_lab (h,w) edge-following patches of the newest frame; thing_lab (h,w) carried outlines.
    Each patch votes for the thing it mostly lies in; each thing gets the average color of its patches."""
    pl = patch_lab.ravel(); tl = thing_lab.ravel(); nt = int(tl.max()) + 1
    votes = np.bincount(pl.astype(np.int64) * nt + tl, minlength=n_patch * nt).reshape(n_patch, nt)
    owner = votes.argmax(1); share = votes[np.arange(n_patch), owner] / (votes.sum(1) + 1e-6)
    owner[share < 0.6] = 0                                   # a patch split between things stays on its own
    cnt = np.bincount(pl, minlength=n_patch).astype(np.float32) + 1e-6
    pm = np.stack([np.bincount(pl, weights=ab[..., c].ravel(), minlength=n_patch) / cnt for c in range(2)], 1)   # color per patch
    tcnt = np.bincount(owner, weights=cnt, minlength=nt) + 1e-6
    tm = np.stack([np.bincount(owner, weights=pm[:, c] * cnt, minlength=nt) / tcnt for c in range(2)], 1)         # color per thing
    if TUNE['solid']:                              # averaging different colors makes them muddy: keep their strength
        pc = np.bincount(pl, weights=cv2.magnitude(ab[..., 0], ab[..., 1]).ravel(), minlength=n_patch) / cnt
        pm = keep_chroma(pm, pc); tm = keep_chroma(tm, np.bincount(owner, weights=pc * cnt, minlength=nt) / tcnt)
    col = np.where((owner > 0)[:, None], tm[owner], pm)
    return col[pl].reshape(ab.shape[0], ab.shape[1], 2).astype(np.float32)

SNAP = True
STEREO_SRC = {'src': None}
TIMES = {}                                         # how long each part of the display work takes (ms, smoothed)
def _tick(name, t):
    ms = (time.perf_counter() - t) * 1000; TIMES[name] = ms if name not in TIMES else TIMES[name] * 0.9 + ms * 0.1
    return time.perf_counter()
_regions = {}
things = None                                      # set when the outline model is running

def blend(mem, fol, gray, R, PW, PH, guide=None):
    """the colors to show for the newest frame: remembered where the room still looks the same, followed elsewhere."""
    t = time.perf_counter()
    gs = cv2.resize(gray, (fol.w, fol.h), interpolation=cv2.INTER_AREA)
    fresh, miss = fol.render(gs, R)
    fresh = fresh * (1 - (0.25 if TUNE['solid'] else 0.5) * _ss(miss.astype(np.float32), 30, 80))[..., None]     # no good match: tone the color down
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
    if TUNE['vivid'] != 1:                         # stronger colors; weak ones are pushed apart, strong ones level off
        fresh = np.ascontiguousarray(fresh, dtype=np.float32)
        c = cv2.magnitude(fresh[..., 0], fresh[..., 1]) + 1e-3
        fresh = fresh * (75.0 * np.tanh(TUNE['vivid'] * c / 75.0) / c)[..., None]
    t = _tick('vivid', t)
    if TUNE['steady'] < 1:                         # smooth colors over time: keep most of the last picture's colors, turned
        fresh = np.ascontiguousarray(fresh, dtype=np.float32)       # with the head, except where something really moved
        P = STEADY
        if P['ab'] is not None and P['ab'].shape == fresh.shape:
            Hm = fol.K @ fol.S @ P['R'].T @ R @ fol.S @ fol.Ki if (R is not None and P['R'] is not None) else None
            pab, pg = (P['ab'], P['g']) if Hm is None else (
                cv2.warpPerspective(P['ab'], Hm, (fol.w, fol.h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE),
                cv2.warpPerspective(P['g'], Hm, (fol.w, fol.h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE))
            moved = _ss(cv2.GaussianBlur(cv2.absdiff(pg, gs), (0, 0), 2).astype(np.float32), 12, 35)
            a = (TUNE['steady'] + (1 - TUNE['steady']) * moved)[..., None]
            fresh = pab + (fresh - pab) * a
        STEADY.update(ab=fresh.copy(), g=gs.copy(), R=None if R is None else R.copy())
        t = _tick('steady', t)
    STEREO_SRC['src'] = (np.ascontiguousarray(fresh, dtype=np.float32), gs)      # the left eye borrows these colors
    if TUNE['sharp'] and guide is not None and guide.shape == (PH, PW):
        out = guided_up(np.ascontiguousarray(fresh, dtype=np.float32), gs, guide)
    else:
        out = cv2.resize(fresh, (PW, PH), interpolation=cv2.INTER_LINEAR)
    _tick('upsize', t)
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

acks = {'n': 0, 'w': 0, 't': time.time(), 'pw': 0, 'pt': time.time(), 'gw': 0}; ack_cv = threading.Condition()
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
        if rec[:4] == b'FTE1':                    # where the eyes look: valid, then x/-z and y/-z relative to the head
            ok, gx, gy = struct.unpack('<fff', rec[4:16])
            state['gaze'] = (time.time(), ok > 0.5, gx, gy)
            if ok > 0.5: acks['gw'] += 1
            continue
        if rec[:4] != b'FTP1': continue
        m = np.frombuffer(rec, np.float32, 12, 4).reshape(3, 4)
        poses.append((time.time(), m[:, :3].astype(np.float64), m[:, 3].astype(np.float64)))
        acks['pw'] += 1
        if time.time() - acks['pt'] >= 2:
            state['pose_fps'] = acks['pw'] / (time.time() - acks['pt']); state['gaze_fps'] = acks['gw'] / (time.time() - acks['pt'])
            acks['pt'], acks['pw'], acks['gw'] = time.time(), 0, 0
        if len(poses) > 200: del poses[:100]

state = {'paused': False, 'raw': None, 'pose': None, 'mem': False, 'seq': 0, 'frames': 0, 'ab': None, 'answers': 0, 'model_ms': 0.0, 'jpeg': None, 'jpeg_n': 0,
         'cam_fps': 0.0, 'ans_fps': 0.0, 'problem': '', 'last': 0.0, 'shown': 0, 'viewers': 0, 'things': -1, 'things_ms': 0.0, 'disp_fps': 0.0, 'raw_l': None, 'last_l': 0.0, 'panel_fps': -1.0, 'ts': 0.0, 'mispairs': 0, 'pair_ms': 0.0, 'pose_fps': 0.0, 'pose_age': 0.0, 'gaze': None, 'gaze_fps': 0.0, 'remote': ''}
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

mem_job = {'job': None}; mem_cv = threading.Condition()
def mem_writer():                                  # writes answers into the color memory (the newest waiting one wins)
    while True:
        with mem_cv:
            mem_cv.wait_for(lambda: mem_job['job'] is not None); job, mem_job['job'] = mem_job['job'], None
        t = time.perf_counter(); memory.write(*job); _ctick('memory', t)

CTIMES = {}                                        # how long each part of a color answer takes (ms, smoothed)
def _ctick(name, t):
    ms = (time.perf_counter() - t) * 1000; CTIMES[name] = ms if name not in CTIMES else CTIMES[name] * 0.8 + ms * 0.2
    return time.perf_counter()

class Remote:
    """the optional PC helper (pc/colorize_server.py): the large color model on a PC's graphics card. Sends the
    black-and-white view, gets colors back. When the PC does not answer, returns None (the headset model is used)
    and tries again 5 seconds later."""
    def __init__(s, addr):
        host, _, port = addr.rpartition(':'); s.host, s.port = host or addr, int(port or 8765)
        s.conn = None; s.retry_at = 0.0; s.ms = 0.0; s.ok = False; s.said = False
    def color(s, view):
        if time.time() < s.retry_at: return None
        import http.client
        try:
            ok, jpg = cv2.imencode('.jpg', cv2.resize(view, (512, 512), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if s.conn is None: s.conn = http.client.HTTPConnection(s.host, s.port, timeout=1.5)
            t = time.time(); s.conn.request('POST', '/color', body=jpg.tobytes(), headers={'Content-Type': 'image/jpeg'})
            r = s.conn.getresponse(); data = r.read()
            if r.status != 200 or data[:4] != b'FTC1': raise IOError('unexpected answer (%d)' % r.status)
            n, pc_ms = struct.unpack('<Hf', data[4:10])
            ab = np.frombuffer(data[10:10 + 2 * n * n], np.int8).reshape(2, n, n).astype(np.float32)
            s.ms = (time.time() - t) * 1000; state['remote'] = 'PC: %.0f ms round trip (%.0f ms on the PC)' % (s.ms, pc_ms)
            if not s.ok: print('colors now come from the PC at %s:%d' % (s.host, s.port), flush=True); s.ok = True
            return ab
        except Exception as e:
            if s.ok or not s.said: print('PC colorizer at %s:%d not reachable (%s: %s); using the headset model, trying again every 5 s' % (s.host, s.port, type(e).__name__, e), flush=True)
            s.ok = False; s.said = True; s.retry_at = time.time() + 5; state['remote'] = 'headset model (PC not reachable)'
            try: s.conn.close()
            except Exception: pass
            s.conn = None; return None
remote = Remote(os.environ['FT_REMOTE'].strip()) if os.environ.get('FT_REMOTE', '').strip() else None

FOV = {'n': 0, 'what': 'no gaze yet', 'rect': None, 'ms': 0, 'combine_ms': 0}
MF = MW / 2 / np.tan(np.radians(VIEW_DEG / 2))   # focal length of the straightened picture the model looks at, in pixels
K_PW = np.array([[VF, 0, PW / 2], [0, VF, PH / 2], [0, 0, 1]]); S_FLIP = np.diag([1.0, -1.0, -1.0])

def fovea_rect(R):
    """the square of the model's straightened picture (x0, y0, side) centred where the eyes look, for a frame taken with
    head turn R; None when there is no fresh gaze or it is outside the picture."""
    g = state['gaze']
    if g is None or not g[1] or time.time() - g[0] > 0.3 or not poses:
        FOV['what'] = 'no gaze: whole view only'; return None
    d = np.array([g[2], g[3], -1.0]); d = R.T @ (poses[-1][1] @ (d / np.linalg.norm(d)))   # gaze now -> this frame's view
    if d[2] > -0.2:
        FOV['what'] = 'looking outside the picture'; return None
    u = MW / 2 + MF * d[0] / -d[2]; v = MH / 2 - MF * d[1] / -d[2]
    if not (0 <= u < MW and 0 <= v < MH):
        FOV['what'] = 'looking outside the picture'; return None
    side = int(min(MH, 2 * MF * np.tan(np.radians(FOVEA_DEG / 2))))
    x0 = int(min(max(u - side / 2, 0), MW - side)); y0 = int(min(max(v - side / 2, 0), MH - side))
    FOV['what'] = 'gaze area %d deg around (%d, %d)' % (FOVEA_DEG, u, v); FOV['rect'] = [x0, y0, side]
    return x0, y0, side

def fovea_combine(ab, rect, last_full, R):
    """a whole-view answer for the frame taken with head turn R: the gaze-area answer ab (2,S,S) for rect, laid with a soft
    edge over the newest whole-view answer turned to this frame. Returns colors (2,PH,PW) and the gaze area's mask (PH,PW)."""
    full, R0 = last_full
    Hm = K_PW @ S_FLIP @ R0.T @ R @ S_FLIP @ np.linalg.inv(K_PW)   # undo the head turn between the two frames
    base = cv2.warpPerspective(np.ascontiguousarray(full.transpose(1, 2, 0)), Hm, (PW, PH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    k = PW / MW; x0, y0 = int(round(rect[0] * k)), int(round(rect[1] * k)); sp = min(int(round(rect[2] * k)), PW - x0, PH - y0)
    patch = cv2.resize(np.ascontiguousarray(ab.transpose(1, 2, 0)), (sp, sp), interpolation=cv2.INTER_LINEAR)
    r = np.clip(np.minimum(np.arange(sp), sp - 1 - np.arange(sp)) / (0.2 * sp), 0, 1); r = r * r * (3 - 2 * r)
    mask = np.zeros((PH, PW), np.float32); mask[y0:y0 + sp, x0:x0 + sp] = np.minimum.outer(r, r)
    out = base.copy(); m = mask[y0:y0 + sp, x0:x0 + sp, None]
    out[y0:y0 + sp, x0:x0 + sp] = base[y0:y0 + sp, x0:x0 + sp] * (1 - m) + patch * m
    return np.ascontiguousarray(out.transpose(2, 0, 1)), mask

_clahe = {}
def stretch(g, amount, levels=None):
    """the camera's picture is flat and milky (grey blacks, little depth), which makes any colors on it look like an old
    tape. Stretch its brightness so the darkest 1% becomes black and the brightest 0.5% white (smoothed over time in
    levels, so it does not pump), then add local contrast (CLAHE) the way a photo has it. amount 0..1 mixes this with the
    picture as it is. levels=None: measure this picture alone (for the color model's input)."""
    v = g[::8, ::8] if g.size > 200000 else g[::2, ::2]; v = v[v > 4]      # not the black corners
    if v.size < 100: return g
    lo, hi = np.percentile(v, (1, 99.5))
    if levels is not None:
        levels['lo'] = lo if levels['lo'] is None else levels['lo'] * 0.9 + lo * 0.1
        levels['hi'] = hi if levels['hi'] is None else levels['hi'] * 0.9 + hi * 0.1
        lo, hi = levels['lo'], levels['hi']
    x = np.arange(256, dtype=np.float32)
    out = cv2.LUT(g, np.clip((x - lo) / max(hi - lo, 20) * 255, 0, 255).astype(np.uint8))
    if threading.get_ident() not in _clahe: _clahe[threading.get_ident()] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    out = _clahe[threading.get_ident()].apply(out)
    return out if amount >= 1 else cv2.addWeighted(out, amount, g, 1 - amount, 0)

def colorer():                                    # the slow part: asks the model for colors, newest frame each time
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    try:                                          # the model gets the fast cores to itself; everything else keeps off them
        mine = set(CORES) & ALL_CORES
        if mine: os.sched_setaffinity(0, mine); so.intra_op_num_threads = len(mine)
        print('color model on cores', sorted(os.sched_getaffinity(0)), 'with', so.intra_op_num_threads, 'threads', flush=True)
    except Exception as e:
        print('could not choose cores:', e, flush=True)
    TUNE['model'] = MODEL; loaded = None
    def load(fname):
        s_ = ort.InferenceSession(model_file(fname), so, providers=['CPUExecutionProvider'])
        print('model ready:', fname, flush=True); return s_, s_.get_inputs()[0].name, s_.get_inputs()[0].shape[2]
    last, t0, n0 = 0, time.time(), 0
    cast = None                                   # the overall color cast, smoothed over the last few answers
    last_full, n_ans = None, 0                    # the newest whole-view answer (colors at PW x PH, head turn)
    mem_thread = None
    while True:
        if state['paused']: time.sleep(0.1); continue
        if TUNE['model'] != loaded:                 # (switchable while running: /set?model=<file in the models folder>)
            try:
                sess, name, S = load(TUNE['model']); loaded = TUNE['model']; cast = None
            except Exception as e:
                print('could not load model %s: %s' % (TUNE['model'], e), flush=True); TUNE['model'] = loaded or MODEL
                if loaded is None: raise
                continue
        with lock:
            raw, seq, pose = state['raw'], state['seq'], state['pose']
        if raw is None or seq == last:
            time.sleep(0.005); continue
        last = seq
        tc = time.perf_counter()
        g = cv2.remap(raw, model_maps[0], model_maps[1], cv2.INTER_LINEAR)
        rect = None                               # every FOVEA_EVERY-th answer: only the area around the gaze
        if GAZE and TUNE['fovea'] and pose is not None and last_full is not None and n_ans % FOVEA_EVERY == FOVEA_EVERY - 1:
            rect = fovea_rect(pose[1])
        view = g if rect is None else g[rect[1]:rect[1] + rect[2], rect[0]:rect[0] + rect[2]]
        ab = remote.color(view) if (remote is not None and TUNE['remote']) else None   # the PC's large model, if it answers
        if ab is not None:
            ms = remote.ms
            x = cv2.resize(view, (ab.shape[2], ab.shape[1]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        else:                                     # the headset's own model
            x = cv2.resize(view, (S, S), interpolation=cv2.INTER_AREA)
            if TUNE['model_contrast']: x = stretch(x, 1.0)  # a photo-like picture: the model guesses less orange on it
            x = x.astype(np.float32) / 255.0
            t = time.time()
            ab = sess.run(None, {name: np.repeat(x[None, None], 3, axis=1)})[0][0]
            ms = (time.time() - t) * 1000
        n_ans += 1; tc = _ctick('model', tc)
        if TUNE['balance'] > 0:                   # white balance: take out part of the most common color (walls, ceiling)
            ok = x[::4, ::4] > 0.03               # not the black corners
            if rect is None and ok.sum() > 100:   # (measured on whole-view answers only)
                med = np.array([np.median(ab[0][::4, ::4][ok]), np.median(ab[1][::4, ::4][ok])], np.float32)
                cast = med if cast is None else cast * 0.7 + med * 0.3
            if cast is not None:
                ab = ab - (TUNE['balance'] * cast)[:, None, None]
        gs = cv2.resize(g, (PW, PH), interpolation=cv2.INTER_AREA)
        weight = None
        if rect is None:
            small = np.stack([cv2.resize(ab[0], (PW, PH), interpolation=cv2.INTER_AREA), cv2.resize(ab[1], (PW, PH), interpolation=cv2.INTER_AREA)])
            if pose is not None: last_full = (small, pose[1].copy())
        else:
            t = time.time()
            small, mask = fovea_combine(ab, rect, last_full, pose[1])
            weight = 0.3 + 1.7 * mask              # the gaze area counts double; the carried-over rest counts little
            FOV['n'] += 1; FOV['ms'] = round(ms); FOV['combine_ms'] = round((time.time() - t) * 1000)
        tc = _ctick('prepare', tc)
        if memory is not None and pose is not None:   # written by its own thread, so the next answer can be asked for at once
            if mem_thread is None: mem_thread = threading.Thread(target=mem_writer, daemon=True); mem_thread.start()
            with mem_cv:
                mem_job['job'] = (small, gs, pose[1], weight); mem_cv.notify()
        if rect is None or memory is None or pose is None:   # gaze-area answers go into the memory only: shown directly they
            follower.new_answer(small, gs, None if pose is None else pose[1])   # alternate with whole-view ones and flicker
        with lock:
            state['mem'] = memory is not None and pose is not None
            state['ab'] = small if state['ab'] is None else state['ab'] * SMOOTH + small * (1 - SMOOTH)
            state['answers'] += 1; state['model_ms'] = ms
        _ctick('finish', tc)
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
        ovl = subprocess.Popen(OVERLAY + ['%.3f' % width, '%.3f' % DIST, '%.3f' % ALPHA, 'poses'] + (['sbs'] if STEREO else []) + (['gaze'] if GAZE else []), stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=HERE,
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
        if TUNE['degrain']: g_out = cv2.medianBlur(g_out, 3)                  # the camera's grain
        if TUNE['contrast'] > 0: g_out = stretch(g_out, TUNE['contrast'], LEVELS)
        t = _tick('straighten', t)
        if follower.ans is not None:              # remembered colors where the room is unchanged, followed colors elsewhere
            ab = blend(memory if state['mem'] else None, follower, g, None if pose is None else pose[1], OW, OH, g_out)
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
        elif self.path.startswith('/set') and self.client_address[0] in ('127.0.0.1', '::1'):   # live tuning, from the headset itself only
            from urllib.parse import urlparse, parse_qs
            for k, v in parse_qs(urlparse(self.path).query).items():
                if k == 'vivid': TUNE['vivid'] = min(4.0, max(0.5, float(v[0])))
                elif k in ('sharp', 'solid', 'degrain', 'model_contrast'): TUNE[k] = v[0] not in ('0', 'false', 'off')
                elif k in ('balance', 'contrast'): TUNE[k] = min(1.0, max(0.0, float(v[0])))
                elif k == 'steady': TUNE[k] = min(1.0, max(0.05, float(v[0])))
                elif k in ('fovea', 'remote'): TUNE[k] = v[0] not in ('0', 'false', 'off')
                elif k == 'model' and os.path.basename(v[0]) == v[0] and v[0].endswith('.onnx') and os.path.exists(os.path.join(ROOT, 'models', v[0])): TUNE[k] = v[0]
            body = json.dumps(TUNE).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(body)
        elif self.path == '/stats':
            d = {k: state[k] for k in ('cam_fps', 'ans_fps', 'model_ms', 'answers', 'frames', 'problem', 'things', 'things_ms')}
            d['remote'] = state['remote'] if remote is not None else 'off'
            d['answer_ms'] = {k: round(v) for k, v in CTIMES.items()}
            d['memory'] = round(memory.filled() * 100) if (memory is not None and state['mem']) else -1
            d['age'] = time.time() - state['last'] if state['last'] else -1
            T = dict(TIMES); d['disp'] = 'display %.0f pictures/s made, %s reach the headset, %.0f ms each (%s), %.0f ms behind the camera; head tracking %.0f/s, off by %.0f ms%s' % (state['disp_fps'], ('%.0f/s' % state['panel_fps']) if state['panel_fps'] >= 0 else 'unknown how many', T.get('total', 0), ', '.join('%s %.0f' % (k, T[k]) for k in ('straighten', 'follow', 'memory', 'edges', 'vivid', 'steady', 'upsize', 'picture', 'panel', 'left eye') if k in T), T.get('behind', 0), state['pose_fps'], abs(state['pose_age']), ('; two eyes %.1f ms apart, %d pictures out of step so far' % (state['pair_ms'], state['mispairs'])) if STEREO else '')
            if GAZE:
                d['fovea'] = dict(FOV, gaze_fps=round(state['gaze_fps'], 1))
                d['disp'] += '; eye tracking %.0f answers/s, %d gaze-area answers (%s)' % (state['gaze_fps'], FOV['n'], FOV['what'])
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
