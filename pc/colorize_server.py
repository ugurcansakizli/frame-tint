# colorize_server.py: optional PC helper for Frame Tint. Colors the headset's camera pictures with the LARGE DDColor
# model on this PC's graphics card, which the headset's own processor cannot run.
#
#   python colorize_server.py --ddcolor <DDColor code folder> --weights <folder with config.json and pytorch_model.bin>
#   then on the headset, in ~/.config/frametint/frametint.conf:  FT_REMOTE=<this PC's address>:8765
#
# The headset sends a black-and-white picture (JPEG or PNG); the answer is the colors as Lab a,b at 128x128 (8 bits
# each). If this PC stops answering, Frame Tint goes back to its own model by itself. Only computers on private
# (home) networks may connect; nothing is stored. Needs PyTorch with CUDA, OpenCV and NumPy.
import argparse, ipaddress, json, os, struct, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, cv2, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--ddcolor', required=True, help='folder of the DDColor code (github.com/piddnad/DDColor)')
ap.add_argument('--weights', required=True, help='folder with config.json and pytorch_model.bin (e.g. ddcolor_modelscope)')
ap.add_argument('--port', type=int, default=8765)
ap.add_argument('--size', type=int, default=512, help='picture size the model works at')
ap.add_argument('--out', type=int, default=128, help='size of the color answer sent back')
ap.add_argument('--fp32', action='store_true', help='run the model in 32-bit (slower; default 16-bit)')
ap.add_argument('--allow', default='10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,127.0.0.0/8', help='networks allowed to connect')
args = ap.parse_args()
sys.path.insert(0, os.path.abspath(args.ddcolor)); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddcolor.model import DDColor
from nlmeans_gpu import nlmeans

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = json.load(open(os.path.join(args.weights, 'config.json'))); cfg['input_size'] = [args.size, args.size]
model = DDColor(**cfg); sd = torch.load(os.path.join(args.weights, 'pytorch_model.bin'), map_location='cpu', weights_only=True)
model.load_state_dict(sd.get('params', sd) if isinstance(sd, dict) else sd, strict=False); model.to(dev).eval()
half = dev.type == 'cuda' and not args.fp32
ALLOW = [ipaddress.ip_network(n.strip()) for n in args.allow.split(',') if n.strip()]
_cl = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)); gpu = threading.Lock()
stats = {'n': 0, 'ms': 0.0, 'started': time.time()}

def camera_fix(g):
    """the tracking camera's picture made photo-like: grain removed (on the GPU), black blacks, local contrast"""
    g = nlmeans(torch.from_numpy(g).to(dev)).clamp(0, 255).byte().cpu().numpy()
    v = g[g > 4]
    if v.size > 100:
        lo, hi = np.percentile(v, (1, 99.5)); g = np.clip((g.astype(np.float32) - lo) / max(hi - lo, 20) * 255, 0, 255).astype(np.uint8)
    return _cl.apply(g)

@torch.no_grad()
def colorize(gray):
    g = cv2.resize(gray, (args.size, args.size), interpolation=cv2.INTER_AREA)
    with gpu:
        g = camera_fix(g)
        x = torch.from_numpy(g).to(dev).float().div(255)[None, None].repeat(1, 3, 1, 1)
        with torch.autocast('cuda', dtype=torch.float16, enabled=half):
            ab = model(x).float()
        ab = F.interpolate(ab, size=(args.out, args.out), mode='area')[0]
        return ab.round().clamp(-128, 127).to(torch.int8).cpu().numpy()

class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'                                         # keep the connection open between pictures
    def log_message(self, *a): pass
    def allowed(self):
        ip = ipaddress.ip_address(self.client_address[0])
        if any(ip in n for n in ALLOW): return True
        self.send_response(403); self.send_header('Content-Length', '0'); self.end_headers(); return False
    def do_GET(self):
        if not self.allowed(): return
        body = ('Frame Tint PC colorizer: %d pictures colored, %.0f ms each on average\n' % (stats['n'], stats['ms'])).encode()
        self.send_response(200); self.send_header('Content-Type', 'text/plain'); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        if not self.allowed(): return
        n = int(self.headers.get('Content-Length', 0))
        if self.path != '/color' or not 0 < n < 8_000_000:
            self.send_response(400); self.send_header('Content-Length', '0'); self.end_headers(); return
        t = time.perf_counter()
        gray = cv2.imdecode(np.frombuffer(self.rfile.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            self.send_response(400); self.send_header('Content-Length', '0'); self.end_headers(); return
        ab = colorize(gray); ms = (time.perf_counter() - t) * 1000
        body = struct.pack('<4sHf', b'FTC1', args.out, ms) + ab.tobytes()
        stats['n'] += 1; stats['ms'] = ms if stats['n'] == 1 else stats['ms'] * 0.95 + ms * 0.05
        self.send_response(200); self.send_header('Content-Type', 'application/octet-stream'); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

colorize(np.full((720, 960), 100, np.uint8)); colorize(np.full((720, 960), 100, np.uint8))     # warm up the GPU
t = time.perf_counter(); colorize(np.full((720, 960), 100, np.uint8)); warm = (time.perf_counter() - t) * 1000
print('Frame Tint PC colorizer ready on port %d (%s, %s, about %.0f ms per picture). Ctrl+C to stop.' % (
    args.port, torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU', '16-bit' if half else '32-bit', warm), flush=True)
ThreadingHTTPServer(('0.0.0.0', args.port), H).serve_forever()
