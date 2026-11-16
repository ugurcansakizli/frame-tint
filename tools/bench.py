# bench.py: times the color model with different numbers of threads and different sets of processor cores.
#   ./venv/bin/python bench.py ddcolor-fast-8bit.onnx ddcolor-fast2-8bit.onnx
import os, sys, time, glob
import numpy as np, onnxruntime as ort

freq = {}
for p in sorted(glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq')):
    freq[int(p.split('/cpu')[2].split('/')[0])] = int(open(p).read()) // 1000
print('RESULT cores and their top speed in MHz:', freq, flush=True)
allc = sorted(os.sched_getaffinity(0))
order = sorted(allc, key=lambda c: -freq.get(c, 0))          # fastest first
x = np.full((1, 3, 512, 512), 0.5, np.float32)

def run(path, cores, threads):
    os.sched_setaffinity(0, cores)
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.log_severity_level = 3
    s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    s.run(None, {'input': x}); t = []
    for _ in range(5):
        a = time.time(); s.run(None, {'input': x}); t.append((time.time() - a) * 1000)
    os.sched_setaffinity(0, allc)
    print('RESULT %-26s cores %-18s threads %d | ms: %s | best %d' % (os.path.basename(path), ','.join(map(str, sorted(cores))), threads, [round(v) for v in t], min(t)), flush=True)

for path in sys.argv[1:]:
    run(path, set(allc), 4)
for path in sys.argv[-1:]:
    for n in (3, 4, 5, 6):
        if n <= len(order): run(path, set(order[:n]), n)
    if len(order) >= 6: run(path, set(order[:6]), 4)
print('RESULT done', flush=True)
