# segmenter.py: outlines every distinct thing in a camera picture (FastSAM-s, a small "segment everything" model).
import numpy as np, cv2, onnxruntime as ort

class Segmenter:
    W, H, PAD = 320, 240, 8                         # the model takes 320x256: the 320x240 picture plus 8 rows above and below
    def __init__(s, path, threads=2, conf=0.25, iou=0.7):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.log_severity_level = 3
        s.sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider']); s.conf, s.iou = conf, iou
        s.x = np.full((1, 3, s.H + 2 * s.PAD, s.W), 114 / 255.0, np.float32)
    def masks(s, gray):
        """gray: any size, same view -> list of (60,80) boolean masks, one per thing"""
        g = cv2.resize(gray, (s.W, s.H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        s.x[0, :, s.PAD:s.PAD + s.H, :] = g
        det, proto = s.sess.run(None, {'images': s.x})
        det = det[0].T; keep = det[:, 4] > s.conf; det = det[keep]
        if len(det) == 0: return []
        boxes = np.stack([det[:, 0] - det[:, 2] / 2, det[:, 1] - det[:, 3] / 2, det[:, 2], det[:, 3]], 1)
        idx = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), det[:, 4].tolist(), s.conf, s.iou)).reshape(-1)[:100]
        p = proto[0]; m = 1 / (1 + np.exp(-(det[idx, 5:] @ p.reshape(32, -1)).reshape(-1, p.shape[1], p.shape[2])))
        out = []
        for k, i in enumerate(idx):
            x0, y0, w, h = boxes[i] / 4
            mk = np.zeros(m.shape[1:], bool); ys, ye, xs, xe = int(max(y0, 0)), int(np.ceil(y0 + h)), int(max(x0, 0)), int(np.ceil(x0 + w))
            mk[ys:ye, xs:xe] = m[k, ys:ye, xs:xe] > 0.5
            out.append(mk[s.PAD // 4:s.PAD // 4 + s.H // 4])
        return out

def label_map(masks, w, h, min_area=40):
    """(h,w) int32 labels, 0 = no thing; small things sit on top of big ones"""
    lab = np.zeros((h, w), np.int32)
    mk = [cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5 for m in masks]
    area = [int(m.sum()) for m in mk]
    for n, i in enumerate(sorted(range(len(mk)), key=lambda i: -area[i])):
        if area[i] >= min_area: lab[mk[i]] = n + 1
    return lab
