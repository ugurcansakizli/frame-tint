# nlmeans_gpu.py: non-local means grain removal (like OpenCV's fastNlMeansDenoising) on the graphics card with PyTorch.
# Each pixel becomes a weighted average of pixels in a search window whose surrounding patches look alike:
#   weight = exp(-(mean squared difference of the two patches) / h^2)
# A whole row of search offsets is worked out at once, so the GPU gets few, large jobs.
import torch, torch.nn.functional as F

@torch.no_grad()
def nlmeans(img, h=7.0, template=7, search=21):
    """img: (H,W) uint8 or float tensor 0..255 on any device -> (H,W) float tensor 0..255, grain removed"""
    x = img.float()[None, None]; r = search // 2; t = template // 2
    P = F.pad(x, (r + t, r + t, r + t, r + t), mode='reflect')[0, 0]
    H, W = x.shape[-2:]; Ht, Wt = H + 2 * t, W + 2 * t
    centre = P[r:r + Ht, r:r + Wt]
    acc = torch.zeros(H, W, device=img.device); wsum = torch.zeros(H, W, device=img.device); inv = 1.0 / (h * h)
    for dy in range(-r, r + 1):
        rows = P[r + dy:r + dy + Ht]                                         # (Ht, W + 2r + 2t)
        sh = rows.unfold(1, Wt, 1)[:, :search].permute(1, 0, 2)              # (search, Ht, Wt): every dx at once
        d = F.avg_pool2d(((centre - sh) ** 2)[:, None], template, stride=1)[:, 0]   # mean squared patch difference
        w = torch.exp(-d * inv)
        acc += (w * sh[:, t:t + H, t:t + W]).sum(0); wsum += w.sum(0)
    return acc / wsum
