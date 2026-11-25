"""CourtNet: a small U-Net-style CNN trained FROM SCRATCH (no pretrained weights).

Input : RGB image, 256 x 448 (height x width), values 0..1
Output: one heatmap per court point at 1/4 resolution (64 x 112). The brightest pixel of
        heatmap k is where court point k is.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4
INPUT_HEIGHT, INPUT_WIDTH = 256, 448   # both must be multiples of 32


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class CourtNet(nn.Module):
    def __init__(self, n_points=12, base=24):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 8]
        self.stem = nn.Sequential(nn.Conv2d(3, c[0], 5, stride=2, padding=2, bias=False),
                                  nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True))        # /2
        self.enc1 = conv_block(c[0], c[0])                                              # /2
        self.enc2 = conv_block(c[0], c[1])                                              # /4
        self.enc3 = conv_block(c[1], c[2])                                              # /8
        self.enc4 = conv_block(c[2], c[3])                                              # /16
        self.enc5 = conv_block(c[3], c[4])                                              # /32
        self.dec4 = conv_block(c[4] + c[3], c[3])                                       # /16
        self.dec3 = conv_block(c[3] + c[2], c[2])                                       # /8
        self.dec2 = conv_block(c[2] + c[1], c[1])                                       # /4
        self.head = nn.Conv2d(c[1], n_points, 1)

    def forward(self, x):
        s1 = self.enc1(self.stem(x))
        s2 = self.enc2(F.max_pool2d(s1, 2))
        s3 = self.enc3(F.max_pool2d(s2, 2))
        s4 = self.enc4(F.max_pool2d(s3, 2))
        s5 = self.enc5(F.max_pool2d(s4, 2))
        d4 = self.dec4(torch.cat([F.interpolate(s5, size=s4.shape[2:], mode="bilinear", align_corners=False), s4], 1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=s3.shape[2:], mode="bilinear", align_corners=False), s3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=s2.shape[2:], mode="bilinear", align_corners=False), s2], 1))
        return self.head(d2)     # logits, (B, n_points, H/4, W/4)


def make_heatmaps(points_in, visible, out_h=INPUT_HEIGHT // STRIDE, out_w=INPUT_WIDTH // STRIDE, sigma=1.5):
    """Gaussian target per point. points_in: (n, 2) in network-input pixels."""
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    maps = np.zeros((len(points_in), out_h, out_w), np.float32)
    for k, ((x, y), vis) in enumerate(zip(points_in, visible)):
        if vis:
            cx, cy = (x + 0.5) / STRIDE - 0.5, (y + 0.5) / STRIDE - 0.5
            maps[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return maps


def decode_heatmaps(logits, window=2):
    """Heatmap logits (B, K, h, w) -> points (B, K, 2) in network-input pixels and
    confidences (B, K). Sub-pixel position = centroid around the peak."""
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    batch, n, h, w = probs.shape
    points = np.zeros((batch, n, 2), np.float32)
    conf = np.zeros((batch, n), np.float32)
    for b in range(batch):
        for k in range(n):
            peak_y, peak_x = np.unravel_index(np.argmax(probs[b, k]), (h, w))
            y0, y1 = max(0, peak_y - window), min(h, peak_y + window + 1)
            x0, x1 = max(0, peak_x - window), min(w, peak_x + window + 1)
            patch = probs[b, k, y0:y1, x0:x1]
            gy, gx = np.mgrid[y0:y1, x0:x1]
            total = patch.sum() + 1e-9
            cx, cy = (patch * gx).sum() / total, (patch * gy).sum() / total
            points[b, k] = ((cx + 0.5) * STRIDE - 0.5, (cy + 0.5) * STRIDE - 0.5)
            conf[b, k] = probs[b, k, peak_y, peak_x]
    return points, conf


def save_checkpoint(path, model, point_names, base, extra=None):
    torch.save({"state_dict": model.state_dict(), "point_names": point_names, "base": base,
                "input_size": [INPUT_HEIGHT, INPUT_WIDTH], **(extra or {})}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CourtNet(len(ckpt["point_names"]), ckpt["base"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["point_names"]
