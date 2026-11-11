"""Small court keypoint CNN, created with random weights; no pretrained backbone."""
import torch
from torch import nn
from torch.nn import functional as F

from .learning_data import POINT_NAMES


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                         nn.GroupNorm(4, cout), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.ReLU())


class CourtNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = block(3, 16, 2)
        self.down2 = block(16, 32, 2)
        self.down3 = block(32, 64, 2)
        self.fuse = block(96, 32)
        # Spatial coordinate channels help distinguish near/far and left/right.
        self.head = nn.Sequential(nn.Conv2d(34, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, len(POINT_NAMES), 1))

    def forward(self, images):
        low = self.down2(self.down1(images))
        deep = self.down3(low)
        features = self.fuse(torch.cat((low, F.interpolate(deep, size=low.shape[-2:],
                                                          mode="bilinear", align_corners=False)), dim=1))
        h, w = features.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=images.device),
                                torch.linspace(-1, 1, w, device=images.device), indexing="ij")
        coords = torch.stack((xx, yy))[None].expand(len(images), -1, -1, -1)
        return self.head(torch.cat((features, coords), dim=1))


def keypoint_loss(logits, targets, mask):
    per_point = -(targets.flatten(2) * F.log_softmax(logits.flatten(2), dim=-1)).sum(-1)
    return (per_point * mask).sum() / mask.sum().clamp_min(1)


def decode_points(logits):
    """Heatmap maxima in input-image pixels. Peaks are not visibility confidence."""
    _, _, h, w = logits.shape
    index = logits.flatten(2).argmax(dim=-1)
    return torch.stack(((index % w) * 4, (index // w) * 4), dim=-1).float()
