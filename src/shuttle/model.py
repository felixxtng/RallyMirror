"""A small TrackNet-style network; this is not an exact TrackNet reproduction."""

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),
    )


class ShuttleNet(nn.Module):
    """Three RGB frames (9 channels) -> one heatmap for the middle frame."""

    def __init__(self, base_channels=16):
        super().__init__()
        b = base_channels
        self.enc1 = conv_block(9, b)
        self.enc2 = conv_block(b, b * 2)
        self.enc3 = conv_block(b * 2, b * 4)
        self.bottom = conv_block(b * 4, b * 8)
        self.dec3 = conv_block(b * 12, b * 4)
        self.dec2 = conv_block(b * 6, b * 2)
        self.dec1 = conv_block(b * 3, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, frames):
        e1 = self.enc1(frames)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        x = self.bottom(F.max_pool2d(e3, 2))
        x = self.dec3(torch.cat((F.interpolate(x, size=e3.shape[-2:],
                                             mode="bilinear", align_corners=False), e3), 1))
        x = self.dec2(torch.cat((F.interpolate(x, size=e2.shape[-2:],
                                             mode="bilinear", align_corners=False), e2), 1))
        x = self.dec1(torch.cat((F.interpolate(x, size=e1.shape[-2:],
                                             mode="bilinear", align_corners=False), e1), 1))
        return self.head(x)


def heatmap_loss(logits, targets):
    """Balance shuttle pixels against the hardest background pixels.

    Averaging every background pixel can hide a few strong false peaks.
    Mining the 1,024 most difficult background pixels per image makes those
    peaks contribute directly. Empty targets teach the model to output nothing.
    """
    losses = []
    for prediction, target in zip(logits, targets):
        foreground = target > 0.5
        background = ~foreground
        negative = F.softplus(prediction[background])
        negative = negative.topk(min(1024, negative.numel())).values.mean()
        if foreground.any():
            positive = F.softplus(-prediction[foreground]).mean()
            losses.append((positive + negative) / 2)
        else:
            losses.append(negative)
    return torch.stack(losses).mean()
