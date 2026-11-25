"""Temporal Convolutional Network (TCN) over pose-keypoint sequences.

This is the learned temporal model behind ``TrainableClassifier``. It is kept in
its own module so that ``classifier.py`` (the swappable interface + the
torch-free rule-based baseline) never requires torch to import. Import this
module only when you actually train or run the learned model.

Why a TCN for this input
------------------------
The input is NOT raw video -- MediaPipe already distilled each frame to a small
pose feature vector (``FEATURE_LEN`` values). So the model only has to learn
*motion patterns over time*, not perceive a body from pixels. A TCN treats a
window as a ``FEATURE_LEN``-channel 1-D signal of length ``window_size`` and
convolves ALONG TIME:

- **dilated** convolutions give an exponentially growing receptive field with
  few layers (so a short stack can see the whole window);
- **causal** padding means a prediction for the current frame only uses current
  and past frames -- matching the online ``PhasePipeline`` which classifies the
  current frame from its trailing window.

Small, fast, and well suited to little data -- the right first model here.

Tensor shapes
-------------
A window is ``window_size x FEATURE_LEN`` (time-major, as produced by the
windower and the dataset builder). For conv1d we transpose to
``(batch, channels=FEATURE_LEN, length=window_size)``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class _CausalConv1d(nn.Module):
    """1-D convolution with left-only (causal) padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad only on the left along the time axis, then trim nothing (valid).
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class _TemporalBlock(nn.Module):
    """Dilated causal conv -> ReLU -> dropout, twice, with a residual link."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 1x1 conv to match channels for the residual add when they differ.
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """Stacked dilated causal conv blocks + global pooling + linear head.

    Parameters
    ----------
    feature_len:
        Channels per frame (``2 * NUM_KEYPOINTS``).
    num_classes:
        Number of phase classes.
    channels:
        Output channels for each temporal block (its depth = len(channels)).
        Dilation doubles per block (1, 2, 4, ...).
    kernel_size, dropout:
        Standard conv hyperparameters.
    """

    def __init__(
        self,
        feature_len: int,
        num_classes: int,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        channels = channels or [64, 64]
        self.hparams = {
            "feature_len": feature_len,
            "num_classes": num_classes,
            "channels": list(channels),
            "kernel_size": kernel_size,
            "dropout": dropout,
        }
        blocks = []
        in_ch = feature_len
        for i, out_ch in enumerate(channels):
            blocks.append(
                _TemporalBlock(in_ch, out_ch, kernel_size,
                               dilation=2 ** i, dropout=dropout)
            )
            in_ch = out_ch
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_size, feature_len) -> logits (batch, num_classes)."""
        # To (batch, channels=feature_len, length=window_size) for conv1d.
        x = x.transpose(1, 2)
        y = self.network(x)
        # Global average over time, then classify.
        y = y.mean(dim=2)
        return self.head(y)
