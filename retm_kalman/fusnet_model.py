from __future__ import annotations
import torch
import torch.nn as nn


class MultiChannelConvolutionModel(nn.Module):
    """
    FuSNet-7 style multichannel convolution model from the user's original code.

    Input shape:
        [batch, 4, T]

    Output shape:
        [batch, 3, T - filter_length + 1]

    The model uses one Conv1d branch per Group-B microphone and sums the branch
    outputs to reconstruct the three Group-A microphone signals.
    """

    def __init__(self, filter_length: int):
        super().__init__()
        self.filter_length = int(filter_length)
        self.conv1 = nn.Conv1d(1, 3, kernel_size=self.filter_length, bias=False)
        self.conv2 = nn.Conv1d(1, 3, kernel_size=self.filter_length, bias=False)
        self.conv3 = nn.Conv1d(1, 3, kernel_size=self.filter_length, bias=False)
        self.conv4 = nn.Conv1d(1, 3, kernel_size=self.filter_length, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input [batch, 4, T], got {tuple(x.shape)}")
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 input channels, got {x.shape[1]}")
        conv1_out = self.conv1(x[:, 0:1, :])
        conv2_out = self.conv2(x[:, 1:2, :])
        conv3_out = self.conv3(x[:, 2:3, :])
        conv4_out = self.conv4(x[:, 3:4, :])
        return conv1_out + conv2_out + conv3_out + conv4_out


def build_fusnet7_from_context(context: int) -> MultiChannelConvolutionModel:
    """
    Original code used:
        filter_length = int(2 * context) + 1
    """
    return MultiChannelConvolutionModel(filter_length=int(2 * context) + 1)
