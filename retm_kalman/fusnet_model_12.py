from __future__ import annotations

import torch
import torch.nn as nn


class MultiChannelConvolutionModel12(nn.Module):
    """
    FuSNet-12 style multichannel convolution model.

    Input shape:
        [batch, 7, T]

    Output shape:
        [batch, 5, T - filter_length + 1]

    This matches the 12-mic setup where Group-B has 7 microphones and
    Group-A has 5 microphones.
    """

    def __init__(self, filter_length: int):
        super().__init__()
        self.filter_length = int(filter_length)
        self.conv1 = nn.Conv1d(1, 5, kernel_size=self.filter_length, bias=False)
        self.conv2 = nn.Conv1d(1, 5, kernel_size=self.filter_length, bias=False)
        self.conv3 = nn.Conv1d(1, 5, kernel_size=self.filter_length, bias=False)
        self.conv4 = nn.Conv1d(1, 5, kernel_size=self.filter_length, bias=False)
        self.conv5 = nn.Conv1d(1, 5, kernel_size=self.filter_length, bias=False)
        self.conv6 = nn.Conv1d(1, 5, kernel_size=self.filter_length, bias=False)
        self.conv7 = nn.Conv1d(1, 5, kernel_size=self.filter_length, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input [batch, 7, T], got {tuple(x.shape)}")
        if x.shape[1] != 7:
            raise ValueError(f"Expected 7 input channels, got {x.shape[1]}")

        conv1_out = self.conv1(x[:, 0:1, :])
        conv2_out = self.conv2(x[:, 1:2, :])
        conv3_out = self.conv3(x[:, 2:3, :])
        conv4_out = self.conv4(x[:, 3:4, :])
        conv5_out = self.conv5(x[:, 4:5, :])
        conv6_out = self.conv6(x[:, 5:6, :])
        conv7_out = self.conv7(x[:, 6:7, :])

        return conv1_out + conv2_out + conv3_out + conv4_out + conv5_out + conv6_out + conv7_out


def build_fusnet12_from_context(context: int) -> MultiChannelConvolutionModel12:
    """
    Original code used:
        filter_length = int(2 * context) + 1
    """
    return MultiChannelConvolutionModel12(filter_length=int(2 * context) + 1)
