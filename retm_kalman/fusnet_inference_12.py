from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .fusnet_model_12 import build_fusnet12_from_context, MultiChannelConvolutionModel12
from .io_utils import frame_signal, overlap_add


def load_checkpoint_state(checkpoint_path: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    return {str(k).replace("module.", ""): v for k, v in state.items()}


def load_fusnet12_model(
    checkpoint_path: str | Path,
    context: int = 4096,
    device: str = "cuda",
):
    """
    Load the 12-mic FuSNet model.

    Expected channel layout:
        mB: 7 channels
        mA: 5 channels
    """
    dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    model = build_fusnet12_from_context(context)
    state = load_checkpoint_state(checkpoint_path, dev)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("Warning: missing checkpoint keys:", missing)
    if unexpected:
        print("Warning: unexpected checkpoint keys:", unexpected)
    model.to(dev)
    model.eval()
    return model, dev


@torch.no_grad()
def predict_fusnet12_original_style(
    model: MultiChannelConvolutionModel12,
    mB: np.ndarray,
    context: int = 4096,
    window_size: int = 16384,
    stride: int = 8192,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """
    FuSNet inference for the 12-mic setup.

    Input:
        mB: [7, T]
    Output:
        mA_f: [5, T]
    """
    dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    mB = np.asarray(mB, dtype=np.float32)
    if mB.ndim != 2 or mB.shape[0] != 7:
        raise ValueError(f"Expected mB shape [7, T], got {mB.shape}")

    T_orig = mB.shape[1]
    mB_pad = np.pad(mB, ((0, 0), (context, context)), mode="constant")
    frames = frame_signal(mB_pad, window_size, stride)

    out_frames = []
    for s in range(0, len(frames), batch_size):
        xb = torch.from_numpy(frames[s:s + batch_size]).float().to(dev)
        yb = model(xb)
        out_frames.append(yb.detach().cpu().float().numpy())

    y_frames = np.concatenate(out_frames, axis=0)
    y_full = overlap_add(y_frames, hop=stride)

    if y_full.shape[1] >= context + T_orig:
        y = y_full[:, context:context + T_orig]
    else:
        y = y_full
        if y.shape[1] < T_orig:
            y = np.pad(y, ((0, 0), (0, T_orig - y.shape[1])), mode="constant")
        y = y[:, :T_orig]

    return y.astype(np.float32)