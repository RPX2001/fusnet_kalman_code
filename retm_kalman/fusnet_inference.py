from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from .fusnet_model import build_fusnet7_from_context, MultiChannelConvolutionModel
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


def load_fusnet7_model(checkpoint_path: str | Path,
                       context: int = 4096,
                       device: str = "cuda"):
    """
    Load the integrated FuSNet-7 model.

    This matches the user's original model construction:
        context = 2**13 // 2 = 4096
        filter_length = 2 * context + 1 = 8193
    """
    dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    model = build_fusnet7_from_context(context)
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
def predict_fusnet7_original_style(model: MultiChannelConvolutionModel,
                                   mB: np.ndarray,
                                   context: int = 4096,
                                   window_size: int = 16384,
                                   stride: int = 8192,
                                   batch_size: int = 8,
                                   device: str | torch.device = "cuda") -> np.ndarray:
    """
    FuSNet inference using the same padding/windowing logic as the provided training code.

    Input:
        mB: [4, T]
    Output:
        mA_f: [3, T] approximately aligned with original signal length.

    Notes:
    - The model convolution is valid convolution, so each window produces
      window_size - filter_length + 1 samples. With context=4096 and
      window_size=16384, this is 8192 samples.
    - We zero-pad the full input by context samples on both sides, frame it,
      run FuSNet, and overlap-add the output windows with hop=stride.
    """
    dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    mB = np.asarray(mB, dtype=np.float32)
    if mB.ndim != 2 or mB.shape[0] != 4:
        raise ValueError(f"Expected mB shape [4, T], got {mB.shape}")

    T_orig = mB.shape[1]
    mB_pad = np.pad(mB, ((0, 0), (context, context)), mode="constant")
    frames = frame_signal(mB_pad, window_size, stride)  # [N, 4, window_size]

    out_frames = []
    for s in range(0, len(frames), batch_size):
        xb = torch.from_numpy(frames[s:s + batch_size]).float().to(dev)
        yb = model(xb)
        out_frames.append(yb.detach().cpu().float().numpy())

    y_frames = np.concatenate(out_frames, axis=0)  # [N, 3, out_len]
    out_len = y_frames.shape[-1]
    y_full = overlap_add(y_frames, hop=stride)     # [3, approx padded output length]

    # The original script applies another trim after concatenation:
    # output = output[:, context:]
    # For stable signal alignment, remove initial context and keep original length.
    if y_full.shape[1] >= context + T_orig:
        y = y_full[:, context:context + T_orig]
    else:
        y = y_full
        if y.shape[1] < T_orig:
            y = np.pad(y, ((0, 0), (0, T_orig - y.shape[1])), mode="constant")
        y = y[:, :T_orig]
    return y.astype(np.float32)
