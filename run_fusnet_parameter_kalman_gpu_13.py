from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from retm_kalman.fusnet_inference_13 import (
    load_fusnet13_model,
    predict_fusnet13_original_style,
)
from retm_kalman.kalman_fusnet_param_full_gpu import FullGPUFUSENetParameterKalman
from retm_kalman.kalman_fusnet_param_block_gpu import BlockGPUFUSENetParameterKalman


# ============================================================
# Speed / reproducibility settings
# ============================================================

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Fast mode. Results may vary slightly between runs.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# For exact reproducibility, use:
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


# ============================================================
# Utility functions
# ============================================================

def load_mic_group(seq_dir: str | Path, mic_indices: list[int], fs_target: int):
    seq_dir = Path(seq_dir)

    signals = []
    fs_out = None

    for mic_id in mic_indices:
        wav_path = seq_dir / f"mic_{mic_id}.wav"

        if not wav_path.exists():
            raise FileNotFoundError(f"Missing microphone file: {wav_path}")

        wav, fs = torchaudio.load(str(wav_path))

        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        if fs != fs_target:
            wav = torchaudio.functional.resample(wav, fs, fs_target)
            fs = fs_target

        signals.append(wav.squeeze(0))
        fs_out = fs

    min_len = min(x.numel() for x in signals)
    signals = [x[:min_len] for x in signals]

    X = torch.stack(signals, dim=0)

    return X, fs_out


def normalize_pair_torch(mA: torch.Tensor, mB: torch.Tensor):
    peak = torch.max(torch.abs(torch.cat([mA, mB], dim=0)))

    if peak > 0:
        return mA / peak, mB / peak, float(peak.item())

    return mA, mB, 1.0


def sdr_db(target: np.ndarray, estimate: np.ndarray, eps: float = 1e-12):
    T = min(target.shape[1], estimate.shape[1])
    target = target[:, :T]
    estimate = estimate[:, :T]

    err = target - estimate

    vals = []
    for ch in range(target.shape[0]):
        sig_pow = np.sum(target[ch] ** 2)
        err_pow = np.sum(err[ch] ** 2)
        vals.append(10 * np.log10((sig_pow + eps) / (err_pow + eps)))

    return np.asarray(vals)


def mse_db(target: np.ndarray, estimate: np.ndarray, eps: float = 1e-12):
    T = min(target.shape[1], estimate.shape[1])
    target = target[:, :T]
    estimate = estimate[:, :T]

    err = target - estimate
    mse = np.mean(err ** 2, axis=1)

    return 10 * np.log10(mse + eps)


def print_metrics(target: np.ndarray, estimate: np.ndarray, name: str):
    sdr = sdr_db(target, estimate)
    mse = mse_db(target, estimate)

    print()
    print(name)
    print("-" * 80)
    print("SDR per channel:", np.round(sdr, 4))
    print("SDR avg        :", float(np.mean(sdr)))
    print("MSE dB/channel :", np.round(mse, 4))
    print("MSE dB avg     :", float(np.mean(mse)))


# ============================================================
# Correct FuSNet13 original-style frame alignment
# ============================================================

def make_fusnet13_frames_original_style(
    mB: torch.Tensor,
    mA: torch.Tensor,
    context: int,
    window_size: int,
    stride: int,
):
    """
    Create FuSNet13 frames using the same alignment as predict_fusnet13_original_style().

    Original inference:
        mB_pad = pad(mB, context, context)
        frames = frame_signal(mB_pad, window_size, stride)
        y_full = overlap_add(y_frames, hop=stride)
        y = y_full[:, context:context + T_orig]

    Therefore, a model output frame from padded start index 'start'
    maps to original target index:

        target_start = start - context
    """

    if mB.ndim != 2 or mB.shape[0] != 8:
        raise ValueError(f"Expected mB shape [8, T], got {tuple(mB.shape)}")

    if mA.ndim != 2 or mA.shape[0] != 5:
        raise ValueError(f"Expected mA shape [5, T], got {tuple(mA.shape)}")

    T = min(mB.shape[1], mA.shape[1])
    mB = mB[:, :T]
    mA = mA[:, :T]

    filter_length = 2 * context + 1
    out_len = window_size - filter_length + 1

    if out_len <= 0:
        raise ValueError("window_size must be larger than filter_length")

    # Same input padding as original inference
    mB_pad = F.pad(mB, (context, context), mode="constant", value=0.0)

    padded_len = mB_pad.shape[1]

    frames_x = []
    frames_d = []
    output_starts_original = []

    for start in range(0, padded_len - window_size + 1, stride):
        x_frame = mB_pad[:, start:start + window_size]

        target_start = start - context
        target_end = target_start + out_len

        d_frame = torch.zeros(
            (mA.shape[0], out_len),
            dtype=mA.dtype,
            device=mA.device,
        )

        valid_start = max(target_start, 0)
        valid_end = min(target_end, T)

        if valid_end > valid_start:
            dst_start = valid_start - target_start
            dst_end = dst_start + (valid_end - valid_start)
            d_frame[:, dst_start:dst_end] = mA[:, valid_start:valid_end]

        frames_x.append(x_frame)
        frames_d.append(d_frame)
        output_starts_original.append(target_start)

    return frames_x, frames_d, output_starts_original, T, out_len


def overlap_add_original_style(
    frames: list[np.ndarray],
    starts_original: list[int],
    T: int,
    qa: int = 5,
):
    """
    Overlap-add using original-time output positions.

    Some first-frame outputs start at negative index because of context padding.
    This function crops those parts correctly.
    """

    y = np.zeros((qa, T), dtype=np.float64)
    w = np.zeros((T,), dtype=np.float64)

    for frame, start in zip(frames, starts_original):
        frame_len = frame.shape[1]
        end = start + frame_len

        dst_start = max(start, 0)
        dst_end = min(end, T)

        if dst_end <= dst_start:
            continue

        src_start = dst_start - start
        src_end = src_start + (dst_end - dst_start)

        y[:, dst_start:dst_end] += frame[:, src_start:src_end]
        w[dst_start:dst_end] += 1.0

    w[w == 0] = 1.0

    return (y / w[None, :]).astype(np.float32)


# ============================================================
# Kalman run functions
# ============================================================

def run_full_param_kalman(
    model,
    frames_x,
    frames_d,
    starts,
    T,
    qa,
    updater,
    device,
):
    """
    Full mode:
        one Kalman-style parameter update per FuSNet frame.
    """

    # Keep eval mode; gradients still work.
    model.eval()

    out_frames = []
    error_trace = []

    for i, (x, d) in enumerate(zip(frames_x, frames_d)):
        x = x.unsqueeze(0).to(device, non_blocking=True)
        d = d.unsqueeze(0).to(device, non_blocking=True)

        y, e, loss = updater.step(x, d)

        out_frames.append(y.squeeze(0).cpu().float().numpy())

        err_rms = torch.sqrt(torch.mean(e ** 2)).cpu().item()
        error_trace.append(err_rms)

        if (i + 1) % 10 == 0 or (i + 1) == len(frames_x):
            print(
                f"Full parameter Kalman frame {i+1:04d}/{len(frames_x)}, "
                f"loss={loss:.6e}, err_rms={err_rms:.6e}"
            )

    mA_hat = overlap_add_original_style(out_frames, starts, T, qa)
    error_trace = np.asarray(error_trace, dtype=np.float32)

    return mA_hat, error_trace


def run_block_param_kalman(
    model,
    frames_x,
    frames_d,
    starts,
    T,
    qa,
    updater,
    device,
):
    """
    Block mode:
        one Kalman-style parameter update per block of FuSNet frames.
    """

    # Keep eval mode; gradients still work.
    model.eval()

    out_frames = []
    out_starts = []
    error_trace = []

    block_frames = updater.block_frames
    num_frames = len(frames_x)

    for block_start in range(0, num_frames, block_frames):
        block_end = min(block_start + block_frames, num_frames)

        xb = torch.stack(frames_x[block_start:block_end], dim=0).to(
            device, non_blocking=True
        )
        db = torch.stack(frames_d[block_start:block_end], dim=0).to(
            device, non_blocking=True
        )

        yb, eb, loss = updater.step_block(xb, db)

        yb_np = yb.cpu().float().numpy()

        for j in range(yb_np.shape[0]):
            out_frames.append(yb_np[j])
            out_starts.append(starts[block_start + j])

        err_rms = torch.sqrt(torch.mean(eb ** 2, dim=(1, 2))).cpu().numpy()
        error_trace.extend(err_rms.tolist())

        print(
            f"Block parameter Kalman frames {block_start+1:04d}-{block_end:04d}/"
            f"{num_frames}, loss={loss:.6e}"
        )

    mA_hat = overlap_add_original_style(out_frames, out_starts, T, qa)
    error_trace = np.asarray(error_trace, dtype=np.float32)

    return mA_hat, error_trace


# ============================================================
# Main
# ============================================================

def main():
    # --------------------------------------------------------
    # USER SETTINGS
    # --------------------------------------------------------

    seq_dir = Path(
        "/home/jaliya/eeg_speech/Julian/RetM_Workspace/Dataset/mic_moving"
    )

    checkpoint_path = Path(
        "/home/jaliya/eeg_speech/Julian/RetM_Workspace/ReTM_Research_Project/best_checkpoint_A1_1_FUSENet_13_rctd.pth"
    )

    out_dir = Path("results_fusnet_parameter_kalman_gpu_13_corrected")
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_target = 16000

    qa_mics = [1, 2, 3, 4, 5]
    qb_mics = [6, 7, 8, 9, 10, 11, 12, 13]

    context = 4096
    filter_length = 2 * context + 1

    window_size = 16384
    stride = 8192

    # Choose "full" or "block"
    mode = "full"

    # For block mode, this is number of FuSNet frames per update,
    # not number of audio samples.
    kalman_block_frames = 4

    transition = 0.995
    process_noise = 1e-8
    observation_noise = 1e-2
    initial_covariance = 1e-3

    kalman_lr = 20
    max_grad_norm = 1

    # "initial" keeps parameters close to original checkpoint.
    # "zero" applies direct G*r decay.
    transition_center = "zero"

    save_adapted_checkpoint = True

    batch_size_for_baseline = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("Corrected GPU FuSNet Full-Parameter Kalman Adaptation")
    print("=" * 90)
    print(f"Mode        : {mode}")
    print(f"Device      : {device}")
    print(f"Sequence    : {seq_dir}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Output dir  : {out_dir}")
    print(f"Filter size : 8 x 5 x {filter_length}")
    print(f"Parameters  : {8 * 5 * filter_length}")
    print("=" * 90)

    # --------------------------------------------------------
    # Load microphone data
    # --------------------------------------------------------

    print("\n[1] Loading microphone WAVs...")

    mB_cpu, fsB = load_mic_group(seq_dir, qb_mics, fs_target)
    mA_cpu, fsA = load_mic_group(seq_dir, qa_mics, fs_target)

    if fsA != fsB:
        raise ValueError("Sampling rates do not match")

    fs = fsA

    T0 = min(mA_cpu.shape[1], mB_cpu.shape[1])
    mA_cpu = mA_cpu[:, :T0]
    mB_cpu = mB_cpu[:, :T0]

    mA_cpu, mB_cpu, scale = normalize_pair_torch(mA_cpu, mB_cpu)

    print(f"mA shape: {tuple(mA_cpu.shape)}")
    print(f"mB shape: {tuple(mB_cpu.shape)}")
    print(f"fs      : {fs}")
    print(f"scale   : {scale:.8f}")

    # --------------------------------------------------------
    # Baseline FuSNet using ORIGINAL inference function
    # --------------------------------------------------------

    print("\n[2] Loading baseline FuSNet model...")

    baseline_model, dev_loaded = load_fusnet13_model(
        checkpoint_path=checkpoint_path,
        context=context,
        device=str(device),
    )

    baseline_model.eval()

    print("\n[3] Running baseline FuSNet using original inference function...")

    mA_fusnet = predict_fusnet13_original_style(
        model=baseline_model,
        mB=mB_cpu.numpy(),
        context=context,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size_for_baseline,
        device=device,
    )

    T = min(mA_cpu.shape[1], mB_cpu.shape[1], mA_fusnet.shape[1])
    mA_cpu = mA_cpu[:, :T]
    mB_cpu = mB_cpu[:, :T]
    mA_fusnet = mA_fusnet[:, :T]

    mA_np = mA_cpu.numpy()

    print_metrics(mA_np, mA_fusnet, "Baseline FuSNet original-style inference")

    # --------------------------------------------------------
    # Move full signals to GPU and create corrected frames
    # --------------------------------------------------------

    print("\n[4] Creating corrected FuSNet frames on GPU...")

    mA_gpu = mA_cpu.to(device, non_blocking=True)
    mB_gpu = mB_cpu.to(device, non_blocking=True)

    frames_x, frames_d, starts, T, out_len = make_fusnet13_frames_original_style(
        mB=mB_gpu,
        mA=mA_gpu,
        context=context,
        window_size=window_size,
        stride=stride,
    )

    print(f"Number of frames: {len(frames_x)}")
    print(f"Output length per frame: {out_len}")
    print(f"Signal length: {T}")

    # --------------------------------------------------------
    # Load adaptive model
    # --------------------------------------------------------

    print("\n[5] Loading adaptive FuSNet model...")

    model, _ = load_fusnet13_model(
        checkpoint_path=checkpoint_path,
        context=context,
        device=str(device),
    )

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable FuSNet parameters to update: {total_params}")

    # --------------------------------------------------------
    # Run Kalman parameter adaptation
    # --------------------------------------------------------

    print("\n[6] Running corrected GPU parameter Kalman update...")

    if mode.lower() == "full":
        updater = FullGPUFUSENetParameterKalman(
            model=model,
            transition=transition,
            process_noise=process_noise,
            observation_noise=observation_noise,
            initial_covariance=initial_covariance,
            kalman_lr=kalman_lr,
            max_grad_norm=max_grad_norm,
            transition_center=transition_center,
        )

        mA_hat, error_trace = run_full_param_kalman(
            model=model,
            frames_x=frames_x,
            frames_d=frames_d,
            starts=starts,
            T=T,
            qa=5,
            updater=updater,
            device=device,
        )

    elif mode.lower() == "block":
        updater = BlockGPUFUSENetParameterKalman(
            model=model,
            block_frames=kalman_block_frames,
            transition=transition,
            process_noise=process_noise,
            observation_noise=observation_noise,
            initial_covariance=initial_covariance,
            kalman_lr=kalman_lr,
            max_grad_norm=max_grad_norm,
            transition_center=transition_center,
        )

        mA_hat, error_trace = run_block_param_kalman(
            model=model,
            frames_x=frames_x,
            frames_d=frames_d,
            starts=starts,
            T=T,
            qa=5,
            updater=updater,
            device=device,
        )

    else:
        raise ValueError("mode must be 'full' or 'block'")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print_metrics(mA_np, mA_fusnet, "Baseline FuSNet original-style inference")
    print_metrics(mA_np, mA_hat, "FuSNet after corrected GPU parameter Kalman")

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    print("\n[7] Saving outputs...")

    np.save(out_dir / "mA_target.npy", mA_np)
    np.save(out_dir / "mB_input.npy", mB_cpu[:, :T].numpy())
    np.save(out_dir / "mA_fusnet_baseline_original_style.npy", mA_fusnet)
    np.save(out_dir / "mA_after_gpu_parameter_kalman.npy", mA_hat)
    np.save(out_dir / "error_trace_rms.npy", error_trace)

    metrics = {
        "mode": mode,
        "fs": fs,
        "num_samples": int(T),
        "context": context,
        "filter_length": filter_length,
        "window_size": window_size,
        "stride": stride,
        "kalman_block_frames": kalman_block_frames,
        "transition": transition,
        "process_noise": process_noise,
        "observation_noise": observation_noise,
        "initial_covariance": initial_covariance,
        "kalman_lr": kalman_lr,
        "max_grad_norm": max_grad_norm,
        "transition_center": transition_center,
        "total_updated_parameters": int(total_params),
        "baseline_sdr_per_channel": sdr_db(mA_np, mA_fusnet).tolist(),
        "baseline_sdr_avg": float(np.mean(sdr_db(mA_np, mA_fusnet))),
        "param_kalman_sdr_per_channel": sdr_db(mA_np, mA_hat).tolist(),
        "param_kalman_sdr_avg": float(np.mean(sdr_db(mA_np, mA_hat))),
        "baseline_mse_db_per_channel": mse_db(mA_np, mA_fusnet).tolist(),
        "param_kalman_mse_db_per_channel": mse_db(mA_np, mA_hat).tolist(),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if save_adapted_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "architecture": "FuSNet13 corrected full-parameter GPU Kalman adapted",
                "mode": mode,
                "context": context,
                "filter_length": filter_length,
                "kalman_settings": {
                    "transition": transition,
                    "process_noise": process_noise,
                    "observation_noise": observation_noise,
                    "initial_covariance": initial_covariance,
                    "kalman_lr": kalman_lr,
                    "max_grad_norm": max_grad_norm,
                    "transition_center": transition_center,
                    "kalman_block_frames": kalman_block_frames,
                },
            },
            out_dir / "adapted_fusnet_gpu_parameter_kalman_corrected.pth",
        )

    # Save wav files
    for ch in range(5):
        torchaudio.save(
            str(out_dir / f"target_mic_{ch+1}.wav"),
            torch.from_numpy(mA_np[ch:ch + 1]).float(),
            fs,
        )

        torchaudio.save(
            str(out_dir / f"baseline_fusnet_mic_{ch+1}.wav"),
            torch.from_numpy(mA_fusnet[ch:ch + 1]).float(),
            fs,
        )

        torchaudio.save(
            str(out_dir / f"after_gpu_param_kalman_mic_{ch+1}.wav"),
            torch.from_numpy(mA_hat[ch:ch + 1]).float(),
            fs,
        )

    print(f"\nDone. Results saved to:\n{out_dir}")


if __name__ == "__main__":
    main()