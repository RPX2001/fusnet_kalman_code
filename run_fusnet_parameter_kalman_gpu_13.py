from __future__ import annotations

import os
import json
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
import torchaudio

from retm_kalman.fusnet_inference_13 import load_fusnet13_model
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

# For maximum speed, benchmark=True is faster but not fully deterministic.
# For exact reproducibility, set benchmark=False and deterministic=True.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


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


def make_frames(mB: torch.Tensor, mA: torch.Tensor, context: int, window_size: int, stride: int):
    """
    Create FuSNet input/target frames.

    mB: [8, T]
    mA: [5, T]

    Input frame:
        [8, window_size]

    Target frame:
        [5, window_size - 2*context]

    For context=4096 and window_size=16384:
        target length = 8192
    """
    T = min(mB.shape[1], mA.shape[1])
    mB = mB[:, :T]
    mA = mA[:, :T]

    out_len = window_size - 2 * context

    if out_len <= 0:
        raise ValueError("window_size must be larger than 2*context")

    mB_pad = F.pad(mB, (context, context), mode="constant", value=0.0)

    frames_x = []
    frames_d = []
    starts = []

    for start in range(0, T, stride):
        x_start = start
        x_end = x_start + window_size

        x_frame = mB_pad[:, x_start:x_end]

        if x_frame.shape[1] < window_size:
            pad_right = window_size - x_frame.shape[1]
            x_frame = F.pad(x_frame, (0, pad_right), mode="constant", value=0.0)

        d_start = start
        d_end = min(start + out_len, T)

        d_frame = mA[:, d_start:d_end]

        if d_frame.shape[1] < out_len:
            pad_right = out_len - d_frame.shape[1]
            d_frame = F.pad(d_frame, (0, pad_right), mode="constant", value=0.0)

        frames_x.append(x_frame)
        frames_d.append(d_frame)
        starts.append(start)

        if start + out_len >= T:
            break

    return frames_x, frames_d, starts, T, out_len


def overlap_add(frames: list[np.ndarray], starts: list[int], T: int, qa: int):
    y = np.zeros((qa, T), dtype=np.float64)
    w = np.zeros((T,), dtype=np.float64)

    for frame, start in zip(frames, starts):
        frame_len = frame.shape[1]
        end = min(start + frame_len, T)
        valid = end - start

        if valid <= 0:
            continue

        y[:, start:end] += frame[:, :valid]
        w[start:end] += 1.0

    w[w == 0] = 1.0

    return (y / w[None, :]).astype(np.float32)


@torch.no_grad()
def run_baseline_fusnet(model, frames_x, starts, T, qa, device):
    model.eval()

    out_frames = []

    for x in frames_x:
        x = x.unsqueeze(0).to(device, non_blocking=True)
        y = model(x)
        out_frames.append(y.squeeze(0).detach().cpu().float().numpy())

    return overlap_add(out_frames, starts, T, qa)


def run_full_param_kalman(model, frames_x, frames_d, starts, T, qa, updater, device):
    model.train()

    out_frames = []
    error_trace = []

    for i, (x, d) in enumerate(zip(frames_x, frames_d)):
        x = x.unsqueeze(0).to(device, non_blocking=True)
        d = d.unsqueeze(0).to(device, non_blocking=True)

        y, e, loss = updater.step(x, d)

        out_frames.append(y.squeeze(0).cpu().float().numpy())
        error_trace.append(torch.sqrt(torch.mean(e ** 2)).cpu().item())

        if (i + 1) % 10 == 0:
            print(f"Full Kalman frame {i+1:04d}/{len(frames_x)}, loss={loss:.6e}")

    mA_hat = overlap_add(out_frames, starts, T, qa)
    error_trace = np.asarray(error_trace, dtype=np.float32)

    return mA_hat, error_trace


def run_block_param_kalman(model, frames_x, frames_d, starts, T, qa, updater, device):
    model.train()

    out_frames = []
    out_starts = []
    error_trace = []

    block_frames = updater.block_frames
    num_frames = len(frames_x)

    for block_start in range(0, num_frames, block_frames):
        block_end = min(block_start + block_frames, num_frames)

        xb = torch.stack(frames_x[block_start:block_end], dim=0).to(device, non_blocking=True)
        db = torch.stack(frames_d[block_start:block_end], dim=0).to(device, non_blocking=True)

        yb, eb, loss = updater.step_block(xb, db)

        yb_np = yb.cpu().float().numpy()

        for j in range(yb_np.shape[0]):
            out_frames.append(yb_np[j])
            out_starts.append(starts[block_start + j])

        err_rms = torch.sqrt(torch.mean(eb ** 2, dim=(1, 2))).cpu().numpy()
        error_trace.extend(err_rms.tolist())

        print(
            f"Block Kalman frames {block_start+1:04d}-{block_end:04d}/{num_frames}, "
            f"loss={loss:.6e}"
        )

    mA_hat = overlap_add(out_frames, out_starts, T, qa)
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

    checkpoint_path = Path("/home/jaliya/eeg_speech/Julian/RetM_Workspace/ReTM_Research_Project/best_checkpoint_A1_1_FUSENet_13_rctd.pth")

    out_dir = Path("results_fusnet_parameter_kalman_gpu_13")
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_target = 16000

    qa_mics = [1, 2, 3, 4, 5]
    qb_mics = [6, 7, 8, 9, 10, 11, 12, 13]

    context = 4096
    filter_length = 2 * context + 1

    window_size = 16384
    stride = 8192

    # Choose "full" or "block"
    mode = "block"

    # Block Kalman setting:
    # This is number of FuSNet frames per Kalman update, not audio samples.
    kalman_block_frames = 4

    transition = 0.995
    process_noise = 1e-8
    observation_noise = 1e-2
    initial_covariance = 1e-3

    kalman_lr = 0.000001
    max_grad_norm = 1.0

    transition_center = "initial"

    save_adapted_checkpoint = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("GPU FuSNet Full-Parameter Kalman Adaptation")
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

    mA_cpu, mB_cpu, scale = normalize_pair_torch(mA_cpu, mB_cpu)

    print(f"mA shape: {tuple(mA_cpu.shape)}")
    print(f"mB shape: {tuple(mB_cpu.shape)}")
    print(f"fs      : {fs}")
    print(f"scale   : {scale:.8f}")

    # Keep full signals on GPU for frame slicing speed
    mA_gpu = mA_cpu.to(device, non_blocking=True)
    mB_gpu = mB_cpu.to(device, non_blocking=True)

    # --------------------------------------------------------
    # Make frames directly on GPU
    # --------------------------------------------------------

    print("\n[2] Creating FuSNet frames...")

    frames_x, frames_d, starts, T, out_len = make_frames(
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
    # Baseline FuSNet
    # --------------------------------------------------------

    print("\n[3] Loading baseline FuSNet model...")

    baseline_model, _ = load_fusnet13_model(
        checkpoint_path=checkpoint_path,
        context=context,
        device=str(device),
    )

    baseline_model.to(device)

    print("\n[4] Running baseline FuSNet...")

    mA_fusnet = run_baseline_fusnet(
        model=baseline_model,
        frames_x=frames_x,
        starts=starts,
        T=T,
        qa=5,
        device=device,
    )

    # --------------------------------------------------------
    # Adaptive FuSNet model
    # --------------------------------------------------------

    print("\n[5] Loading adaptive FuSNet model...")

    model, _ = load_fusnet13_model(
        checkpoint_path=checkpoint_path,
        context=context,
        device=str(device),
    )

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable FuSNet parameters to update: {total_params}")

    # --------------------------------------------------------
    # Kalman update
    # --------------------------------------------------------

    print("\n[6] Running GPU parameter Kalman update...")

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

    mA_np = mA_cpu[:, :T].cpu().numpy()

    print_metrics(mA_np, mA_fusnet, "Baseline FuSNet")
    print_metrics(mA_np, mA_hat, "FuSNet after GPU parameter Kalman")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n[7] Saving outputs...")

    np.save(out_dir / "mA_target.npy", mA_np)
    np.save(out_dir / "mB_input.npy", mB_cpu[:, :T].cpu().numpy())
    np.save(out_dir / "mA_fusnet_baseline.npy", mA_fusnet)
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
                "architecture": "FuSNet13 full-parameter GPU Kalman adapted",
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
            out_dir / "adapted_fusnet_gpu_parameter_kalman.pth",
        )

    # Save wav files
    for ch in range(5):
        torchaudio.save(
            str(out_dir / f"target_mic_{ch+1}.wav"),
            torch.from_numpy(mA_np[ch:ch+1]).float(),
            fs,
        )

        torchaudio.save(
            str(out_dir / f"baseline_fusnet_mic_{ch+1}.wav"),
            torch.from_numpy(mA_fusnet[ch:ch+1]).float(),
            fs,
        )

        torchaudio.save(
            str(out_dir / f"after_gpu_param_kalman_mic_{ch+1}.wav"),
            torch.from_numpy(mA_hat[ch:ch+1]).float(),
            fs,
        )

    print(f"\nDone. Results saved to:\n{out_dir}")


if __name__ == "__main__":
    main()