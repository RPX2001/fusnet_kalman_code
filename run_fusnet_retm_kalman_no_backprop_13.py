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
from retm_kalman.kalman_fusnet_retm_partitioned_gpu import (
    PartitionedBlockReTMKalmanFromFuSNet,
)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_mic_group(seq_dir, mic_indices, fs_target):
    seq_dir = Path(seq_dir)
    signals, fs_out = [], None
    for mic_id in mic_indices:
        wav_path = seq_dir / f"mic_{mic_id}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing: {wav_path}")
        wav, fs = torchaudio.load(str(wav_path))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if fs != fs_target:
            wav = torchaudio.functional.resample(wav, fs, fs_target)
        signals.append(wav.squeeze(0))
        fs_out = fs_target
    min_len = min(x.numel() for x in signals)
    return torch.stack([x[:min_len] for x in signals], dim=0), fs_out


def normalize_pair(mA, mB):
    peak = torch.max(torch.abs(torch.cat([mA, mB], dim=0)))
    if peak > 0:
        return mA / peak, mB / peak, float(peak)
    return mA, mB, 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def sdr_db(target, est, eps=1e-12):
    T = min(target.shape[1], est.shape[1])
    err = target[:, :T] - est[:, :T]
    sig = np.sum(target[:, :T] ** 2, axis=1)
    noise = np.sum(err ** 2, axis=1)
    return 10 * np.log10((sig + eps) / (noise + eps))


def mse_db(target, est, eps=1e-12):
    T = min(target.shape[1], est.shape[1])
    mse = np.mean((target[:, :T] - est[:, :T]) ** 2, axis=1)
    return 10 * np.log10(mse + eps)


def print_metrics(target, est, label):
    sdr = sdr_db(target, est)
    mse = mse_db(target, est)
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  SDR per ch : {np.round(sdr, 3).tolist()}")
    print(f"  SDR avg    : {np.mean(sdr):.4f} dB")
    print(f"  MSE per ch : {np.round(mse, 3).tolist()}")
    print(f"  MSE avg    : {np.mean(mse):.4f} dB")


# ──────────────────────────────────────────────────────────────────────────────
# Kalman runner
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_retm_kalman_samplewise(
    kalman,
    mB,
    mA,
    context,
    filter_length,
    update_stride=1,
    log_every=4000,
):
    device, dtype = kalman.device, kalman.dtype
    mB = mB.to(device=device, dtype=dtype)
    mA = mA.to(device=device, dtype=dtype)

    T = min(mB.shape[1], mA.shape[1])
    mB, mA = mB[:, :T], mA[:, :T]

    # Same padding as baseline FuSNet inference
    mB_pad = F.pad(mB, (context, context), mode="constant", value=0.0)

    y_out   = torch.zeros((kalman.qa, T), device=device, dtype=dtype)
    err_out = torch.zeros((kalman.qa, T), device=device, dtype=dtype)
    error_rms_log = []

    for t in range(T):
        start  = context + t
        x_full = mB_pad[:, start : start + filter_length]

        if x_full.shape[1] < filter_length:
            x_full = F.pad(x_full, (0, filter_length - x_full.shape[1]))

        d = mA[:, t]

        if update_stride <= 1 or (t % update_stride == 0):
            y_hat, e = kalman.update_one_sample(x_full=x_full, d=d)
        else:
            y_hat = kalman.predict_one_sample(x_full=x_full)
            e = d - y_hat

        y_out[:, t]   = y_hat
        err_out[:, t] = e

        if t % log_every == 0 or t == T - 1:
            rms = float(torch.sqrt(torch.mean(e ** 2)).item())
            error_rms_log.append(rms)
            if kalman.adaptive_noise:
                rv_mean = float(kalman._err_power_ema.mean().item())
                print(f"  t={t+1:07d}/{T}  err_rms={rms:.4e}  Rv_ema={rv_mean:.4e}")
            else:
                print(f"  t={t+1:07d}/{T}  err_rms={rms:.4e}")

    return (
        y_out.detach().cpu().float().numpy(),
        err_out.detach().cpu().float().numpy(),
        np.array(error_rms_log, dtype=np.float32),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():

    # ── Paths ──────────────────────────────────────────────────────────
    seq_dir = Path(
        "/home/jaliya/eeg_speech/Julian/RetM_Workspace/Dataset/Dataset_Folder/Moving_noise_sources/F/1"
    )
    checkpoint_path = Path(
        "/home/jaliya/eeg_speech/Julian/RetM_Workspace/ReTM_Research_Project/"
        "best_checkpoint_A1_1_FUSENet_13_F.pth"
    )
    out_dir = Path("results_fusnet_retm_kalman_F_1")
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_target     = 16000
    qa_mics       = [1, 2, 3, 4, 5]
    qb_mics       = [6, 7, 8, 9, 10, 11, 12, 13]
    context       = 4096
    filter_length = 2 * context + 1

    window_size             = 16384
    stride                  = 8192
    batch_size_for_baseline = 8

    # ── Kalman hyperparameters ─────────────────────────────────────────
    #
    # TUNING GUIDE FOR MOVING SOURCE
    # ─────────────────────────────────────────────────────────────────
    # transition (G):
    #   How fast old estimates are forgotten.
    #   0.999  → slow tracking, stable (good for slow movement)
    #   0.990  → faster tracking, slightly noisier
    #   Start: 0.995 for a half-circle sweep.
    #
    # process_noise (Q):
    #   Variance injected into P each step — makes filter more willing
    #   to update. Raise from 1e-8 toward 1e-6 if tracking lags.
    #
    # observation_noise (Rv floor):
    #   Floor for adaptive Rv. Should approximate FuSNet residual
    #   variance on clean static signal (~1e-3 to 1e-2).
    #
    # adaptive_noise = True:
    #   Automatically raises Rv when FuSNet is confused (fast motion)
    #   and lowers it when error is small. Strongly recommended.
    #
    # innovation_momentum (beta):
    #   0.0  = standard Kalman
    #   0.3  = light momentum — helps on smooth arc trajectories
    #   0.5  = stronger smoothing
    #   Too high (>0.7) can cause lag.
    #
    # block_length:
    #   64 is a good balance of speed and memory.
    #   Larger → faster loop but more GPU RAM for covariance blocks.
    # ─────────────────────────────────────────────────────────────────

    block_length        = 64
    transition          = 0.995
    process_noise       = 1e-7
    observation_noise   = 1e-2
    initial_covariance  = 1e-3

    adaptive_noise        = True
    adaptive_alpha        = 0.999
    adaptive_noise_floor  = 1e-4
    adaptive_noise_ceil   = 1.0

    innovation_momentum   = 0.3

    update_stride         = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Banner ─────────────────────────────────────────────────────────
    print("=" * 80)
    print("FuSNet ReTM Kalman — moving-source adaptation")
    print("=" * 80)
    print(f"  Device         : {device}")
    print(f"  Filter length  : {filter_length}")
    print(f"  Block length   : {block_length}")
    print(f"  Transition G   : {transition}")
    print(f"  Process noise  : {process_noise}")
    print(f"  Obs noise Rv   : {observation_noise}  adaptive={adaptive_noise}")
    print(f"  Momentum beta  : {innovation_momentum}")
    print(f"  Update stride  : {update_stride}")
    print()

    # ── 1. Data ────────────────────────────────────────────────────────
    print("[1] Loading WAVs...")
    mB_cpu, fsB = load_mic_group(seq_dir, qb_mics, fs_target)
    mA_cpu, fsA = load_mic_group(seq_dir, qa_mics, fs_target)
    assert fsA == fsB
    fs = fsA

    T0 = min(mA_cpu.shape[1], mB_cpu.shape[1])
    mA_cpu, mB_cpu = mA_cpu[:, :T0], mB_cpu[:, :T0]
    mA_cpu, mB_cpu, scale = normalize_pair(mA_cpu, mB_cpu)
    print(f"  mA {tuple(mA_cpu.shape)}, mB {tuple(mB_cpu.shape)}, scale={scale:.6f}")

    # ── 2. Baseline FuSNet ─────────────────────────────────────────────
    print("\n[2] Loading model + running baseline FuSNet inference...")
    model, _ = load_fusnet13_model(
        checkpoint_path=checkpoint_path, context=context, device=str(device),
    )
    model.eval()

    mA_fusnet = predict_fusnet13_original_style(
        model=model, mB=mB_cpu.numpy(),
        context=context, window_size=window_size, stride=stride,
        batch_size=batch_size_for_baseline, device=device,
    )

    T = min(mA_cpu.shape[1], mB_cpu.shape[1], mA_fusnet.shape[1])
    mA_cpu    = mA_cpu[:, :T]
    mB_cpu    = mB_cpu[:, :T]
    mA_fusnet = mA_fusnet[:, :T]
    mA_np     = mA_cpu.numpy()

    print_metrics(mA_np, mA_fusnet, "Baseline FuSNet")

    # ── 3. Kalman ──────────────────────────────────────────────────────
    print("\n[3] Initialising partitioned Kalman from FuSNet weights...")
    kalman = PartitionedBlockReTMKalmanFromFuSNet(
        model=model,
        qa=5, qb=8,
        filter_length=filter_length,
        block_length=block_length,
        transition=transition,
        process_noise=process_noise,
        observation_noise=observation_noise,
        initial_covariance=initial_covariance,
        device=device,
        dtype=torch.float32,
        adaptive_noise=adaptive_noise,
        adaptive_alpha=adaptive_alpha,
        adaptive_noise_floor=adaptive_noise_floor,
        adaptive_noise_ceil=adaptive_noise_ceil,
        innovation_momentum=innovation_momentum,
        symmetrize_covariance=True,
    )

    print("\n[4] Running ReTM Kalman update...")
    mA_hat, err, error_trace = run_retm_kalman_samplewise(
        kalman=kalman, mB=mB_cpu, mA=mA_cpu,
        context=context, filter_length=filter_length,
        update_stride=update_stride,
    )

    # ── 4. Results ─────────────────────────────────────────────────────
    print_metrics(mA_np, mA_fusnet, "Baseline FuSNet")
    print_metrics(mA_np, mA_hat,    "FuSNet + Kalman (fixed + adaptive)")

    # ── 5. Save ────────────────────────────────────────────────────────
    print("\n[5] Saving outputs...")
    kalman.copy_retm_state_to_fusnet()
    R_adapted = kalman.get_retm_tensor().cpu().float().numpy()

    np.save(out_dir / "mA_target.npy",          mA_np)
    np.save(out_dir / "mB_input.npy",           mB_cpu[:, :T].numpy())
    np.save(out_dir / "mA_fusnet_baseline.npy", mA_fusnet)
    np.save(out_dir / "mA_kalman.npy",          mA_hat)
    np.save(out_dir / "error_kalman.npy",        err)
    np.save(out_dir / "error_trace_rms.npy",     error_trace)
    np.save(out_dir / "R_adapted.npy",           R_adapted)

    metrics = {
        "fs": fs, "num_samples": int(T),
        "context": context, "filter_length": filter_length,
        "block_length": block_length,
        "transition_G": transition,
        "process_noise": process_noise,
        "observation_noise": observation_noise,
        "adaptive_noise": adaptive_noise,
        "adaptive_alpha": adaptive_alpha,
        "innovation_momentum": innovation_momentum,
        "update_stride": update_stride,
        "baseline_sdr_avg":  float(np.mean(sdr_db(mA_np, mA_fusnet))),
        "kalman_sdr_avg":    float(np.mean(sdr_db(mA_np, mA_hat))),
        "baseline_mse_avg":  float(np.mean(mse_db(mA_np, mA_fusnet))),
        "kalman_mse_avg":    float(np.mean(mse_db(mA_np, mA_hat))),
        "baseline_sdr_per_ch": sdr_db(mA_np, mA_fusnet).tolist(),
        "kalman_sdr_per_ch":   sdr_db(mA_np, mA_hat).tolist(),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        {"model_state_dict": model.state_dict(), "kalman_settings": metrics},
        out_dir / "adapted_fusnet_kalman.pth",
    )

    for ch in range(5):
        for tag, arr in [
            ("target",   mA_np),
            ("baseline", mA_fusnet),
            ("kalman",   mA_hat),
        ]:
            torchaudio.save(
                str(out_dir / f"{tag}_mic_{ch+1}.wav"),
                torch.from_numpy(arr[ch:ch+1]).float(), fs,
            )

    print(f"\nDone.  Results → {out_dir}")


if __name__ == "__main__":
    main()