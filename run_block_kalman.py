"""
run_kalman_v2.py
================
Runner for the improved v2 Kalman — includes per-channel diagnostics
and a tuning guide tailored to your half-circle moving-source scenario.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from retm_kalman.fusnet_inference_13 import (
    load_fusnet13_model,
    predict_fusnet13_original_style,
)
from retm_kalman.kalman_block_retm import PartitionedKalmanReTMv2


SEED = 0
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_mic_group(seq_dir, mic_indices, fs_target):
    seq_dir = Path(seq_dir)
    signals = []
    for mic_id in mic_indices:
        wav, fs = torchaudio.load(str(seq_dir / f"mic_{mic_id}.wav"))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if fs != fs_target:
            wav = torchaudio.functional.resample(wav, fs, fs_target)
        signals.append(wav.squeeze(0))
    L = min(x.numel() for x in signals)
    return torch.stack([x[:L] for x in signals], dim=0), fs_target


def normalize_pair(mA, mB):
    peak = torch.max(torch.abs(torch.cat([mA, mB], dim=0)))
    return (mA/peak, mB/peak, float(peak)) if peak > 0 else (mA, mB, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def sdr_db(ref, est, eps=1e-12):
    T = min(ref.shape[1], est.shape[1])
    err = ref[:,:T] - est[:,:T]
    return 10*np.log10(
        (np.sum(ref[:,:T]**2, 1)+eps) / (np.sum(err**2, 1)+eps)
    )

def mse_db(ref, est, eps=1e-12):
    T = min(ref.shape[1], est.shape[1])
    return 10*np.log10(np.mean((ref[:,:T]-est[:,:T])**2, 1)+eps)

def print_metrics(ref, est, label):
    sdr = sdr_db(ref, est); mse = mse_db(ref, est)
    print(f"\n{'─'*72}\n  {label}\n{'─'*72}")
    print(f"  SDR per ch : {np.round(sdr,3).tolist()}")
    print(f"  SDR avg    : {np.mean(sdr):.4f} dB")
    print(f"  MSE per ch : {np.round(mse,3).tolist()}")
    print(f"  MSE avg    : {np.mean(mse):.4f} dB")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_kalman(
    kalman: PartitionedKalmanReTMv2,
    mB: torch.Tensor,
    mA: torch.Tensor,
    context: int,
    filter_length: int,
    update_stride: int = 1,
    log_every: int = 8000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    dev, dtype = kalman.device, kalman.dtype
    mB = mB.to(dev, dtype); mA = mA.to(dev, dtype)
    T  = min(mB.shape[1], mA.shape[1])
    mB, mA = mB[:,:T], mA[:,:T]

    mB_pad = F.pad(mB, (context, context))

    y_out   = torch.zeros((kalman.qa, T), device=dev, dtype=dtype)
    err_out = torch.zeros((kalman.qa, T), device=dev, dtype=dtype)
    rms_log = []

    t0 = time.perf_counter()
    for t in range(T):
        s      = context + t
        x_full = mB_pad[:, s : s + filter_length]
        if x_full.shape[1] < filter_length:
            x_full = F.pad(x_full, (0, filter_length - x_full.shape[1]))

        d = mA[:, t]

        if update_stride <= 1 or t % update_stride == 0:
            y_hat, e = kalman.update_one(x_full, d)
        else:
            y_hat = kalman.predict_one(x_full)
            e     = d - y_hat

        y_out[:,t]   = y_hat
        err_out[:,t] = e

        if t % log_every == 0 or t == T-1:
            rms = float(torch.sqrt(torch.mean(e**2)).item())
            rms_log.append(rms)
            ch_sdr = [
                f"{10*np.log10((float((mA[c,:t+1]**2).mean())+1e-12)/(float((err_out[c,:t+1]**2).mean())+1e-12)):.1f}"
                for c in range(kalman.qa)
            ]
            rv_str = ""
            if kalman.adaptive:
                rv = kalman._err_ema.cpu().tolist()
                rv_str = f"  Rv=[{', '.join(f'{v:.1e}' for v in rv)}]"
            print(
                f"  t={t+1:07d}/{T}"
                f"  rms={rms:.3e}"
                f"  ch_SDR(dB)={ch_sdr}"
                f"{rv_str}"
                f"  {time.perf_counter()-t0:.0f}s"
            )

    return (
        y_out.cpu().float().numpy(),
        err_out.cpu().float().numpy(),
        np.array(rms_log, np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():

    seq_dir = Path(
        "/home/jaliya/eeg_speech/Julian/RetM_Workspace/Dataset/A2_rctd"
    )
    checkpoint_path = Path(
        "/home/jaliya/eeg_speech/Julian/RetM_Workspace/ReTM_Research_Project/"
        "best_checkpoint_A1_1_FUSENet_13_rctd.pth"
    )
    out_dir = Path("results_kalman_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_target     = 16000
    qa_mics       = [1, 2, 3, 4, 5]
    qb_mics       = [6, 7, 8, 9, 10, 11, 12, 13]
    context       = 4096
    filter_length = 2 * context + 1
    window_size   = 16384
    stride        = 8192

    # ─────────────────────────────────────────────────────────────────
    # TUNING GUIDE (half-circle moving source, 1 music + 1 moving speech)
    # ─────────────────────────────────────────────────────────────────
    #
    # transition G
    #   Controls how fast old ReTM state is forgotten.
    #   0.9990 → slow decay, less noise, may lag the arc
    #   0.9950 → (default) good balance for moderate sweep speed
    #   0.9900 → fast adaptation, tracks aggressive motion, noisier
    #   Try: start at 0.995, raise if SDR plateaus early in the arc.
    #
    # process_noise Q
    #   Variance injected into P each step (predict step).
    #   Keeps the Kalman gain from collapsing to zero over time.
    #   1e-7 → recommended starting point
    #   1e-6 → more responsive, more noise
    #   Try: raise Q if channels stall after early improvement.
    #
    # observation_noise (Rv floor)
    #   Per-channel floor for the adaptive Rv.
    #   Should reflect FuSNet's residual error on a clean static signal.
    #   1e-3 → tight floor, trusts FuSNet more
    #   1e-2 → conservative (your current setting, works well)
    #
    # adaptive_noise = True
    #   Per-channel Rv tracks each mic's squared error separately.
    #   Channel 5 (weak in your results) will automatically get higher Rv,
    #   reducing how much it pulls the filter.
    #
    # innovation_momentum beta
    #   0.0 → pure Kalman
    #   0.3 → light smoothing along the arc (default, works well)
    #   0.5 → stronger smoothing; try if channel 5 is still noisy
    #
    # channel_gain_alpha
    #   Per-channel gain scaling based on running error magnitude.
    #   Channels with persistent high error get gain scaled to ~0.5×.
    #   0.999 → slow reliability tracking (recommended)
    #   0.0   → disable (all channels get equal gain)
    #
    # p_floor
    #   Minimum diagonal value of P, enforced before the update.
    #   Prevents P from collapsing and freezing the Kalman gain.
    #   1e-10 → recommended default
    #
    # block_length
    #   Tap partition size.  D = qb * block_length covariance dimension.
    #   64 → D=512, good balance of speed and tracking resolution.
    #   32 → finer, slightly slower, marginally more accurate.
    # ─────────────────────────────────────────────────────────────────

    block_length        = 64
    transition          = 0.995
    process_noise       = 1e-7
    observation_noise   = 1e-2
    initial_covariance  = 1e-3
    adaptive_noise      = True
    adaptive_alpha      = 0.999
    adaptive_noise_floor = 1e-3      # tighter than v1 (was 1e-4)
    adaptive_noise_ceil  = 1.0
    innovation_momentum  = 0.3
    channel_gain_alpha   = 0.999     # NEW — per-channel reliability scaling
    p_floor              = 1e-10     # NEW — covariance lower bound
    update_stride        = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Kalman ReTM v2 — Joseph form + per-channel reliability")
    print("=" * 80)
    print(f"  Transition G         : {transition}")
    print(f"  Process noise Q      : {process_noise}")
    print(f"  Obs noise (Rv floor) : {observation_noise}  adaptive={adaptive_noise}")
    print(f"  Innovation momentum  : {innovation_momentum}")
    print(f"  Channel gain alpha   : {channel_gain_alpha}")
    print(f"  P floor              : {p_floor}")
    print(f"  Block length         : {block_length}")
    print()

    # ── Data ─────────────────────────────────────────────────────────
    print("[1] Loading WAVs...")
    mB_cpu, _ = load_mic_group(seq_dir, qb_mics, fs_target)
    mA_cpu, _ = load_mic_group(seq_dir, qa_mics, fs_target)
    T0 = min(mA_cpu.shape[1], mB_cpu.shape[1])
    mA_cpu, mB_cpu = mA_cpu[:,:T0], mB_cpu[:,:T0]
    mA_cpu, mB_cpu, scale = normalize_pair(mA_cpu, mB_cpu)
    print(f"  mA {tuple(mA_cpu.shape)}, mB {tuple(mB_cpu.shape)}")

    # ── Baseline FuSNet ───────────────────────────────────────────────
    print("\n[2] FuSNet baseline inference...")
    model, _ = load_fusnet13_model(
        checkpoint_path=checkpoint_path, context=context, device=str(device)
    )
    model.eval()
    mA_fusnet = predict_fusnet13_original_style(
        model=model, mB=mB_cpu.numpy(),
        context=context, window_size=window_size, stride=stride,
        batch_size=8, device=device,
    )
    T = min(mA_cpu.shape[1], mB_cpu.shape[1], mA_fusnet.shape[1])
    mA_cpu    = mA_cpu[:, :T]
    mB_cpu    = mB_cpu[:, :T]
    mA_fusnet = mA_fusnet[:, :T]
    mA_np     = mA_cpu.numpy()
    print_metrics(mA_np, mA_fusnet, "Baseline FuSNet")

    # ── Kalman v2 ─────────────────────────────────────────────────────
    print("\n[3] Initialising Kalman v2...")
    kalman = PartitionedKalmanReTMv2(
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
        channel_gain_alpha=channel_gain_alpha,
        p_floor=p_floor,
        symmetrize=True,
    )

    print("\n[4] Running Kalman v2...")
    t0 = time.perf_counter()
    mA_hat, err, rms_log = run_kalman(
        kalman=kalman, mB=mB_cpu, mA=mA_cpu,
        context=context, filter_length=filter_length,
        update_stride=update_stride,
    )
    print(f"\n  Done in {time.perf_counter()-t0:.1f}s")

    # ── Results ───────────────────────────────────────────────────────
    print_metrics(mA_np, mA_fusnet, "Baseline FuSNet")
    print_metrics(mA_np, mA_hat,    "Kalman v2 (Joseph + per-channel)")

    # Per-channel SDR table
    sdr_base = sdr_db(mA_np, mA_fusnet)
    sdr_v2   = sdr_db(mA_np, mA_hat)
    print("\n  Per-channel SDR improvement:")
    print(f"  {'Ch':>4} {'Baseline':>10} {'Kalman v2':>10} {'Gain':>8}")
    for ch in range(5):
        print(
            f"  {ch+1:>4}"
            f"  {sdr_base[ch]:>9.2f}"
            f"  {sdr_v2[ch]:>9.2f}"
            f"  {sdr_v2[ch]-sdr_base[ch]:>+7.2f}"
        )

    # ── Save ──────────────────────────────────────────────────────────
    print("\n[5] Saving...")
    kalman.copy_to_fusnet()
    R_adapted = kalman.get_retm_tensor().cpu().float().numpy()

    np.save(out_dir / "mA_target.npy",      mA_np)
    np.save(out_dir / "mA_baseline.npy",    mA_fusnet)
    np.save(out_dir / "mA_kalman_v2.npy",   mA_hat)
    np.save(out_dir / "error_v2.npy",       err)
    np.save(out_dir / "rms_log.npy",        rms_log)
    np.save(out_dir / "R_adapted.npy",      R_adapted)

    metrics = {
        "baseline_sdr_avg":  float(np.mean(sdr_base)),
        "kalman_v2_sdr_avg": float(np.mean(sdr_v2)),
        "baseline_mse_avg":  float(np.mean(mse_db(mA_np, mA_fusnet))),
        "kalman_v2_mse_avg": float(np.mean(mse_db(mA_np, mA_hat))),
        "baseline_sdr_per_ch":  sdr_base.tolist(),
        "kalman_v2_sdr_per_ch": sdr_v2.tolist(),
        "settings": dict(
            block_length=block_length, transition=transition,
            process_noise=process_noise, observation_noise=observation_noise,
            adaptive_noise=adaptive_noise, adaptive_alpha=adaptive_alpha,
            adaptive_noise_floor=adaptive_noise_floor,
            innovation_momentum=innovation_momentum,
            channel_gain_alpha=channel_gain_alpha, p_floor=p_floor,
        )
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        {"model_state_dict": model.state_dict(), "metrics": metrics},
        out_dir / "adapted_fusnet_kalman_v2.pth",
    )
    for ch in range(5):
        for tag, arr in [
            ("target", mA_np), ("baseline", mA_fusnet), ("kalman_v2", mA_hat)
        ]:
            torchaudio.save(
                str(out_dir / f"{tag}_ch{ch+1}.wav"),
                torch.from_numpy(arr[ch:ch+1]).float(), fs_target,
            )

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()