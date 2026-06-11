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


@torch.no_grad()
def run_retm_kalman_samplewise(
    kalman: PartitionedBlockReTMKalmanFromFuSNet,
    mB: torch.Tensor,
    mA: torch.Tensor,
    context: int,
    filter_length: int,
    update_stride: int = 1,
):
    """
    Run no-backprop ReTM Kalman update.

    mB:
        [QB, T]

    mA:
        [QA, T]

    Original FuSNet inference alignment:
        output at original time t uses:
            mB_pad[:, context+t : context+t+L]

    Therefore, this function uses the same alignment.
    """

    device = kalman.device
    dtype = kalman.dtype

    mB = mB.to(device=device, dtype=dtype)
    mA = mA.to(device=device, dtype=dtype)

    T = min(mB.shape[1], mA.shape[1])
    mB = mB[:, :T]
    mA = mA[:, :T]

    qb = mB.shape[0]
    qa = mA.shape[0]
    L = int(filter_length)

    if qb != kalman.qb:
        raise ValueError(f"Expected QB={kalman.qb}, got {qb}")

    if qa != kalman.qa:
        raise ValueError(f"Expected QA={kalman.qa}, got {qa}")

    # Same right/left context padding style as original FuSNet inference.
    mB_pad = F.pad(mB, (context, context), mode="constant", value=0.0)

    y_out = torch.zeros(
        (qa, T),
        device=device,
        dtype=dtype,
    )

    err_out = torch.zeros(
        (qa, T),
        device=device,
        dtype=dtype,
    )

    error_rms = []

    for t in range(T):
        # Same regressor alignment as original FuSNet output crop:
        # y[t] corresponds to y_full[context+t].
        start = context + t
        x_full = mB_pad[:, start:start + L]

        if x_full.shape[1] < L:
            pad_right = L - x_full.shape[1]
            x_full = F.pad(x_full, (0, pad_right), mode="constant", value=0.0)

        d = mA[:, t]

        if update_stride <= 1 or (t % update_stride == 0):
            y_hat, e = kalman.update_one_sample(x_full=x_full, d=d)
        else:
            y_hat = kalman.predict_one_sample(x_full=x_full)
            e = d - y_hat

        y_out[:, t] = y_hat
        err_out[:, t] = e

        if t % 1000 == 0 or t == T - 1:
            rms = torch.sqrt(torch.mean(e ** 2)).detach().cpu().item()
            error_rms.append(rms)
            print(f"sample {t+1:07d}/{T}, error_rms={rms:.6e}")

    return (
        y_out.detach().cpu().float().numpy(),
        err_out.detach().cpu().float().numpy(),
        np.asarray(error_rms, dtype=np.float32),
    )


def main():
    # --------------------------------------------------------
    # User settings
    # --------------------------------------------------------

    seq_dir = Path(
        "/home/jaliya/eeg_speech/Julian/RetM_Workspace/Dataset/mic_moving"
    )

    checkpoint_path = Path(
        "/home/jaliya/eeg_speech/Julian/RetM_Workspace/ReTM_Research_Project/best_checkpoint_A1_1_FUSENet_13_rctd.pth"
    )

    out_dir = Path("results_fusenet_kalman")
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_target = 16000

    qa_mics = [1, 2, 3, 4, 5]
    qb_mics = [6, 7, 8, 9, 10, 11, 12, 13]

    context = 4096
    filter_length = 2 * context + 1

    window_size = 16384
    stride = 8192

    # Partition length along the FuSNet filter.
    #
    # Recommended:
    #   32  -> lower memory, slower
    #   64  -> good starting point
    #   128 -> faster but higher GPU memory
    #
    # For block_length=64:
    #   D = QB * block_length = 8 * 64 = 512
    #   each covariance block is 512 x 512 per output mic.
    block_length = 64

    transition = 0.995
    process_noise = 1e-8
    observation_noise = 1e-2
    initial_covariance = 1e-3

    # update_stride = 1 updates every sample.
    # For faster initial tests, use 8, 16, or 32.
    # But the document-style recursive method is update_stride=1.
    update_stride = 1

    batch_size_for_baseline = 8

    save_adapted_checkpoint = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("FuSNet ReTM Kalman Update Without Backpropagation")
    print("=" * 90)
    print(f"Device      : {device}")
    print(f"Sequence    : {seq_dir}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Output dir  : {out_dir}")
    print(f"Filter size : 5 x 8 x {filter_length}")
    print(f"Parameters  : {5 * 8 * filter_length}")
    print(f"Block length: {block_length}")
    print()
    print("Update equation:")
    print("  r(t+1) = G * ( r(t) + K(t)e(t) )")
    print("No backpropagation.")
    print("No transition center.")
    print("FuSNet conv weights are the ReTM state.")
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
    # Baseline FuSNet using original inference function
    # --------------------------------------------------------

    print("\n[2] Loading FuSNet model...")

    model, _ = load_fusnet13_model(
        checkpoint_path=checkpoint_path,
        context=context,
        device=str(device),
    )

    model.to(device)
    model.eval()

    print("\n[3] Running baseline FuSNet original-style inference...")

    mA_fusnet = predict_fusnet13_original_style(
        model=model,
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
    # Create no-backprop Kalman filter from FuSNet weights
    # --------------------------------------------------------

    print("\n[4] Initializing partitioned ReTM Kalman from FuSNet weights...")

    kalman = PartitionedBlockReTMKalmanFromFuSNet(
        model=model,
        qa=5,
        qb=8,
        filter_length=filter_length,
        block_length=block_length,
        transition=transition,
        process_noise=process_noise,
        observation_noise=observation_noise,
        initial_covariance=initial_covariance,
        device=device,
        dtype=torch.float32,
        symmetrize_covariance=True,
    )

    # --------------------------------------------------------
    # Run recursive ReTM Kalman update
    # --------------------------------------------------------

    print("\n[5] Running ReTM Kalman update...")

    mA_hat, err, error_trace = run_retm_kalman_samplewise(
        kalman=kalman,
        mB=mB_cpu,
        mA=mA_cpu,
        context=context,
        filter_length=filter_length,
        update_stride=update_stride,
    )

    print_metrics(mA_np, mA_fusnet, "Baseline FuSNet original-style inference")
    print_metrics(mA_np, mA_hat, "ReTM Kalman output")

    # --------------------------------------------------------
    # Copy adapted state back to FuSNet and save
    # --------------------------------------------------------

    print("\n[6] Saving outputs...")

    kalman.copy_retm_state_to_fusnet()

    R_adapted = kalman.get_retm_tensor().detach().cpu().float().numpy()

    np.save(out_dir / "mA_target.npy", mA_np)
    np.save(out_dir / "mB_input.npy", mB_cpu[:, :T].numpy())
    np.save(out_dir / "mA_fusnet_baseline_original_style.npy", mA_fusnet)
    np.save(out_dir / "mA_no_backprop_retm_kalman.npy", mA_hat)
    np.save(out_dir / "error_no_backprop_retm_kalman.npy", err)
    np.save(out_dir / "error_trace_rms.npy", error_trace)
    np.save(out_dir / "R_adapted_retm_from_fusnet.npy", R_adapted)

    metrics = {
        "method": "no_backprop_partitioned_retm_kalman_from_fusnet_weights",
        "fs": fs,
        "num_samples": int(T),
        "context": context,
        "filter_length": filter_length,
        "window_size": window_size,
        "stride": stride,
        "block_length": block_length,
        "transition_G": transition,
        "process_noise": process_noise,
        "observation_noise": observation_noise,
        "initial_covariance": initial_covariance,
        "update_stride": update_stride,
        "total_retm_parameters": int(5 * 8 * filter_length),
        "baseline_sdr_per_channel": sdr_db(mA_np, mA_fusnet).tolist(),
        "baseline_sdr_avg": float(np.mean(sdr_db(mA_np, mA_fusnet))),
        "kalman_sdr_per_channel": sdr_db(mA_np, mA_hat).tolist(),
        "kalman_sdr_avg": float(np.mean(sdr_db(mA_np, mA_hat))),
        "baseline_mse_db_per_channel": mse_db(mA_np, mA_fusnet).tolist(),
        "kalman_mse_db_per_channel": mse_db(mA_np, mA_hat).tolist(),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if save_adapted_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "architecture": "FuSNet13 adapted using no-backprop ReTM Kalman",
                "kalman_settings": {
                    "state_model": "r(t+1)=G*(r(t)+K(t)e(t))",
                    "no_backprop": True,
                    "partitioned_block_retm": True,
                    "block_length": block_length,
                    "transition_G": transition,
                    "process_noise": process_noise,
                    "observation_noise": observation_noise,
                    "initial_covariance": initial_covariance,
                    "update_stride": update_stride,
                },
            },
            out_dir / "adapted_fusnet_no_backprop_retm_kalman.pth",
        )

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
            str(out_dir / f"no_backprop_retm_kalman_mic_{ch+1}.wav"),
            torch.from_numpy(mA_hat[ch:ch + 1]).float(),
            fs,
        )

    print(f"\nDone. Results saved to:\n{out_dir}")


if __name__ == "__main__":
    main()