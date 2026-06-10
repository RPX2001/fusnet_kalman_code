from __future__ import annotations

import os
import random
import json
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np

np.random.seed(0)
random.seed(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

from retm_kalman.io_utils import read_mic_wavs, normalize_pair, write_mic_wavs
from retm_kalman.metrics import print_basic_metrics, sdr_db, mse_db
from retm_kalman.fusnet_filter_extraction import load_or_extract_fusnet_retm
from retm_kalman.kalman_weight_full import FullKalmanFusenetWeightReTM
from retm_kalman.kalman_weight_block import BlockKalmanFusenetWeightReTM


def load_config(path: str | Path = "config.json") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(config_path: str = "config.json"):
    cfg = load_config(config_path)

    system_cfg = cfg.get("system", {})
    data_cfg = cfg["data"]
    fus_cfg = cfg["fusnet"]
    kal_cfg = cfg["kalman"]
    out_cfg = cfg.get("outputs", {})

    kalman_dtype_str = str(kal_cfg.get("dtype", "float32")).lower()
    kalman_dtype_np = np.float32 if kalman_dtype_str == "float32" else np.float64

    mic_config = int(system_cfg.get("mic_config", 13))

    profile_defaults = {
        7: {
            "qa_mics": (1, 2, 3),
            "qb_mics": (4, 5, 6, 7),
        },
        12: {
            "qa_mics": (1, 2, 3, 4, 5),
            "qb_mics": (6, 7, 8, 9, 10, 11, 12),
        },
        13: {
            "qa_mics": (1, 2, 3, 4, 5),
            "qb_mics": (6, 7, 8, 9, 10, 11, 12, 13),
        },
    }

    if mic_config not in profile_defaults:
        raise ValueError("system.mic_config must be 7, 12, or 13")

    if mic_config == 13:
        from retm_kalman.fusnet_inference_13 import (
            load_fusnet13_model as load_fusnet_model,
            predict_fusnet13_original_style as predict_fusnet,
        )
    elif mic_config == 12:
        from retm_kalman.fusnet_inference_12 import (
            load_fusnet12_model as load_fusnet_model,
            predict_fusnet12_original_style as predict_fusnet,
        )
    else:
        from retm_kalman.fusnet_inference import (
            load_fusnet7_model as load_fusnet_model,
            predict_fusnet7_original_style as predict_fusnet,
        )

    seq_dir = Path(data_cfg["seq_dir"])
    out_dir = Path(data_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("FuSNet-Weight-Driven Kalman ReTM System")
    print("=" * 90)
    print("Model:")
    print("  R_ab^F      = equivalent FuSNet checkpoint filter")
    print("  R_ab^K(t)   = R_ab^F ⊙ R_t(t)")
    print("  mA_hat(t)   = R_ab^K(t) mB(t)")
    print("  e(t)        = mA(t) - mA_hat(t)")
    print("  Kalman updates R_t(t)")
    print("=" * 90)

    print(f"Mic config: {mic_config}")
    print(f"Sequence : {seq_dir}")
    print(f"Output   : {out_dir}")

    default_qa_mics = profile_defaults[mic_config]["qa_mics"]
    default_qb_mics = profile_defaults[mic_config]["qb_mics"]

    qa_mics = tuple(data_cfg.get("qa_mics", default_qa_mics))
    qb_mics = tuple(data_cfg.get("qb_mics", default_qb_mics))

    print("\n[1] Loading microphone WAV files...")

    mA, mB, fs = read_mic_wavs(
        seq_dir,
        fs_target=int(data_cfg.get("fs", 16000)),
        qa_mics=qa_mics,
        qb_mics=qb_mics,
    )

    print(f"mA shape: {mA.shape}, mB shape: {mB.shape}, fs={fs}")

    scale = 1.0
    if bool(data_cfg.get("normalize", True)):
        mA, mB, scale = normalize_pair(mA, mB)
        print(f"Normalized by global peak = {scale:.8f}")

    print("\n[2] Loading FuSNet checkpoint...")

    model, device = load_fusnet_model(
        checkpoint_path=fus_cfg["checkpoint"],
        context=int(fus_cfg.get("context", 4096)),
        device=fus_cfg.get("device", "cuda"),
    )

    print("\n[3] Running normal FuSNet inference for comparison...")

    mA_fusnet = predict_fusnet(
        model=model,
        mB=mB,
        context=int(fus_cfg.get("context", 4096)),
        window_size=int(fus_cfg.get("window_size", 16384)),
        stride=int(fus_cfg.get("stride", 8192)),
        batch_size=int(fus_cfg.get("batch_size", 8)),
        device=device,
    )

    T = min(mA.shape[1], mB.shape[1], mA_fusnet.shape[1])
    mA = mA[:, :T]
    mB = mB[:, :T]
    mA_fusnet = mA_fusnet[:, :T]

    print_basic_metrics(mA, mA_fusnet, name="FuSNet normal output")

    print("\n[4] Extracting/loading FuSNet checkpoint ReTM filter R_ab^F...")

    L = int(kal_cfg.get("L", 8192))

    filter_cfg = cfg.get("fusnet_filter", {})
    cache_path = Path(
        filter_cfg.get(
            "cache_path",
            out_dir / f"R_fusnet_base_QA{mA.shape[0]}_QB{mB.shape[0]}_L{L}.npy",
        )
    )

    R_fusnet = load_or_extract_fusnet_retm(
        cache_path=cache_path,
        model=model,
        predict_fn=predict_fusnet,
        qb=mB.shape[0],
        qa=mA.shape[0],
        L=L,
        context=int(fus_cfg.get("context", 4096)),
        window_size=int(fus_cfg.get("window_size", 16384)),
        stride=int(fus_cfg.get("stride", 8192)),
        batch_size=int(fus_cfg.get("batch_size", 8)),
        device=device,
        start_offset=int(filter_cfg.get("start_offset", 0)),
        force_extract=bool(filter_cfg.get("force_extract", False)),
        output_dtype=kalman_dtype_np,
    )

    print(f"R_fusnet shape: {R_fusnet.shape}")

    print("\n[5] Running Kalman update on R_t filter...")

    mode = kal_cfg.get("mode", "block").lower()

    common_kwargs = dict(
        R_fusnet=R_fusnet,
        transition=float(kal_cfg.get("transition", 0.999)),
        process_noise=float(kal_cfg.get("process_noise", 1e-8)),
        observation_noise=float(kal_cfg.get("observation_noise", 1e-2)),
        initial_covariance=float(kal_cfg.get("initial_covariance", 1e-3)),
        dtype=kalman_dtype_np,
        device=str(device),
    )

    if mode == "full":
        kf = FullKalmanFusenetWeightReTM(**common_kwargs)

    elif mode == "block":
        kf = BlockKalmanFusenetWeightReTM(
            **common_kwargs,
            block_size=int(kal_cfg.get("block_size", 256)),
        )

    else:
        raise ValueError("kalman.mode must be 'full' or 'block'")

    mA_hat, mA_base_filter, err, Rt_final = kf.process(mB=mB, mA=mA)

    print_basic_metrics(mA, mA_base_filter, name="Fixed extracted FuSNet ReTM filter")
    print_basic_metrics(mA, mA_hat, name="FuSNet-weight Kalman final estimate")

    error_trace = np.sqrt(np.mean(err.astype(np.float64) ** 2, axis=0))

    smooth_window = int(out_cfg.get("error_smooth_window", 1))
    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        error_trace_smooth = np.convolve(error_trace, kernel, mode="same")
    else:
        error_trace_smooth = error_trace

    sample_times_sec = np.arange(mA_hat.shape[1], dtype=np.float64) / float(fs)

    print("\n[6] Saving outputs...")

    if bool(out_cfg.get("save_npy", True)):
        np.save(out_dir / "mA_target.npy", mA)
        np.save(out_dir / "mB_input.npy", mB)
        np.save(out_dir / "mA_fusnet_normal_output.npy", mA_fusnet)
        np.save(out_dir / "R_fusnet_base.npy", R_fusnet.astype(np.float32))
        np.save(out_dir / "Rt_final.npy", Rt_final.astype(np.float32))
        np.save(out_dir / "mA_base_filter_output.npy", mA_base_filter)
        np.save(out_dir / "mA_final_weight_kalman.npy", mA_hat)
        np.save(out_dir / "error_final.npy", err)
        np.save(out_dir / "error_trace_rms.npy", error_trace)
        np.save(out_dir / "error_trace_rms_smooth.npy", error_trace_smooth)

    if bool(out_cfg.get("save_error_plot", True)):
        plt.figure(figsize=(10, 4))
        plt.plot(sample_times_sec, error_trace, linewidth=1.0, alpha=0.35, label="sample RMS error")

        if smooth_window > 1:
            plt.plot(
                sample_times_sec,
                error_trace_smooth,
                linewidth=2.0,
                label=f"smoothed RMS, window={smooth_window}",
            )
        else:
            plt.plot(sample_times_sec, error_trace_smooth, linewidth=2.0, label="sample RMS error")

        plt.xlabel("Time (s)")
        plt.ylabel("Estimation error RMS")
        plt.title("FuSNet-weight Kalman estimation error over time")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "kalman_error_trace.png", dpi=200)
        plt.close()

    if bool(out_cfg.get("save_wav", True)):
        write_mic_wavs(out_dir / "wav_target_mA", mA, fs, prefix="target_mic")
        write_mic_wavs(out_dir / "wav_fusnet_normal", mA_fusnet, fs, prefix="fusnet_mic")
        write_mic_wavs(out_dir / "wav_base_filter", mA_base_filter, fs, prefix="base_filter_mic")
        write_mic_wavs(out_dir / "wav_weight_kalman_final", mA_hat, fs, prefix="weight_kalman_mic")

    metrics = {
        "fs": fs,
        "num_samples": int(mA_hat.shape[1]),
        "mode": mode,
        "architecture": "R_abK(t) = R_abF checkpoint filter multiplied by adaptive Rt(t)",
        "L": L,
        "block_size": int(kal_cfg.get("block_size", 256)),
        "transition": float(kal_cfg.get("transition", 0.999)),
        "process_noise": float(kal_cfg.get("process_noise", 1e-8)),
        "observation_noise": float(kal_cfg.get("observation_noise", 1e-2)),
        "initial_covariance": float(kal_cfg.get("initial_covariance", 1e-3)),
        "filter_shape_R_fusnet": list(R_fusnet.shape),
        "filter_shape_Rt": list(Rt_final.shape),
        "fusnet_normal_sdr_per_channel": sdr_db(mA, mA_fusnet).tolist(),
        "fusnet_normal_sdr_avg": float(np.mean(sdr_db(mA, mA_fusnet))),
        "base_filter_sdr_per_channel": sdr_db(mA, mA_base_filter).tolist(),
        "base_filter_sdr_avg": float(np.mean(sdr_db(mA, mA_base_filter))),
        "weight_kalman_sdr_per_channel": sdr_db(mA, mA_hat).tolist(),
        "weight_kalman_sdr_avg": float(np.mean(sdr_db(mA, mA_hat))),
        "fusnet_normal_mse_db_per_channel": mse_db(mA, mA_fusnet).tolist(),
        "base_filter_mse_db_per_channel": mse_db(mA, mA_base_filter).tolist(),
        "weight_kalman_mse_db_per_channel": mse_db(mA, mA_hat).tolist(),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "used_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nDone. Results written to:\n{out_dir}")


if __name__ == "__main__":
    main("config.json")