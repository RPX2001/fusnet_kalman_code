from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from retm_kalman.io_utils import read_mic_wavs, normalize_pair, write_mic_wavs
from retm_kalman.metrics import print_basic_metrics, sdr_db, mse_db

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

    mic_config = int(system_cfg.get("mic_config", 7))
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
        from retm_kalman.kalman_full_13 import FullKalmanCorrectionReTM
        from retm_kalman.kalman_block_13 import BlockKalmanCorrectionReTM
    elif mic_config == 12:
        from retm_kalman.fusnet_inference_12 import (
            load_fusnet12_model as load_fusnet_model,
            predict_fusnet12_original_style as predict_fusnet,
        )
        from retm_kalman.kalman_full_12 import FullKalmanCorrectionReTM
        from retm_kalman.kalman_block_12 import BlockKalmanCorrectionReTM
    else:
        from retm_kalman.fusnet_inference import (
            load_fusnet7_model as load_fusnet_model,
            predict_fusnet7_original_style as predict_fusnet,
        )
        from retm_kalman.kalman_full import FullKalmanCorrectionReTM
        from retm_kalman.kalman_block import BlockKalmanCorrectionReTM

    seq_dir = Path(data_cfg["seq_dir"])
    out_dir = Path(data_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FuSNet + Kalman Dynamic ReTM Correction System")
    print("=" * 80)
    print(f"Mic config: {mic_config}-mic")
    print(f"Sequence : {seq_dir}")
    print(f"Output   : {out_dir}")
    print(f"Mode     : {kal_cfg['mode']}")

    print("\n[1] Loading microphone WAV files...")
    default_qa_mics = profile_defaults[mic_config]["qa_mics"]
    default_qb_mics = profile_defaults[mic_config]["qb_mics"]
    qa_mics = tuple(data_cfg.get("qa_mics", default_qa_mics))
    qb_mics = tuple(data_cfg.get("qb_mics", default_qb_mics))
    if len(qa_mics) != len(default_qa_mics) or len(qb_mics) != len(default_qb_mics):
        raise ValueError(
            f"For mic_config={mic_config}, expected {len(default_qa_mics)} QA mics and {len(default_qb_mics)} QB mics, "
            f"got QA={len(qa_mics)} QB={len(qb_mics)}"
        )
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

    print("\n[2] Loading integrated FuSNet model...")
    model, device = load_fusnet_model(
        checkpoint_path=fus_cfg["checkpoint"],
        context=int(fus_cfg.get("context", 4096)),
        device=fus_cfg.get("device", "cuda"),
    )

    print("\n[3] Running FuSNet initial Group-A estimation...")
    mA_f = predict_fusnet(
        model=model,
        mB=mB,
        context=int(fus_cfg.get("context", 4096)),
        window_size=int(fus_cfg.get("window_size", 16384)),
        stride=int(fus_cfg.get("stride", 8192)),
        batch_size=int(fus_cfg.get("batch_size", 8)),
        device=device,
    )
    T = min(mA.shape[1], mB.shape[1], mA_f.shape[1])
    mA, mB, mA_f = mA[:, :T], mB[:, :T], mA_f[:, :T]
    print_basic_metrics(mA, mA_f, name="FuSNet initial estimate")

    print("\n[4] Running Kalman correction ReTM estimator...")
    mode = kal_cfg.get("mode", "block").lower()
    common_kwargs = dict(
        qb=mB.shape[0],
        qa=mA.shape[0],
        L=int(kal_cfg.get("L", 1024)),
        transition=float(kal_cfg.get("transition", 0.995)),
        process_noise=float(kal_cfg.get("process_noise", 1e-7)),
        observation_noise=float(kal_cfg.get("observation_noise", 1e-3)),
        initial_covariance=float(kal_cfg.get("initial_covariance", 1e-2)),
    )
    if mode == "full":
        kf = FullKalmanCorrectionReTM(**common_kwargs, device=device)
    elif mode == "block":
        kf = BlockKalmanCorrectionReTM(
            **common_kwargs,
            block_size=int(kal_cfg.get("block_size", 128)),
            device=device,
        )
    else:
        raise ValueError("kalman.mode must be 'full' or 'block'")

    mA_hat, delta_hat, err = kf.process(mB=mB, mA=mA, mA_fusnet=mA_f)
    print_basic_metrics(mA, mA_hat, name="FuSNet + Kalman final estimate")

    # Frame-wise error trace for checking convergence toward zero.
    # We use per-sample RMSE across channels, then smooth it a little so the
    # trend over time is easier to inspect.
    error_trace = np.sqrt(np.mean(err.astype(np.float64) ** 2, axis=0))
    smooth_window = int(out_cfg.get("error_smooth_window", 1))
    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        error_trace_smooth = np.convolve(error_trace, kernel, mode="same")
    else:
        error_trace_smooth = error_trace

    sample_times_sec = np.arange(T, dtype=np.float64) / float(fs)

    print("\n[5] Saving outputs...")
    if bool(out_cfg.get("save_npy", True)):
        np.save(out_dir / "mA_target.npy", mA)
        np.save(out_dir / "mB_input.npy", mB)
        np.save(out_dir / "mA_fusnet_initial.npy", mA_f)
        np.save(out_dir / "delta_kalman.npy", delta_hat)
        np.save(out_dir / "mA_final_kalman.npy", mA_hat)
        np.save(out_dir / "error_final.npy", err)
        np.save(out_dir / "error_trace_rms.npy", error_trace)
        np.save(out_dir / "error_trace_rms_smooth.npy", error_trace_smooth)

    if bool(out_cfg.get("save_error_plot", True)):
        plt.figure(figsize=(10, 4))
        plt.plot(sample_times_sec, error_trace, linewidth=1.0, alpha=0.35, label="sample RMS error")
        if smooth_window > 1:
            plt.plot(sample_times_sec, error_trace_smooth, linewidth=2.0, label=f"smoothed RMS (window={smooth_window})")
        else:
            plt.plot(sample_times_sec, error_trace_smooth, linewidth=2.0, label="sample RMS error")
        plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Estimation error RMS")
        plt.title("Kalman estimation error over time")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "kalman_error_trace.png", dpi=200)
        plt.close()

    if bool(out_cfg.get("save_wav", True)):
        write_mic_wavs(out_dir / "wav_target_mA", mA, fs, prefix="target_mic")
        write_mic_wavs(out_dir / "wav_fusnet_initial", mA_f, fs, prefix="fusnet_mic")
        write_mic_wavs(out_dir / "wav_kalman_final", mA_hat, fs, prefix="kalman_mic")
        write_mic_wavs(out_dir / "wav_kalman_delta", delta_hat, fs, prefix="delta_mic")

    metrics = {
        "fs": fs,
        "num_samples": int(T),
        "mode": mode,
        "L": int(kal_cfg.get("L", 1024)),
        "block_size": int(kal_cfg.get("block_size", 128)),
        "fusnet_sdr_per_channel": sdr_db(mA, mA_f).tolist(),
        "fusnet_sdr_avg": float(np.mean(sdr_db(mA, mA_f))),
        "kalman_sdr_per_channel": sdr_db(mA, mA_hat).tolist(),
        "kalman_sdr_avg": float(np.mean(sdr_db(mA, mA_hat))),
        "fusnet_mse_db_per_channel": mse_db(mA, mA_f).tolist(),
        "kalman_mse_db_per_channel": mse_db(mA, mA_hat).tolist(),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "used_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Done. Results written to: {out_dir}")


if __name__ == "__main__":
    main("config.json")
