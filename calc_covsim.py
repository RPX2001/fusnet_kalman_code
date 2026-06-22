from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


# ============================================================
# USER SETTINGS
# ============================================================

RESULT_DIR = Path("results_fusnet_retm_kalman_P_0.5_16")

TARGET_FILE = RESULT_DIR / "mA_target.npy"
BASELINE_FILE = RESULT_DIR / "mA_fusnet_baseline.npy"
KALMAN_FILE = RESULT_DIR / "mA_kalman.npy"

OUT_JSON = RESULT_DIR / "covsim_results.json"

FS = 16000

# STFT settings
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

# Ignore very silent time-frequency bins
USE_ENERGY_THRESHOLD = True
ENERGY_THRESHOLD_DB = -80.0

EPS = 1e-12


# ============================================================
# STFT
# ============================================================

def stft_multichannel(x: np.ndarray) -> torch.Tensor:
    """
    Convert multichannel time-domain signal to STFT.

    Parameters
    ----------
    x : np.ndarray
        Shape [M, T], where M is number of microphones.

    Returns
    -------
    X : torch.Tensor
        Shape [M, F, TT], complex STFT.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected input shape [M, T], got {x.shape}")

    x_torch = torch.from_numpy(x).float()
    window = torch.hann_window(WIN_LENGTH)

    X_list = []

    for ch in range(x_torch.shape[0]):
        X_ch = torch.stft(
            x_torch[ch],
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            center=True,
            return_complex=True,
        )
        X_list.append(X_ch)

    X = torch.stack(X_list, dim=0)

    return X


# ============================================================
# Covariance and CovSim
# ============================================================

def covariance_from_vector(x_ft: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance matrix at one time-frequency bin.

    x_ft shape:
        [M]

    R = x x^H

    Returns:
        [M, M]
    """

    return x_ft[:, None] @ x_ft.conj()[None, :]


def covsim_matrix(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Calculate covariance similarity:

        sigma(R1, R2)
        =
        Re{tr(R1^H R2)} / (||R1||_F ||R2||_F)

    R1, R2 shape:
        [M, M]
    """

    numerator = torch.real(torch.trace(R1.conj().T @ R2))

    denominator = (
        torch.linalg.norm(R1, ord="fro")
        *
        torch.linalg.norm(R2, ord="fro")
    )

    return numerator / (denominator + EPS)


def compute_covsim(
    target: np.ndarray,
    estimate: np.ndarray,
    label: str = "estimate",
) -> dict:
    """
    Compute CovSim between target and estimated multichannel signals.

    target shape:
        [M, T]

    estimate shape:
        [M, T]
    """

    if target.ndim != 2:
        raise ValueError(f"target must have shape [M, T], got {target.shape}")

    if estimate.ndim != 2:
        raise ValueError(f"estimate must have shape [M, T], got {estimate.shape}")

    M = min(target.shape[0], estimate.shape[0])
    T = min(target.shape[1], estimate.shape[1])

    target = target[:M, :T].astype(np.float32)
    estimate = estimate[:M, :T].astype(np.float32)

    X_target = stft_multichannel(target)
    X_est = stft_multichannel(estimate)

    _, F, TT = X_target.shape

    # Energy threshold from target signal
    target_mag2 = torch.sum(torch.abs(X_target) ** 2, dim=0)  # [F, TT]
    max_energy = torch.max(target_mag2)
    threshold = max_energy * (10.0 ** (ENERGY_THRESHOLD_DB / 10.0))

    sims = []

    for f in range(F):
        for tt in range(TT):

            if USE_ENERGY_THRESHOLD:
                if target_mag2[f, tt] < threshold:
                    continue

            x_target = X_target[:, f, tt]
            x_est = X_est[:, f, tt]

            R_target = covariance_from_vector(x_target)
            R_est = covariance_from_vector(x_est)

            sim = covsim_matrix(R_est, R_target)

            if torch.isfinite(sim):
                sims.append(float(sim.item()))

    if len(sims) == 0:
        raise RuntimeError("No valid time-frequency bins were found for CovSim.")

    sims = np.array(sims, dtype=np.float64)

    result = {
        "label": label,
        "num_mics": int(M),
        "num_samples": int(T),
        "num_frequency_bins": int(F),
        "num_time_frames": int(TT),
        "num_evaluated_tf_bins": int(len(sims)),
        "covsim_mean": float(np.mean(sims)),
        "covsim_median": float(np.median(sims)),
        "covsim_std": float(np.std(sims)),
        "covsim_min": float(np.min(sims)),
        "covsim_max": float(np.max(sims)),
    }

    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("CovSim calculation")
    print("=" * 80)

    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Missing target file: {TARGET_FILE}")

    target = np.load(TARGET_FILE).astype(np.float32)

    print(f"Target file : {TARGET_FILE}")
    print(f"Target shape: {target.shape}")

    results = {}

    # --------------------------------------------------------
    # FuSNet baseline CovSim
    # --------------------------------------------------------

    if BASELINE_FILE.exists():
        baseline = np.load(BASELINE_FILE).astype(np.float32)

        print("\nCalculating CovSim for FuSNet baseline...")
        baseline_result = compute_covsim(
            target=target,
            estimate=baseline,
            label="FuSNet baseline",
        )

        results["fusnet_baseline"] = baseline_result

        print(f"CovSim mean   : {baseline_result['covsim_mean']:.6f}")
        print(f"CovSim median : {baseline_result['covsim_median']:.6f}")
    else:
        print(f"\nBaseline file not found, skipping: {BASELINE_FILE}")

    # --------------------------------------------------------
    # FuSNet + Kalman CovSim
    # --------------------------------------------------------

    if KALMAN_FILE.exists():
        kalman = np.load(KALMAN_FILE).astype(np.float32)

        print("\nCalculating CovSim for FuSNet + Kalman...")
        kalman_result = compute_covsim(
            target=target,
            estimate=kalman,
            label="FuSNet + Kalman",
        )

        results["fusnet_kalman"] = kalman_result

        print(f"CovSim mean   : {kalman_result['covsim_mean']:.6f}")
        print(f"CovSim median : {kalman_result['covsim_median']:.6f}")
    else:
        print(f"\nKalman file not found, skipping: {KALMAN_FILE}")

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    if "fusnet_baseline" in results and "fusnet_kalman" in results:
        delta_mean = (
            results["fusnet_kalman"]["covsim_mean"]
            -
            results["fusnet_baseline"]["covsim_mean"]
        )

        delta_median = (
            results["fusnet_kalman"]["covsim_median"]
            -
            results["fusnet_baseline"]["covsim_median"]
        )

        results["improvement"] = {
            "delta_covsim_mean": float(delta_mean),
            "delta_covsim_median": float(delta_median),
        }

        print("\nCovSim improvement")
        print("-" * 60)
        print(f"Delta mean   : {delta_mean:.6f}")
        print(f"Delta median : {delta_median:.6f}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results:")
    print(OUT_JSON)
    print("\nDone.")


if __name__ == "__main__":
    main()