from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

# This should match the output folder in your Kalman run code
RESULT_DIR = Path("results_fusnet_retm_kalman_B_1")

# Figure output folder
FIG_DIR = RESULT_DIR / "plots_target_kalman"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Sampling rate
FS = 16000

# STFT settings
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

# Spectrogram dynamic range
TOP_DB = 80.0

# Number of Group-A target microphones
NUM_MICS = 5

# Plot all mics or only one selected mic
PLOT_ALL_MICS = True
SELECTED_MIC = 1   # used only if PLOT_ALL_MICS = False


# ============================================================
# Loading functions
# ============================================================

def load_npy_outputs(result_dir: Path):
    """
    Load target and FuSNet+Kalman outputs.

    Expected files:
        mA_target.npy
        mA_kalman.npy

    Alternative Kalman filenames are also supported.
    """

    target_path = result_dir / "mA_target.npy"

    kalman_candidates = [
        result_dir / "mA_kalman.npy",
        result_dir / "mA_no_backprop_retm_kalman.npy",
        result_dir / "mA_after_G_update_parameter_kalman.npy",
        result_dir / "mA_after_gpu_parameter_kalman.npy",
        result_dir / "mA_after_randomwalk_parameter_kalman.npy",
    ]

    if not target_path.exists():
        raise FileNotFoundError(f"Missing target file: {target_path}")

    kalman_path = None
    for p in kalman_candidates:
        if p.exists():
            kalman_path = p
            break

    if kalman_path is None:
        raise FileNotFoundError(
            "Could not find FuSNet+Kalman output. Expected one of:\n"
            + "\n".join(str(p) for p in kalman_candidates)
        )

    print(f"Loading target : {target_path}")
    print(f"Loading result : {kalman_path}")

    target = np.load(target_path).astype(np.float32)
    kalman = np.load(kalman_path).astype(np.float32)

    T = min(target.shape[1], kalman.shape[1])

    target = target[:, :T]
    kalman = kalman[:, :T]

    return target, kalman, kalman_path.name


def load_wav_outputs(result_dir: Path, mic_idx: int):
    """
    Load target and FuSNet+Kalman outputs from WAV files.

    Expected:
        target_mic_1.wav
        kalman_mic_1.wav

    Alternative Kalman WAV names are also supported.
    """

    target_path = result_dir / f"target_mic_{mic_idx}.wav"

    kalman_candidates = [
        result_dir / f"kalman_mic_{mic_idx}.wav",
        result_dir / f"no_backprop_retm_kalman_mic_{mic_idx}.wav",
        result_dir / f"after_G_update_param_kalman_mic_{mic_idx}.wav",
        result_dir / f"after_gpu_param_kalman_mic_{mic_idx}.wav",
        result_dir / f"after_randomwalk_param_kalman_mic_{mic_idx}.wav",
    ]

    if not target_path.exists():
        raise FileNotFoundError(f"Missing target WAV: {target_path}")

    kalman_path = None
    for p in kalman_candidates:
        if p.exists():
            kalman_path = p
            break

    if kalman_path is None:
        raise FileNotFoundError(
            "Could not find FuSNet+Kalman WAV file for mic "
            f"{mic_idx}"
        )

    target, fs1 = torchaudio.load(str(target_path))
    kalman, fs2 = torchaudio.load(str(kalman_path))

    if fs1 != FS:
        target = torchaudio.functional.resample(target, fs1, FS)

    if fs2 != FS:
        kalman = torchaudio.functional.resample(kalman, fs2, FS)

    target = target.squeeze(0).numpy().astype(np.float32)
    kalman = kalman.squeeze(0).numpy().astype(np.float32)

    T = min(len(target), len(kalman))

    return target[:T], kalman[:T], kalman_path.name


# ============================================================
# Spectrogram functions
# ============================================================

def compute_spectrogram_db(x: np.ndarray):
    """
    Compute magnitude spectrogram in dB.
    """

    x = np.asarray(x, dtype=np.float32)

    x_tensor = torch.from_numpy(x).float()
    window = torch.hann_window(WIN_LENGTH)

    X = torch.stft(
        x_tensor,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        return_complex=True,
    )

    mag = torch.abs(X).numpy()
    mag_db = 20.0 * np.log10(mag + 1e-8)

    time_axis = np.arange(mag_db.shape[1]) * HOP_LENGTH / FS
    freq_axis = np.arange(mag_db.shape[0]) * FS / N_FFT

    return mag_db, time_axis, freq_axis


def plot_spectrogram_target_kalman(
    target: np.ndarray,
    kalman: np.ndarray,
    mic_idx: int,
    save_path: Path,
):
    """
    Plot only:
        1. Target spectrogram
        2. FuSNet + Kalman estimated spectrogram
    """

    target_db, time_axis, freq_axis = compute_spectrogram_db(target)
    kalman_db, _, _ = compute_spectrogram_db(kalman)

    vmax = max(np.max(target_db), np.max(kalman_db))
    vmin = vmax - TOP_DB

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].imshow(
        target_db,
        origin="lower",
        aspect="auto",
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"Target signal - Mic {mic_idx}")
    axes[0].set_ylabel("Frequency (Hz)")

    im = axes[1].imshow(
        kalman_db,
        origin="lower",
        aspect="auto",
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"FuSNet + Kalman estimated signal - Mic {mic_idx}")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time (s)")

    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Magnitude (dB)")

    fig.suptitle(
        f"Target vs FuSNet + Kalman Spectrogram - Group-A mic {mic_idx}",
        fontsize=14,
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Time-domain waveform functions
# ============================================================

def plot_waveform_target_kalman(
    target: np.ndarray,
    kalman: np.ndarray,
    mic_idx: int,
    save_path: Path,
):
    """
    Plot only:
        1. Target waveform
        2. FuSNet + Kalman estimated waveform
    """

    T = min(len(target), len(kalman))

    target = target[:T]
    kalman = kalman[:T]

    t = np.arange(T) / FS

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, target, linewidth=0.8)
    axes[0].set_title(f"Target signal - Mic {mic_idx}")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, kalman, linewidth=0.8)
    axes[1].set_title(f"FuSNet + Kalman estimated signal - Mic {mic_idx}")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")

    fig.suptitle(
        f"Target vs FuSNet + Kalman Waveform - Group-A mic {mic_idx}",
        fontsize=14,
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_estimated_signal_only(
    kalman: np.ndarray,
    mic_idx: int,
    save_path: Path,
):
    """
    Plot only the final estimated signal from FuSNet + Kalman.
    """

    T = len(kalman)
    t = np.arange(T) / FS

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ax.plot(t, kalman, linewidth=0.8)
    ax.set_title(f"FuSNet + Kalman estimated signal - Mic {mic_idx}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_target_vs_kalman(
    target: np.ndarray,
    kalman: np.ndarray,
    mic_idx: int,
    save_path: Path,
):
    """
    Overlay target and FuSNet+Kalman estimate in one time-domain plot.
    """

    T = min(len(target), len(kalman))
    target = target[:T]
    kalman = kalman[:T]

    t = np.arange(T) / FS

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ax.plot(t, target, linewidth=0.8, label="Target")
    ax.plot(t, kalman, linewidth=0.8, label="FuSNet + Kalman estimate")

    ax.set_title(f"Target vs FuSNet + Kalman estimate - Mic {mic_idx}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Plot target and FuSNet + Kalman results")
    print("=" * 80)
    print(f"Result folder : {RESULT_DIR}")
    print(f"Figure folder : {FIG_DIR}")
    print("=" * 80)

    # Prefer NPY outputs because they contain all microphones in one file.
    target_all, kalman_all, kalman_file = load_npy_outputs(RESULT_DIR)

    print(f"Target shape : {target_all.shape}")
    print(f"Kalman shape : {kalman_all.shape}")
    print(f"Result file  : {kalman_file}")

    if PLOT_ALL_MICS:
        mic_indices = range(1, min(NUM_MICS, target_all.shape[0]) + 1)
    else:
        mic_indices = [SELECTED_MIC]

    for mic_idx in mic_indices:
        ch = mic_idx - 1

        target = target_all[ch]
        kalman = kalman_all[ch]

        spec_path = FIG_DIR / f"spectrogram_target_kalman_mic_{mic_idx}.png"
        wave_path = FIG_DIR / f"waveform_target_kalman_mic_{mic_idx}.png"
        est_path = FIG_DIR / f"estimated_signal_kalman_mic_{mic_idx}.png"
        overlay_path = FIG_DIR / f"overlay_target_kalman_mic_{mic_idx}.png"

        plot_spectrogram_target_kalman(
            target=target,
            kalman=kalman,
            mic_idx=mic_idx,
            save_path=spec_path,
        )

        plot_waveform_target_kalman(
            target=target,
            kalman=kalman,
            mic_idx=mic_idx,
            save_path=wave_path,
        )

        plot_estimated_signal_only(
            kalman=kalman,
            mic_idx=mic_idx,
            save_path=est_path,
        )

        plot_overlay_target_vs_kalman(
            target=target,
            kalman=kalman,
            mic_idx=mic_idx,
            save_path=overlay_path,
        )

        print(f"Saved: {spec_path}")
        print(f"Saved: {wave_path}")
        print(f"Saved: {est_path}")
        print(f"Saved: {overlay_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()