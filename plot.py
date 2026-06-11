from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

# Change this to your result folder
out_dir = Path("results_fusnet_retm_kalman_no_backprop")

# Output image folder
fig_dir = out_dir / "spectrogram_plots"
fig_dir.mkdir(parents=True, exist_ok=True)

# Sampling rate
fs = 16000

# STFT settings
n_fft = 1024
hop_length = 256
win_length = 1024

# Dynamic range for plotting
top_db = 80.0

# Number of target mics
num_target_mics = 5


# ============================================================
# Utility functions
# ============================================================

def load_wav_mono(path: Path) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))

    if sr != fs:
        wav = torchaudio.functional.resample(wav, sr, fs)

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    return wav.squeeze(0).cpu().numpy().astype(np.float32)


def load_signal_pair_from_wav(out_dir: Path, mic_idx: int):
    """
    Load target and result signals from WAV files.
    """

    target_path = out_dir / f"target_mic_{mic_idx}.wav"

    candidate_result_paths = [
        out_dir / f"result_mic_{mic_idx}.wav",
        out_dir / f"kalman_mic_{mic_idx}.wav",
        out_dir / f"after_G_update_param_kalman_mic_{mic_idx}.wav",
        out_dir / f"after_randomwalk_param_kalman_mic_{mic_idx}.wav",
        out_dir / f"after_gpu_param_kalman_mic_{mic_idx}.wav",
        out_dir / f"after_param_kalman_mic_{mic_idx}.wav",
        out_dir / f"no_backprop_retm_kalman_mic_{mic_idx}.wav",
        out_dir / f"baseline_fusnet_mic_{mic_idx}.wav",
    ]

    if not target_path.exists():
        return None, None

    result_path = None
    for p in candidate_result_paths:
        if p.exists():
            result_path = p
            break

    if result_path is None:
        return None, None

    target = load_wav_mono(target_path)
    result = load_wav_mono(result_path)

    T = min(len(target), len(result))
    target = target[:T]
    result = result[:T]

    return target, result


def load_signal_pair_from_npy(out_dir: Path):
    """
    Load target and result signals from NPY files.
    """

    target_path = out_dir / "mA_target.npy"

    candidate_result_paths = [
        out_dir / "mA_result.npy",
        out_dir / "mA_no_backprop_retm_kalman.npy",
        out_dir / "mA_after_G_update_parameter_kalman.npy",
        out_dir / "mA_after_randomwalk_parameter_kalman.npy",
        out_dir / "mA_after_gpu_parameter_kalman.npy",
        out_dir / "mA_fusnet_baseline_original_style.npy",
        out_dir / "mA_fusnet_baseline.npy",
    ]

    if not target_path.exists():
        return None, None

    result_path = None
    for p in candidate_result_paths:
        if p.exists():
            result_path = p
            break

    if result_path is None:
        return None, None

    target = np.load(target_path).astype(np.float32)
    result = np.load(result_path).astype(np.float32)

    T = min(target.shape[1], result.shape[1])
    target = target[:, :T]
    result = result[:, :T]

    return target, result


def compute_spectrogram_db(x: np.ndarray):
    """
    Compute log spectrogram in dB.
    """

    x_torch = torch.from_numpy(x).float()
    window = torch.hann_window(win_length)

    X = torch.stft(
        x_torch,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )

    mag = torch.abs(X).numpy()
    mag_db = 20.0 * np.log10(mag + 1e-8)

    return mag_db


def plot_one_mic_spectrogram(
    target: np.ndarray,
    result: np.ndarray,
    mic_idx: int,
    save_path: Path,
):
    """
    Plot target and result spectrograms for one microphone.
    """

    T = min(len(target), len(result))
    target = target[:T]
    result = result[:T]

    target_db = compute_spectrogram_db(target)
    result_db = compute_spectrogram_db(result)

    vmax = max(np.max(target_db), np.max(result_db))
    vmin = vmax - top_db

    time_axis = np.arange(target_db.shape[1]) * hop_length / fs
    freq_axis = np.arange(target_db.shape[0]) * fs / n_fft

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    im0 = axes[0].imshow(
        target_db,
        origin="lower",
        aspect="auto",
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"Target mic {mic_idx}")
    axes[0].set_ylabel("Frequency (Hz)")

    im1 = axes[1].imshow(
        result_db,
        origin="lower",
        aspect="auto",
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"Result mic {mic_idx}")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time (s)")

    cbar = fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Magnitude (dB)")

    fig.suptitle(f"Spectrogram comparison - mic {mic_idx}", fontsize=14)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_waveform_comparison(
    target: np.ndarray,
    result: np.ndarray,
    mic_idx: int,
    save_path: Path,
):
    """
    Plot target and result waveforms for one microphone.
    """

    T = min(len(target), len(result))
    target = target[:T]
    result = result[:T]

    t = np.arange(T) / fs

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, target, linewidth=0.8)
    axes[0].set_title(f"Target mic {mic_idx}")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, result, linewidth=0.8)
    axes[1].set_title(f"Result mic {mic_idx}")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")

    fig.suptitle(f"Waveform comparison - mic {mic_idx}", fontsize=14)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Plot target and result spectrograms")
    print("=" * 80)
    print(f"Input result folder : {out_dir}")
    print(f"Output figure folder: {fig_dir}")
    print("=" * 80)

    # --------------------------------------------------------
    # First try WAV files
    # --------------------------------------------------------

    found_wav = False

    for mic_idx in range(1, num_target_mics + 1):
        target, result = load_signal_pair_from_wav(out_dir, mic_idx)

        if target is None:
            continue

        found_wav = True

        spec_path = fig_dir / f"spectrogram_target_result_mic_{mic_idx}.png"
        wave_path = fig_dir / f"waveform_target_result_mic_{mic_idx}.png"

        plot_one_mic_spectrogram(
            target=target,
            result=result,
            mic_idx=mic_idx,
            save_path=spec_path,
        )

        plot_waveform_comparison(
            target=target,
            result=result,
            mic_idx=mic_idx,
            save_path=wave_path,
        )

        print(f"Saved: {spec_path}")
        print(f"Saved: {wave_path}")

    if found_wav:
        print("\nDone using WAV files.")
        return

    # --------------------------------------------------------
    # If WAV files are not found, try NPY files
    # --------------------------------------------------------

    target_all, result_all = load_signal_pair_from_npy(out_dir)

    if target_all is None:
        raise FileNotFoundError(
            "Could not find target/result WAV or NPY files in the output folder."
        )

    for ch in range(target_all.shape[0]):
        mic_idx = ch + 1

        target = target_all[ch]
        result = result_all[ch]

        spec_path = fig_dir / f"spectrogram_target_result_mic_{mic_idx}.png"
        wave_path = fig_dir / f"waveform_target_result_mic_{mic_idx}.png"

        plot_one_mic_spectrogram(
            target=target,
            result=result,
            mic_idx=mic_idx,
            save_path=spec_path,
        )

        plot_waveform_comparison(
            target=target,
            result=result,
            mic_idx=mic_idx,
            save_path=wave_path,
        )

        print(f"Saved: {spec_path}")
        print(f"Saved: {wave_path}")

    print("\nDone using NPY files.")


if __name__ == "__main__":
    main()