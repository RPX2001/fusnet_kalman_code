from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None


def read_mic_wavs(seq_dir: str | Path, fs_target: int = 16000,
                  qa_mics=(1, 2, 3), qb_mics=(4, 5, 6, 7)) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Read mic_1.wav ... mic_7.wav from one sequence folder.

    Returns
    -------
    mA : ndarray, shape [QA, T]
        Group A target microphone signals.
    mB : ndarray, shape [QB, T]
        Group B input microphone signals.
    fs : int
        Output sampling rate.
    """
    seq_dir = Path(seq_dir)

    def _read(mic_idx: int):
        path = seq_dir / f"mic_{mic_idx}.wav"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        x, fs = sf.read(path, always_2d=False)
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        x = x.astype(np.float32)
        if fs != fs_target:
            if resample_poly is None:
                raise RuntimeError("scipy is required for resampling. Install scipy or provide 16 kHz wavs.")
            from math import gcd
            g = gcd(fs_target, fs)
            x = resample_poly(x, fs_target // g, fs // g).astype(np.float32)
            fs = fs_target
        return x, fs

    signals = {}
    fs0 = None
    for m in sorted(set(qa_mics) | set(qb_mics)):
        x, fs = _read(m)
        if fs0 is None:
            fs0 = fs
        elif fs != fs0:
            raise ValueError("All microphone files must have the same sampling rate.")
        signals[m] = x

    T = min(len(signals[m]) for m in signals)
    for m in signals:
        signals[m] = signals[m][:T]

    mA = np.stack([signals[m] for m in qa_mics], axis=0)
    mB = np.stack([signals[m] for m in qb_mics], axis=0)
    return mA.astype(np.float32), mB.astype(np.float32), fs0


def normalize_pair(mA: np.ndarray, mB: np.ndarray, eps: float = 1e-8):
    peak = max(float(np.max(np.abs(mA))), float(np.max(np.abs(mB))), eps)
    return mA / peak, mB / peak, peak


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    C, T = x.shape
    if T < frame_len:
        x = np.pad(x, ((0, 0), (0, frame_len - T)))
        T = x.shape[1]

    n_frames = 1 + int(np.ceil((T - frame_len) / hop))
    total_len = (n_frames - 1) * hop + frame_len
    if total_len > T:
        x = np.pad(x, ((0, 0), (0, total_len - T)))

    frames = []
    for i in range(n_frames):
        s = i * hop
        frames.append(x[:, s:s + frame_len])
    return np.stack(frames, axis=0).astype(np.float32)


def overlap_add(frames: np.ndarray, hop: int, target_len: int | None = None) -> np.ndarray:
    n_frames, C, frame_len = frames.shape
    total_len = (n_frames - 1) * hop + frame_len
    y = np.zeros((C, total_len), dtype=np.float32)
    w = np.zeros(total_len, dtype=np.float32)

    win = np.hanning(frame_len).astype(np.float32)
    if np.max(win) <= 0:
        win = np.ones(frame_len, dtype=np.float32)

    for i in range(n_frames):
        s = i * hop
        y[:, s:s + frame_len] += frames[i] * win[None, :]
        w[s:s + frame_len] += win

    y = y / np.maximum(w[None, :], 1e-8)
    if target_len is not None:
        y = y[:, :target_len]
    return y.astype(np.float32)


def write_mic_wavs(out_dir: str | Path, signals: np.ndarray, fs: int = 16000, prefix: str = "est_mic"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(signals.shape[0]):
        sf.write(out_dir / f"{prefix}_{i+1}.wav", signals[i].astype(np.float32), fs)
