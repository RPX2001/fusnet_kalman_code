from __future__ import annotations
import numpy as np


def mse_db(ref: np.ndarray, est: np.ndarray, eps: float = 1e-12):
    err = ref - est
    mse = np.mean(err ** 2, axis=-1)
    return 10 * np.log10(mse + eps)


def sdr_db(ref: np.ndarray, est: np.ndarray, eps: float = 1e-12):
    err = ref - est
    num = np.sum(ref ** 2, axis=-1)
    den = np.sum(err ** 2, axis=-1)
    return 10 * np.log10((num + eps) / (den + eps))


def print_basic_metrics(ref: np.ndarray, est: np.ndarray, name: str = "estimate"):
    sdr = sdr_db(ref, est)
    mse = mse_db(ref, est)
    print(f"\n{name}")
    print("SDR per channel:", [round(float(v), 4) for v in sdr])
    print("SDR avg:", round(float(np.mean(sdr)), 4))
    print("MSE dB per channel:", [round(float(v), 4) for v in mse])
    print("MSE dB avg:", round(float(np.mean(mse)), 4))
