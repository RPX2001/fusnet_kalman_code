from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def extract_fusnet_retm_by_impulse(
    model,
    predict_fn: Callable,
    qb: int,
    qa: int,
    L: int,
    context: int,
    window_size: int,
    stride: int,
    batch_size: int,
    device,
    start_offset: int = 0,
    impulse_sample: int | None = None,
    output_dtype=np.float64,
) -> np.ndarray:
    """
    Extract an equivalent FuSNet filter R_ab^F by impulse probing.

    For each Group-B input channel b:
        input = impulse on channel b
        output = FuSNet(input)
        extracted output response becomes R_ab^F[:, b, :]

    Output:
        R_fusnet shape = [QA, QB, L]

    Notes:
        This treats FuSNet as an equivalent linear multichannel FIR system.
        This is suitable if your FuSNet architecture is mainly convolutional/linear.
    """

    qb = int(qb)
    qa = int(qa)
    L = int(L)

    if impulse_sample is None:
        impulse_sample = max(2 * context + window_size, 2 * L + context)

    total_len = impulse_sample + L + abs(start_offset) + window_size + context + stride
    total_len = int(total_len)

    R_fusnet = np.zeros((qa, qb, L), dtype=output_dtype)

    print("=" * 80)
    print("Extracting FuSNet equivalent ReTM using impulse probing")
    print("=" * 80)
    print(f"QB={qb}, QA={qa}, L={L}")
    print(f"Impulse sample = {impulse_sample}")
    print(f"Start offset   = {start_offset}")
    print(f"Probe length   = {total_len}")

    for b in range(qb):
        x = np.zeros((qb, total_len), dtype=np.float32)
        x[b, impulse_sample] = 1.0

        y = predict_fn(
            model=model,
            mB=x,
            context=context,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            device=device,
        )

        if y.shape[0] != qa:
            raise ValueError(f"Expected FuSNet output QA={qa}, got {y.shape[0]}")

        start = impulse_sample + start_offset
        end = start + L

        if start < 0:
            raise ValueError("start_offset is too negative; extraction start < 0")

        if y.shape[1] < end:
            y = np.pad(y, ((0, 0), (0, end - y.shape[1])), mode="constant")

        R_fusnet[:, b, :] = y[:, start:end].astype(output_dtype)

        print(f"Extracted FuSNet filter for input mic b={b + 1}/{qb}")

    return R_fusnet


def load_or_extract_fusnet_retm(
    cache_path: str | Path,
    model,
    predict_fn: Callable,
    qb: int,
    qa: int,
    L: int,
    context: int,
    window_size: int,
    stride: int,
    batch_size: int,
    device,
    start_offset: int = 0,
    force_extract: bool = False,
    output_dtype=np.float64,
) -> np.ndarray:
    """
    Load cached FuSNet ReTM filter if available.
    Otherwise extract it from checkpoint and save it.
    """

    cache_path = Path(cache_path)

    if cache_path.exists() and not force_extract:
        print(f"Loading cached FuSNet ReTM filter:\n{cache_path}")
        R = np.load(cache_path)

        expected_shape = (qa, qb, L)
        if R.shape != expected_shape:
            raise ValueError(
                f"Cached FuSNet filter has shape {R.shape}, expected {expected_shape}. "
                f"Delete cache or set force_extract=true."
            )

        return R.astype(output_dtype)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    R = extract_fusnet_retm_by_impulse(
        model=model,
        predict_fn=predict_fn,
        qb=qb,
        qa=qa,
        L=L,
        context=context,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        device=device,
        start_offset=start_offset,
        output_dtype=output_dtype,
    )

    np.save(cache_path, R.astype(np.float32))
    print(f"Saved FuSNet ReTM filter cache:\n{cache_path}")

    return R.astype(output_dtype)