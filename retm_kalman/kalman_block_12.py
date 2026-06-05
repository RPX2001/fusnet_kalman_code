from __future__ import annotations
from .kalman_block import BlockKalmanCorrectionReTM as _BaseBlockKalmanCorrectionReTM


class BlockKalmanCorrectionReTM(_BaseBlockKalmanCorrectionReTM):
    """12-mic default wrapper for the block Kalman correction model."""

    def __init__(
        self,
        qb: int = 7,
        qa: int = 5,
        L: int = 1024,
        block_size: int = 128,
        transition: float = 0.995,
        process_noise: float = 1e-7,
        observation_noise: float = 1e-3,
        initial_covariance: float = 1e-2,
        dtype=None,
        device: str | None = None,
    ):
        if dtype is None:
            import numpy as np

            dtype = np.float64
        super().__init__(
            qb=qb,
            qa=qa,
            L=L,
            block_size=block_size,
            transition=transition,
            process_noise=process_noise,
            observation_noise=observation_noise,
            initial_covariance=initial_covariance,
            dtype=dtype,
            device=device,
        )