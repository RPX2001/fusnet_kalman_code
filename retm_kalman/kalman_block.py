from __future__ import annotations

import numpy as np
import torch


class BlockKalmanCorrectionReTM:
    """
    Block Kalman ReTM tracker using FuSNet output directly.

    New model:
        u(t)      = FuSNet(mB)(t)
        mA_hat(t) = R(t) * u_buffer(t)
        e(t)      = mA(t) - mA_hat(t)

    R dimension:
        R: QA x QA x L

    Difference from FullKalmanCorrectionReTM:
        - Prediction is done once per block.
        - Then measurement updates are done sample-by-sample inside the block.
    """

    def __init__(
        self,
        qb: int = 8,          # kept for compatibility
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
            dtype = np.float64

        self.qb = int(qb)
        self.qa = int(qa)
        self.L = int(L)
        self.block_size = int(block_size)

        self.transition = float(transition)
        self.process_noise = float(process_noise)
        self.observation_noise = float(observation_noise)
        self.initial_covariance = float(initial_covariance)

        self.dtype = dtype
        self.device = device
        self.use_torch = False
        self.torch_device = None
        self.torch_dtype = torch.float64 if dtype == np.float64 else torch.float32

        if device is not None:
            try:
                requested_device = torch.device(device)
                if requested_device.type == "cuda" and torch.cuda.is_available():
                    self.use_torch = True
                    self.torch_device = requested_device
            except Exception:
                self.use_torch = False

        # New filter dimension:
        # R: QA x QA x L
        self.input_dim = self.qa * self.L

        if self.use_torch:
            self.R = torch.zeros(
                (self.qa, self.input_dim),
                dtype=self.torch_dtype,
                device=self.torch_device,
            )
            self.P = torch.full(
                (self.qa, self.input_dim),
                self.initial_covariance,
                dtype=self.torch_dtype,
                device=self.torch_device,
            )
        else:
            self.R = np.zeros((self.qa, self.input_dim), dtype=self.dtype)

            self.P = self.initial_covariance * np.ones(
                (self.qa, self.input_dim), dtype=self.dtype
            )

        # Identity initialization
        for q in range(self.qa):
            idx = q * self.L
            self.R[q, idx] = 1.0

        if self.use_torch:
            self.u_buffer = torch.zeros(
                (self.qa, self.L), dtype=self.torch_dtype, device=self.torch_device
            )
        else:
            self.u_buffer = np.zeros((self.qa, self.L), dtype=self.dtype)

    def _reset_buffer(self):
        self.u_buffer[:, :] = 0.0

    def _update_buffer(self, u_t: np.ndarray):
        if self.use_torch:
            u_t = torch.as_tensor(u_t, dtype=self.torch_dtype, device=self.torch_device)
        self.u_buffer[:, 1:] = self.u_buffer[:, :-1]
        self.u_buffer[:, 0] = u_t

    def _make_phi(self) -> np.ndarray:
        return self.u_buffer.reshape(-1)

    def process(
        self,
        mB: np.ndarray,
        mA: np.ndarray,
        mA_fusnet: np.ndarray,
    ):
        """
        Args:
            mB        : Group-B microphones, shape [QB, T]
                        Not used in the new method.
            mA        : Actual Group-A target, shape [QA, T]
            mA_fusnet : FuSNet initial Group-A estimate, shape [QA, T]

        Returns:
            mA_hat    : Kalman-filtered estimate, shape [QA, T]
            delta_hat : mA_hat - mA_fusnet, shape [QA, T]
            err       : mA - mA_hat, shape [QA, T]
        """

        mA = np.asarray(mA, dtype=self.dtype)
        mA_fusnet = np.asarray(mA_fusnet, dtype=self.dtype)

        if mA.ndim != 2:
            raise ValueError(f"Expected mA shape [QA, T], got {mA.shape}")

        if mA_fusnet.ndim != 2:
            raise ValueError(f"Expected mA_fusnet shape [QA, T], got {mA_fusnet.shape}")

        if mA.shape[0] != self.qa:
            raise ValueError(f"Expected mA with QA={self.qa}, got {mA.shape[0]}")

        if mA_fusnet.shape[0] != self.qa:
            raise ValueError(
                f"Expected mA_fusnet with QA={self.qa}, got {mA_fusnet.shape[0]}"
            )

        T = min(mA.shape[1], mA_fusnet.shape[1])
        mA = mA[:, :T]
        mA_fusnet = mA_fusnet[:, :T]

        mA_hat = np.zeros((self.qa, T), dtype=self.dtype)
        err = np.zeros((self.qa, T), dtype=self.dtype)

        self._reset_buffer()

        G = self.transition
        Qn = self.process_noise
        Rv = self.observation_noise

        if self.use_torch:
            for block_start in range(0, T, self.block_size):
                block_end = min(block_start + self.block_size, T)

                # -----------------------------------
                # Block prediction step
                # -----------------------------------
                R_pred_block = G * self.R
                P_pred_block = (G ** 2) * self.P + Qn

                self.R = R_pred_block.clone()
                self.P = P_pred_block.clone()

                # -----------------------------------
                # Sequential measurement updates inside block
                # -----------------------------------
                for t in range(block_start, block_end):
                    u_t = mA_fusnet[:, t]
                    d_t = torch.as_tensor(mA[:, t], dtype=self.torch_dtype, device=self.torch_device)

                    self._update_buffer(u_t)
                    phi = self._make_phi()

                    for q in range(self.qa):
                        y_hat = torch.dot(self.R[q], phi)
                        e = d_t[q] - y_hat

                        S = torch.sum(self.P[q] * (phi ** 2)) + Rv
                        K = (self.P[q] * phi) / (S + 1e-12)

                        self.R[q] = self.R[q] + K * e

                        P_new = (1.0 - K * phi) * self.P[q]
                        self.P[q] = torch.clamp(P_new, min=1e-12)

                        mA_hat[q, t] = float(y_hat.detach().cpu().item())
                        err[q, t] = float(e.detach().cpu().item())
        else:
            for block_start in range(0, T, self.block_size):
                block_end = min(block_start + self.block_size, T)

                # -----------------------------------
                # Block prediction step
                # -----------------------------------
                R_pred_block = G * self.R
                P_pred_block = (G ** 2) * self.P + Qn

                self.R = R_pred_block.copy()
                self.P = P_pred_block.copy()

                # -----------------------------------
                # Sequential measurement updates inside block
                # -----------------------------------
                for t in range(block_start, block_end):
                    u_t = mA_fusnet[:, t]
                    d_t = mA[:, t]

                    self._update_buffer(u_t)
                    phi = self._make_phi()

                    for q in range(self.qa):
                        y_hat = float(np.dot(self.R[q], phi))
                        e = float(d_t[q] - y_hat)

                        S = float(np.sum(self.P[q] * (phi ** 2)) + Rv)
                        K = (self.P[q] * phi) / (S + 1e-12)

                        self.R[q] = self.R[q] + K * e

                        P_new = (1.0 - K * phi) * self.P[q]
                        self.P[q] = np.maximum(P_new, 1e-12)

                        mA_hat[q, t] = y_hat
                        err[q, t] = e

        delta_hat = mA_hat - mA_fusnet

        return (
            mA_hat.astype(np.float32),
            delta_hat.astype(np.float32),
            err.astype(np.float32),
        )