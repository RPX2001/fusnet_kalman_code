from __future__ import annotations

import numpy as np
import torch


class BlockKalmanFusenetWeightReTM:
    """
    Block Kalman filtering on FuSNet checkpoint-derived ReTM weights.

    Effective filter:
        R_abK(t) = R_fusnet ⊙ Rt(t)

    Output:
        mA_hat(t) = R_abK(t) * mB(t)

    Difference from full version:
        Prediction is done once per block.
        Measurement update is done sample-by-sample inside the block.
    """

    def __init__(
        self,
        R_fusnet: np.ndarray,
        block_size: int = 256,
        transition: float = 0.999,
        process_noise: float = 1e-8,
        observation_noise: float = 1e-2,
        initial_covariance: float = 1e-3,
        dtype=None,
        device: str | None = None,
    ):
        if dtype is None:
            dtype = np.float64

        R_fusnet = np.asarray(R_fusnet, dtype=dtype)

        if R_fusnet.ndim != 3:
            raise ValueError(f"Expected R_fusnet shape [QA, QB, L], got {R_fusnet.shape}")

        self.R_fusnet = R_fusnet
        self.qa, self.qb, self.L = R_fusnet.shape

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

        self.input_dim = self.qb * self.L

        if self.use_torch:
            # Rt multiplier, initialized as identity multiplier
            self.Rt = torch.ones(
                (self.qa, self.input_dim), dtype=self.torch_dtype, device=self.torch_device
            )

            self.P = torch.full(
                (self.qa, self.input_dim),
                self.initial_covariance,
                dtype=self.torch_dtype,
                device=self.torch_device,
            )

            self.x_buffer = torch.zeros(
                (self.qb, self.L), dtype=self.torch_dtype, device=self.torch_device
            )

            self.Rf_flat = torch.as_tensor(
                self.R_fusnet.reshape(self.qa, self.input_dim),
                dtype=self.torch_dtype,
                device=self.torch_device,
            )
        else:
            # Rt multiplier, initialized as identity multiplier
            self.Rt = np.ones((self.qa, self.input_dim), dtype=self.dtype)

            self.P = self.initial_covariance * np.ones(
                (self.qa, self.input_dim), dtype=self.dtype
            )

            self.x_buffer = np.zeros((self.qb, self.L), dtype=self.dtype)

            self.Rf_flat = self.R_fusnet.reshape(self.qa, self.input_dim)

    def _reset_buffer(self):
        self.x_buffer[:, :] = 0.0

    def _update_buffer(self, x_t: np.ndarray):
        if self.use_torch:
            x_t = torch.as_tensor(x_t, dtype=self.torch_dtype, device=self.torch_device)
        self.x_buffer[:, 1:] = self.x_buffer[:, :-1]
        self.x_buffer[:, 0] = x_t

    def _make_phi(self) -> np.ndarray:
        return self.x_buffer.reshape(-1)

    def process(self, mB: np.ndarray, mA: np.ndarray):
        mB = np.asarray(mB, dtype=self.dtype)
        mA = np.asarray(mA, dtype=self.dtype)

        if mB.shape[0] != self.qb:
            raise ValueError(f"Expected mB with QB={self.qb}, got {mB.shape[0]}")

        if mA.shape[0] != self.qa:
            raise ValueError(f"Expected mA with QA={self.qa}, got {mA.shape[0]}")

        T = min(mB.shape[1], mA.shape[1])
        mB = mB[:, :T]
        mA = mA[:, :T]

        self._reset_buffer()

        G = self.transition
        Qn = self.process_noise
        Rv = self.observation_noise


        if self.use_torch:
            mA_hat_t = torch.zeros((self.qa, T), dtype=self.torch_dtype, device=self.torch_device)
            mA_base_t = torch.zeros((self.qa, T), dtype=self.torch_dtype, device=self.torch_device)
            err_t = torch.zeros((self.qa, T), dtype=self.torch_dtype, device=self.torch_device)

            for block_start in range(0, T, self.block_size):
                block_end = min(block_start + self.block_size, T)

                # Block prediction around identity:
                # Rt^- = 1 + G(Rt - 1)
                self.Rt = 1.0 + G * (self.Rt - 1.0)
                self.P = (G ** 2) * self.P + Qn

                for t in range(block_start, block_end):
                    x_t = mB[:, t]
                    d_t = torch.as_tensor(mA[:, t], dtype=self.torch_dtype, device=self.torch_device)

                    self._update_buffer(x_t)
                    phi = self._make_phi()

                    for a in range(self.qa):
                        base_regressor = self.Rf_flat[a] * phi

                        y_base = torch.sum(base_regressor)
                        y_hat = torch.dot(self.Rt[a], base_regressor)

                        e = d_t[a] - y_hat

                        S = torch.sum(self.P[a] * (base_regressor ** 2)) + Rv
                        K = (self.P[a] * base_regressor) / (S + 1e-12)

                        self.Rt[a] = self.Rt[a] + K * e

                        P_new = (1.0 - K * base_regressor) * self.P[a]
                        self.P[a] = torch.clamp(P_new, min=1e-12)

                        mA_base_t[a, t] = y_base
                        mA_hat_t[a, t] = y_hat
                        err_t[a, t] = e

            Rt_final_t = self.Rt.reshape(self.qa, self.qb, self.L)

            return (
                mA_hat_t.detach().cpu().numpy().astype(np.float32),
                mA_base_t.detach().cpu().numpy().astype(np.float32),
                err_t.detach().cpu().numpy().astype(np.float32),
                Rt_final_t.detach().cpu().numpy().astype(np.float32),
            )
        else:
            mA_hat = np.zeros((self.qa, T), dtype=self.dtype)
            mA_base = np.zeros((self.qa, T), dtype=self.dtype)
            err = np.zeros((self.qa, T), dtype=self.dtype)

            for block_start in range(0, T, self.block_size):
                block_end = min(block_start + self.block_size, T)

                # Block prediction around identity:
                # Rt^- = 1 + G(Rt - 1)
                self.Rt = 1.0 + G * (self.Rt - 1.0)
                self.P = (G ** 2) * self.P + Qn

                for t in range(block_start, block_end):
                    x_t = mB[:, t]
                    d_t = mA[:, t]

                    self._update_buffer(x_t)
                    phi = self._make_phi()

                    for a in range(self.qa):
                        base_regressor = self.Rf_flat[a] * phi

                        y_base = float(np.sum(base_regressor))
                        y_hat = float(np.dot(self.Rt[a], base_regressor))

                        e = float(d_t[a] - y_hat)

                        S = float(np.sum(self.P[a] * (base_regressor ** 2)) + Rv)
                        K = (self.P[a] * base_regressor) / (S + 1e-12)

                        self.Rt[a] = self.Rt[a] + K * e

                        P_new = (1.0 - K * base_regressor) * self.P[a]
                        self.P[a] = np.maximum(P_new, 1e-12)

                        mA_base[a, t] = y_base
                        mA_hat[a, t] = y_hat
                        err[a, t] = e

            Rt_final = self.Rt.reshape(self.qa, self.qb, self.L)

            return (
                mA_hat.astype(np.float32),
                mA_base.astype(np.float32),
                err.astype(np.float32),
                Rt_final.astype(np.float32),
            )