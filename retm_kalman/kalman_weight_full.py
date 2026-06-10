from __future__ import annotations

import numpy as np
import torch


class FullKalmanFusenetWeightReTM:
    """
    Sample-by-sample Kalman filtering on FuSNet checkpoint-derived ReTM weights.

    Base filter:
        R_fusnet = R_ab^F
        shape: [QA, QB, L]

    Adaptive multiplier:
        Rt(t)
        shape: [QA, QB, L]

    Effective filter:
        R_abK(t) = R_fusnet ⊙ Rt(t)

    Output:
        mA_hat(t) = R_abK(t) * mB(t)

    Error:
        e(t) = mA(t) - mA_hat(t)

    Rt is initialized to ones, so initially:
        R_abK(0) = R_fusnet
    """

    def __init__(
        self,
        R_fusnet: np.ndarray,
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
            # Rt multiplier, shape [QA, QB*L]
            # Initial Rt = 1, so R_abK = R_fusnet
            self.Rt = torch.ones(
                (self.qa, self.input_dim), dtype=self.torch_dtype, device=self.torch_device
            )

            # Diagonal covariance for Rt coefficients
            self.P = torch.full(
                (self.qa, self.input_dim),
                self.initial_covariance,
                dtype=self.torch_dtype,
                device=self.torch_device,
            )

            self.x_buffer = torch.zeros(
                (self.qb, self.L), dtype=self.torch_dtype, device=self.torch_device
            )

            # Flatten base filter per output channel
            self.Rf_flat = torch.as_tensor(
                self.R_fusnet.reshape(self.qa, self.input_dim),
                dtype=self.torch_dtype,
                device=self.torch_device,
            )
        else:
            # Rt multiplier, shape [QA, QB*L]
            # Initial Rt = 1, so R_abK = R_fusnet
            self.Rt = np.ones((self.qa, self.input_dim), dtype=self.dtype)

            # Diagonal covariance for Rt coefficients
            self.P = self.initial_covariance * np.ones(
                (self.qa, self.input_dim), dtype=self.dtype
            )

            self.x_buffer = np.zeros((self.qb, self.L), dtype=self.dtype)

            # Flatten base filter per output channel
            self.Rf_flat = self.R_fusnet.reshape(self.qa, self.input_dim)

    def _reset_buffer(self):
        self.x_buffer[:, :] = 0.0

    def _update_buffer(self, x_t: np.ndarray):
        if self.use_torch:
            x_t = torch.as_tensor(x_t, dtype=self.torch_dtype, device=self.torch_device)
        self.x_buffer[:, 1:] = self.x_buffer[:, :-1]
        self.x_buffer[:, 0] = x_t

    def _make_phi(self) -> np.ndarray:
        """
        Group-B regressor:
            [mB1(t), mB1(t-1), ..., mB1(t-L+1),
             mB2(t), ..., mB_QB(t-L+1)]
        """
        return self.x_buffer.reshape(-1)

    def process(self, mB: np.ndarray, mA: np.ndarray):
        """
        Args:
            mB: Group-B microphone signals, shape [QB, T]
            mA: actual Group-A microphone signals, shape [QA, T]

        Returns:
            mA_hat: Kalman output, shape [QA, T]
            mA_base: fixed FuSNet-filter output using R_fusnet only, shape [QA, T]
            err: mA - mA_hat, shape [QA, T]
            Rt_final: final adaptive multiplier, shape [QA, QB, L]
        """

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

            for t in range(T):
                x_t = mB[:, t]
                d_t = torch.as_tensor(mA[:, t], dtype=self.torch_dtype, device=self.torch_device)

                self._update_buffer(x_t)
                phi = self._make_phi()

                for a in range(self.qa):
                    # Base FuSNet filter contribution
                    base_regressor = self.Rf_flat[a] * phi

                    # Fixed FuSNet-weight filter output
                    y_base = torch.sum(base_regressor)

                    # Prediction around identity multiplier:
                    # Rt^- = 1 + G(Rt - 1)
                    Rt_pred = 1.0 + G * (self.Rt[a] - 1.0)
                    P_pred = (G ** 2) * self.P[a] + Qn

                    # Effective output
                    y_hat = torch.dot(Rt_pred, base_regressor)

                    e = d_t[a] - y_hat

                    # Diagonal Kalman gain
                    S = torch.sum(P_pred * (base_regressor ** 2)) + Rv
                    K = (P_pred * base_regressor) / (S + 1e-12)

                    # Rt update
                    Rt_new = Rt_pred + K * e

                    # Covariance update
                    P_new = (1.0 - K * base_regressor) * P_pred
                    P_new = torch.clamp(P_new, min=1e-12)

                    self.Rt[a] = Rt_new
                    self.P[a] = P_new

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

            for t in range(T):
                x_t = mB[:, t]
                d_t = mA[:, t]

                self._update_buffer(x_t)
                phi = self._make_phi()

                for a in range(self.qa):
                    # Base FuSNet filter contribution
                    base_regressor = self.Rf_flat[a] * phi

                    # Fixed FuSNet-weight filter output
                    y_base = float(np.sum(base_regressor))

                    # Prediction around identity multiplier:
                    # Rt^- = 1 + G(Rt - 1)
                    Rt_pred = 1.0 + G * (self.Rt[a] - 1.0)
                    P_pred = (G ** 2) * self.P[a] + Qn

                    # Effective output
                    y_hat = float(np.dot(Rt_pred, base_regressor))

                    e = float(d_t[a] - y_hat)

                    # Diagonal Kalman gain
                    S = float(np.sum(P_pred * (base_regressor ** 2)) + Rv)
                    K = (P_pred * base_regressor) / (S + 1e-12)

                    # Rt update
                    Rt_new = Rt_pred + K * e

                    # Covariance update
                    P_new = (1.0 - K * base_regressor) * P_pred
                    P_new = np.maximum(P_new, 1e-12)

                    self.Rt[a] = Rt_new
                    self.P[a] = P_new

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