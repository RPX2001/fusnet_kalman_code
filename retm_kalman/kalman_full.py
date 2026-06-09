from __future__ import annotations

import numpy as np
import torch


class FullKalmanCorrectionReTM:
    """
    Sample-by-sample Kalman ReTM tracker using FuSNet output directly.

    New model:
        u(t)      = FuSNet(mB)(t)          shape [QA]
        mA_hat(t) = R(t) * u_buffer(t)     shape [QA]
        e(t)      = mA(t) - mA_hat(t)

    Here R has size:
        R: QA x QA x L

    For 13-mic setup:
        QA = 5
        R  = 5 x 5 x L

    This implementation uses a diagonal Kalman covariance approximation
    to avoid the huge full covariance matrix.
    """

    def __init__(
        self,
        qb: int = 8,          # kept only for compatibility with old run_system.py
        qa: int = 5,
        L: int = 256,
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
        # each output channel has QA input channels, each with L taps
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
            # R shape: [QA, QA*L]
            # Equivalent to QA x QA x L
            self.R = np.zeros((self.qa, self.input_dim), dtype=self.dtype)

            # Diagonal covariance for each output filter coefficient
            self.P = self.initial_covariance * np.ones(
                (self.qa, self.input_dim), dtype=self.dtype
            )

        # Initialize R close to identity:
        # output channel q initially follows FuSNet output channel q at delay 0
        for q in range(self.qa):
            idx = q * self.L
            self.R[q, idx] = 1.0

        # FuSNet output buffer: [QA, L]
        if self.use_torch:
            self.u_buffer = torch.zeros(
                (self.qa, self.L), dtype=self.torch_dtype, device=self.torch_device
            )
        else:
            self.u_buffer = np.zeros((self.qa, self.L), dtype=self.dtype)

    def _reset_buffer(self):
        self.u_buffer[:, :] = 0.0

    def _update_buffer(self, u_t: np.ndarray):
        """
        u_t: FuSNet output sample, shape [QA]
        """
        if self.use_torch:
            u_t = torch.as_tensor(u_t, dtype=self.torch_dtype, device=self.torch_device)
        self.u_buffer[:, 1:] = self.u_buffer[:, :-1]
        self.u_buffer[:, 0] = u_t

    def _make_phi(self) -> np.ndarray:
        """
        Construct regressor vector from FuSNet output buffer.

        phi = [
            u1(t), u1(t-1), ..., u1(t-L+1),
            u2(t), u2(t-1), ..., u2(t-L+1),
            ...
            uQA(t), ..., uQA(t-L+1)
        ]

        shape: [QA*L]
        """
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
                        Not used in new method, kept for interface compatibility.
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
            for t in range(T):
                u_t = mA_fusnet[:, t]
                d_t = torch.as_tensor(mA[:, t], dtype=self.torch_dtype, device=self.torch_device)

                self._update_buffer(u_t)
                phi = self._make_phi()

                for q in range(self.qa):
                    # -----------------------------
                    # Prediction
                    # -----------------------------
                    R_pred = G * self.R[q]
                    P_pred = (G ** 2) * self.P[q] + Qn

                    # -----------------------------
                    # Output estimate
                    # -----------------------------
                    y_hat = torch.dot(R_pred, phi)
                    e = d_t[q] - y_hat

                    # -----------------------------
                    # Kalman gain
                    # -----------------------------
                    S = torch.sum(P_pred * (phi ** 2)) + Rv
                    K = (P_pred * phi) / (S + 1e-12)

                    # -----------------------------
                    # Update R filter
                    # -----------------------------
                    R_new = R_pred + K * e

                    # -----------------------------
                    # Update covariance
                    # -----------------------------
                    P_new = (1.0 - K * phi) * P_pred
                    P_new = torch.clamp(P_new, min=1e-12)

                    self.R[q] = R_new
                    self.P[q] = P_new

                    mA_hat[q, t] = float(y_hat.detach().cpu().item())
                    err[q, t] = float(e.detach().cpu().item())
        else:
            for t in range(T):
                u_t = mA_fusnet[:, t]
                d_t = mA[:, t]

                self._update_buffer(u_t)
                phi = self._make_phi()

                for q in range(self.qa):
                    # -----------------------------
                    # Prediction
                    # -----------------------------
                    R_pred = G * self.R[q]
                    P_pred = (G ** 2) * self.P[q] + Qn

                    # -----------------------------
                    # Output estimate
                    # -----------------------------
                    y_hat = float(np.dot(R_pred, phi))
                    e = float(d_t[q] - y_hat)

                    # -----------------------------
                    # Kalman gain
                    # -----------------------------
                    # Diagonal covariance approximation:
                    # S = phi^T P phi + observation_noise
                    S = float(np.sum(P_pred * (phi ** 2)) + Rv)

                    K = (P_pred * phi) / (S + 1e-12)

                    # -----------------------------
                    # Update R filter
                    # -----------------------------
                    R_new = R_pred + K * e

                    # -----------------------------
                    # Update covariance
                    # -----------------------------
                    P_new = (1.0 - K * phi) * P_pred
                    P_new = np.maximum(P_new, 1e-12)

                    self.R[q] = R_new
                    self.P[q] = P_new

                    mA_hat[q, t] = y_hat
                    err[q, t] = e

        delta_hat = mA_hat - mA_fusnet

        return (
            mA_hat.astype(np.float32),
            delta_hat.astype(np.float32),
            err.astype(np.float32),
        )