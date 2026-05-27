from __future__ import annotations
import numpy as np
import torch


class FullKalmanCorrectionReTM:
    """
    Standard Kalman correction ReTM estimator.

    FuSNet gives:
        mA_f(t)

    Kalman estimates residual/correction ReTM:
        delta_mA(t) = R_delta(t) mB(t)

    Final:
        mA_hat(t) = mA_f(t) + delta_mA(t)

    This is "without block Kalman".
    Each Group A output uses a full covariance matrix of size (QB*L) x (QB*L).
    """

    def __init__(self,
                 qb: int = 4,
                 qa: int = 3,
                 L: int = 256,
                 transition: float = 0.995,
                 process_noise: float = 1e-7,
                 observation_noise: float = 1e-3,
                 initial_covariance: float = 1e-2,
                 dtype=np.float64,
                 device: str | None = None):
        self.qb = qb
        self.qa = qa
        self.L = L
        self.n = qb * L
        self.transition = transition
        self.q = process_noise
        self.r = observation_noise
        self.dtype = dtype
        # device: if provided and torch available, run Kalman on that device
        self.device = device
        self.use_torch = False
        if device is not None:
            try:
                dev = torch.device(device) if not isinstance(device, torch.device) else device
                if dev.type == "cuda" and torch.cuda.is_available():
                    self.use_torch = True
                    self.torch_device = dev
                    self.torch_dtype = torch.float64 if dtype == np.float64 else torch.float32
            except Exception:
                self.use_torch = False

        if self.use_torch:
            self.w = torch.zeros((qa, self.n), dtype=self.torch_dtype, device=self.torch_device)
            self.P = torch.stack([torch.eye(self.n, dtype=self.torch_dtype, device=self.torch_device) * initial_covariance
                                   for _ in range(qa)], dim=0)
            self.xbuf = torch.zeros((qb, L), dtype=self.torch_dtype, device=self.torch_device)
        else:
            self.w = np.zeros((qa, self.n), dtype=dtype)
            self.P = np.stack([np.eye(self.n, dtype=dtype) * initial_covariance for _ in range(qa)], axis=0)
            self.xbuf = np.zeros((qb, L), dtype=dtype)

    def _update_buffer(self, mB_t: np.ndarray):
        if self.use_torch:
            # mB_t is numpy; convert and operate on torch buffer
            mt = torch.as_tensor(mB_t, dtype=self.torch_dtype, device=self.torch_device)
            self.xbuf[:, 1:] = self.xbuf[:, :-1]
            self.xbuf[:, 0] = mt
            return self.xbuf.reshape(-1)
        else:
            self.xbuf[:, 1:] = self.xbuf[:, :-1]
            self.xbuf[:, 0] = mB_t
            return self.xbuf.reshape(-1)  # [QB*L]

    def process(self, mB: np.ndarray, mA: np.ndarray, mA_fusnet: np.ndarray):
        qb, T = mB.shape
        qa, T2 = mA.shape
        assert qb == self.qb
        assert qa == self.qa
        assert T == T2 == mA_fusnet.shape[1]

        mA_hat = np.zeros_like(mA, dtype=np.float32)
        delta_hat = np.zeros_like(mA, dtype=np.float32)
        err_all = np.zeros_like(mA, dtype=np.float32)

        if self.use_torch:
            I = torch.eye(self.n, dtype=self.torch_dtype, device=self.torch_device)
            for t in range(T):
                x = self._update_buffer(mB[:, t])

                # 1) state prediction
                self.w *= self.transition
                self.P = (self.transition ** 2) * self.P
                for a in range(self.qa):
                    self.P[a] = self.P[a] + I * self.q

                # 2) predicted output and innovation
                y_delta = torch.mv(self.w, x)
                y_hat = torch.as_tensor(mA_fusnet[:, t], dtype=self.torch_dtype, device=self.torch_device) + y_delta
                e = torch.as_tensor(mA[:, t], dtype=self.torch_dtype, device=self.torch_device) - y_hat

                # 3) Kalman update per target mic
                for a in range(self.qa):
                    P = self.P[a]
                    Px = P @ x
                    S = float((x @ Px).cpu().item() + self.r)
                    denom = S if S > 1e-12 else 1e-12
                    K = Px / denom

                    self.w[a] = self.w[a] + K * e[a]
                    self.P[a] = (I - torch.ger(K, x)) @ P

                # output after update
                y_delta = torch.mv(self.w, x)
                y_hat = torch.as_tensor(mA_fusnet[:, t], dtype=self.torch_dtype, device=self.torch_device) + y_delta
                e_final = torch.as_tensor(mA[:, t], dtype=self.torch_dtype, device=self.torch_device) - y_hat

                delta_hat[:, t] = y_delta.cpu().numpy().astype(np.float32)
                mA_hat[:, t] = y_hat.cpu().numpy().astype(np.float32)
                err_all[:, t] = e_final.cpu().numpy().astype(np.float32)
        else:
            I = np.eye(self.n, dtype=self.dtype)

            for t in range(T):
                x = self._update_buffer(mB[:, t].astype(self.dtype))

                # 1) state prediction
                self.w *= self.transition
                self.P = (self.transition ** 2) * self.P
                for a in range(self.qa):
                    self.P[a] += I * self.q

                # 2) predicted output and innovation
                y_delta = np.einsum("an,n->a", self.w, x)
                y_hat = mA_fusnet[:, t].astype(self.dtype) + y_delta
                e = mA[:, t].astype(self.dtype) - y_hat

                # 3) Kalman update per target mic
                for a in range(self.qa):
                    P = self.P[a]
                    Px = P @ x
                    S = float(x @ Px + self.r)
                    K = Px / max(S, 1e-12)

                    self.w[a] = self.w[a] + K * e[a]
                    self.P[a] = (I - np.outer(K, x)) @ P

                # output after update
                y_delta = np.einsum("an,n->a", self.w, x)
                y_hat = mA_fusnet[:, t].astype(self.dtype) + y_delta
                e_final = mA[:, t].astype(self.dtype) - y_hat

                delta_hat[:, t] = y_delta.astype(np.float32)
                mA_hat[:, t] = y_hat.astype(np.float32)
                err_all[:, t] = e_final.astype(np.float32)

        return mA_hat, delta_hat, err_all

    def get_correction_filters(self):
        return self.w.reshape(self.qa, self.qb, self.L).copy()
