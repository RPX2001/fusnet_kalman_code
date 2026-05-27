from __future__ import annotations
import numpy as np


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
                 dtype=np.float64):
        self.qb = qb
        self.qa = qa
        self.L = L
        self.n = qb * L
        self.transition = transition
        self.q = process_noise
        self.r = observation_noise
        self.dtype = dtype

        self.w = np.zeros((qa, self.n), dtype=dtype)
        self.P = np.stack([np.eye(self.n, dtype=dtype) * initial_covariance for _ in range(qa)], axis=0)
        self.xbuf = np.zeros((qb, L), dtype=dtype)

    def _update_buffer(self, mB_t: np.ndarray):
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
