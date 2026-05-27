from __future__ import annotations
import numpy as np


class BlockKalmanCorrectionReTM:
    """
    Block Kalman correction ReTM estimator.

    Same correction model as FullKalmanCorrectionReTM, but the state is partitioned
    into smaller blocks. This is much lighter for long FIR filters.
    """

    def __init__(self,
                 qb: int = 4,
                 qa: int = 3,
                 L: int = 1024,
                 block_size: int = 128,
                 transition: float = 0.995,
                 process_noise: float = 1e-7,
                 observation_noise: float = 1e-3,
                 initial_covariance: float = 1e-2,
                 dtype=np.float64):
        self.qb = qb
        self.qa = qa
        self.L = L
        self.n = qb * L
        self.block_size = int(block_size)
        self.transition = transition
        self.q = process_noise
        self.r = observation_noise
        self.dtype = dtype

        self.slices = []
        start = 0
        while start < self.n:
            end = min(start + self.block_size, self.n)
            self.slices.append(slice(start, end))
            start = end

        self.w = np.zeros((qa, self.n), dtype=dtype)

        self.P = []
        for _a in range(qa):
            Pa = []
            for sl in self.slices:
                dim = sl.stop - sl.start
                Pa.append(np.eye(dim, dtype=dtype) * initial_covariance)
            self.P.append(Pa)

        self.xbuf = np.zeros((qb, L), dtype=dtype)

    def _update_buffer(self, mB_t: np.ndarray):
        self.xbuf[:, 1:] = self.xbuf[:, :-1]
        self.xbuf[:, 0] = mB_t
        return self.xbuf.reshape(-1)

    def process(self, mB: np.ndarray, mA: np.ndarray, mA_fusnet: np.ndarray):
        qb, T = mB.shape
        qa, T2 = mA.shape
        assert qb == self.qb
        assert qa == self.qa
        assert T == T2 == mA_fusnet.shape[1]

        mA_hat = np.zeros_like(mA, dtype=np.float32)
        delta_hat = np.zeros_like(mA, dtype=np.float32)
        err_all = np.zeros_like(mA, dtype=np.float32)

        eye_blocks = [np.eye(sl.stop - sl.start, dtype=self.dtype) for sl in self.slices]

        for t in range(T):
            x = self._update_buffer(mB[:, t].astype(self.dtype))

            # 1) block state prediction
            self.w *= self.transition
            for a in range(self.qa):
                for b, sl in enumerate(self.slices):
                    self.P[a][b] = (self.transition ** 2) * self.P[a][b] + eye_blocks[b] * self.q

            # 2) predicted output and innovation
            y_delta = np.einsum("an,n->a", self.w, x)
            y_hat = mA_fusnet[:, t].astype(self.dtype) + y_delta
            e = mA[:, t].astype(self.dtype) - y_hat

            # 3) block-coordinate Kalman update
            for a in range(self.qa):
                for b, sl in enumerate(self.slices):
                    xb = x[sl]
                    P = self.P[a][b]
                    Pxb = P @ xb
                    S = float(xb @ Pxb + self.r)
                    Kb = Pxb / max(S, 1e-12)

                    self.w[a, sl] = self.w[a, sl] + Kb * e[a]
                    self.P[a][b] = (eye_blocks[b] - np.outer(Kb, xb)) @ P

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
