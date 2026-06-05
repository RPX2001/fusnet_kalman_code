from __future__ import annotations

import math
import numpy as np
import torch


class BlockKalmanCorrectionReTM:
    """
    Partitioned Block Frequency-Domain Kalman Filter for FuSNet + ReTM correction.

    Correction model:

        residual_target(t) = mA(t) - mA_fusnet(t)

        delta_mA_hat(t) = R_delta(t) * mB(t)

        mA_hat(t) = mA_fusnet(t) + delta_mA_hat(t)

    Inputs to process():
        mB        : [QB, T]
        mA        : [QA, T]
        mA_fusnet : [QA, T]

    Outputs:
        mA_hat    : [QA, T]
        delta_hat : [QA, T]
        err_all   : [QA, T]

    Important:
        Here block_size is used as hop_len.
        L must be an integer multiple of block_size.

        Example:
            L = 1024
            block_size = 256
            n_block = 4
            fft_len = 512
    """

    def __init__(
        self,
        qb: int = 4,
        qa: int = 3,
        L: int = 1024,
        block_size: int = 256,
        transition: float = 0.999,
        process_noise: float = 1e-7,
        observation_noise: float = 1e-3,
        initial_covariance: float = 1.6e-2,
        alpha_e: float = 0.7,
        beta_f: float = 0.1,
        dtype=np.float32,
        device: str | None = None,
        eps: float = 1e-10,
    ):
        self.qb = int(qb)
        self.qa = int(qa)
        self.L = int(L)
        self.block_size = int(block_size)

        if self.L % self.block_size != 0:
            raise ValueError("L must be divisible by block_size.")

        self.hop_len = self.block_size
        self.n_block = self.L // self.hop_len
        self.fft_len = 2 * self.hop_len
        self.n_bin = self.fft_len // 2 + 1

        self.transition = float(transition)
        self.q = float(process_noise)
        self.r = float(observation_noise)
        self.initial_covariance = float(initial_covariance)
        self.alpha_e = float(alpha_e)
        self.beta_f = float(beta_f)
        self.eps = float(eps)

        self.dtype_np = dtype
        self.torch_dtype = torch.float64 if dtype == np.float64 else torch.float32
        self.complex_dtype = torch.complex128 if dtype == np.float64 else torch.complex64

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self.reset_states()

    def reset_states(self):
        self.phi_e = None
        self.phi_f = None
        self.phi_n = None
        self.H_prior = None
        self.H_post = None
        self.P = None

    def _init_states(self):
        self.P = torch.full(
            (self.n_bin, self.n_block, self.qb, self.qa),
            self.initial_covariance,
            dtype=self.torch_dtype,
            device=self.device,
        )

        # Same idea as AFCSPEX: latest partition gets larger covariance.
        # In this implementation, block index n_block-1 is the newest partition.
        if self.n_block == 4:
            self.P[:, 0, :, :] = 0.1
            self.P[:, 1, :, :] = 0.2
            self.P[:, 2, :, :] = 0.4
            self.P[:, 3, :, :] = 0.8

        self.phi_e = torch.zeros(
            self.n_bin,
            self.qa,
            dtype=self.torch_dtype,
            device=self.device,
        )

        self.phi_f = torch.zeros(
            self.n_bin,
            self.n_block,
            self.qb,
            self.qa,
            dtype=self.torch_dtype,
            device=self.device,
        )

        self.phi_n = torch.zeros_like(self.phi_f)

        self.H_prior = torch.zeros(
            self.n_bin,
            self.n_block,
            self.qb,
            self.qa,
            dtype=self.complex_dtype,
            device=self.device,
        )

        self.H_post = torch.zeros_like(self.H_prior)

    def _to_torch_signal(self, x: np.ndarray, expected_channels: int, name: str) -> torch.Tensor:
        x = np.asarray(x)

        if x.ndim != 2:
            raise ValueError(f"{name} must have shape [channels, T]. Got {x.shape}")

        if x.shape[0] != expected_channels:
            raise ValueError(
                f"{name} expected {expected_channels} channels, got {x.shape[0]}"
            )

        # [C, T] -> [T, C]
        return torch.as_tensor(
            x.T,
            dtype=self.torch_dtype,
            device=self.device,
        )

    def _make_mB_buffer(self, mB_tc: torch.Tensor, frame_start: int) -> torch.Tensor:
        """
        Make Group-B buffer.

        Args:
            mB_tc:
                [T, QB]

            frame_start:
                start index of current output frame

        Returns:
            buffer:
                [L + hop_len, QB]
        """
        T, QB = mB_tc.shape
        assert QB == self.qb

        buffer_len = self.L + self.hop_len

        start = frame_start - self.L
        end = frame_start + self.hop_len

        valid_start = max(start, 0)
        valid_end = min(end, T)

        chunk = mB_tc[valid_start:valid_end, :]

        if start < 0:
            pad_left = -start
            left = torch.zeros(
                pad_left,
                QB,
                dtype=self.torch_dtype,
                device=self.device,
            )
            chunk = torch.cat([left, chunk], dim=0)

        if chunk.shape[0] < buffer_len:
            pad_right = buffer_len - chunk.shape[0]
            right = torch.zeros(
                pad_right,
                QB,
                dtype=self.torch_dtype,
                device=self.device,
            )
            chunk = torch.cat([chunk, right], dim=0)

        if chunk.shape[0] != buffer_len:
            raise RuntimeError(
                f"Internal buffer error. Expected {buffer_len}, got {chunk.shape[0]}"
            )

        return chunk

    def _make_frame(self, x_tc: torch.Tensor, frame_start: int, channels: int) -> torch.Tensor:
        """
        Make one hop frame.

        Args:
            x_tc:
                [T, C]

        Returns:
            frame:
                [hop_len, C]
        """
        T, C = x_tc.shape
        assert C == channels

        end = min(frame_start + self.hop_len, T)
        frame = x_tc[frame_start:end, :]

        if frame.shape[0] < self.hop_len:
            pad = torch.zeros(
                self.hop_len - frame.shape[0],
                C,
                dtype=self.torch_dtype,
                device=self.device,
            )
            frame = torch.cat([frame, pad], dim=0)

        return frame

    def _stft_mB_buffer(self, mB_buffer: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mB_buffer:
                [L + hop_len, QB]

        Returns:
            X:
                [n_bin, n_block, QB]
        """
        if mB_buffer.shape != (self.L + self.hop_len, self.qb):
            raise ValueError(
                f"mB_buffer must be [{self.L + self.hop_len}, {self.qb}], "
                f"got {tuple(mB_buffer.shape)}"
            )

        # [T, QB] -> [QB, T]
        x = mB_buffer.T.contiguous()

        X = torch.stft(
            x,
            n_fft=self.fft_len,
            hop_length=self.hop_len,
            win_length=self.fft_len,
            window=torch.ones(self.fft_len, dtype=self.torch_dtype, device=self.device),
            center=False,
            return_complex=True,
        )

        # [QB, n_bin, n_block] -> [n_bin, n_block, QB]
        X = X.permute(1, 2, 0).contiguous()

        if X.shape != (self.n_bin, self.n_block, self.qb):
            raise RuntimeError(
                f"Unexpected STFT shape {tuple(X.shape)}. "
                f"Expected {(self.n_bin, self.n_block, self.qb)}"
            )

        return X

    def _rfft_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame:
                [hop_len, C]

        Returns:
            FD frame:
                [n_bin, C]
        """
        if frame.shape[0] != self.hop_len:
            raise ValueError(f"Expected hop_len={self.hop_len}, got {frame.shape[0]}")

        zeros = torch.zeros(
            self.fft_len - self.hop_len,
            frame.shape[1],
            dtype=self.torch_dtype,
            device=self.device,
        )

        padded = torch.cat([zeros, frame], dim=0)
        return torch.fft.rfft(padded, n=self.fft_len, dim=0)

    def _fd_to_valid_hop(self, Y_fd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y_fd:
                [n_bin, C]

        Returns:
            valid time-domain hop:
                [hop_len, C]
        """
        y = torch.fft.irfft(Y_fd, n=self.fft_len, dim=0)
        return y[self.hop_len:, :]

    def _estimate_delta_fd(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X:
                [n_bin, n_block, QB]

            H:
                [n_bin, n_block, QB, QA]

        Returns:
            delta_fd:
                [n_bin, QA]
        """
        return torch.sum(X.unsqueeze(-1) * H, dim=(1, 2))

    def _kalman_gain(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X:
                [n_bin, n_block, QB]

        Returns:
            K:
                [n_bin, n_block, QB, QA]
        """
        self.phi_f = (
            self.beta_f * self.phi_f
            + (1.0 - self.beta_f) * self.H_prior.abs().pow(2)
        )

        # Dynamic process noise estimate + small fixed process noise
        self.phi_n = (1.0 - self.transition ** 2) * self.phi_f + self.q

        X_power = X.abs().pow(2).unsqueeze(-1)  # [n_bin, n_block, QB, 1]

        U = X_power * self.P

        R = U + 2.0 * self.phi_e.unsqueeze(1).unsqueeze(2) + self.r + self.eps

        K = self.P * X.unsqueeze(-1).conj() / R

        # AFCSPEX-style covariance update
        P0 = self.P.detach() - 0.5 * K.detach().abs().pow(2) * R.detach()
        P0 = torch.clamp(P0, min=self.eps)

        self.P = (self.transition ** 2) * P0 + self.phi_n
        self.P = torch.clamp(self.P, min=self.eps)

        return K

    def _process_frame(
        self,
        mB_buffer: torch.Tensor,
        mA_frame: torch.Tensor,
        mA_fusnet_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one hop.

        Args:
            mB_buffer:
                [L + hop_len, QB]

            mA_frame:
                [hop_len, QA]

            mA_fusnet_frame:
                [hop_len, QA]

        Returns:
            mA_hat_frame:
                [hop_len, QA]

            delta_frame:
                [hop_len, QA]

            err_frame:
                [hop_len, QA]
        """
        if self.H_post is None:
            self._init_states()

        X = self._stft_mB_buffer(mB_buffer)

        # residual target = mA - FuSNet estimate
        residual_target = mA_frame - mA_fusnet_frame
        residual_target_fd = self._rfft_frame(residual_target)

        # prediction
        self.H_prior = self.H_post

        delta_prior_fd = self._estimate_delta_fd(X, self.H_prior)
        delta_prior_td = self._fd_to_valid_hop(delta_prior_fd)

        # Re-transform valid hop like AFCSPEX implementation
        delta_prior_fd_valid = self._rfft_frame(delta_prior_td)

        # innovation in frequency domain
        E = residual_target_fd - delta_prior_fd_valid

        self.phi_e = (
            self.alpha_e * self.phi_e
            + (1.0 - self.alpha_e) * E.abs().pow(2)
        )

        K = self._kalman_gain(X)

        dH_td = torch.fft.irfft(
            K * E.unsqueeze(1).unsqueeze(2),
            n=self.fft_len,
            dim=0,
        )

        # Partitioned block constraint:
        # keep only the first hop_len samples of each partition update
        dH_td[self.hop_len:, :, :, :] = 0.0

        dH_fd = torch.fft.rfft(dH_td, n=self.fft_len, dim=0)

        self.H_post = self.transition * (self.H_prior + dH_fd)

        # output after update
        delta_post_fd = self._estimate_delta_fd(X, self.H_post)
        delta_frame = self._fd_to_valid_hop(delta_post_fd)

        mA_hat_frame = mA_fusnet_frame + delta_frame
        err_frame = mA_frame - mA_hat_frame

        return mA_hat_frame, delta_frame, err_frame

    def process(self, mB: np.ndarray, mA: np.ndarray, mA_fusnet: np.ndarray):
        """
        Process full sequence.

        Args:
            mB:
                [QB, T]

            mA:
                [QA, T]

            mA_fusnet:
                [QA, T]

        Returns:
            mA_hat:
                [QA, T]

            delta_hat:
                [QA, T]

            err_all:
                [QA, T]
        """
        mB = np.asarray(mB)
        mA = np.asarray(mA)
        mA_fusnet = np.asarray(mA_fusnet)

        if mB.ndim != 2 or mA.ndim != 2 or mA_fusnet.ndim != 2:
            raise ValueError("mB, mA, and mA_fusnet must have shape [channels, T].")

        qb, T = mB.shape
        qa, T2 = mA.shape

        if qb != self.qb:
            raise ValueError(f"Expected mB with {self.qb} channels, got {qb}")

        if qa != self.qa:
            raise ValueError(f"Expected mA with {self.qa} channels, got {qa}")

        if T != T2 or mA_fusnet.shape != (self.qa, T):
            raise ValueError(
                f"Shape mismatch: mB={mB.shape}, mA={mA.shape}, "
                f"mA_fusnet={mA_fusnet.shape}"
            )

        self.reset_states()

        mB_tc = self._to_torch_signal(mB, self.qb, "mB")
        mA_tc = self._to_torch_signal(mA, self.qa, "mA")
        mF_tc = self._to_torch_signal(mA_fusnet, self.qa, "mA_fusnet")

        mA_hat_tc = torch.zeros_like(mA_tc)
        delta_tc = torch.zeros_like(mA_tc)
        err_tc = torch.zeros_like(mA_tc)

        n_frames = math.ceil(T / self.hop_len)

        for frame_idx in range(n_frames):
            frame_start = frame_idx * self.hop_len
            valid_len = min(self.hop_len, T - frame_start)

            mB_buffer = self._make_mB_buffer(mB_tc, frame_start)
            mA_frame = self._make_frame(mA_tc, frame_start, self.qa)
            mF_frame = self._make_frame(mF_tc, frame_start, self.qa)

            y_frame, d_frame, e_frame = self._process_frame(
                mB_buffer=mB_buffer,
                mA_frame=mA_frame,
                mA_fusnet_frame=mF_frame,
            )

            mA_hat_tc[frame_start:frame_start + valid_len, :] = y_frame[:valid_len, :]
            delta_tc[frame_start:frame_start + valid_len, :] = d_frame[:valid_len, :]
            err_tc[frame_start:frame_start + valid_len, :] = e_frame[:valid_len, :]

        mA_hat = mA_hat_tc.T.detach().cpu().numpy().astype(np.float32)
        delta_hat = delta_tc.T.detach().cpu().numpy().astype(np.float32)
        err_all = err_tc.T.detach().cpu().numpy().astype(np.float32)

        return mA_hat, delta_hat, err_all

    def get_correction_filters(self):
        """
        Return approximate time-domain correction filters.

        Output:
            filters:
                [QA, QB, L]

        The returned filters are ordered newest-to-oldest partition so that
        filters[:, :, 0] corresponds to the most recent/direct part.
        """
        if self.H_post is None:
            return np.zeros((self.qa, self.qb, self.L), dtype=np.float32)

        h_td = torch.fft.irfft(self.H_post, n=self.fft_len, dim=0)
        h_td = h_td[:self.hop_len, :, :, :]  # [hop_len, n_block, QB, QA]

        # Newest partition is n_block-1. Return direct/recent samples first.
        h_td = h_td[:, torch.arange(self.n_block - 1, -1, -1, device=self.device), :, :]

        # [hop_len, n_block, QB, QA] -> [QA, QB, n_block, hop_len]
        h = h_td.permute(3, 2, 1, 0).contiguous()
        h = h.reshape(self.qa, self.qb, self.L)

        return h.detach().cpu().numpy().astype(np.float32)