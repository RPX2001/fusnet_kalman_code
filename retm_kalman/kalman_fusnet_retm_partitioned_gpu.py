from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class ReTMBlock:
    """
    One partitioned ReTM block.

    R:
        [QA, D]
        where D = QB * block_length

    P:
        [QA, D, D]
        full covariance matrix for this block and each target mic.
    """
    l0: int
    l1: int
    R: torch.Tensor
    P: torch.Tensor


class PartitionedBlockReTMKalmanFromFuSNet:
    """
    No-backprop Kalman update for FuSNet13 convolution weights.

    This implements the document-style ReTM Kalman update.

    Observation model:
        mA_hat(t) = R_AB(t) mB(t)

    Error:
        e(t) = mA(t) - mA_hat(t)

    Kalman gain:
        K_b(t) = P_b(t) x_b(t)^T
                 [ sum_b x_b(t) P_b(t) x_b(t)^T + Phi_v ]^{-1}

    State update:
        r_b(t+1) = G * ( r_b(t) + K_b(t) e(t) )

    Covariance update:
        P_b(t+1) = G^2 * [ P_b(t) - K_b(t) x_b(t) P_b(t) ] + Phi_A I

    Notes:
        - No backpropagation is used.
        - FuSNet convolution weights are used as the initial ReTM state.
        - The model is linear, so the ReTM state can be updated explicitly.
        - Partitioned-block covariance is used instead of one impossible full covariance.
    """

    def __init__(
        self,
        model: nn.Module,
        qa: int = 5,
        qb: int = 8,
        filter_length: int = 8193,
        block_length: int = 64,
        transition: float = 0.999,
        process_noise: float = 1e-8,
        observation_noise: float = 1e-2,
        initial_covariance: float = 1e-3,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        symmetrize_covariance: bool = True,
    ):
        self.model = model
        self.qa = int(qa)
        self.qb = int(qb)
        self.L = int(filter_length)
        self.block_length = int(block_length)

        self.G = float(transition)
        self.Q = float(process_noise)
        self.Rv = float(observation_noise)
        self.P0 = float(initial_covariance)

        self.device = torch.device(device)
        self.dtype = dtype
        self.symmetrize_covariance = bool(symmetrize_covariance)

        self.blocks: List[ReTMBlock] = []

        R_init = self._extract_retm_from_fusnet()

        self._create_partitioned_blocks(R_init)

    def _extract_retm_from_fusnet(self) -> torch.Tensor:
        """
        Extract FuSNet13 convolution weights as ReTM.

        FuSNet13 has conv1 ... conv8.
        Each conv has weight shape:
            [5, 1, L]

        We build:
            R[QA, QB, L]
        """

        conv_names = [
            "conv1", "conv2", "conv3", "conv4",
            "conv5", "conv6", "conv7", "conv8",
        ]

        R = torch.zeros(
            (self.qa, self.qb, self.L),
            device=self.device,
            dtype=self.dtype,
        )

        for q, name in enumerate(conv_names):
            if not hasattr(self.model, name):
                raise AttributeError(f"FuSNet model does not have layer {name}")

            conv = getattr(self.model, name)

            if not isinstance(conv, nn.Conv1d):
                raise TypeError(f"{name} is not nn.Conv1d")

            w = conv.weight.detach().to(self.device, dtype=self.dtype)

            if w.shape != (self.qa, 1, self.L):
                raise ValueError(
                    f"{name}.weight has shape {tuple(w.shape)}, "
                    f"expected {(self.qa, 1, self.L)}"
                )

            R[:, q, :] = w[:, 0, :]

        return R

    def _create_partitioned_blocks(self, R_init: torch.Tensor):
        """
        Partition R[QA, QB, L] along filter length.

        For each block:
            R_block: [QA, QB * block_len]
            P_block: [QA, D, D]
        """

        self.blocks.clear()

        for l0 in range(0, self.L, self.block_length):
            l1 = min(l0 + self.block_length, self.L)
            B = l1 - l0
            D = self.qb * B

            R_block = R_init[:, :, l0:l1].reshape(self.qa, D).contiguous()

            eye = torch.eye(
                D,
                device=self.device,
                dtype=self.dtype,
            )

            P_block = eye.unsqueeze(0).repeat(self.qa, 1, 1) * self.P0

            self.blocks.append(
                ReTMBlock(
                    l0=l0,
                    l1=l1,
                    R=R_block,
                    P=P_block,
                )
            )

    @torch.no_grad()
    def predict_one_sample(
        self,
        x_full: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict mA_hat(t) using current ReTM state.

        x_full:
            [QB, L]

        returns:
            y_hat [QA]
        """

        y = torch.zeros(
            (self.qa,),
            device=self.device,
            dtype=self.dtype,
        )

        for blk in self.blocks:
            x_b = x_full[:, blk.l0:blk.l1].reshape(-1)

            y = y + torch.sum(blk.R * x_b.unsqueeze(0), dim=1)

        return y

    @torch.no_grad()
    def update_one_sample(
        self,
        x_full: torch.Tensor,
        d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One document-style Kalman update.

        x_full:
            [QB, L]

        d:
            true Group-A sample, shape [QA]

        returns:
            y_hat [QA]
            error [QA]
        """

        # ----------------------------------------------------
        # 1. Current output estimate
        # ----------------------------------------------------

        y_hat = self.predict_one_sample(x_full)
        error = d - y_hat

        # ----------------------------------------------------
        # 2. Compute denominator S for each output channel
        #
        # S_a = sum_b x_b^T P_{a,b} x_b + Phi_v
        # ----------------------------------------------------

        S = torch.full(
            (self.qa,),
            fill_value=self.Rv,
            device=self.device,
            dtype=self.dtype,
        )

        PX_cache = []

        for blk in self.blocks:
            x_b = x_full[:, blk.l0:blk.l1].reshape(-1)

            # PX: [QA, D]
            PX = torch.matmul(blk.P, x_b)

            # x^T P x for each target channel: [QA]
            s_b = torch.sum(x_b.unsqueeze(0) * PX, dim=1)

            S = S + s_b
            PX_cache.append((blk, x_b, PX))

        S = torch.clamp(S, min=1e-12)

        # ----------------------------------------------------
        # 3. Update each partitioned block
        #
        # K_b = P_b x_b / S
        # r_b(t+1) = G * (r_b(t) + K_b e)
        # ----------------------------------------------------

        for blk, x_b, PX in PX_cache:
            # K: [QA, D]
            K = PX / S.unsqueeze(1)

            # r(t+1) = G * (r(t) + K e)
            blk.R.add_(K * error.unsqueeze(1))
            blk.R.mul_(self.G)

            # P update:
            # P = G^2 * (P - K (x^T P)) + Q I
            #
            # Since P is symmetric, x^T P = (P x)^T = PX^T.
            correction = K.unsqueeze(2) * PX.unsqueeze(1)

            blk.P.sub_(correction)
            blk.P.mul_(self.G ** 2)

            D = blk.P.shape[-1]
            eye = torch.eye(
                D,
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0)

            blk.P.add_(self.Q * eye)

            if self.symmetrize_covariance:
                blk.P.copy_(0.5 * (blk.P + blk.P.transpose(1, 2)))

        return y_hat, error

    @torch.no_grad()
    def copy_retm_state_to_fusnet(self):
        """
        Copy the adapted ReTM state back into FuSNet conv weights.

        This lets you save the adapted FuSNet checkpoint.
        """

        R_full = torch.zeros(
            (self.qa, self.qb, self.L),
            device=self.device,
            dtype=self.dtype,
        )

        for blk in self.blocks:
            B = blk.l1 - blk.l0
            R_full[:, :, blk.l0:blk.l1] = blk.R.reshape(self.qa, self.qb, B)

        conv_names = [
            "conv1", "conv2", "conv3", "conv4",
            "conv5", "conv6", "conv7", "conv8",
        ]

        for q, name in enumerate(conv_names):
            conv = getattr(self.model, name)
            conv.weight.data.copy_(R_full[:, q:q + 1, :])

    @torch.no_grad()
    def get_retm_tensor(self) -> torch.Tensor:
        """
        Return adapted ReTM tensor:
            [QA, QB, L]
        """

        R_full = torch.zeros(
            (self.qa, self.qb, self.L),
            device=self.device,
            dtype=self.dtype,
        )

        for blk in self.blocks:
            B = blk.l1 - blk.l0
            R_full[:, :, blk.l0:blk.l1] = blk.R.reshape(self.qa, self.qb, B)

        return R_full