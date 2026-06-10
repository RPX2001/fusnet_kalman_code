from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class BlockGPUFUSENetParameterKalman:
    """
    GPU block diagonal Kalman-style adaptation of ALL FuSNet parameters.

    Block version:
        - Prediction is done once per block.
        - Several FuSNet frames are processed as a mini-batch.
        - One Kalman-style parameter update is done using the average block loss.

    This is faster and usually more stable than updating every frame.
    """

    def __init__(
        self,
        model: nn.Module,
        block_frames: int = 4,
        transition: float = 0.999,
        process_noise: float = 1e-8,
        observation_noise: float = 1e-2,
        initial_covariance: float = 1e-3,
        kalman_lr: float = 0.1,
        max_grad_norm: Optional[float] = 1.0,
        transition_center: str = "initial",
    ):
        self.model = model
        self.block_frames = int(block_frames)

        self.G = float(transition)
        self.Q = float(process_noise)
        self.Rv = float(observation_noise)
        self.P0 = float(initial_covariance)
        self.kalman_lr = float(kalman_lr)
        self.max_grad_norm = max_grad_norm

        if transition_center not in {"initial", "zero"}:
            raise ValueError("transition_center must be 'initial' or 'zero'")

        self.transition_center = transition_center

        self.P: Dict[str, torch.Tensor] = {}
        self.r0: Dict[str, torch.Tensor] = {}

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            self.P[name] = torch.full_like(
                p.data,
                fill_value=self.P0,
                device=p.device,
                dtype=p.dtype,
            )

            self.r0[name] = p.data.detach().clone()

    @torch.no_grad()
    def predict_parameters(self):
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            if self.transition_center == "initial":
                center = self.r0[name]
                p.data.copy_(center + self.G * (p.data - center))
            else:
                p.data.mul_(self.G)

            self.P[name].mul_(self.G ** 2).add_(self.Q)

    def update_from_loss(self, loss: torch.Tensor):
        self.model.zero_grad(set_to_none=True)

        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float(self.max_grad_norm),
            )

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue

                if p.grad is None:
                    continue

                grad = p.grad
                P_pred = self.P[name]

                K_diag = P_pred / (P_pred + self.Rv)

                p.data.add_(-self.kalman_lr * K_diag * grad)

                self.P[name] = torch.clamp(
                    (1.0 - K_diag) * P_pred,
                    min=1e-12,
                )

    def step_block(self, x_block: torch.Tensor, d_block: torch.Tensor):
        """
        One block Kalman-style update.

        Args:
            x_block: [B, 8, window_size]
            d_block: [B, 5, output_length]

        Returns:
            y_hat, error, loss
        """
        self.predict_parameters()

        y_hat = self.model(x_block)

        if y_hat.shape != d_block.shape:
            raise RuntimeError(
                f"FuSNet output shape {tuple(y_hat.shape)} does not match "
                f"target shape {tuple(d_block.shape)}"
            )

        error = d_block - y_hat
        loss = 0.5 * torch.mean(error ** 2)

        self.update_from_loss(loss)

        return y_hat.detach(), error.detach(), float(loss.detach().cpu().item())