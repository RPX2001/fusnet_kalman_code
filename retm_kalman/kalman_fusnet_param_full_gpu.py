from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn


class FullGPUFUSENetParameterKalman:
    """
    GPU diagonal Kalman-style adaptation of ALL FuSNet parameters.

    State:
        r(t) = all FuSNet trainable parameters

    Model:
        mA_hat(t) = FuSNet_{r(t)}(mB(t))

    Error:
        e(t) = mA(t) - mA_hat(t)

    Practical update:
        Prediction:
            r_minus = r0 + G * (r - r0)
            P_minus = G^2 P + Q

        Update:
            loss = 0.5 * mean(e^2)
            grad = d(loss) / d(r)

            K_diag = P_minus / (P_minus + Rv)
            r_new = r_minus - kalman_lr * K_diag * grad

            P_new = (1 - K_diag) * P_minus

    Notes:
        - All parameters are updated.
        - All tensors stay on GPU if the model is on GPU.
        - r0 is the original checkpoint parameter value.
        - Using r0-centered transition avoids destroying pretrained weights.
    """

    def __init__(
        self,
        model: nn.Module,
        transition: float = 0.999,
        process_noise: float = 1e-8,
        observation_noise: float = 1e-2,
        initial_covariance: float = 1e-3,
        kalman_lr: float = 0.1,
        max_grad_norm: Optional[float] = 1.0,
        transition_center: str = "initial",
    ):
        self.model = model

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
        """
        Predict all FuSNet parameters on GPU.
        """
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
        """
        Backpropagate observation error and update all FuSNet parameters on GPU.
        """
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

    def step(self, x_frame: torch.Tensor, d_frame: torch.Tensor):
        """
        One full Kalman-style update.

        Args:
            x_frame: [B, 8, window_size]
            d_frame: [B, 5, output_length]

        Returns:
            y_hat, error, loss
        """
        self.predict_parameters()

        y_hat = self.model(x_frame)

        if y_hat.shape != d_frame.shape:
            raise RuntimeError(
                f"FuSNet output shape {tuple(y_hat.shape)} does not match "
                f"target shape {tuple(d_frame.shape)}"
            )

        error = d_frame - y_hat
        loss = 0.5 * torch.mean(error ** 2)

        self.update_from_loss(loss)

        return y_hat.detach(), error.detach(), float(loss.detach().cpu().item())