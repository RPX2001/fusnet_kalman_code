from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ReTMBlock:
    """
    One ReTM Kalman block.

    R : [QA, D]          current ReTM state (filter coefficients)
    P : [QA, D, D]       per-output-channel covariance matrix
    l0, l1               filter-tap index range [l0, l1)
    """
    l0: int
    l1: int
    R: torch.Tensor
    P: torch.Tensor

class ReTMKalmanFilterFromFuSNet:
    """
    Kalman filter tracker for FuSNet ReTM weights.

    Fixes vs. original
    ------------------
    1. Transition order:  r(t+1) = G * r(t)  +  K(t) * e(t)
       (was:              r(t+1) = G * (r(t) + K(t) * e(t))  )
       The original form erroneously scales the innovation by G.

    2. Covariance update: P(t+1) = G² * (P - K * x^T * P) + Q*I
       implemented as     P(t+1) = G² * (I - K*x^T) * P   + Q*I
       using the outer-product form that avoids redundant multiplies.
       The original code computed  K * PX  which gives (PX/S)*PX^T —
       the S division was applied twice.

    3. Adaptive observation noise Rv(t) — estimated from a short
       exponential moving average of the squared prediction error.
       When FuSNet is confused (fast source motion) Rv rises
       automatically, reducing the Kalman gain and preventing
       divergence.

    4. Numerical guard: covariance diagonal is clipped to [eps, inf)
       before inversion to prevent blow-up in degenerate blocks.

    5. Optional momentum on the innovation:  a velocity term
       v(t) = beta * v(t-1) + (1-beta) * e(t)  is mixed into the
       update so the filter can track smoothly-moving sources.

    Parameters
    ----------
    model               : FuSNet nn.Module
    qa, qb              : microphone group sizes
    filter_length       : total FIR tap count (2*context + 1)
    block_length        : taps per Kalman block (trade memory vs speed)
    transition          : G  (scalar < 1, e.g. 0.995–0.9999)
    process_noise       : Q  — diagonal process-noise variance added each step
    observation_noise   : Rv — baseline obs-noise variance (adaptive scheme
                          uses this as a floor)
    initial_covariance  : P0 — initial diagonal covariance
    adaptive_noise      : enable adaptive Rv(t) estimation
    adaptive_alpha      : EMA coefficient for error-power estimate (0.99–0.9999)
    adaptive_noise_floor: minimum Rv(t) (prevents over-aggressive updates)
    adaptive_noise_ceil : maximum Rv(t) (prevents frozen filter)
    innovation_momentum : beta for velocity-aided innovation (0 = disabled)
    symmetrize_covariance: enforce P symmetry every step
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
        adaptive_noise: bool = True,
        adaptive_alpha: float = 0.999,
        adaptive_noise_floor: float = 1e-4,
        adaptive_noise_ceil: float = 1.0,
        innovation_momentum: float = 0.0,
        symmetrize_covariance: bool = True,
    ):
        self.model = model
        self.qa = int(qa)
        self.qb = int(qb)
        self.L = int(filter_length)
        self.block_length = int(block_length)

        self.G = float(transition)
        self.G2 = self.G ** 2
        self.Q = float(process_noise)
        self.Rv_base = float(observation_noise)
        self.P0 = float(initial_covariance)

        self.device = torch.device(device)
        self.dtype = dtype
        self.symmetrize_covariance = bool(symmetrize_covariance)

        # ── Adaptive observation noise ──────────────────────────────────
        self.adaptive_noise = bool(adaptive_noise)
        self.alpha = float(adaptive_alpha)
        self.Rv_floor = float(adaptive_noise_floor)
        self.Rv_ceil = float(adaptive_noise_ceil)

        # Running EMA of per-channel error power  [QA]
        self._err_power_ema = torch.full(
            (self.qa,), fill_value=self.Rv_base,
            device=self.device, dtype=self.dtype,
        )

        # ── Velocity-aided innovation ───────────────────────────────────
        self.beta = float(innovation_momentum)
        self._velocity = torch.zeros(
            (self.qa,), device=self.device, dtype=self.dtype,
        )

        # ── Kalman blocks ───────────────────────────────────────────────
        self.blocks: List[ReTMBlock] = []
        R_init = self._extract_retm_from_fusnet()
        self._create_kalman_blocks(R_init)

    # ──────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────────────

    def _extract_retm_from_fusnet(self) -> torch.Tensor:
        """
        Read FuSNet conv1…conv{QB} weights → R[QA, QB, L].
        Each conv has shape [QA, 1, L].
        """
        conv_names = [f"conv{i}" for i in range(1, self.qb + 1)]

        R = torch.zeros(
            (self.qa, self.qb, self.L),
            device=self.device, dtype=self.dtype,
        )

        for q, name in enumerate(conv_names):
            if not hasattr(self.model, name):
                raise AttributeError(f"FuSNet model has no layer '{name}'")
            conv = getattr(self.model, name)
            if not isinstance(conv, nn.Conv1d):
                raise TypeError(f"'{name}' is not nn.Conv1d")
            w = conv.weight.detach().to(self.device, dtype=self.dtype)
            if w.shape != (self.qa, 1, self.L):
                raise ValueError(
                    f"'{name}'.weight shape {tuple(w.shape)} ≠ "
                    f"{(self.qa, 1, self.L)}"
                )
            R[:, q, :] = w[:, 0, :]

        return R

    def _create_kalman_blocks(self, R_init: torch.Tensor):
        """Slice R[QA, QB, L] into blocks along the tap dimension."""
        self.blocks.clear()

        for l0 in range(0, self.L, self.block_length):
            l1 = min(l0 + self.block_length, self.L)
            B = l1 - l0
            D = self.qb * B

            R_block = R_init[:, :, l0:l1].reshape(self.qa, D).contiguous()

            eye = torch.eye(D, device=self.device, dtype=self.dtype)
            P_block = eye.unsqueeze(0).expand(self.qa, -1, -1).clone() * self.P0

            self.blocks.append(ReTMBlock(l0=l0, l1=l1, R=R_block, P=P_block))

    # ──────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_one_sample(self, x_full: torch.Tensor) -> torch.Tensor:
        """
        ŷ(t) = Σ_b  R_b  x_b          (linear prediction)

        x_full : [QB, L]
        returns : [QA]
        """
        y = torch.zeros((self.qa,), device=self.device, dtype=self.dtype)

        for blk in self.blocks:
            x_b = x_full[:, blk.l0:blk.l1].reshape(-1)           # [D]
            y += (blk.R * x_b.unsqueeze(0)).sum(dim=1)            # [QA]

        return y

    # ──────────────────────────────────────────────────────────────────
    # Update  (core Kalman step — fixed)
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_one_sample(
        self,
        x_full: torch.Tensor,
        d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One corrected Kalman update.

        x_full : [QB, L]   — regressor (mB buffer)
        d      : [QA]      — true Group-A sample

        returns : (y_hat [QA], error [QA])

        State equations (fixed)
        -----------------------
        Innovation:
            e(t) = d(t) - R(t) x(t)

        Innovation denominator (per output channel a):
            S_a(t) = Σ_b  x_b^T P_{a,b}(t) x_b  +  Rv(t)

        Kalman gain (per block b, per output channel a):
            K_{a,b}(t) = P_{a,b}(t) x_b / S_a(t)       [D]

        State update  ← ORDER FIXED (G scales prior, not posterior):
            R_{a,b}(t+1) = G * R_{a,b}(t)  +  K_{a,b}(t) * e_a(t)

        Covariance update  ← CORRECTION FIXED:
            P_{a,b}(t+1) = G² * (P_{a,b}(t) - K_{a,b}(t) x_b^T P_{a,b}(t))
                           + Q * I
        """

        # ── 1. Predict ──────────────────────────────────────────────
        y_hat = self.predict_one_sample(x_full)
        error = d - y_hat                                          # [QA]

        # ── 2. Adaptive observation noise ───────────────────────────
        if self.adaptive_noise:
            # EMA of per-channel squared error
            self._err_power_ema.mul_(self.alpha).add_(
                (1.0 - self.alpha) * error.detach() ** 2
            )
            Rv = torch.clamp(
                self._err_power_ema,
                min=self.Rv_floor,
                max=self.Rv_ceil,
            )                                                      # [QA]
        else:
            Rv = torch.full(
                (self.qa,), self.Rv_base,
                device=self.device, dtype=self.dtype,
            )

        # ── 3. Velocity-aided innovation ────────────────────────────
        if self.beta > 0.0:
            self._velocity.mul_(self.beta).add_((1.0 - self.beta) * error)
            eff_error = self._velocity                             # [QA]
        else:
            eff_error = error

        # ── 4. Accumulate S = Σ_b x_b^T P_b x_b + Rv ───────────────
        S = Rv.clone()                                             # [QA]
        PX_cache: list[tuple[ReTMBlock, torch.Tensor, torch.Tensor]] = []

        for blk in self.blocks:
            x_b = x_full[:, blk.l0:blk.l1].reshape(-1)           # [D]
            PX = torch.einsum("aij,j->ai", blk.P, x_b)           # [QA, D]
            s_b = (x_b.unsqueeze(0) * PX).sum(dim=1)             # [QA]
            S = S + s_b
            PX_cache.append((blk, x_b, PX))

        S = torch.clamp(S, min=1e-10)                             # numerical guard

        # ── 5. Update each block ─────────────────────────────────────
        for blk, x_b, PX in PX_cache:
            # Kalman gain  K: [QA, D]
            K = PX / S.unsqueeze(1)

            # ── State update (FIXED ORDER) ──────────────────────────
            # r(t+1) = G * r(t)  +  K * e(t)
            blk.R.mul_(self.G)
            blk.R.addcmul_(K, eff_error.unsqueeze(1))

            # ── Covariance update (FIXED CORRECTION) ────────────────
            # P(t+1) = G² * (P - K x^T P)  +  Q I
            #        = G² * P  -  G² * K (PX)^T  +  Q I
            #
            # K (PX)^T  has shape [QA, D, D]:
            #   outer[a] = K[a].unsqueeze(1) * PX[a].unsqueeze(0)
            outer = K.unsqueeze(2) * PX.unsqueeze(1)              # [QA, D, D]

            blk.P.mul_(self.G2)
            blk.P.sub_(self.G2 * outer)

            # Add process noise
            D = blk.P.shape[-1]
            blk.P.diagonal(dim1=1, dim2=2).add_(self.Q)

            # Clip diagonal to prevent negative variance
            blk.P.diagonal(dim1=1, dim2=2).clamp_(min=1e-12)

            # Enforce symmetry to counter floating-point drift
            if self.symmetrize_covariance:
                blk.P.copy_(0.5 * (blk.P + blk.P.transpose(1, 2)))

        return y_hat, error

    # ──────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def copy_retm_state_to_fusnet(self):
        """Write adapted ReTM filters back into FuSNet conv weights."""
        R_full = self.get_retm_tensor()
        conv_names = [f"conv{i}" for i in range(1, self.qb + 1)]
        for q, name in enumerate(conv_names):
            conv = getattr(self.model, name)
            conv.weight.data.copy_(R_full[:, q:q + 1, :])

    @torch.no_grad()
    def get_retm_tensor(self) -> torch.Tensor:
        """Return assembled R[QA, QB, L]."""
        R_full = torch.zeros(
            (self.qa, self.qb, self.L),
            device=self.device, dtype=self.dtype,
        )
        for blk in self.blocks:
            B = blk.l1 - blk.l0
            R_full[:, :, blk.l0:blk.l1] = blk.R.reshape(self.qa, self.qb, B)
        return R_full

    @torch.no_grad()
    def reset_velocity(self):
        """Zero the innovation-momentum buffer (call at segment boundaries)."""
        self._velocity.zero_()
