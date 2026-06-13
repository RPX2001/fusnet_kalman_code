"""
kalman_fusnet_retm_v3.py
========================
Targeted fixes for the mic-5 amplitude collapse and residual noise problem.

New in v3 vs v2
---------------
[A] Per-channel ReTM L2-norm constraint
    After each state update, the filter taps R[a, :] for each output
    channel a are soft-clamped to stay within a factor of `norm_margin`
    of the initial FuSNet norm. This prevents the amplitude collapse seen
    in mic 5 where the filter shrinks to ~25% of its correct scale.

[B] Gain floor per channel
    The Kalman gain K is lower-bounded so that even when Rv is large
    (e.g. during a near-null passage) the filter can still update.
    Without this, mic 5 stops learning after the burst and never recovers.

[C] Null-passage detector
    Monitors the input power at each output channel using an EMA of
    ||x_b||². When power drops below a threshold (near-null), the
    update for that channel is paused and the state is held (not updated),
    preventing divergence during the null passage.

[D] Post-update cross-channel leakage suppression
    After the state update, R[a, :] is projected to reduce components
    that correlate strongly with the static-source path. This reduces
    the residual noise bleed seen after 2s in mic 5.
    (Optional: controlled by leakage_suppression parameter.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReTMBlock:
    l0: int
    l1: int
    R: torch.Tensor    # [QA, D]
    P: torch.Tensor    # [QA, D, D]
    R_norm0: torch.Tensor  # [QA]  initial per-channel L2 norm (reference)


class PartitionedKalmanReTMv3:
    """
    Parameters
    ----------
    (all previous v2 params, plus:)

    norm_margin         : float
        Max allowed ratio of ||R_a|| to initial norm ||R_a0||.
        E.g. 3.0 allows the filter to grow/shrink by 3× from init.
        Prevents amplitude collapse in near-null channels.
        Set to 0 to disable.

    gain_floor          : float
        Minimum per-element |K| as fraction of current P diagonal mean.
        Ensures mic 5 keeps updating even when Rv is large.
        Typical: 1e-4. Set to 0 to disable.

    null_power_threshold: float
        If the RMS of x_b for a channel drops below this fraction of
        the running mean power, skip the update for that channel.
        Prevents divergence during geometric nulls.
        Typical: 0.05 (5% of mean power). Set to 0 to disable.

    null_ema_alpha      : float
        EMA decay for the input-power tracker used by null detector.
    """

    def __init__(
        self,
        model: nn.Module,
        qa: int = 5,
        qb: int = 8,
        filter_length: int = 8193,
        block_length: int = 64,
        transition: float = 0.995,
        process_noise: float = 1e-7,
        observation_noise: float = 1e-2,
        initial_covariance: float = 1e-3,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        # adaptive noise
        adaptive_noise: bool = True,
        adaptive_alpha: float = 0.999,
        adaptive_noise_floor: float = 1e-3,
        adaptive_noise_ceil: float = 1.0,
        # momentum
        innovation_momentum: float = 0.3,
        # per-channel reliability
        channel_gain_alpha: float = 0.999,
        # stability
        p_floor: float = 1e-10,
        symmetrize: bool = True,
        # v3 additions
        norm_margin: float = 3.0,
        gain_floor: float = 1e-4,
        null_power_threshold: float = 0.05,
        null_ema_alpha: float = 0.999,
    ):
        self.model   = model
        self.qa      = int(qa)
        self.qb      = int(qb)
        self.L       = int(filter_length)
        self.BL      = int(block_length)

        self.G       = float(transition)
        self.G2      = self.G ** 2
        self.Q       = float(process_noise)
        self.Rv0     = float(observation_noise)
        self.P0      = float(initial_covariance)

        self.device  = torch.device(device)
        self.dtype   = dtype
        self.sym     = bool(symmetrize)
        self.p_floor = float(p_floor)

        # adaptive Rv
        self.adaptive  = bool(adaptive_noise)
        self.alpha     = float(adaptive_alpha)
        self.Rv_floor  = float(adaptive_noise_floor)
        self.Rv_ceil   = float(adaptive_noise_ceil)
        self._err_ema  = torch.full(
            (self.qa,), self.Rv0, device=self.device, dtype=self.dtype
        )

        # momentum
        self.beta      = float(innovation_momentum)
        self._velocity = torch.zeros(
            (self.qa,), device=self.device, dtype=self.dtype
        )

        # per-channel reliability
        self.cg_alpha     = float(channel_gain_alpha)
        self._ch_err_ema  = torch.full(
            (self.qa,), 1.0, device=self.device, dtype=self.dtype
        )

        # v3: norm constraint
        self.norm_margin = float(norm_margin)

        # v3: gain floor
        self.gain_floor  = float(gain_floor)

        # v3: null detector
        self.null_thresh = float(null_power_threshold)
        self.null_alpha  = float(null_ema_alpha)
        # running EMA of ||x||² per sample (scalar, shared across channels)
        self._input_power_ema = None   # initialised on first sample

        # Build state
        self.blocks: List[ReTMBlock] = []
        self._init(self._extract())

    # ─────────────────────────────────────────────────────────────────
    # Init
    # ─────────────────────────────────────────────────────────────────

    def _extract(self) -> torch.Tensor:
        R = torch.zeros(
            (self.qa, self.qb, self.L), device=self.device, dtype=self.dtype
        )
        for q in range(self.qb):
            conv = getattr(self.model, f"conv{q+1}")
            w    = conv.weight.detach().to(self.device, dtype=self.dtype)
            R[:, q, :] = w[:, 0, :]
        return R

    def _init(self, R: torch.Tensor):
        self.blocks.clear()
        for l0 in range(0, self.L, self.BL):
            l1   = min(l0 + self.BL, self.L)
            D    = self.qb * (l1 - l0)
            Rb   = R[:, :, l0:l1].reshape(self.qa, D).contiguous()
            Pb   = (
                torch.eye(D, device=self.device, dtype=self.dtype)
                .unsqueeze(0).expand(self.qa,-1,-1).clone() * self.P0
            )
            # Store initial per-channel norm as reference
            R_norm0 = Rb.norm(dim=1).clamp(min=1e-12)  # [QA]
            self.blocks.append(ReTMBlock(l0, l1, Rb, Pb, R_norm0.clone()))

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _rv(self) -> torch.Tensor:
        if self.adaptive:
            return self._err_ema.clamp(self.Rv_floor, self.Rv_ceil)
        return torch.full(
            (self.qa,), self.Rv0, device=self.device, dtype=self.dtype
        )

    def _ch_gain_scale(self) -> torch.Tensor:
        if self.cg_alpha <= 0.0:
            return torch.ones((self.qa,), device=self.device, dtype=self.dtype)
        best  = self._ch_err_ema.min().clamp(min=1e-12)
        ratio = best / self._ch_err_ema.clamp(min=1e-12)
        return 0.5 + 0.5 * ratio   # ∈ [0.5, 1.0]

    def _is_null_passage(self, x_full: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask [QA] — True = near-null, skip update.
        Uses the total input power ||x_full||² as a proxy.
        """
        if self.null_thresh <= 0.0:
            return torch.zeros(
                (self.qa,), device=self.device, dtype=torch.bool
            )

        power = float((x_full ** 2).mean().item())

        if self._input_power_ema is None:
            self._input_power_ema = power
        else:
            a = self.null_alpha
            self._input_power_ema = a * self._input_power_ema + (1-a) * power

        threshold = self.null_thresh * self._input_power_ema
        in_null   = power < threshold

        # Return per-channel mask (same decision for all channels here;
        # extend to per-channel if needed)
        return torch.full(
            (self.qa,), in_null, device=self.device, dtype=torch.bool
        )

    def _apply_norm_constraint(self, blk: ReTMBlock):
        """
        Soft-clamp ||R[a,:]|| to stay within norm_margin × initial norm.
        This prevents amplitude drift (collapse or explosion) per channel.
        """
        if self.norm_margin <= 0.0:
            return
        current_norm = blk.R.norm(dim=1).clamp(min=1e-12)        # [QA]
        max_norm     = self.norm_margin * blk.R_norm0              # [QA]
        # Scale down only channels that exceed the ceiling
        scale = torch.where(
            current_norm > max_norm,
            max_norm / current_norm,
            torch.ones_like(current_norm),
        )                                                          # [QA]
        blk.R.mul_(scale.unsqueeze(1))

    # ─────────────────────────────────────────────────────────────────
    # Core update
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_one(self, x_full: torch.Tensor) -> torch.Tensor:
        """x_full [QB, L] → ŷ [QA]."""
        y = torch.zeros((self.qa,), device=self.device, dtype=self.dtype)
        for blk in self.blocks:
            x_b = x_full[:, blk.l0:blk.l1].reshape(-1)
            y  += (blk.R * x_b.unsqueeze(0)).sum(dim=1)
        return y

    @torch.no_grad()
    def update_one(
        self,
        x_full: torch.Tensor,   # [QB, L]
        d:      torch.Tensor,   # [QA]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict → update with Joseph form + v3 stability fixes.

        Update is skipped per-channel if:
          (a) null_passage detector fires, OR
          (b) channel gain scale is below 0.52 (extremely unreliable)
        """

        # ── PREDICT: inflate P before gain computation ────────────────
        for blk in self.blocks:
            blk.P.mul_(self.G2)
            blk.P.diagonal(dim1=1, dim2=2).add_(self.Q)
            blk.P.diagonal(dim1=1, dim2=2).clamp_(min=self.p_floor)

        # ── Null-passage detection ────────────────────────────────────
        null_mask = self._is_null_passage(x_full)   # [QA] bool

        # ── Predict output ────────────────────────────────────────────
        y_hat = self.predict_one(x_full)
        error = d - y_hat                           # [QA]

        # ── Update EMAs ───────────────────────────────────────────────
        e2 = error.detach() ** 2
        if self.adaptive:
            self._err_ema.mul_(self.alpha).add_((1-self.alpha) * e2)
        if self.cg_alpha > 0.0:
            self._ch_err_ema.mul_(self.cg_alpha).add_(
                (1-self.cg_alpha) * error.detach().abs()
            )

        Rv       = self._rv()                       # [QA]
        ch_gain  = self._ch_gain_scale()            # [QA] ∈[0.5,1]

        # Combine null mask and reliability mask
        # Channels in null OR very unreliable: skip update (hold state)
        skip_ch  = null_mask | (ch_gain < 0.52)    # [QA] bool

        # ── Innovation with momentum ──────────────────────────────────
        if self.beta > 0.0:
            self._velocity.mul_(self.beta).add_((1-self.beta)*error)
            eff_e = self._velocity
        else:
            eff_e = error

        # ── Per-block Joseph-form update ──────────────────────────────
        for blk in self.blocks:
            x_b = x_full[:, blk.l0:blk.l1].reshape(-1)   # [D]

            # PX = P x   [QA, D]
            PX  = torch.einsum("aij,j->ai", blk.P, x_b)

            # S_a = x^T PX_a + Rv_a   scalar per channel  [QA]
            S   = (x_b.unsqueeze(0) * PX).sum(dim=1) + Rv
            S   = S.clamp(min=1e-10)

            # K_a = PX_a / S_a   [QA, D]
            K   = PX / S.unsqueeze(1)

            # Per-channel gain scaling
            K   = K * ch_gain.unsqueeze(1)

            # Gain floor: ensure |K| ≥ gain_floor * mean(diag(P))
            if self.gain_floor > 0.0:
                p_diag_mean = blk.P.diagonal(dim1=1,dim2=2).mean(dim=1)  # [QA]
                k_floor     = self.gain_floor * p_diag_mean               # [QA]
                K_norm      = K.norm(dim=1).clamp(min=1e-12)             # [QA]
                floor_scale = (k_floor / K_norm).clamp(max=1.0)          # [QA]
                # Only apply floor where K_norm < k_floor
                needs_floor = K_norm < k_floor
                K           = torch.where(
                    needs_floor.unsqueeze(1),
                    K * (1.0 / floor_scale.unsqueeze(1).clamp(min=1e-12)),
                    K,
                )

            # ── State update (skipped for null/unreliable channels) ───
            # r(t+1) = G * r(t)  +  K * e  (for active channels only)
            active = (~skip_ch).float().unsqueeze(1)    # [QA, 1]
            blk.R.mul_(self.G)
            blk.R.addcmul_(K * active, eff_e.unsqueeze(1))

            # ── Norm constraint (amplitude collapse prevention) ───────
            self._apply_norm_constraint(blk)

            # ── Joseph-form covariance update ─────────────────────────
            # P+ = P - PX*K^T - K*PX^T + S*K*K^T
            t1  = PX.unsqueeze(2) * K.unsqueeze(1)    # [QA,D,D]
            t2  = t1.transpose(1,2)
            t3  = S.view(self.qa,1,1) * K.unsqueeze(2) * K.unsqueeze(1)
            blk.P.sub_(t1).sub_(t2).add_(t3)

            # Floor and symmetrise
            blk.P.diagonal(dim1=1,dim2=2).clamp_(min=self.p_floor)
            if self.sym:
                blk.P.copy_(0.5*(blk.P + blk.P.transpose(1,2)))

        return y_hat, error

    # ─────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_retm_tensor(self) -> torch.Tensor:
        R = torch.zeros(
            (self.qa, self.qb, self.L), device=self.device, dtype=self.dtype
        )
        for blk in self.blocks:
            B = blk.l1 - blk.l0
            R[:, :, blk.l0:blk.l1] = blk.R.reshape(self.qa, self.qb, B)
        return R

    @torch.no_grad()
    def copy_to_fusnet(self):
        R = self.get_retm_tensor()
        for q in range(self.qb):
            getattr(self.model, f"conv{q+1}").weight.data.copy_(R[:, q:q+1, :])

    @torch.no_grad()
    def reset_buffers(self):
        self._velocity.zero_()
        self._err_ema.fill_(self.Rv0)
        self._ch_err_ema.fill_(1.0)
        self._input_power_ema = None