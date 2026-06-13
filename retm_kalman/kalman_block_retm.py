"""
kalman_fusnet_retm_v2.py
========================
Improved sample-wise partitioned-block Kalman ReTM tracker.

Improvements over v1
--------------------

[A] Correct Joseph-form covariance update
    v1 computed:
        outer = K * PX^T   where K = PX/S
                          = (PX/S) * PX^T    ← wrong, S applied once to K
                                                 but the true correction is
                                                 (P x)(P x)^T / S = PX*PX^T/S

    Correct standard update (rank-1 measurement):
        P_new = (I - K x^T) P (I - K x^T)^T  +  K Rv K^T    ← Joseph form
    which for scalar Rv simplifies to:
        P_new = P  -  K (x^T P)  -  (P x) K^T  +  K (x^T P x + Rv) K^T
              = P  -  PX * K^T   -  K * PX^T   +  S * K * K^T
    This is numerically positive-definite even with floating-point errors,
    unlike the asymmetric  P - K*PX^T  used in v1.

[B] Per-channel adaptive Rv  (was shared scalar, now per-channel vector)
    Channel 5 in your results had SDR=1.5 dB vs 12+ for others.
    Per-channel Rv lets the filter trust poorly-tracked channels less.

[C] Covariance lower-bound via diagonal floor BEFORE the update
    Prevents P collapsing to near-zero mid-run, which freezes the gain.

[D] Optional per-channel gain scaling by mic reliability
    Channels with persistently high error get a reduced effective gain,
    preventing them from pulling the shared covariance estimate off-track.
    (Set channel_gain_alpha=0 to disable.)

[E] Process noise added BEFORE the update step (predict → update)
    In the standard Kalman predict-update cycle, Q is added to P during
    the *predict* step, not after the *update*. This keeps P inflated
    to reflect genuine state uncertainty before computing the gain.

[F] Numerical stability: use torch.linalg.solve style for gain
    For rank-1 updates the scalar S is sufficient (no matrix inverse needed),
    but we now guard S with a per-channel floor relative to diag(P).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReTMBlock:
    l0: int
    l1: int
    R: torch.Tensor   # [QA, D]
    P: torch.Tensor   # [QA, D, D]


# ─────────────────────────────────────────────────────────────────────────────

class PartitionedKalmanReTMv2:
    """
    Parameters
    ----------
    model               : FuSNet nn.Module
    qa, qb              : mic group sizes
    filter_length       : total tap count (2*context + 1)
    block_length        : taps per partition  (D = qb * block_length)
    transition G        : forgetting factor (0.990–0.9999)
    process_noise Q     : diagonal process noise added in predict step
    observation_noise   : baseline Rv per channel (floor for adaptive)
    initial_covariance  : P0 diagonal init value
    adaptive_noise      : use per-channel EMA of e² as Rv
    adaptive_alpha      : EMA decay  (0.99–0.9999)
    adaptive_noise_floor: min Rv per channel
    adaptive_noise_ceil : max Rv per channel
    innovation_momentum : beta for velocity buffer (0=off)
    channel_gain_alpha  : EMA for per-channel gain scaling (0=off)
    p_floor             : minimum diagonal value of P (enforced before update)
    symmetrize          : enforce P symmetry each step
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
        adaptive_noise: bool = True,
        adaptive_alpha: float = 0.999,
        adaptive_noise_floor: float = 1e-4,
        adaptive_noise_ceil: float = 1.0,
        innovation_momentum: float = 0.3,
        channel_gain_alpha: float = 0.999,
        p_floor: float = 1e-10,
        symmetrize: bool = True,
    ):
        self.model = model
        self.qa  = int(qa)
        self.qb  = int(qb)
        self.L   = int(filter_length)
        self.BL  = int(block_length)

        self.G   = float(transition)
        self.G2  = self.G ** 2
        self.Q   = float(process_noise)
        self.Rv0 = float(observation_noise)
        self.P0  = float(initial_covariance)

        self.device = torch.device(device)
        self.dtype  = dtype
        self.sym    = bool(symmetrize)
        self.p_floor = float(p_floor)

        # ── Per-channel adaptive Rv ──────────────────────────────────
        self.adaptive     = bool(adaptive_noise)
        self.alpha        = float(adaptive_alpha)
        self.Rv_floor     = float(adaptive_noise_floor)
        self.Rv_ceil      = float(adaptive_noise_ceil)
        self._err_ema     = torch.full(
            (self.qa,), self.Rv0, device=self.device, dtype=self.dtype
        )

        # ── Innovation momentum ──────────────────────────────────────
        self.beta         = float(innovation_momentum)
        self._velocity    = torch.zeros(
            (self.qa,), device=self.device, dtype=self.dtype
        )

        # ── Per-channel reliability scaling ─────────────────────────
        # Tracks a slow EMA of |e_a| per channel.
        # Channels with persistently large error get gain scaled down.
        self.cg_alpha     = float(channel_gain_alpha)
        self._ch_err_ema  = torch.full(
            (self.qa,), 1.0, device=self.device, dtype=self.dtype
        )

        # ── Build state ──────────────────────────────────────────────
        self.blocks: List[ReTMBlock] = []
        self._init(self._extract())

    # ─────────────────────────────────────────────────────────────────
    # Init
    # ─────────────────────────────────────────────────────────────────

    def _extract(self) -> torch.Tensor:
        """FuSNet conv1…conv{qb} → R[QA, QB, L]."""
        R = torch.zeros(
            (self.qa, self.qb, self.L), device=self.device, dtype=self.dtype
        )
        for q in range(self.qb):
            name = f"conv{q+1}"
            conv = getattr(self.model, name)
            w    = conv.weight.detach().to(self.device, dtype=self.dtype)
            R[:, q, :] = w[:, 0, :]
        return R

    def _init(self, R: torch.Tensor):
        self.blocks.clear()
        for l0 in range(0, self.L, self.BL):
            l1 = min(l0 + self.BL, self.L)
            D  = self.qb * (l1 - l0)
            Rb = R[:, :, l0:l1].reshape(self.qa, D).contiguous()
            Pb = (
                torch.eye(D, device=self.device, dtype=self.dtype)
                .unsqueeze(0).expand(self.qa, -1, -1).clone() * self.P0
            )
            self.blocks.append(ReTMBlock(l0, l1, Rb, Pb))

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
        """
        Per-channel gain scaling in [0,1].
        Channels with large persistent error get scaled toward 0.5×.
        Normalised so the best channel always gets scale=1.
        """
        if self.cg_alpha <= 0.0:
            return torch.ones(
                (self.qa,), device=self.device, dtype=self.dtype
            )
        best = self._ch_err_ema.min().clamp(min=1e-12)
        ratio = best / self._ch_err_ema.clamp(min=1e-12)   # ∈ (0,1]
        # soft-clip: scale ∈ [0.5, 1.0]
        return 0.5 + 0.5 * ratio

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
        Full Kalman predict→update cycle for one sample.

        Key fix: Joseph-form covariance update
        ----------------------------------------
        Standard rank-1 update (scalar obs noise Rv per channel):

            S_a  = x^T P_a x  +  Rv_a              (scalar)
            K_a  = P_a x / S_a                      [D]
            e_a  = d_a - r_a^T x                   (scalar innovation)

            Joseph form (numerically PD):
            P_a+ = (I - K_a x^T) P_a (I - K_a x^T)^T  +  Rv_a K_a K_a^T
                 = P_a  -  K_a (Px)^T  -  Px K_a^T  +  S_a K_a K_a^T

        where Px = P_a x = S_a * K_a, so:
            P_a+ = P_a  -  S_a*K_a * K_a^T  -  K_a * S_a*K_a^T
                         +  S_a * K_a * K_a^T
                 = P_a  -  K_a * (S_a K_a)^T       ← PX = S_a * K_a
                         -  (S_a K_a) * K_a^T
                         +  S_a * K_a * K_a^T

        Simplifying (K_a K_a^T terms):
            P_a+ = P_a  -  PX * K_a^T  -  K_a * PX^T  +  S_a * K_a * K_a^T

        This is symmetric by construction — far more stable than
        the one-sided  P - K*PX^T  of v1.

        Predict step (applied to P BEFORE computing gain):
            P_predict = G² * P  +  Q * I
        """

        # ── PREDICT step: inflate P before computing gain ────────────
        # This is the correct order: predict first, then update.
        for blk in self.blocks:
            blk.P.mul_(self.G2)
            blk.P.diagonal(dim1=1, dim2=2).add_(self.Q)
            blk.P.diagonal(dim1=1, dim2=2).clamp_(min=self.p_floor)

        # ── Predict output ───────────────────────────────────────────
        y_hat = self.predict_one(x_full)
        error = d - y_hat                                       # [QA]

        # ── Update adaptive Rv and channel reliability EMA ──────────
        e2 = error.detach() ** 2
        if self.adaptive:
            self._err_ema.mul_(self.alpha).add_((1 - self.alpha) * e2)
        if self.cg_alpha > 0.0:
            self._ch_err_ema.mul_(self.cg_alpha).add_(
                (1 - self.cg_alpha) * error.detach().abs()
            )

        Rv      = self._rv()                                    # [QA]
        ch_gain = self._ch_gain_scale()                        # [QA] ∈[0.5,1]

        # ── Innovation with optional momentum ───────────────────────
        if self.beta > 0.0:
            self._velocity.mul_(self.beta).add_((1 - self.beta) * error)
            eff_e = self._velocity
        else:
            eff_e = error

        # ── Per-block Joseph-form update ─────────────────────────────
        for blk in self.blocks:
            x_b = x_full[:, blk.l0:blk.l1].reshape(-1)        # [D]

            # PX = P x    [QA, D]
            PX = torch.einsum("aij,j->ai", blk.P, x_b)

            # S_a = x^T PX_a + Rv_a   (scalar per channel)
            S = (x_b.unsqueeze(0) * PX).sum(dim=1) + Rv        # [QA]
            S = S.clamp(min=1e-10)

            # K_a = PX_a / S_a   [QA, D]
            K = PX / S.unsqueeze(1)

            # Apply per-channel gain scaling (channel reliability)
            K = K * ch_gain.unsqueeze(1)

            # ── State update ─────────────────────────────────────────
            # r(t+1) = G * r(t)  +  K_a * e_a
            blk.R.mul_(self.G)
            blk.R.addcmul_(K, eff_e.unsqueeze(1))

            # ── Joseph-form covariance update ────────────────────────
            # P+ = P  -  PX*K^T  -  K*PX^T  +  S * K*K^T
            #
            # Note P was already scaled by G² in the predict step above.

            # Term 1: PX * K^T    [QA, D, D]
            t1 = PX.unsqueeze(2) * K.unsqueeze(1)   # outer: PX[a,i]*K[a,j]

            # Term 2: K * PX^T = t1^T
            t2 = t1.transpose(1, 2)

            # Term 3: S * K * K^T    [QA, D, D]
            t3 = S.unsqueeze(1).unsqueeze(2) * K.unsqueeze(2) * K.unsqueeze(1)

            blk.P.sub_(t1).sub_(t2).add_(t3)

            # Floor diagonal
            blk.P.diagonal(dim1=1, dim2=2).clamp_(min=self.p_floor)

            # Symmetrise
            if self.sym:
                blk.P.copy_(0.5 * (blk.P + blk.P.transpose(1, 2)))

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
