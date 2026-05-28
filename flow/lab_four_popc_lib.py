"""
flow/lab_four_popc_lib.py — POPC (Partial Observation, Partial Control) variant
of the Burgers FM lab.

FOPC (what lab_four already does):
    - control w: zeroed in middle 50% of space  (partial CONTROL)
    - state u:   FULLY observed at all 128 points  (full observation)
    - c = (u_0, u_T*) at all 128 points → inpaint full boundary rows

POPC (this module):
    - control w: zeroed in middle 50%  (same partial control)
    - state u:   observed ONLY at front+rear quarter (x ∈ [0,1/4] ∪ [3/4,1])
                 middle 50% UNOBSERVED → the model must GENERATE it
    - c carries boundary only at observed positions; inpaint only those columns

The only thing that changes vs FOPC is the OBSERVATION mask on u. The control
channel handling and everything else is inherited from lab_four_explore.

Paper reference: `run_train_POPC_10k.sh`, `notes_baseline_summary.md §3.2`
(DDPM POPC baseline γ=1: J=0.0201, E=1409).
"""

from __future__ import annotations
from typing import Tuple

import torch

from flow.lab_four_explore import (
    BurgersDataset, BurgersFlowTrainer, BurgersVectorField, BurgersEulerSampler,
    BurgersPriorDataset, BurgersPriorTrainer,
    GaussianConditionalProbabilityPath, VectorFieldNet,
    _LivePlotMixin,
    T_IDX,
)
from model.burgers_1d.unet import Unet2D


def observed_mask(Nx: int = 128, device=None) -> torch.Tensor:
    """Boolean mask, True at OBSERVED positions (front+rear quarter), False in middle 50%.

    POPC observes u only at x ∈ [0, Nx/4) ∪ [3Nx/4, Nx).
    """
    mask = torch.ones(Nx, dtype=torch.bool, device=device)
    mask[Nx // 4 : (3 * Nx) // 4] = False
    return mask


def inpaint_overwrite_popc(x: torch.Tensor, c: torch.Tensor, T_idx: int = T_IDX) -> torch.Tensor:
    """Like FOPC inpaint_overwrite, but only overwrites OBSERVED columns of u.

    The unobserved middle 50% of rows 0 and T_idx is left as-is (the model
    generates it).
    """
    x = x.clone()
    Nx = x.shape[-1]
    obs = observed_mask(Nx, device=x.device)
    x[:, 0, 0,     obs] = c[:, 0, obs]
    x[:, 0, T_idx, obs] = c[:, 1, obs]
    return x


class BurgersPOPCDataset(BurgersDataset):
    """Same (z, c) as FOPC, but c's unobserved middle 50% is zeroed.

    z stays FULL (the target trajectory includes the unobserved middle — the
    model is trained to reconstruct it). Only the conditioning c is masked, so
    the model never "sees" the middle boundary as a clean input.
    """
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z, c = super().sample(num_samples)
        obs = observed_mask(c.shape[-1], device=c.device)
        c = c.clone()
        c[:, :, ~obs] = 0.0     # zero the unobserved middle of (u_0, u_T*)
        return z, c


class BurgersPOPCFlowTrainer(BurgersFlowTrainer):
    """Joint POPC trainer. Inpaint trick zeros target velocity ONLY at observed
    boundary positions (the unobserved middle is generated, so its target
    velocity must remain non-zero)."""
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z, c = self.data.sample(batch_size)
        t = torch.rand(batch_size, 1, 1, 1, device=z.device)
        x_t, eps = self.path.sample_conditional_path(z, t)
        u_target = self.path.target_velocity(x_t, z, t, eps)
        obs = observed_mask(z.shape[-1], device=z.device)
        # zero target velocity only at OBSERVED boundary positions
        u_target[:, 0, 0,     obs] = 0
        u_target[:, 0, T_IDX, obs] = 0
        u_pred = self.net(x_t, t, c)
        return ((u_pred - u_target) ** 2).mean()


class BurgersPOPCVectorField(VectorFieldNet):
    """Wraps Unet2D; injects c only at observed boundary columns."""
    def __init__(self, dim: int = 64, dim_mults: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.unet = Unet2D(dim=dim, dim_mults=dim_mults, channels=2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x_in = x.clone()
        Nx = x.shape[-1]
        obs = observed_mask(Nx, device=x.device)
        x_in[:, 0, 0,     obs] = c[:, 0, obs]
        x_in[:, 0, T_IDX, obs] = c[:, 1, obs]
        t_flat = t.view(-1).to(x_in.dtype)
        return self.unet(x_in, t_flat)


class BurgersPOPCEulerSampler(BurgersEulerSampler):
    """Euler sampler using the POPC inpaint (only observed columns)."""
    @torch.no_grad()
    def sample(self, c: torch.Tensor, shape: Tuple[int, ...] = (2, 16, 128)) -> torch.Tensor:
        b = c.shape[0]
        device = c.device
        dtau = (1.0 - 2 * self.tau_min) / self.n_steps
        x = torch.randn(b, *shape, device=device)
        x = inpaint_overwrite_popc(x, c)
        for i in range(self.n_steps):
            tau = self.tau_min + i * dtau
            tau_tensor = torch.full((b,), tau, device=device)
            v = self.net(x, tau_tensor, c)
            x = x + v * dtau
            x = inpaint_overwrite_popc(x, c)
        return x


class BurgersPOPCPriorDataset(BurgersPOPCDataset):
    """Prior dataset for POPC: u-channel zeroed (only learn p(w|c)), c masked to observed."""
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z, c = super().sample(num_samples)
        z[:, 0] = 0
        return z, c


class BurgersPOPCPriorTrainer(BurgersPOPCFlowTrainer):
    """Prior trainer for POPC: force u-channel target + prediction to 0."""
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z, c = self.data.sample(batch_size)
        t = torch.rand(batch_size, 1, 1, 1, device=z.device)
        x_t, eps = self.path.sample_conditional_path(z, t)
        u_target = self.path.target_velocity(x_t, z, t, eps)
        obs = observed_mask(z.shape[-1], device=z.device)
        u_target[:, 0, 0,     obs] = 0
        u_target[:, 0, T_IDX, obs] = 0
        u_target[:, 0] = 0        # prior: zero entire u-channel target
        u_pred = self.net(x_t, t, c)
        u_pred[:, 0] = 0          # prior: zero entire u-channel prediction
        return ((u_pred - u_target) ** 2).mean()


# Live-loss-plot versions for jupyter training
class LiveLossTrainerPOPC(_LivePlotMixin, BurgersPOPCFlowTrainer):
    """POPC joint trainer with live loss plot."""
    pass


class LiveLossTrainerPOPCPrior(_LivePlotMixin, BurgersPOPCPriorTrainer):
    """POPC prior trainer with live loss plot."""
    _prior_plot = True


# Convenience: a POPC-aware ReweightedVectorField is NOT needed — the existing
# ReweightedVectorField from lab_four_explore works as-is (it just combines two
# velocity fields; the POPC-ness lives in how net_joint/net_prior inject c, which
# is handled by BurgersPOPCVectorField.forward).
