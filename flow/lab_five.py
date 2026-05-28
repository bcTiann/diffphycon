"""
lab_five.py — Flow Matching for 2D Jellyfish Control (DiffPhyCon Experiment 3)
==============================================================================

Companion to lab_four.py (Burgers 1D, Experiment 1).

This lab is the **toy / local** version of Experiment 3:
  - Smaller Unet3D (dim=32, dim_mults=(1,2), ~1.9M params vs paper's 22M)
  - Trains on the 200 test_data samples (split 160 train / 40 eval),
    instead of the full 30k-sample training set that lives on Modal
  - Skips prior-reweighting + LilyPad eval (lab_five_modal.py future work)
  - Designed to run end-to-end on M4 Pro MPS in one overnight

Why toy: the goal here is to verify FM scaffolding works on a 3D (T,H,W)
data layout and learn how the jellyfish data + Unet3D differ from Burgers.
For paper-faithful Table 28 comparison, see future lab_five_modal.py.

## How to use this lab

1. Read the Part 0 markdown — most of the abstractions are imported from
   `flow.lab_four`, so you can refresh how `Trainer`, `GaussianConditionalProbabilityPath`,
   and friends work by glancing back at lab_four.

2. Fill in `raise NotImplementedError(...)` for Q5.1 → Q5.5 (6 spots total).
   Each spot has a `# Step N:` comment that says exactly what to compute.

3. After Part 2 fills, run `sanity_check_2_4()` — should see loss drop to ~0.05.
4. After Part 3 fills, run `sanity_check_3_3()` — should see a vaguely
   wing-shaped sample (not pure noise).

5. When sanity passes, call `train_jellyfish_for_part5(num_steps=30000, ...)`
   to actually train (overnight on M4 Pro), then visualize with `Part 5` helpers.

## Data shape conventions

Throughout this file:

  - `z` has shape `(b, 20, 4, 64, 64)`  — layout `(b, T, C, H, W)`:
      - time axis = 20 (first 20 of each sim's 40 frames)
      - channel 0..2 = state (vx, vy, pressure) — normalized to [-1, 1]
      - channel 3    = theta-broadcast — θ(t) repeated across H, W
      - spatial = 64 × 64

  - `c` has shape `(b, 20, 3, 64, 64)`:
      - 3 channels = boundary mask + 2 offsets — pre-padded 62×62 → 64×64

  - `t` (FM time) shape: `(b, 1, 1, 1, 1)` — broadcasts against z's 5D shape

  - ⚠️ **Layout is `(b, T, C, H, W)`, not `(b, C, T, H, W)`**!
    `Unet3D_with_Conv3D.forward` does `x.permute(0,2,1,3,4)` internally,
    so we feed it T-first. This matches paper's `dataset/data_2d.py::Jellyfish`
    which also returns `(T, C, H, W)` per sample. **Channel dim = dim 2**,
    not dim 1 — that's what trips up most "Burgers-instinct" implementations.

  - Boundary in jellyfish ≠ boundary in Burgers:
      * Burgers c is just `(u_0, u_T*)` (2 slices of u-field at t=0, t=10)
        → injected via `inpaint_overwrite` (replace specific rows of x)
      * Jellyfish c is the FULL boundary trajectory over all 20 frames
        → injected via channel-concat (dim=2!) in `JellyfishVectorField.forward`
"""

# pyright: reportUnknownMemberType=false
from __future__ import annotations
import math
import os
import pickle
import sys
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- path setup (works as .py or in Jupyter notebook) ---
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.path.abspath(os.getcwd())
    if os.path.basename(HERE) != "flow":
        HERE = os.path.join(HERE, "flow")
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ============================================================================
#                          Part 0: Setup + Re-imports
# ============================================================================
#
# We reuse all the FM scaffolding from lab_four. If anything below is unclear,
# open flow/lab_four.py and read the corresponding class.
#
# Imports:
#   - Sampleable, LabeledSampleable: ABCs for distributions
#   - ConditionalProbabilityPath, GaussianConditionalProbabilityPath: FM paths
#   - LinearAlpha, LinearBeta: α_τ = τ, β_τ = 1-τ schedules
#   - VectorFieldNet: ABC for velocity-field networks
#   - Trainer: ABC with tqdm + loss_history + optimizer loop
#   - EMA, finetune_with_ema: EMA helpers (optional for toy mode)
# ============================================================================

from flow.lab_four import (
    Sampleable,
    LabeledSampleable,
    ConditionalProbabilityPath,
    GaussianConditionalProbabilityPath,
    LinearAlpha,
    LinearBeta,
    VectorFieldNet,
    Trainer,
    EMA,
    finetune_with_ema,
)

from model.video_diffusion_pytorch.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D


# ============================================================================
#                   Jellyfish-specific constants + data loader
# ============================================================================

JELLYFISH_DATA_PATH = os.path.join(ROOT, "data", "jellyfish")
H, W = 64, 64               # spatial size of state
BD_H, BD_W = 62, 62         # spatial size of raw boundary (pad to 64)
T_FRAMES = 20               # FM operates on 20-frame windows (first 20 of 40)
TRAJ_LEN = 40               # each sim has 40 total frames
N_TOTAL = 200               # 200 sims available in test_data/
N_TRAIN = 160
N_EVAL = 40
THETA_NORM = 1.6            # rad — empirically θ ∈ [-π/2, π/2] approx; this normalizes to [-1, 1]


def load_jellyfish_normalization() -> dict:
    """Load (vx/vy/pressure)_max/min from the pkl that paper provides."""
    p = os.path.join(JELLYFISH_DATA_PATH, "test_data", "normalization_max_min.pkl")
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Missing {p}. Expected normalization pkl from paper data dump."
        )
    return pickle.load(open(p, "rb"))


def _load_one_sim(sim_id: int, norm: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one sim's (state, bd_mask_offset, theta) — all unbatched, numpy."""
    base = os.path.join(JELLYFISH_DATA_PATH, "test_data")

    # state: (40, 3, 64, 64) raw → normalize each channel to [-1, 1]
    state = np.load(os.path.join(base, f"states/sim_{sim_id:06d}.npz"))["a"].astype(np.float32)
    vx_lo, vx_hi = norm["vx_min"], norm["vx_max"]
    vy_lo, vy_hi = norm["vy_min"], norm["vy_max"]
    p_lo,  p_hi  = norm["p_min"],  norm["p_max"]
    state[:, 0] = np.clip((state[:, 0] - vx_lo) / (vx_hi - vx_lo), 0, 1) * 2 - 1
    state[:, 1] = np.clip((state[:, 1] - vy_lo) / (vy_hi - vy_lo), 0, 1) * 2 - 1
    state[:, 2] = np.clip((state[:, 2] - p_lo)  / (p_hi  - p_lo),  0, 1) * 2 - 1
    state = np.nan_to_num(state, nan=0.0)

    # bd_mask_offset: raw (40, 62, 62, 3) → transpose to (40, 3, 62, 62)
    bd = np.load(os.path.join(base, f"bdry_merged_mask_offsets/sim_{sim_id:06d}.npz"))["a"]
    bd = np.transpose(bd, (0, 3, 1, 2)).astype(np.float32)
    bd = np.nan_to_num(bd, nan=0.0)

    # theta: (40,) — normalize to [-1, 1] using THETA_NORM
    theta = np.load(os.path.join(base, f"bdry_head_thetas/sim_{sim_id:06d}.npz"))["thetas"].astype(np.float32)
    theta = theta / THETA_NORM

    return state, bd, theta


def load_jellyfish_all(verbose: bool = True):
    """Load all 200 sims into RAM. Returns 3 numpy arrays."""
    norm = load_jellyfish_normalization()
    states = np.zeros((N_TOTAL, TRAJ_LEN, 3, H, W), dtype=np.float32)
    bds    = np.zeros((N_TOTAL, TRAJ_LEN, 3, BD_H, BD_W), dtype=np.float32)
    thetas = np.zeros((N_TOTAL, TRAJ_LEN), dtype=np.float32)
    for i in range(N_TOTAL):
        s, b, t = _load_one_sim(i, norm)
        states[i] = s; bds[i] = b; thetas[i] = t
        if verbose and (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{N_TOTAL} sims")
    return states, bds, thetas


def split_train_eval(states, bds, thetas, seed: int = 42):
    """Deterministic 160 / 40 split."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N_TOTAL)
    tr, ev = idx[:N_TRAIN], idx[N_TRAIN:]
    return ((states[tr], bds[tr], thetas[tr]),
            (states[ev], bds[ev], thetas[ev]))


# ============================================================================
#                  Part 1: Get a Feel for Jellyfish Data
# ============================================================================
#
# No fill-ins here — just visualize to internalize the data shape.
#
# Per sim:
#   - state (vx, vy, pressure): (40, 3, 64, 64)
#   - boundary mask+offsets:    (40, 3, 62, 62)
#   - theta (wing angle):       (40,) scalar per frame
#
# We work with the first 20 frames per sim → (20, ...) windows.
# ============================================================================


def visualize_jellyfish_sample(
    state: np.ndarray,   # (20, 3, 64, 64) — normalized [-1, 1]
    bd:    np.ndarray,   # (20, 3, 62, 62)
    theta: np.ndarray,   # (20,) — normalized
    title: str = "",
    save_path: Optional[str] = None,
):
    """4-row plot:
        row 0: vx        at t={0, 5, 10, 15, 19}
        row 1: vy        at t={0, 5, 10, 15, 19}
        row 2: pressure  at t={0, 5, 10, 15, 19}
        row 3: bd_mask   at t={0, 5, 10, 15, 19}  (channel 0 of bd)
       Plus θ vs t line plot at bottom.
    """
    import matplotlib.pyplot as plt
    cols = [0, 5, 10, 15, 19]
    fig, axes = plt.subplots(5, 5, figsize=(15, 14))
    row_labels = ["vx", "vy", "pressure", "bd_mask", ""]
    for r, label in enumerate(row_labels[:4]):
        for ci, t_idx in enumerate(cols):
            ax = axes[r, ci]
            if r < 3:
                img = state[t_idx, r]
            else:
                img = bd[t_idx, 0]   # boundary mask
            im = ax.imshow(img, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"{label} t={t_idx}", fontsize=9)
            ax.axis("off")
    # θ vs t plot on the last row (spans 5 cols, but we just use middle)
    for ci in range(5):
        axes[4, ci].axis("off")
    ax_theta = fig.add_subplot(5, 1, 5)
    ax_theta.plot(np.arange(20), theta * THETA_NORM, "o-", linewidth=2)
    ax_theta.axhline(0, color="gray", lw=0.5)
    ax_theta.set_xlabel("frame")
    ax_theta.set_ylabel("θ (rad)")
    ax_theta.set_title("Wing angle over time (un-normalized)")
    ax_theta.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=14, y=0.99)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"  saved {save_path}")
    plt.show()


def sanity_check_part1():
    """Load + visualize 2 sims to confirm data shape."""
    print("\n=== Sanity Check Part 1: Jellyfish data ===")
    print(f"  Loading {N_TOTAL} sims (this takes ~10 sec)...")
    states, bds, thetas = load_jellyfish_all(verbose=False)
    print(f"  shapes:  states={states.shape}  bds={bds.shape}  thetas={thetas.shape}")
    print(f"  state  range: [{states.min():.3f}, {states.max():.3f}]")
    print(f"  bd     range: [{bds.min():.3f}, {bds.max():.3f}]")
    print(f"  theta  range (normalized): [{thetas.min():.3f}, {thetas.max():.3f}]")
    print(f"  theta  range (un-normalized rad): [{(thetas*THETA_NORM).min():.3f}, {(thetas*THETA_NORM).max():.3f}]")

    for idx in [0, 100]:
        visualize_jellyfish_sample(
            states[idx, :T_FRAMES],
            bds[idx, :T_FRAMES],
            thetas[idx, :T_FRAMES],
            title=f"Jellyfish sim_{idx:06d} (first 20 frames)",
            save_path=os.path.join(HERE, f"lab_five_part1_sim{idx}.png"),
        )
    print("  ✅ Part 1 visualization done.")


# ============================================================================
#                  Part 2: Train Joint FM (Q5.1 - Q5.4)
# ============================================================================
#
# Mirrors Part 2 of lab_four. Three classes to fill:
#   Q5.1 — JellyfishDataset.sample  (extract (z, c) batches)
#   Q5.2 — JellyfishFlowTrainer.get_train_loss  (6 steps, same as Q2.2 but 5D)
#   Q5.3 — JellyfishVectorField.forward  (channel-concat conditioning)
#   Q5.4 — sanity check (provided, no fill-in)
# ============================================================================


# ----------------------------- Question 5.1 ---------------------------------

class JellyfishDataset(LabeledSampleable):
    """Wraps the 160-sim train pool, exposes (z, c) batches.

    z: (b, 20, 4, 64, 64) — state (3) + theta-broadcast (1), layout (b,T,C,H,W)
    c: (b, 20, 3, 64, 64) — bd_mask_offset, padded from 62×62 to 64×64

    Design note:
      Unlike Burgers (c = 2 boundary slices, injected via inpaint_overwrite),
      jellyfish c is the **full boundary trajectory** over all 20 frames.
      The boundary changes every frame as the wings rotate. So we treat c
      as auxiliary input channels concat'd to x in `JellyfishVectorField.forward`.
    """
    def __init__(self, states: np.ndarray, bds: np.ndarray, thetas: np.ndarray,
                 device: str = "cpu"):
        # all tensors live on `device`. Pre-convert + cache.
        # states: (N, 40, 3, 64, 64) — keep on CPU until sampled (saves device memory)
        self.states = torch.from_numpy(states).to(device)
        self.bds    = torch.from_numpy(bds).to(device)
        self.thetas = torch.from_numpy(thetas).to(device)
        self.N = states.shape[0]
        self.device = device

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z: (num_samples, 20, 4, 64, 64) — joint = state + theta-broadcast
            c: (num_samples, 20, 3, 64, 64) — boundary padded to 64×64
        """
        # Step 1: Uniformly sample `num_samples` indices in [0, self.N).
        #         Use them to gather state, bd, theta windows of length T_FRAMES=20
        #         from the FIRST T_FRAMES frames of each sampled sim.
        #
        # The cached tensors are already in (T, C, H, W) per-sim layout, so
        # `self.states[idx]` returns (b, 40, 3, 64, 64) = (b, T_total, C, H, W).
        # No permute needed.
        #
        # Hint:
        #     idx   = torch.randint(0, self.N, (num_samples,), device=self.device)
        #     state = self.states[idx][:, :T_FRAMES]    # (b, 20, 3, 64, 64)
        #     bd    = self.bds   [idx][:, :T_FRAMES]    # (b, 20, 3, 62, 62)
        #     theta = self.thetas[idx][:, :T_FRAMES]    # (b, 20)
        raise NotImplementedError("Fill me in! (Q5.1 Step 1)")

        # Step 2: Pad bd from (62, 62) to (64, 64) — 1 pixel each side, value 0.
        #         This gives `c` the right shape for Unet3D.
        #
        # Hint:
        #     c = F.pad(bd, (1, 1, 1, 1), value=0.0)    # pad last 2 dims → (b, 20, 3, 64, 64)
        raise NotImplementedError("Fill me in! (Q5.1 Step 2)")

        # Step 3: Build z by concatenating state and theta-broadcast on channel dim.
        #         theta is (b, 20). Broadcast it to (b, 20, 1, 64, 64) and cat along
        #         **dim=2 (channel)** → (b, 20, 4, 64, 64).
        #
        # Hint:
        #     theta_full = theta[:, :, None, None, None].expand(-1, T_FRAMES, 1, H, W)
        #     z = torch.cat([state, theta_full], dim=2)   # cat on CHANNEL dim (dim=2!)
        raise NotImplementedError("Fill me in! (Q5.1 Step 3)")

        return z, c


# ----------------------------- Question 5.2 ---------------------------------

class JellyfishFlowTrainer(Trainer):
    """Trains the joint FM model u_τ^θ(x | c) on jellyfish data.

    Structurally **identical** to BurgersFlowTrainer (lab_four Q2.2).
    Only difference: shapes are 5D `(b, 4, 20, 64, 64)` not 4D `(b, 2, 16, 128)`.
    So when sampling FM time, shape it as `(b, 1, 1, 1, 1)` not `(b, 1, 1, 1)`.

    Inpainting trick from Burgers does NOT apply here:
        - Burgers: clean boundary lives in specific rows of x → must zero
          target velocity at those rows
        - Jellyfish: boundary lives in c (separate from x), never noisy
          → no row-zeroing needed.
    So no Step 5 from Q2.2 here.
    """
    def __init__(
        self,
        net: VectorFieldNet,
        path: GaussianConditionalProbabilityPath,
        data: JellyfishDataset,
        lr: float = 1e-3,
    ):
        super().__init__(net, lr=lr)
        self.path = path
        self.data = data

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample a batch from self.data — z (clean) and c (boundary).
        #         Use self.data.sample(batch_size).
        raise NotImplementedError("Fill me in! (Q5.2 Step 1)")

        # Step 2: Sample FM time t ~ Uniform[0, 1] with shape (b, 1, 1, 1, 1)
        #         so it broadcasts against x of shape (b, 20, 4, 64, 64).
        #         **5 ones, not 4** — that's the only thing that changes from Q2.2.
        #
        # Hint:
        #     t = torch.rand(batch_size, device=z.device).view(-1, 1, 1, 1, 1)
        raise NotImplementedError("Fill me in! (Q5.2 Step 2)")

        # Step 3: Sample x_t ~ p_t(x|z). Use self.path.sample_conditional_path(z, t).
        #         Returns (x_t, eps).
        raise NotImplementedError("Fill me in! (Q5.2 Step 3)")

        # Step 4: Compute target velocity:
        #             u_target = self.path.target_velocity(x_t, z, t, eps)
        raise NotImplementedError("Fill me in! (Q5.2 Step 4)")

        # Step 5: Forward through self.net (the JellyfishVectorField) with (x_t, t, c).
        #         Then return MSE: ((u_pred - u_target) ** 2).mean()
        raise NotImplementedError("Fill me in! (Q5.2 Step 5)")


# ----------------------------- Question 5.3 ---------------------------------

class JellyfishVectorField(VectorFieldNet):
    """Wraps Unet3D_with_Conv3D as a conditional velocity field.

    Input pattern (layout (b, T, C, H, W) — see top-of-file docstring):
        x: (b, 20, 4, 64, 64) — noisy joint (3 state + 1 theta), dim 2 = channel
        c: (b, 20, 3, 64, 64) — boundary (always clean), dim 2 = channel
      We concat **along dim 2 (channel)** → (b, 20, 7, 64, 64) and feed Unet3D.
      Out_dim=4 → Unet3D returns velocity in z-space only (no velocity for c).

    Toy config:
        dim=32, dim_mults=(1, 2)   → ~1.9M params, fits M4 Pro
    Paper config (for reference):
        dim=64, dim_mults=(1, 2, 4) → ~22M params, needs A100
    """
    def __init__(self, dim: int = 32, dim_mults: Tuple[int, ...] = (1, 2)):
        super().__init__()
        self.unet = Unet3D_with_Conv3D(
            dim=dim,
            dim_mults=dim_mults,
            channels=7,    # 4 z-channels + 3 c-channels concat'd
            out_dim=4,     # only output z-velocity (state + theta), not c
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, 20, 4, 64, 64) noisy z at FM time t  — channel dim is dim 2
            t: (b,) or (b, 1, 1, 1, 1) FM time
            c: (b, 20, 3, 64, 64) boundary (clean)      — channel dim is dim 2
        Returns:
            v: (b, 20, 4, 64, 64) predicted velocity
        """
        # Step 1: Concat c onto x along the **channel dim (dim=2)** → (b, 20, 7, 64, 64).
        #         NOT dim=1! Dim 1 is time. See top-of-file docstring on layout.
        #
        # Hint:
        #     x_in = torch.cat([x, c], dim=2)
        raise NotImplementedError("Fill me in! (Q5.3 Step 1)")

        # Step 2: Unet3D_with_Conv3D.forward(x, time, ...) expects `time` as 1-D (b,).
        #         If t came in as (b, 1, 1, 1, 1), flatten it.
        #
        # Hint:
        #     t_flat = t.view(-1).to(x_in.dtype)
        raise NotImplementedError("Fill me in! (Q5.3 Step 2)")

        # Step 3: Call self.unet(x_in, t_flat). Output is (b, 20, 4, 64, 64)
        #         because out_dim=4. Return it directly.
        raise NotImplementedError("Fill me in! (Q5.3 Step 3)")


# ----------------------------- sanity_check_2_4 (Q5.4, provided) ------------

def sanity_check_2_4(num_train_steps: int = 200, batch_size: int = 2):
    """After filling Q5.1, Q5.2, Q5.3: train ~200 steps on small subset, check loss drops.

    Self-contained. Loads all 200 sims, trains on the 160-sim train pool with
    small batch (3D conv is memory-hungry on MPS), and prints loss trace.
    """
    print("\n=== Sanity Check 2.4: jellyfish FM trainer ===")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  device: {device}")
    print("  loading 200 sims...")
    states, bds, thetas = load_jellyfish_all(verbose=False)
    (st_tr, bd_tr, th_tr), _ = split_train_eval(states, bds, thetas)
    ds = JellyfishDataset(st_tr, bd_tr, th_tr, device=device)
    print(f"  train pool: {ds.N} sims  (each contributes a 20-frame window)")

    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    net = JellyfishVectorField(dim=32, dim_mults=(1, 2)).to(device)
    print(f"  net params: {sum(p.numel() for p in net.parameters()):,}")

    trainer = JellyfishFlowTrainer(net, path, ds, lr=1e-3)
    trainer.train(num_steps=num_train_steps, batch_size=batch_size, print_every=50)

    print(f"  ✅ Sanity 2.4 done. Final loss should be < 0.5.")
    return net, ds


# ============================================================================
#                  Part 3: ODE Sampling (Q5.5 + Q5.6)
# ============================================================================
#
# Unlike Burgers, we don't need inpaint_overwrite here — the boundary lives
# in c (separate from x), so it never gets noised. Sampling = pure Euler ODE.
# ============================================================================


# ----------------------------- Question 5.5 ---------------------------------

class JellyfishEulerSampler:
    """Euler ODE sampler. No inpaint_overwrite (boundary lives in c, not x)."""
    def __init__(self, net: VectorFieldNet, n_steps: int = 100, tau_min: float = 1e-3):
        self.net = net
        self.n_steps = n_steps
        self.tau_min = tau_min

    @torch.no_grad()
    def sample(self, c: torch.Tensor,
               shape: Tuple[int, ...] = (T_FRAMES, 4, H, W)) -> torch.Tensor:
        """
        Args:
            c: (b, 20, 3, 64, 64) boundary condition  — layout (b, T, C, H, W)
            shape: per-sample shape, default (20, 4, 64, 64)
        Returns:
            x_final: (b, *shape) = (b, 20, 4, 64, 64)
        """
        b = c.shape[0]
        device = c.device
        dtau = (1.0 - 2 * self.tau_min) / self.n_steps

        # Step 1: Initialize x with random noise of shape (b, *shape) on device.
        #
        # Hint:
        #     x = torch.randn(b, *shape, device=device)
        raise NotImplementedError("Fill me in! (Q5.5 Step 1)")

        # Step 2: Loop self.n_steps times, taking Euler steps:
        #
        #     for i in range(self.n_steps):
        #         tau = self.tau_min + i * dtau
        #         t = torch.full((b,), tau, device=device)        # (b,) — net flattens internally
        #         v = self.net(x, t, c)                           # (b, *shape)
        #         x = x + v * dtau
        raise NotImplementedError("Fill me in! (Q5.5 Step 2)")

        # Step 3: Return x.
        raise NotImplementedError("Fill me in! (Q5.5 Step 3)")


# ----------------------------- sanity_check_3_3 (Q5.6, provided) ------------

def sanity_check_3_3(num_train_steps: int = 300, n_sample_steps: int = 30,
                     batch_size: int = 2):
    """After filling Q5.5: train 300 steps, sample 2 trajectories, visualize."""
    print("\n=== Sanity Check 3.3: jellyfish ODE sampling ===")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  device: {device}")

    states, bds, thetas = load_jellyfish_all(verbose=False)
    (st_tr, bd_tr, th_tr), (st_ev, bd_ev, th_ev) = split_train_eval(states, bds, thetas)
    ds = JellyfishDataset(st_tr, bd_tr, th_tr, device=device)
    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    net = JellyfishVectorField(dim=32, dim_mults=(1, 2)).to(device)

    print(f"  training {num_train_steps} steps...")
    trainer = JellyfishFlowTrainer(net, path, ds, lr=1e-3)
    trainer.train(num_steps=num_train_steps, batch_size=batch_size, print_every=50)

    # sample 2 trajectories using eval c
    ds_eval = JellyfishDataset(st_ev, bd_ev, th_ev, device=device)
    net.eval()
    sampler = JellyfishEulerSampler(net, n_steps=n_sample_steps)
    z_gt, c_eval = ds_eval.sample(2)
    print(f"  sampling 2 trajectories with n_steps={n_sample_steps}...")
    x_pred = sampler.sample(c_eval)
    print(f"  sample shape: {x_pred.shape}  (expect (2, 20, 4, 64, 64))")

    # quick L2 distance check
    l2 = ((x_pred - z_gt) ** 2).mean().item()
    l2_random = ((torch.randn_like(z_gt) - z_gt) ** 2).mean().item()
    print(f"  L2(predict, gt)  = {l2:.4f}")
    print(f"  L2(random, gt)   = {l2_random:.4f}")
    if l2 < l2_random:
        print(f"  ✅ FM beats random. Toy training is learning something.")
    else:
        print(f"  ⚠️  FM no better than random — try more train steps.")

    # visualize 1st sample: gt vs predicted
    _visualize_compare(
        z_gt[0].cpu().numpy(),
        x_pred[0].cpu().numpy(),
        c_eval[0].cpu().numpy(),
        title=f"Sanity 3.3: gt vs predicted (after {num_train_steps} train steps)",
        save_path=os.path.join(HERE, "lab_five_part3_sample.png"),
    )
    return net


# ============================================================================
#                  Part 4 [SKIPPED in toy v1]
# ============================================================================
#
# Prior model + ReweightedVectorField + γ sweep + LilyPad eval.
#
# Why skipped:
#   (1) LilyPad eval requires Processing IDE + 30 min per 50-sample sweep —
#       not iteration-friendly for tiny toy model
#   (2) Without LilyPad we can't compute v̄ / R(θ) / J — no quantitative
#       way to validate γ sweep results
#   (3) Toy net (1.9M params, 30k step training) may not produce realistic
#       enough θ for γ sweep to matter
#
# When you go Modal-paper-faithful (lab_five_modal.py):
#   - Add JellyfishPriorDataset (zero state channels)
#   - Add JellyfishPriorTrainer
#   - Add ReweightedVectorField for 3D
#   - Use existing jellyfish_modal.py + lilypad_prepare.py + lilypad_parse.py
# ============================================================================


# ============================================================================
#                  Part 5: Visual Evaluation
# ============================================================================


def _visualize_compare(z_gt: np.ndarray, z_pred: np.ndarray, c: np.ndarray,
                       title: str = "", save_path: Optional[str] = None):
    """Side-by-side: gt vs predicted state + theta. z shape (20, 4, 64, 64) — (T, C, H, W)."""
    import matplotlib.pyplot as plt
    cols = [0, 5, 10, 15, 19]
    fig, axes = plt.subplots(4, 10, figsize=(24, 10))
    # left half (cols 0-4): gt;  right half (cols 5-9): predicted
    row_names = ["vx", "vy", "pressure", "θ (broadcast)"]
    for r in range(4):
        for ci, t_idx in enumerate(cols):
            axes[r, ci  ].imshow(z_gt  [t_idx, r], cmap="RdBu_r", vmin=-1, vmax=1)
            axes[r, ci+5].imshow(z_pred[t_idx, r], cmap="RdBu_r", vmin=-1, vmax=1)
            axes[r, ci  ].set_title(f"gt {row_names[r]} t={t_idx}", fontsize=8)
            axes[r, ci+5].set_title(f"pred {row_names[r]} t={t_idx}", fontsize=8)
            axes[r, ci  ].axis("off"); axes[r, ci+5].axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"  saved {save_path}")
    plt.show()


def visualize_sample_vs_gt(net: nn.Module, ds_eval: JellyfishDataset,
                            idx: int = 0, n_steps: int = 100,
                            save_path: Optional[str] = None):
    """Sample one trajectory using ds_eval's idx and side-by-side with gt."""
    device = next(net.parameters()).device
    net.eval()
    sampler = JellyfishEulerSampler(net, n_steps=n_steps)
    # use a fresh batch of 1 from eval
    z_gt_batch, c_batch = ds_eval.sample(idx + 1)
    z_gt = z_gt_batch[idx:idx+1]
    c    = c_batch[idx:idx+1]
    x_pred = sampler.sample(c)
    _visualize_compare(
        z_gt[0].cpu().numpy(),
        x_pred[0].cpu().numpy(),
        c[0].cpu().numpy(),
        title=f"Part 5: sample (n_steps={n_steps})",
        save_path=save_path or os.path.join(HERE, f"lab_five_part5_sample_idx{idx}.png"),
    )
    return x_pred


def compute_l2_per_channel(net: nn.Module, ds_eval: JellyfishDataset,
                            n_samples: int = 20, n_steps: int = 100) -> dict:
    """Mean L2 distance per z-channel between FM predictions and gt over eval set.

    Also reports baselines:
      - identity:  L2(gt, gt) = 0   (lower bound)
      - random:    L2(noise, gt)    (upper bound)
    """
    device = next(net.parameters()).device
    net.eval()
    sampler = JellyfishEulerSampler(net, n_steps=n_steps)

    n_samples = min(n_samples, ds_eval.N)
    z_gt_batch, c_batch = ds_eval.sample(n_samples)
    print(f"  sampling {n_samples} eval trajectories with n_steps={n_steps}...")
    x_pred = sampler.sample(c_batch)

    # L2 per channel — z layout is (n, T=20, C=4, H, W), so channel is dim=2.
    # Average over batch (0), time (1), H (3), W (4) → leaves (4,) per channel.
    diff = (x_pred - z_gt_batch) ** 2
    l2_per_chan = diff.mean(dim=(0, 1, 3, 4)).cpu().numpy()  # (4,)

    z_random = torch.randn_like(z_gt_batch)
    l2_random = ((z_random - z_gt_batch) ** 2).mean(dim=(0, 1, 3, 4)).cpu().numpy()

    names = ["vx", "vy", "pressure", "theta"]
    print(f"  {'channel':<10} {'FM L2':>10} {'random L2':>12} {'ratio':>8}")
    print(f"  {'-'*42}")
    for n, fm, rd in zip(names, l2_per_chan, l2_random):
        ratio = fm / rd if rd > 0 else float("nan")
        print(f"  {n:<10} {fm:>10.4f} {rd:>12.4f} {ratio:>7.2f}×")
    return {"l2_per_channel": l2_per_chan, "l2_random": l2_random, "names": names}


# ============================================================================
#                  Training helper for Part 5 (toy full training)
# ============================================================================


def train_jellyfish_for_part5(
    num_steps: int = 15000,
    batch_size: int = 2,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
    print_every: int = 100,
):
    """Toy full training. ~2.5 sec/step on M4 Pro MPS → 15000 steps ≈ 10 hr (overnight).

    If you want a shorter run to iterate, try num_steps=3000 (~2 hr).
    Saves checkpoint to save_path if given.
    """
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== train_jellyfish_for_part5: {num_steps} steps on {device} ===")
    states, bds, thetas = load_jellyfish_all(verbose=False)
    (st_tr, bd_tr, th_tr), _ = split_train_eval(states, bds, thetas)
    ds = JellyfishDataset(st_tr, bd_tr, th_tr, device=device)
    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    net = JellyfishVectorField(dim=32, dim_mults=(1, 2)).to(device)
    print(f"  net params: {sum(p.numel() for p in net.parameters()):,}")
    trainer = JellyfishFlowTrainer(net, path, ds, lr=lr)
    trainer.train(num_steps=num_steps, batch_size=batch_size, print_every=print_every)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({"state_dict": net.state_dict(),
                    "loss_history": trainer.loss_history}, save_path)
        print(f"  saved {save_path}")
    return net, trainer


# ============================================================================
#                              Main entry
# ============================================================================


if __name__ == "__main__":
    print("=== flow/lab_five.py ===")
    print("Run sanity checks in order. Fill in NotImplementedError between them.")
    print()
    print("After Part 1 visualize:")
    print("    sanity_check_part1()")
    print()
    print("After filling Q5.1, Q5.2, Q5.3:")
    print("    sanity_check_2_4()")
    print()
    print("After filling Q5.5:")
    print("    sanity_check_3_3()")
    print()
    print("Then for full toy training (~10 hr overnight on M4 Pro):")
    print("    net, trainer = train_jellyfish_for_part5(")
    print("        num_steps=15000,")
    print("        save_path='flow/checkpoints/fm_jellyfish_toy.pt'")
    print("    )")
    print()
    print("And Part 5 visual eval:")
    print("    states, bds, thetas = load_jellyfish_all()")
    print("    _, (st_ev, bd_ev, th_ev) = split_train_eval(states, bds, thetas)")
    print("    ds_eval = JellyfishDataset(st_ev, bd_ev, th_ev, device='mps')")
    print("    visualize_sample_vs_gt(net, ds_eval, idx=0)")
    print("    compute_l2_per_channel(net, ds_eval, n_samples=20)")
