# AUTO-GENERATED from flow/lab_four.ipynb by lab_four_explore._ensure_lab_four_solved()
# DO NOT EDIT BY HAND — regenerated whenever the .ipynb is newer.
# Only DEFINITIONS are kept; top-level execution (train_*, sanity_check_*) is dropped.

from __future__ import annotations
import math
import os
import sys
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.path.abspath(os.getcwd())
    if os.path.basename(HERE) != 'flow':
        HERE = os.path.join(HERE, 'flow')
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

class Sampleable(ABC):
    """A distribution we can sample from."""

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        ...

class LabeledSampleable(ABC):
    """A joint distribution over (x, c) we can sample from.

    For Burgers: x = full trajectory (u, w), c = boundary condition (u_0, u_T*).
    Unlike MNIST (where c is a discrete class label), our c is a continuous vector.
    """

    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (x, c)."""
        ...

class ConditionalProbabilityPath(ABC):
    """p_t(x | z): a continuous interpolation from p_init to delta_z.

    Subclasses can implement two equivalent forms of `target_velocity`:
      - Form A (ε form): signature (x_t, z, t, eps)
      - Form B (x_t form, lab_three style): signature (x_t, z, t)

    The dispatcher `target_velocity(x_t, z, t, eps=None)` picks one.
    See markdown cell above for derivation.
    """

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor):
        ...

    @abstractmethod
    def target_velocity(self, x_t: torch.Tensor, z: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None=None) -> torch.Tensor:
        ...

class LinearAlpha:
    """α_τ = τ. So α̇ = 1."""

    def __call__(self, t):
        return t

    def dt(self, t):
        return torch.ones_like(t)

class LinearBeta:
    """β_τ = 1 - τ. So β̇ = -1."""

    def __call__(self, t):
        return 1.0 - t

    def dt(self, t):
        return -torch.ones_like(t)

class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """p_t(x|z) = N(α_t z, β_t² I).

    Two equivalent forms of the conditional vector field are implemented:
        - `target_velocity_formA` : ε form (uses x_t, z, t, eps)
        - `target_velocity_formB` : x_t form (uses x_t, z, t)  — lab_three style

    See the markdown cell above ("Two equivalent forms of the target velocity")
    for the derivation. The `target_velocity` dispatcher below picks
    which form to use — swap the body to compare them.
    """

    def __init__(self, alpha: LinearAlpha, beta: LinearBeta):
        self.alpha = alpha
        self.beta = beta

    def sample_conditional_path(self, z, t):
        """x_t = α_t·z + β_t·ε,  ε ~ N(0, I).

        Returns (x_t, eps) so the caller can pass eps into Form A if they want.
        Form B doesn't need eps; just ignore the second return value.
        """
        eps = torch.randn_like(z)
        x_t = self.alpha(t) * z + self.beta(t) * eps
        return (x_t, eps)

    def target_velocity_formA(self, x_t, z, t, eps):
        """u^target(x_t|z) = α̇_t · z + β̇_t · ε      (Form A, ε form)

        Direct derivative of the conditional flow ψ_t(x_0|z) = α_t·z + β_t·x_0.
        For our CondOT path (α=t, β=1-t, α̇=1, β̇=-1): u^target = z - ε.
        No division by β_t, so no t→1 singularity. Requires ε from sampler.
        """
        return self.alpha.dt(t) * z + self.beta.dt(t) * eps

    def target_velocity_formB(self, x_t, z, t):
        """u^target(x_t|z) = (α̇ − β̇·α/β)·z + (β̇/β)·x_t     (Form B, x_t form)

        Obtained by substituting ε = (x_t − α·z)/β into Form A.
        For our CondOT path: u^target = (z − x_t) / (1 − t).
        Mathematically identical to Form A. Diverges at t = 1 (β → 0);
        callers should clamp t away from 1.
        """
        a = self.alpha(t)
        b = self.beta(t)
        da = self.alpha.dt(t)
        db = self.beta.dt(t)
        return (da - db * a / b) * z + db / b * x_t

    def target_velocity(self, x_t, z, t, eps=None):
        """Default = Form A (ε form). Swap to Form B by editing this body.

        eps is required if using Form A; ignored if using Form B.
        """
        return self.target_velocity_formA(x_t, z, t, eps)

def sanity_check_two_forms(seed: int=0, tau: float=0.5) -> float:
    """Verify Form A and Form B give numerically identical results.

    Returns max |Form A − Form B| over a random batch. Should be ≤ 1e-5 (float32).
    """
    torch.manual_seed(seed)
    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    z = torch.randn(4, 2, 16, 128)
    t = torch.full((4, 1, 1, 1), tau)
    (x_t, eps) = path.sample_conditional_path(z, t)
    u_A = path.target_velocity_formA(x_t, z, t, eps)
    u_B = path.target_velocity_formB(x_t, z, t)
    err = (u_A - u_B).abs().max().item()
    print(f'sanity_check_two_forms(tau={tau}): max |A - B| = {err:.2e}')
    if err < 0.0001:
        print('  ✅ Form A ≡ Form B  (as expected)')
    else:
        print('  ⚠️  Forms diverge by more than 1e-4 — bug somewhere.')
    return err

class VectorFieldNet(nn.Module, ABC):
    """Abstract: a network that predicts the velocity field u_t^θ(x | c)."""

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ...

class Trainer(ABC):
    """Generic FM trainer skeleton — handles optimizer + step loop.
    Subclasses implement `get_train_loss(batch_size)`."""

    def __init__(self, net: nn.Module, lr: float=0.001):
        self.net = net
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_history: list = []

    @abstractmethod
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        ...

    def train(self, num_steps: int, batch_size: int=64, print_every: int=100):
        """Train for num_steps. Live tqdm progress bar with loss postfix.

        `print_every` is now the rolling-mean window for the smoothed "avg" loss
        shown in tqdm's postfix (no longer a print stride — tqdm self-throttles).
        """
        from tqdm.auto import tqdm
        self.net.train()
        pbar = tqdm(range(num_steps), desc='train', leave=True)
        for step in pbar:
            self.opt.zero_grad()
            loss = self.get_train_loss(batch_size)
            loss.backward()
            self.opt.step()
            self.loss_history.append(loss.item())
            window = min(print_every, len(self.loss_history))
            avg = float(np.mean(self.loss_history[-window:]))
            pbar.set_postfix({'loss': f'{loss.item():.4f}', f'avg{window}': f'{avg:.4f}'})
        return self.loss_history
from dataset.data_1d import Burgers1D
from model.burgers_1d.unet import Unet2D
from diffusion.diffusion_1d_burgers import sigmoid_schedule_flip
BURGERS_DATASET_NAME = 'free_u_f_1e5_front_rear_quarter'
T_IDX = 10
BURGERS_DATASET_NAME = 'free_u_f_1e5_front_rear_quarter'
T_IDX = 10

def load_burgers_test(device: str='cpu', dataset: str=BURGERS_DATASET_NAME) -> Burgers1D:
    """Same as load_burgers_train but loads the test split (held-out 2000 samples)."""
    return Burgers1D(dataset='burgers', input_steps=1, output_steps=10, time_interval=1, is_y_diff=False, split='test', transform=None, pre_transform=None, verbose=False, root_path=os.path.join(ROOT, 'data', dataset), device=device, rescaler=10.0, stack_u_and_f=True, pad_for_2d_conv=True, partially_observed_fill_zero_unobserved=None, nt_total=11)

def load_burgers_train(device: str='cpu', dataset: str=BURGERS_DATASET_NAME) -> Burgers1D:
    """Load the Burgers training dataset (8000 samples)."""
    return Burgers1D(dataset='burgers', input_steps=1, output_steps=10, time_interval=1, is_y_diff=False, split='train', transform=None, pre_transform=None, verbose=False, root_path=os.path.join(ROOT, 'data', dataset), device=device, rescaler=10.0, stack_u_and_f=True, pad_for_2d_conv=True, partially_observed_fill_zero_unobserved=None, nt_total=11)

def visualize_trajectory(x: torch.Tensor, title: str='', save_path: str | None=None):
    """Visualize one Burgers trajectory (or a small batch of them).

    Args:
        x: shape (2, 16, 128) or (b, 2, 16, 128)
    """
    import matplotlib.pyplot as plt
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.detach().cpu().numpy()
    b = x.shape[0]
    (fig, axes) = plt.subplots(b, 2, figsize=(12, 3 * b))
    if b == 1:
        axes = axes.reshape(1, -1)
    for i in range(b):
        axes[i, 0].imshow(x[i, 0, :11], aspect='auto', cmap='RdBu_r', vmin=-x[i, 0, :11].max(), vmax=x[i, 0, :11].max())
        axes[i, 0].set_title(f'sample {i}: u(t, x)')
        axes[i, 0].set_xlabel('space')
        axes[i, 0].set_ylabel('time')
        axes[i, 1].imshow(x[i, 1, :11], aspect='auto', cmap='PiYG', vmin=-x[i, 1, :11].max(), vmax=x[i, 1, :11].max())
        axes[i, 1].set_title(f'sample {i}: w(t, x)  [control]')
        axes[i, 1].set_xlabel('space')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f'  saved {save_path}')
    else:
        plt.show()
    plt.close()

def visualize_noisy_samples(z: torch.Tensor, save_path: str | None=None):
    """Show how x_t = α_t·z + β_t·ε looks at several τ values.
    Builds intuition for what the FM model sees during training.
    """
    import matplotlib.pyplot as plt
    (alpha, beta) = (LinearAlpha(), LinearBeta())
    taus = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    (fig, axes) = plt.subplots(1, len(taus), figsize=(3 * len(taus), 3))
    for (i, tau) in enumerate(taus):
        t = torch.tensor(tau)
        eps = torch.randn_like(z)
        x_t = alpha(t) * z + beta(t) * eps
        axes[i].imshow(x_t[0, 0, :11].detach().cpu().numpy(), aspect='auto', cmap='RdBu_r')
        axes[i].set_title(f'τ={tau}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    fig.suptitle('u-channel at increasing τ (noise → data)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f'  saved {save_path}')
    else:
        plt.show()
    plt.close()

class BurgersDataset(LabeledSampleable):
    """Wraps the Burgers1D dataset, exposing (z, c) pairs.

    z: (b, 2, 16, 128)  full trajectory
    c:   (b, 2, 128)      (u_0, u_T*) — extracted from z itself

    The reason c lives separately even though it's "in" z: at training time
    the model sees noisy x_t (where boundary rows are also noised), so we keep
    the *clean* boundary in c and re-inject it via inpainting.
    """

    def __init__(self, dataset: Burgers1D, device: str='cpu'):
        self.ds = dataset
        self.device = device
        all_z = torch.stack([self.ds[i] for i in range(len(self.ds))], dim=0)
        self.all_z = all_z.to(device)
        self.N = self.all_z.shape[0]

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z: (b, 2, 16, 128) — full clean trajectory
            c:   (b, 2, 128)     — (u_0, u_T*) extracted from z's u-channel
        """
        idx = torch.randint(0, self.N, (num_samples,))
        z = self.all_z[idx]
        c = torch.stack([z[:, 0, 0, :], z[:, 0, T_IDX, :]], dim=1)
        return (z, c)

class BurgersFlowTrainer(Trainer):
    """Trains the joint Flow Matching model u_t^θ(x | c).

    Loss = ||u_t^θ(x_t | c) - u^target(x_t | z)||²  (averaged over t, z, ε).

    Inpainting trick (§4.4 in `notes_diffphycon_flow_bridge.md`):
        We also want the model to learn `if c is overwritten into x at row 0
        and row T, then your output at those rows should be 0` (= don't push
        the clean boundary anywhere). We enforce this by zeroing the *target*
        velocity at row 0 and row T_IDX in u-channel.
    """

    def __init__(self, net: VectorFieldNet, path: GaussianConditionalProbabilityPath, data: BurgersDataset, lr: float=0.001):
        super().__init__(net, lr=lr)
        self.path = path
        self.data = data

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        (z, c) = self.data.sample(batch_size)
        t = torch.rand(batch_size, 1, 1, 1, device=z.device)
        (x_t, eps) = self.path.sample_conditional_path(z, t)
        u_target = self.path.target_velocity(x_t, z, t, eps)
        u_target[:, 0, 0, :] = 0
        u_target[:, 0, T_IDX, :] = 0
        u_pred = self.net(x_t, t, c)
        return ((u_pred - u_target) ** 2).mean()

class BurgersVectorField(VectorFieldNet):
    """Wraps Unet2D as a conditional velocity field.

    The trick: there is no embedding for c. Instead, every forward call
    *overwrites the boundary rows of x* with the clean c values before
    passing through the Unet. The Unet then operates as if its input
    already had the boundary baked in. This is exactly what we do at
    sampling time too (see Q3.1), so train/inference are consistent.

    See `notes_diffphycon_flow_bridge.md §4.3`.
    """

    def __init__(self, dim: int=64, dim_mults: Tuple[int, ...]=(1, 2, 4, 8)):
        super().__init__()
        self.unet = Unet2D(dim=dim, dim_mults=dim_mults, channels=2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, 2, 16, 128)  noisy trajectory at FM time t
            t: (b,) or (b, 1, 1, 1)  FM time
            c: (b, 2, 128)  boundary (u_0, u_T*)
        Returns:
            v: (b, 2, 16, 128)  predicted velocity field
        """
        x_in = x.clone()
        x_in[:, 0, 0, :] = c[:, 0]
        x_in[:, 0, T_IDX, :] = c[:, 1]
        t_flat = t.view(-1).to(x_in.dtype)
        return self.unet(x_in, t_flat)

def inpaint_overwrite(x: torch.Tensor, c: torch.Tensor, T_idx: int=T_IDX) -> torch.Tensor:
    """Force x's u-channel boundary rows to equal c (clean values, no noise).

    Called *before every Euler step* during sampling. This is the DiffPhyCon
    way of injecting the hard conditioning constraint — see
    `notes_diffphycon_flow_bridge.md §4.3`.

    Args:
        x: (b, 2, 16, 128) current sample
        c: (b, 2, 128) where c[:, 0] = u_0 and c[:, 1] = u_T*
        T_idx: index of u_T* row (default 10)

    Returns:
        x with rows 0 and T_idx of channel 0 replaced by c.
    """
    x = x.clone()
    x[:, 0, 0, :] = c[:, 0]
    x[:, 0, T_idx, :] = c[:, 1]
    return x

class BurgersEulerSampler:
    """Euler ODE sampler with inpainting overwrite.

    The flow ODE is:
        dx/dτ = v_τ^θ(x | c)
    We integrate from τ=0 (noise) to τ=1 (clean data) using Euler steps,
    with inpainting overwrite *before each step* to enforce the boundary.
    """

    def __init__(self, net: VectorFieldNet, n_steps: int=100, tau_min: float=0.001):
        self.net = net
        self.n_steps = n_steps
        self.tau_min = tau_min

    @torch.no_grad()
    def sample(self, c: torch.Tensor, shape: Tuple[int, ...]=(2, 16, 128)) -> torch.Tensor:
        """
        Args:
            c: (b, 2, 128) boundary condition
            shape: per-sample shape, default (2, 16, 128)
        Returns:
            x_final: (b, *shape)
        """
        b = c.shape[0]
        device = c.device
        dtau = (1.0 - 2 * self.tau_min) / self.n_steps
        x = torch.randn(b, *shape, device=device)
        x = inpaint_overwrite(x, c)
        for i in range(self.n_steps):
            tau = self.tau_min + i * dtau
            tau_tensor = torch.full((b,), tau, device=device)
            v = self.net(x, tau_tensor, c)
            x = x + v * dtau
            x = inpaint_overwrite(x, c)
        return x

class BurgersPriorDataset(BurgersDataset):
    """Dataset for training the prior p(w | c). Same as BurgersDataset but with u-channel zeroed.

    Why: the prior model should only see (and predict) w. By zeroing u in both
    input and output, we cleanly embed `u_prior` into the joint (u,w) space
    with u-block = 0 — matching the math in `notes_fm_prior_reweighting.md §2.4`.

    The DDPM equivalent is `diffusion_1d_burgers.py:400-402` (see comments there).
    """

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        (z, c) = super().sample(num_samples)
        z[:, 0] = 0
        return (z, c)

class BurgersPriorTrainer(BurgersFlowTrainer):
    """Trains net_prior. Inherits from BurgersFlowTrainer; only difference is
    the target velocity for the u-channel must be forced to 0 (we don't want
    the prior model to learn anything about u dynamics)."""

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        (z, c) = self.data.sample(batch_size)
        t = torch.rand(batch_size, 1, 1, 1, device=z.device)
        (x_t, eps) = self.path.sample_conditional_path(z, t)
        u_target = self.path.target_velocity(x_t, z, t, eps)
        u_target[:, 0, 0, :] = 0
        u_target[:, 0, T_IDX, :] = 0
        u_target[:, 0] = 0
        u_pred = self.net(x_t, t, c)
        u_pred[:, 0] = 0
        loss = ((u_pred - u_target) ** 2).mean()
        return loss

def w_scheduler_fm(tau: torch.Tensor) -> torch.Tensor:
    """FM-time equivalent of DDPM's `sigmoid_schedule_flip`.

    DDPM sigmoid_flip(t) is small at t→999 (noisy end) and large at t→0 (clean end).
    In FM time τ ∈ [0, 1]: τ=0 corresponds to DDPM t=999, τ=1 corresponds to t=0.

    See `notes_fm_prior_reweighting.md §3 Step 5`.

    Args:
        tau: (b,) or scalar in [0, 1]
    Returns:
        η̃(τ) — same shape as tau
    """
    orig_device = tau.device
    t_ddpm = ((1.0 - tau) * 999).round().long().clamp(min=0, max=999).cpu()
    eta = sigmoid_schedule_flip(t_ddpm).float().to(orig_device)
    return eta

class ReweightedVectorField(VectorFieldNet):
    """The γ-reweighted velocity field:

        ũ_τ(x|c) = u_joint(x|c) + (γ-1) · η̃(τ) · [u_prior(x|c) - b_τ · [0, w]]

    where:
      - b_τ = α̇_τ / α_τ = 1/τ   (for CondOT path α_τ = τ)
      - η̃(τ) is the FM-time sigmoid_flip schedule from Q4.3
      - [0, w] = the x vector with u-channel zeroed (only w survives)

    Full derivation: `notes_fm_prior_reweighting.md §3`.

    When γ = 1 the correction term vanishes and this reduces to net_joint(x, t, c)
    exactly (sanity-checked in Q4.5).
    """

    def __init__(self, net_joint: VectorFieldNet, net_prior: VectorFieldNet, gamma: float=1.0, use_scheduler: bool=True, tau_min: float=0.001):
        super().__init__()
        self.net_joint = net_joint
        self.net_prior = net_prior
        self.gamma = gamma
        self.use_scheduler = use_scheduler
        self.tau_min = tau_min

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, 2, 16, 128)
            t: (b,)  in [tau_min, 1 - tau_min]
            c: (b, 2, 128)
        """
        v_joint = self.net_joint(x, t, c)
        x_for_prior = x.clone()
        x_for_prior[:, 0] = 0
        v_prior = self.net_prior(x_for_prior, t, c)
        x_w_only = x.clone()
        x_w_only[:, 0] = 0
        t_safe = t.clamp(min=self.tau_min)
        b_t = (1.0 / t_safe).view(-1, 1, 1, 1)
        eta = w_scheduler_fm(t.view(-1)) if self.use_scheduler else torch.ones_like(t.view(-1))
        eta = eta.view(-1, 1, 1, 1).to(x.dtype)
        correction = v_prior - b_t * x_w_only
        v_rw = v_joint + (self.gamma - 1) * eta * correction
        return v_rw
DDPM_BASELINE_FOPC = {0.3: {'J': 0.0083, 'Energy': 1670.7}, 0.5: {'J': 0.00828, 'Energy': 1666.0}, 0.7: {'J': 0.00825, 'Energy': 1661.8}, 0.9: {'J': 0.00821, 'Energy': 1658.0}, 1.0: {'J': 0.0082, 'Energy': 1656.2}, 1.5: {'J': 0.00811, 'Energy': 1648.1}, 2.5: {'J': 0.00796, 'Energy': 1634.0}}

def simulate_with_predicted_w(x_pred: torch.Tensor, c: torch.Tensor, rescaler: float=10.0):
    """Run the PDE solver forward with predicted w and return the simulated u(t,x)."""
    from dataset.apps.generate_burgers import burgers_numeric_solve_free
    x_pred = x_pred.detach().cpu() * rescaler
    c_un = c.detach().cpu() * rescaler
    u_0 = c_un[:, 0]
    w = x_pred[:, 1, :10, :].clone()
    Nx = w.shape[-1]
    w[:, :, Nx // 4:3 * Nx // 4] = 0
    u_sim = burgers_numeric_solve_free(u_0, w, visc=0.01, T=1.0, dt=0.0001, num_t=10)
    return (u_sim, w)

def visualize_trajectory_with_simulation(x_pred: torch.Tensor, c: torch.Tensor, title: str='', save_path: str | None=None, rescaler: float=10.0):
    """5-panel viz (B+A enhanced):
    col 0: predicted u(t, x)     ┐ shared color scale
    col 1: simulated u(t, x)     ┘ (so col 0 vs col 1 visually comparable)
    col 2: diff = sim − pred     — highlights where model diverges from physics
    col 3: predicted w(t, x)     — FOPC dashed lines at x=0.25, 0.75
    col 4: terminal comparison   — u_0, u_T*, sim u(T), pred u(T) line plot
    """
    import matplotlib.pyplot as plt
    (u_sim_t, w_used) = simulate_with_predicted_w(x_pred, c, rescaler=rescaler)
    u_sim = u_sim_t.numpy()
    u_pred = (x_pred[:, 0, :11].detach().cpu() * rescaler).numpy()
    w_pred = w_used.numpy()
    u_0 = (c[:, 0].detach().cpu() * rescaler).numpy()
    u_T = (c[:, 1].detach().cpu() * rescaler).numpy()
    diff = u_sim - u_pred
    b = x_pred.shape[0]
    nx = u_pred.shape[-1]
    x_axis = np.linspace(0, 1, nx)
    (fig, axes) = plt.subplots(b, 5, figsize=(22, 3.5 * b))
    if b == 1:
        axes = axes.reshape(1, -1)
    for row in range(b):
        u_vmax = max(abs(u_pred[row]).max(), abs(u_sim[row]).max(), 1e-06)
        ax = axes[row, 0]
        im0 = ax.imshow(u_pred[row], aspect='auto', origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=-u_vmax, vmax=u_vmax)
        ax.set_title(f'#{row}: predicted u(t,x)')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
        ax = axes[row, 1]
        im1 = ax.imshow(u_sim[row], aspect='auto', origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=-u_vmax, vmax=u_vmax)
        ax.set_title('simulated u(t,x)\n(PDE solve, same color ←)')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        ax = axes[row, 2]
        diff_vmax = max(abs(diff[row]).max(), 1e-06)
        im2 = ax.imshow(diff[row], aspect='auto', origin='lower', extent=[0, 1, 0, 1], cmap='seismic', vmin=-diff_vmax, vmax=diff_vmax)
        ax.set_title(f'diff = sim − pred\nmax|err| = {diff_vmax:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        ax = axes[row, 3]
        w_vmax = max(abs(w_pred[row]).max(), 1e-06)
        im3 = ax.imshow(w_pred[row], aspect='auto', origin='lower', extent=[0, 1, 0, 1], cmap='PiYG', vmin=-w_vmax, vmax=w_vmax)
        ax.set_title('predicted w(t,x)\n(middle 1/2 = 0 — FOPC)')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.axvline(0.25, color='k', lw=0.6, ls='--')
        ax.axvline(0.75, color='k', lw=0.6, ls='--')
        plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
        ax = axes[row, 4]
        ax.plot(x_axis, u_0[row], label='$u_0$', color='gray', lw=1)
        ax.plot(x_axis, u_T[row], label='$u_T^*$ (target)', color='k', lw=2)
        ax.plot(x_axis, u_sim[row, -1], label='sim $u(T)$', color='C0', ls='--')
        ax.plot(x_axis, u_pred[row, 10], label='pred $u(T)$', color='C1', ls=':')
        ax.set_title('terminal-state comparison')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=110, bbox_inches='tight')
        print(f'  saved {save_path}')
    else:
        plt.show()
    plt.close()

def compute_J_and_energy(x_pred: torch.Tensor, c: torch.Tensor, rescaler: float=10.0) -> Tuple[float, float]:
    """Compute J = ||u_sim(T) - u_T*||² and Energy = ||w||² for a single prediction batch.

    Now with FOPC partial-control alignment: w is zeroed in the middle 50% of space
    before PDE simulation (matches DDPM baseline).
    """
    from utils import burgers_metric
    b = x_pred.shape[0]
    x_pred = x_pred.detach().cpu() * rescaler
    c_un = c.detach().cpu() * rescaler
    Nt = 11
    Nx = c_un.shape[-1]
    u_target = torch.zeros(b, Nt, Nx)
    u_target[:, 0] = c_un[:, 0]
    u_target[:, 10] = c_un[:, 1]
    w = x_pred[:, 1, :10, :]
    (J_per_sample, E_per_sample) = burgers_metric(u_target, w, target='final_u', partial_control='front_rear_quarter')
    return (float(J_per_sample.mean()), float(E_per_sample.mean()))

def plot_loss_history(losses, save_path: Optional[str]=None, title: str='Training loss', window: int=100):
    """Plot a loss curve from a list/array OR from a .pt checkpoint path."""
    import matplotlib.pyplot as plt
    if isinstance(losses, (str, os.PathLike)):
        ckpt = torch.load(losses, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'loss_history' in ckpt:
            losses = ckpt['loss_history']
        else:
            raise ValueError(f"{losses} doesn't contain a 'loss_history' key")
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        print('  (empty loss history)')
        return
    w = min(window, losses.size)
    kernel = np.ones(w) / w
    smoothed = np.convolve(losses, kernel, mode='valid')
    smoothed_x = np.arange(w - 1, losses.size)
    (fig, ax) = plt.subplots(figsize=(9, 4.5))
    ax.plot(losses, color='lightsteelblue', lw=0.5, alpha=0.6, label='per-step')
    ax.plot(smoothed_x, smoothed, color='navy', lw=1.5, label=f'rolling mean (w={w})')
    ax.set_yscale('log')
    ax.set_xlabel('training step')
    ax.set_ylabel('loss (log scale)')
    ax.set_title(f'{title}  (n={losses.size}, final smoothed = {smoothed[-1]:.4f})')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=110, bbox_inches='tight')
        print(f'  saved loss plot: {save_path}')
    else:
        plt.show()
    plt.close()

def _train_with_checkpoints(trainer, num_steps, batch_size, save_path, checkpoint_every, print_every=200):
    if save_path is None or checkpoint_every <= 0 or checkpoint_every >= num_steps:
        trainer.train(num_steps=num_steps, batch_size=batch_size, print_every=print_every)
        return
    (base, ext) = os.path.splitext(save_path)
    loss_plot_path = f'{base}_losses.png'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    done = 0
    while done < num_steps:
        chunk = min(checkpoint_every, num_steps - done)
        trainer.train(num_steps=chunk, batch_size=batch_size, print_every=print_every)
        done += chunk
        ckpt = {'state_dict': trainer.net.state_dict(), 'loss_history': list(trainer.loss_history), 'step': done}
        torch.save(ckpt, save_path)
        torch.save(ckpt, f'{base}_step{done}{ext}')
        plot_loss_history(trainer.loss_history, save_path=loss_plot_path, title=f'{os.path.basename(base)}  (step {done}/{num_steps})')
        print(f'  [checkpoint] step {done}/{num_steps}  →  {base}_step{done}{ext}')

def train_joint_for_part5(num_steps: int=5000, batch_size: int=64, lr: float=0.0001, save_path: Optional[str]=None, checkpoint_every: int=1000, device: Optional[str]=None):
    """Joint p(u,w|c) FM training. Periodic checkpoint + loss plot."""
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'  device: {device}')
    ds_full = load_burgers_train(device=device, dataset='free_u_f_1e4_front_rear_quarter')
    ds = BurgersDataset(ds_full, device=device)
    print(f"  dataset size: {ds.N} samples (vs sanity's 160)")
    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    net = BurgersVectorField(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
    print(f'  net params: {sum((p.numel() for p in net.parameters())):,}')
    print(f'  training {num_steps} steps (checkpoint every {checkpoint_every})...')
    trainer = BurgersFlowTrainer(net, path, ds, lr=lr)
    _train_with_checkpoints(trainer, num_steps, batch_size, save_path, checkpoint_every)
    print(f'  done. final loss (smoothed): {float(np.mean(trainer.loss_history[-100:])):.5f}')
    return net

def train_prior_for_part5(num_steps: int=1500, batch_size: int=64, lr: float=0.0001, save_path: Optional[str]=None, checkpoint_every: int=500, device: Optional[str]=None):
    """Prior p(w|c) FM training. Periodic checkpoint + loss plot."""
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'  device: {device}')
    ds_full = load_burgers_train(device=device, dataset='free_u_f_1e4_front_rear_quarter')
    ds = BurgersPriorDataset(ds_full, device=device)
    print(f'  dataset size: {ds.N} samples')
    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    net = BurgersVectorField(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
    print(f'  net params: {sum((p.numel() for p in net.parameters())):,}')
    print(f'  training {num_steps} steps (prior — u-channel = 0; ckpt every {checkpoint_every})...')
    trainer = BurgersPriorTrainer(net, path, ds, lr=lr)
    _train_with_checkpoints(trainer, num_steps, batch_size, save_path, checkpoint_every)
    print(f'  done. final loss (smoothed): {float(np.mean(trainer.loss_history[-100:])):.5f}')
    return net

class EMA:
    """Maintains an exponential moving average of model parameters.
    
    Standard DDPM choice: decay=0.995, effective averaging window ~200 steps.
    Smooths out gradient-noise wobble → cleaner sampling outputs.
    """

    def __init__(self, model: nn.Module, decay: float=0.995):
        self.decay = decay
        self.shadow = {name: p.detach().clone() for (name, p) in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for (name, p) in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for (name, p) in model.named_parameters():
            p.data.copy_(self.shadow[name])

def finetune_with_ema(net: nn.Module, trainer_class: type, dataset, num_steps: int=2000, batch_size: int=64, lr: float=0.0001, ema_decay: float=0.995, save_path_ema: Optional[str]=None):
    """Continue training `net` for `num_steps` while building EMA shadow.

    Returns a NEW net with EMA-smoothed weights (original `net` is also
    trained in-place, so caller has both for comparison).
    """
    import copy
    from tqdm.auto import tqdm
    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    trainer = trainer_class(net, path, dataset, lr=lr)
    ema = EMA(net, decay=ema_decay)
    print(f'  fine-tuning {num_steps} steps with EMA decay={ema_decay}...')
    pbar = tqdm(range(num_steps), desc='finetune+EMA')
    net.train()
    for step in pbar:
        trainer.opt.zero_grad()
        loss = trainer.get_train_loss(batch_size)
        loss.backward()
        trainer.opt.step()
        trainer.loss_history.append(loss.item())
        ema.update(net)
        window = min(100, len(trainer.loss_history))
        avg = float(np.mean(trainer.loss_history[-window:]))
        pbar.set_postfix({'loss': f'{loss.item():.4f}', f'avg{window}': f'{avg:.4f}'})
    net_ema = copy.deepcopy(net).eval()
    ema.copy_to(net_ema)
    if save_path_ema:
        os.makedirs(os.path.dirname(save_path_ema) or '.', exist_ok=True)
        torch.save({'state_dict': net_ema.state_dict()}, save_path_ema)
        print(f'  saved EMA-weighted net: {save_path_ema}')
    print(f'  done. final loss (smoothed): {float(np.mean(trainer.loss_history[-100:])):.5f}')
    return net_ema

def inference_and_plot(net_joint: VectorFieldNet, net_prior: Optional[VectorFieldNet]=None, gamma: float=1.0, n_samples: int=3, n_steps: int=100, use_scheduler: bool=True, save_path: Optional[str]=None, device: Optional[str]=None, dataset_name: str='free_u_f_1e4_front_rear_quarter', split: str='train', seed: Optional[int]=None):
    """Run inference on trained net(s) → produce 5-panel viz. Returns (x_pred, c, J, E).
    
    seed: if set, fixes BOTH the boundary c draw AND the sampler's initial x_0
          (since torch RNG is global). Use this to compare across n_steps fairly.
    """
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    if save_path is None:
        save_path = os.path.join(HERE, 'lab_four_inference.png')
    loader = load_burgers_train if split == 'train' else load_burgers_test
    ds_full = loader(device=device, dataset=dataset_name)
    ds = BurgersDataset(ds_full, device=device)
    if seed is not None:
        torch.manual_seed(seed)
    (_, c) = ds.sample(n_samples)
    print(f'  dataset split: {split} ({ds.N} samples), seed={seed}')
    if net_prior is None or gamma == 1.0:
        print(f'  inference: joint only (γ=1, no reweighting)')
        net = net_joint
    else:
        print(f'  inference: reweighted with γ={gamma}, scheduler={use_scheduler}')
        net = ReweightedVectorField(net_joint, net_prior, gamma=gamma, use_scheduler=use_scheduler).to(device)
    net.eval()
    sampler = BurgersEulerSampler(net, n_steps=n_steps)
    x_pred = sampler.sample(c)
    print(f'  sampled {n_samples} trajectories, shape={tuple(x_pred.shape)}')
    (J, E) = compute_J_and_energy(x_pred, c)
    (baseline_J, baseline_E) = (0.0082, 1656)
    print(f'  J      = {J:.4f}   (DDPM baseline γ=1: {baseline_J})  →  {J / baseline_J:.1f}x')
    print(f'  Energy = {E:.1f}     (DDPM baseline γ=1: {baseline_E})')
    title = f'Inference: γ={gamma}, {n_samples} samples, {n_steps}-step Euler, seed={seed}'
    visualize_trajectory_with_simulation(x_pred, c, title=title, save_path=save_path)
    return (x_pred, c, J, E)

def compare_n_steps_visually(net_joint: VectorFieldNet, net_prior: Optional[VectorFieldNet]=None, gamma: float=2.5, n_steps_list: list=[100, 500, 1000], seed: int=42, n_samples: int=3, save_dir: Optional[str]=None):
    """Loop inference_and_plot over n_steps with shared seed → N PNGs for visual comparison.

    Because seed is fixed, all calls use the same boundary c and same initial
    noise x_0 — only n_steps differs, so visual diff is purely ODE precision.
    """
    if save_dir is None:
        save_dir = os.path.join(ROOT, 'flow')
    print(f'\n=== compare_n_steps_visually: γ={gamma}, seed={seed} ===')
    results = []
    for n in n_steps_list:
        save_path = os.path.join(save_dir, f'lab_four_inf_gamma{gamma}_n{n}_seed{seed}.png')
        print(f'\n--- n_steps={n} ---')
        (_, _, J, E) = inference_and_plot(net_joint, net_prior, gamma=gamma, n_samples=n_samples, n_steps=n, save_path=save_path, seed=seed)
        results.append({'n_steps': n, 'J': J, 'Energy': E, 'path': save_path})
    print(f'\n--- summary ---')
    print(f"{'n_steps':>8s} | {'J':>10s} | {'Energy':>10s}")
    for r in results:
        print(f"{r['n_steps']:>8d} | {r['J']:>10.5f} | {r['Energy']:>10.1f}")
    return results

def part5_gamma_sweep(net_joint: VectorFieldNet, net_prior: VectorFieldNet, n_samples: int=8, n_steps: int=100, device: Optional[str]=None, dataset_name: str='free_u_f_1e4_front_rear_quarter', split: str='train', seed: Optional[int]=None, save_path: Optional[str]=None):
    """Sweep γ ∈ {0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.5} and compare to DDPM baseline.

    Auto-detects device from net_joint if device=None.
    `split="test"` uses held-out 2000 test samples.
    `seed` (int): fixes which n_samples boundary pairs get drawn.
    `save_path`: where to save the gamma-sweep PNG. Defaults to flow/lab_four_gamma_sweep.png
                 (gets overwritten across runs — pass distinct names to keep multiple).
    """
    if device is None:
        device = str(next(net_joint.parameters()).device)
    loader = load_burgers_train if split == 'train' else load_burgers_test
    ds = BurgersDataset(loader(device=device, dataset=dataset_name), device=device)
    if seed is not None:
        torch.manual_seed(seed)
    (_, c) = ds.sample(n_samples)
    print(f'  using split={split!r} ({ds.N} samples available), seed={seed}')
    gammas = [0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.5]
    results = {}
    import time
    from tqdm.auto import tqdm
    print('\n' + '=' * 92)
    print(f"{'γ':>5s} | {'FM J':>10s} | {'DDPM J':>10s} | {'FM Energy':>10s} | {'DDPM Energy':>12s} | {'wall (s)':>9s}")
    print('-' * 92)
    pbar = tqdm(gammas, desc='γ sweep')
    for g in pbar:
        t0 = time.time()
        rw = ReweightedVectorField(net_joint, net_prior, gamma=g, use_scheduler=True).to(device)
        sampler = BurgersEulerSampler(rw, n_steps=n_steps)
        x_pred = sampler.sample(c)
        (J, E) = compute_J_and_energy(x_pred, c)
        elapsed = time.time() - t0
        results[g] = {'J': J, 'Energy': E, 'wall_s': elapsed}
        b = DDPM_BASELINE_FOPC[g]
        pbar.set_postfix({'γ': g, 'FM J': f'{J:.4f}'})
        tqdm.write(f"{g:>5.1f} | {J:>10.5f} | {b['J']:>10.5f} | {E:>10.1f} | {b['Energy']:>12.1f} | {elapsed:>8.1f}s")
    print('=' * 92)
    if save_path is None:
        save_path = os.path.join(ROOT, 'flow', 'lab_four_gamma_sweep.png')
    try:
        import matplotlib.pyplot as plt
        (fig, axes) = plt.subplots(1, 2, figsize=(12, 4))
        gs = list(results.keys())
        axes[0].plot(gs, [results[g]['J'] for g in gs], 'o-', label='FM (ours)')
        axes[0].plot(gs, [DDPM_BASELINE_FOPC[g]['J'] for g in gs], 's--', label='DDPM (baseline)')
        axes[0].set_xlabel('γ')
        axes[0].set_ylabel('J')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(gs, [results[g]['Energy'] for g in gs], 'o-', label='FM')
        axes[1].plot(gs, [DDPM_BASELINE_FOPC[g]['Energy'] for g in gs], 's--', label='DDPM')
        axes[1].set_xlabel('γ')
        axes[1].set_ylabel('Energy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.suptitle(f'FM vs DDPM, n_samples={n_samples}, n_steps={n_steps}, seed={seed}')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'\nSaved plot: {save_path}')
        plt.close()
    except Exception as e:
        print(f'(plot failed: {e})')
    return results

def sanity_check_part1():
    """Just load data and visualize — no fill-ins needed."""
    print('\n=== Sanity Check Part 1: visualize Burgers data ===')
    ds = load_burgers_train()
    print(f'  dataset has {len(ds)} samples; one sample shape = {ds[0].shape}')
    x = torch.stack([ds[i] for i in range(3)])
    visualize_trajectory(x, 'Burgers training data — 3 samples', save_path=os.path.join(HERE, 'lab_four_part1_data.png'))
    visualize_noisy_samples(x[:1], save_path=os.path.join(HERE, 'lab_four_part1_noisy.png'))

def sanity_check_2_4(num_steps: int=500, batch_size: int=32):
    """After filling Q2.1-Q2.3: train the joint FM model for a few hundred steps
    and verify (a) loss curve goes down, (b) sample output is not pure noise."""
    print('\n=== Sanity Check 2.4: joint FM trains and samples ===')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'  device: {device}')
    ds = BurgersDataset(load_burgers_train(device=device), device=device)
    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    net = BurgersVectorField(dim=32, dim_mults=(1, 2, 4)).to(device)
    trainer = BurgersFlowTrainer(net, path, ds, lr=0.001)
    losses = trainer.train(num_steps=num_steps, batch_size=batch_size, print_every=50)
    (_, c_test) = ds.sample(1)
    sampler = BurgersEulerSampler(net, n_steps=50)
    x_pred = sampler.sample(c_test)
    print(f'  predicted trajectory range: u in [{x_pred[0, 0, :11].min():.2f}, {x_pred[0, 0, :11].max():.2f}]')
    print(f'  predicted trajectory range: w in [{x_pred[0, 1, :11].min():.2f}, {x_pred[0, 1, :11].max():.2f}]')
    early = float(np.mean(losses[:20]))
    late = float(np.mean(losses[-20:]))
    print(f'  loss early avg = {early:.4f}, late avg = {late:.4f}')
    if late < early * 0.5:
        print('  ✅ Loss dropped > 2x — joint FM training is working.')
    else:
        print("  ⚠️  Loss didn't drop much; check Q2.2 implementation.")

def sanity_check_3_3(num_train_steps: int=300, n_sample_steps: int=50):
    """After filling Q3.1, Q3.2: train small net, sample with γ=1, visualize, compute J.

    Self-contained — trains a small net inside so you don't need to wire one in
    from sanity_check_2_4. Fast sanity, NOT paper quality.

    NOTE: viz is now 4-panel (mirrors plot_inference.py): pred u | pred w | sim u | terminal.
    """
    print('\n=== Sanity Check 3.3: sample with γ=1 and compute J ===')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'  device: {device}')
    ds = BurgersDataset(load_burgers_train(device=device), device=device)
    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    net = BurgersVectorField(dim=32, dim_mults=(1, 2, 4)).to(device)
    trainer = BurgersFlowTrainer(net, path, ds, lr=0.001)
    print(f'  training {num_train_steps} steps (small net, ~30s on M4 Pro)...')
    trainer.train(num_steps=num_train_steps, batch_size=32, print_every=100)
    net.eval()
    sampler = BurgersEulerSampler(net, n_steps=n_sample_steps)
    (_, c) = ds.sample(3)
    x_pred = sampler.sample(c)
    print(f'  sampled shape: {tuple(x_pred.shape)}')
    (J, E) = compute_J_and_energy(x_pred, c)
    baseline_J = 0.0082
    baseline_E = 1656
    print(f'  J      = {J:.4f}   (DDPM baseline γ=1: {baseline_J})')
    print(f'  Energy = {E:.1f}     (DDPM baseline γ=1: {baseline_E})')
    print(f'  J / baseline = {J / baseline_J:.1f}x')
    if J < baseline_J * 50:
        print(f'  ✅ Within 50x of baseline — Q3 pipeline works (small net, {num_train_steps} train steps).')
    else:
        print(f'  ⚠️  More than 50x worse than baseline — check Q3.1/Q3.2 fill-ins.')
    visualize_trajectory_with_simulation(x_pred, c, title=f'Sampled trajectories (γ=1, joint only, {num_train_steps} train steps)', save_path=os.path.join(HERE, 'lab_four_part3_sample.png'))
    return net

def sanity_check_4_5(net_joint, net_prior, device: str='cpu'):
    """After filling Q4.1-Q4.4: verify γ=1 reweighted == joint output exactly, and γ≠1 differs."""
    print('\n=== Sanity Check 4.5: γ=1 reduces to joint, γ≠1 differs ===')
    ds = BurgersDataset(load_burgers_train(device=device), device=device)
    (_, c) = ds.sample(2)
    x = torch.randn(2, 2, 16, 128, device=device)
    t = torch.full((2,), 0.5, device=device)
    rw_g1 = ReweightedVectorField(net_joint, net_prior, gamma=1.0, use_scheduler=True).to(device).eval()
    rw_g25 = ReweightedVectorField(net_joint, net_prior, gamma=2.5, use_scheduler=True).to(device).eval()
    with torch.no_grad():
        v_g1 = rw_g1(x, t, c)
        v_g25 = rw_g25(x, t, c)
        v_joint = net_joint(x, t, c)
    err_g1 = (v_g1 - v_joint).abs().max().item()
    diff_g25 = (v_g25 - v_joint).abs().max().item()
    print(f'  ||v_γ=1 - v_joint||_∞  = {err_g1:.6e}  (should be ≈ 0)')
    print(f'  ||v_γ=2.5 - v_joint||_∞ = {diff_g25:.6e}  (should be > 0)')
    if err_g1 < 1e-05 and diff_g25 > 0.0001:
        print('  ✅ γ-reweighting math checks out.')
    else:
        print('  ⚠️  Something off; double-check Q4.4 formula and signs.')
if __name__ == '__main__':
    print(__doc__)
    print('\n>>> Running Part 1 sanity (no fill-in needed):')
    print('>>> Once Q4.1-Q4.4 are filled, run:  sanity_check_4_5(net_joint, net_prior)')
    print('>>> When all sanity checks pass, run: part5_gamma_sweep(net_joint, net_prior)')
import os
import torch
device = 'mps'

def load_ckpt(path, device='mps'):
    net = BurgersVectorField(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    return net
import torch
device = 'mps'

def load_ckpt(path):
    net = BurgersVectorField(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    return net
LARGE_DATASET = 'free_u_f_1e5_front_rear_quarter'
