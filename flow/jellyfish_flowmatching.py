
# pyright: reportUnknownMemberType=false
from __future__ import annotations
import math
import os
import sys
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable
import time, json, glob, pickle
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make project root importable so we can use existing diffphycon modules.
# Works both as a .py script (__file__ defined) and inside a Jupyter notebook
# cell (no __file__; fall back to CWD which should be the repo root).
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Running inside Jupyter — assume CWD is the project root or flow/
    HERE = os.path.abspath(os.getcwd())
    if os.path.basename(HERE) != "flow":
        # CWD is repo root, point HERE at flow/
        HERE = os.path.join(HERE, "flow")
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion.diffusion_2d_jellyfish import Unet, ForceUnet, sigmoid_beta_schedule
from model.video_diffusion_pytorch.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D


def load_force_model(device):
    model = ForceUnet(
        dim=64,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
        channels=4,  # physical pressure + boundary(3)
    )

    ckpt = os.path.join(
        DATA_DIR,
        "checkpoints/force_surrogate_model/force_model_epoch_9.pth",
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def jellyfish_objective(
    x_clean,
    c,
    bd_0,
    bd_updater,
    force_model,
    p_min,
    p_max,
    zeta=1000.0,
):
    """
    Returns J per sample: [B]
    J = -average_velocity + zeta * R(theta) + periodicity
    """
    x_clean = inpaint_cond(x_clean, c)
    B, T = x_clean.shape[:2]

    theta = x_clean[:, :, THETA_CH].mean(dim=(-1, -2))  # [B,T]

    # Important: keep grad enabled through boundary updater
    bd_clean = boundary_from_current_theta(
        x_clean, c, bd_0, bd_updater
    )

    # State channel 2 is normalized pressure
    pressure_norm = x_clean[:, :, 2]
    pressure = (
        (0.5 * pressure_norm + 0.5) * (p_max - p_min)
        + p_min
    )

    force_input = torch.cat(
        [pressure.unsqueeze(2), bd_clean],
        dim=2,
    )                                                   # [B,T,4,H,W]

    force = force_model(
        force_input.reshape(B * T, 4, H, W)
    ).reshape(B, T)

    # Compatible with official force_fn
    weights = torch.arange(
        T, 0, -1,
        device=x_clean.device,
        dtype=x_clean.dtype,
    )
    average_velocity = (force * weights).mean(dim=1)

    reg = ((theta[:, 1:] - theta[:, :-1]) ** 2).sum(dim=1)

    theta_0 = c[:, THETA_CH].mean(dim=(-1, -2))
    periodic = F.relu(
        (theta[:, -1] - theta_0).abs() - 0.01
    )

    return -average_velocity + zeta * reg + periodic
# [PROD]  (these imports go into jellyfish_train.py too)

# # (1) STABLE abstractions — reuse, never rewrite
# from flow.jellyfish_flowmatching import (
#     GaussianConditionalProbabilityPath,  # x_t = alpha*z + beta*eps; target velocity = z - eps
#     LinearAlpha, LinearBeta,             # CondOT: alpha=tau, beta=1-tau
#     Trainer,                             # base: Adam + train(num_steps,batch) loop + abstract get_train_loss
#     EMA,                                 # exponential moving average of weights
# )

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
DATA_DIR = os.path.join(ROOT, 'data', 'jellyfish')
T_WIN, H, W, THETA_CH = 20, 64, 64, 3   # 20-frame window; 64x64 grid; theta = channel 3 of (vx,vy,p,theta)
print(f'device={DEVICE}  data_dir={DATA_DIR}')


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
    def target_velocity(
        self,
        x_t: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        return x_t, eps

    # -------- Form A: ε form --------
    def target_velocity_formA(self, x_t, z, t, eps):
        """u^target(x_t|z) = α̇_t · z + β̇_t · ε      (Form A, ε form)

        Direct derivative of the conditional flow ψ_t(x_0|z) = α_t·z + β_t·x_0.
        For our CondOT path (α=t, β=1-t, α̇=1, β̇=-1): u^target = z - ε.
        No division by β_t, so no t→1 singularity. Requires ε from sampler.
        """
        return self.alpha.dt(t) * z + self.beta.dt(t) * eps

    # -------- Form B: x_t form (lab_three style) --------
    def target_velocity_formB(self, x_t, z, t):
        """u^target(x_t|z) = (α̇ − β̇·α/β)·z + (β̇/β)·x_t     (Form B, x_t form)

        Obtained by substituting ε = (x_t − α·z)/β into Form A.
        For our CondOT path: u^target = (z − x_t) / (1 − t).
        Mathematically identical to Form A. Diverges at t = 1 (β → 0);
        callers should clamp t away from 1.
        """
        a   = self.alpha(t)
        b   = self.beta(t)
        da  = self.alpha.dt(t)
        db  = self.beta.dt(t)
        return (da - db * a / b) * z + (db / b) * x_t

    # -------- Dispatcher: which form is "active" --------
    # Change this body to switch between A and B. Both give identical results;
    # see `sanity_check_two_forms()` below to verify.
    def target_velocity(self, x_t, z, t, eps=None):
        """Default = Form A (ε form). Swap to Form B by editing this body.

        eps is required if using Form A; ignored if using Form B.
        """
        # === Active form (uncomment one) ===
        return self.target_velocity_formA(x_t, z, t, eps)
        # return self.target_velocity_formB(x_t, z, t)


def sanity_check_two_forms(seed: int = 0, tau: float = 0.5) -> float:
    """Verify Form A and Form B give numerically identical results.

    Returns max |Form A − Form B| over a random batch. Should be ≤ 1e-5 (float32).
    """
    torch.manual_seed(seed)
    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    z = torch.randn(4, 2, 16, 128)
    t = torch.full((4, 1, 1, 1), tau)
    x_t, eps = path.sample_conditional_path(z, t)
    u_A = path.target_velocity_formA(x_t, z, t, eps)
    u_B = path.target_velocity_formB(x_t, z, t)
    err = (u_A - u_B).abs().max().item()
    print(f"sanity_check_two_forms(tau={tau}): max |A - B| = {err:.2e}")
    if err < 1e-4:
        print("  ✅ Form A ≡ Form B  (as expected)")
    else:
        print("  ⚠️  Forms diverge by more than 1e-4 — bug somewhere.")
    return err


class VectorFieldNet(nn.Module, ABC):
    """Abstract: a network that predicts the velocity field u_t^θ(x | c)."""
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ...


class Trainer(ABC):
    """Generic FM trainer skeleton — handles optimizer + step loop.
    Subclasses implement `get_train_loss(batch_size)`."""
    def __init__(self, net: nn.Module, lr: float = 1e-3):
        self.net = net
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_history: list = []

    @abstractmethod
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        ...

    def train(self, num_steps: int, batch_size: int = 64, print_every: int = 100):
        """Train for `num_steps`. Shows a tqdm progress bar with live loss postfix.

        `print_every` controls the rolling-mean window for the smoothed "avg" loss
        displayed in tqdm's postfix (it's no longer a print stride — tqdm throttles).
        """
        from tqdm.auto import tqdm    # picks notebook-friendly version inside Jupyter

        self.net.train()
        pbar = tqdm(range(num_steps), desc="train", leave=True)
        for step in pbar:
            self.opt.zero_grad()
            loss = self.get_train_loss(batch_size)
            loss.backward()
            self.opt.step()
            self.loss_history.append(loss.item())
            # Live postfix: current loss + smoothed (rolling-mean) loss
            window = min(print_every, len(self.loss_history))
            avg = float(np.mean(self.loss_history[-window:]))
            pbar.set_postfix({"loss": f"{loss.item():.4f}", f"avg{window}": f"{avg:.4f}"})
        return self.loss_history





# ============================================================================
#         EMA (Exponential Moving Average) — for smoother sampling
# ============================================================================
#
# DDPM Trainer uses `ema_decay=0.995` (see diffusion_1d_burgers.py:870). EMA
# averages model weights across recent training steps to suppress the small
# gradient-noise wobble that survives even after loss converges → cleaner
# (less high-freq grainy) samples at inference time.
#
# Math:  θ_ema ← decay · θ_ema + (1 − decay) · θ_current  (each training step)
# Effective averaging window:  ~ 1 / (1 − decay) = 200 steps for decay=0.995.
# ============================================================================



class EMA:
    """Maintains an exponential moving average of model parameters.

    Usage:
        ema = EMA(net, decay=0.995)
        for step in range(num_steps):
            # ... normal training ...
            ema.update(net)                       # after each opt.step()
        net_ema = copy.deepcopy(net).eval()
        ema.copy_to(net_ema)                      # before sampling
    """
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        # ⚠️ state_dict()-based(不是 named_parameters):必须包含模型的 buffer / 共享 RotaryEmbedding
        # 的 freqs(它们在 state_dict 里、不在 named_parameters 里)。否则 ckpt["ema"] 缺 key,
        # jellyfish_fm.py --sample 的 strict load_state_dict 会崩。和 jellyfish_fm.py 的 EMA 对齐。
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)                 # 整型 buffer(如计数器)直接拷,不做平均

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


def finetune_with_ema(
    net: nn.Module,
    trainer_class: type,
    dataset,
    num_steps: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-4,
    ema_decay: float = 0.995,
    save_path_ema: Optional[str] = None,
):
    """Continue training `net` for `num_steps` while building an EMA shadow.

    Returns a NEW net with EMA-smoothed weights (original `net` is also
    trained in-place, so caller has both versions for comparison).

    Args:
        net:           already-trained model (will be further trained in-place)
        trainer_class: `BurgersFlowTrainer` (joint) or `BurgersPriorTrainer` (prior)
        dataset:       matching `BurgersDataset` or `BurgersPriorDataset`
        num_steps:     fine-tune step count. 2000 is enough for EMA to settle.
        ema_decay:     standard 0.995 (window ≈ 200 steps)

    Workflow:
      1. Start from `net` (already trained 25000 / 6250 steps).
      2. Continue training while computing EMA shadow on the fly.
      3. After training, copy EMA shadow into a fresh deepcopy of `net` → `net_ema`.
      4. Use `net_ema` for sampling — gives smoother outputs.

    The original `net` is also fine-tuned, but that's a side effect — main
    deliverable is `net_ema`.
    """
    import copy
    from tqdm.auto import tqdm

    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    trainer = trainer_class(net, path, dataset, lr=lr)
    ema = EMA(net, decay=ema_decay)

    print(f"  fine-tuning {num_steps} steps with EMA decay={ema_decay}...")
    pbar = tqdm(range(num_steps), desc="finetune+EMA")
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
        pbar.set_postfix({"loss": f"{loss.item():.4f}", f"avg{window}": f"{avg:.4f}"})

    # Build NEW net with EMA weights (keeps original `net` trainable)
    net_ema = copy.deepcopy(net).eval()
    ema.copy_to(net_ema)

    if save_path_ema:
        os.makedirs(os.path.dirname(save_path_ema) or ".", exist_ok=True)
        torch.save({"state_dict": net_ema.state_dict()}, save_path_ema)
        print(f"  saved EMA-weighted net: {save_path_ema}")

    print(f"  done. final loss (smoothed): {float(np.mean(trainer.loss_history[-100:])):.5f}")
    return net_ema







# [PROD] Part 1 — JellyfishDataset

def cycle(dl):
    """官方做法(diffusion_2d_jellyfish.py:56):把 DataLoader 包成无限迭代器。"""
    while True:
        for data in dl:
            yield data


class JellyfishDataset(Dataset):
    """torch Dataset(官方 dataset/data_2d.py 风格):__getitem__ 懒读一条 (sim,窗口)。
    配 DataLoader(num_workers) 多进程并行预取 → RAM 有界、全量 30000 可用、速度≈全 load 进 RAM。"""
    def __init__(self, split="train_data", n_sim=1000, device="cpu", load_bd=False):
        self.dirpath = os.path.join(ROOT, "data", "jellyfish", split)
        self.load_bd = load_bd                              # device 参数保留但不用(worker 产 CPU 张量,trainer 再 .to)
        norm = pickle.load(open(os.path.join(self.dirpath, "normalization_max_min.pkl"), "rb"))
        self.vx_max, self.vx_min = norm["vx_max"], norm["vx_min"]
        self.vy_max, self.vy_min = norm["vy_max"], norm["vy_min"]
        self.p_max,  self.p_min  = norm["p_max"],  norm["p_min"]
        files = sorted(glob.glob(os.path.join(self.dirpath, "states", "sim_*.npz")))[:n_sim]
        self.ids = [int(os.path.basename(f)[4:-4]) for f in files]
        self.n_sim = len(self.ids)
        self.windows_per_sim = 40 - T_WIN                                                   # 20, matches paper/official dataset
        print(f"[JellyfishDataset] {self.n_sim} sims × {self.windows_per_sim} windows = {len(self)} items "
              f"(lazy, DataLoader num_workers)", flush=True)

    def __len__(self):
        return self.n_sim * self.windows_per_sim

    def _norm(self, ch, lo, hi):                                                            # Step 2
        return (torch.clamp((ch - lo) / (hi - lo), 0, 1) - 0.5) * 2.0

    def _load_z(self, sim_id):                                                              # Step 3
        st = torch.from_numpy(np.load(os.path.join(self.dirpath, "states", f"sim_{sim_id:06d}.npz"))["a"]).float()
        vx = self._norm(st[:, 0], self.vx_min, self.vx_max).unsqueeze(1)
        vy = self._norm(st[:, 1], self.vy_min, self.vy_max).unsqueeze(1)
        p  = self._norm(st[:, 2], self.p_min,  self.p_max).unsqueeze(1)
        th = torch.from_numpy(np.load(os.path.join(self.dirpath, "bdry_head_thetas", f"sim_{sim_id:06d}.npz"))["thetas"]).float()
        th_plane = th.view(-1, 1, 1, 1).expand(-1, 1, H, W)
        return torch.cat([vx, vy, p, th_plane], dim=1)                                      # [40,4,64,64]

    def _load_bd(self, sim_id):                                                             # Step 5
        bd = torch.from_numpy(np.load(os.path.join(self.dirpath, "bdry_merged_mask_offsets", f"sim_{sim_id:06d}.npz"))["a"]).float().permute(0, 3, 1, 2)
        bd[torch.isnan(bd)] = 0.0
        out = torch.zeros(bd.shape[0], 3, H, W)
        out[:, :, 1:-1, 1:-1] = bd
        return out                                                                          # [40,3,64,64]

    def __getitem__(self, idx):                                                             # 懒读一条 (sim,窗口) → 一个样本
        a = idx // self.windows_per_sim                  # sim 序号
        b = idx % self.windows_per_sim                   # 窗口起点
        sim_id = self.ids[a]
        z = self._load_z(sim_id)[b:b + T_WIN].contiguous()                                  # [20,4,64,64]
        c = z[0].clone()                                                                    # [4,64,64] 窗口首帧=条件
        if self.load_bd:
            bd0 = self._load_bd(sim_id)[b].unsqueeze(0).expand(T_WIN, -1, -1, -1).contiguous()  # [20,3,64,64]
        else:
            bd0 = torch.zeros(T_WIN, 3, H, W)
        return z, c, bd0
    

def inpaint_cond(x, c):
    """Pin the known conditions back onto trajectory x (used in the training loss and at
    every sampling step). Returns a clone; x is NOT modified in place.
    Two things are pinned (all other frames/channels keep x's own values = generated by the model):
      - frame 0, ALL 4 channels (vx,vy,p,theta) = c        -> initial state is fully given
      - last frame, theta channel ONLY = c's theta (theta_0) -> periodic opening-angle boundary (same angle at both ends)
    Args:
        x: [B,20,4,64,64] trajectory being denoised/integrated.
        c: [B,4,64,64]    initial frame (= trajectory frame 0, the conditioning).
    """
    x = x.clone()
    x[:, 0] = c                            # frame 0 = full initial state (vx,vy,p,theta)
    x[:, -1, THETA_CH] = c[:, THETA_CH]    # last frame's theta = theta_0 (periodic)
    return x

def load_bd_updater(device):
    m = Unet(dim=64, out_dim=3, dim_mults=(1, 2, 4, 8), channels=3)
    ckpt = os.path.join(DATA_DIR, "checkpoints", "boundary_updater", "boundary_updater_epoch_9.pth")
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)                          # 冻结,从不训练
    return m


def boundary_from_current_theta(x_t, c, bd0, bd_updater):
    """
    x_t: [B,T,4,H,W], current FM variable
    c:   [B,4,H,W], initial state + theta0
    bd0: [B,T,3,H,W], initial boundary repeated over T
    """
    B, T = x_t.shape[:2]

    theta_t = x_t[:, :, THETA_CH].mean(dim=(-1, -2))     # [B,T]
    theta_0 = c[:, THETA_CH].mean(dim=(-1, -2))[:, None] # [B,1]
    delta = theta_t - theta_0                             # [B,T]

    # FM early-time theta may be outside the updater's training range.
    # Prefer bounds measured from training data.
    # delta = delta.clamp(delta_min, delta_max)

    bd_t = bd_updater(
        bd0.reshape(B * T, 3, H, W),
        delta.reshape(B * T),
    ).reshape(B, T, 3, H, W)

    # Exact endpoint conditions: periodic theta implies b_T = b_0.
    bd_t = bd_t.clone()
    bd_t[:, 0] = bd0[:, 0]
    bd_t[:, -1] = bd0[:, 0]

    return bd_t


class JellyfishVectorField(VectorFieldNet):
    def __init__(self, dim=64, dim_mults=(1, 2, 4)):
        super().__init__()
        self.net = Unet3D_with_Conv3D(
                    dim=dim,
                    out_dim=4,      # velocity of state3 + theta1
                    dim_mults=dim_mults,
                    channels=7,     # state3 + boundary3 + theta1
                )
        
    def forward(self, x, t, c, bd):
        x_in = inpaint_cond(x, c)                            # [B,20,4,64,64] 钉好条件
        # 插入 boundary → [state(3), bd(3), theta(1)]
        net_in = torch.cat([
            x_in[:, :, :3], 
            bd, 
            x_in[:, :, 3:4]
        ], dim=2)

        return self.net(net_in, t.view(-1).to(x_in.dtype))   # [B,20,4,64,64] velocity


# [PROD] Part 3 — JellyfishFlowTrainer
class JellyfishFlowTrainer(Trainer):
    """7ch joint CFM: target velocity = z−ε; boundary comes from current noisy theta."""
    def __init__(self, net, data, bd_updater, lr=1e-4):
        super().__init__(net, lr=lr)                           # 基类:self.net / self.opt(Adam) / self.loss_history
        self.data = data
        self.bd_updater = bd_updater
        self.path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
        self._dl = None                                       # 首次 get_train_loss 时按 batch 建 DataLoader
        self.num_workers = 8                                  # 官方用 16;按容器 CPU 调

    def _loss_mask(self, z):                                   # = Step 2(B)
        m = torch.ones_like(z)
        m[:, 0] = 0                                            # frame0 整帧 = 钉死的初始状态
        m[:, -1, THETA_CH] = 0                                 # 末帧 θ = 周期钉死
        return m

    def get_train_loss(self, batch_size):                     # = Step 1 + Step 2(A) + masked MSE
        if self._dl is None:                                  # 官方做法:DataLoader 多进程并行预取(替代手搓 sample)
            dl = DataLoader(self.data, batch_size=batch_size, shuffle=True, drop_last=True,
                            pin_memory=True, num_workers=self.num_workers,
                            persistent_workers=(self.num_workers > 0))
            self._dl = cycle(dl)
        z, c, bd0 = next(self._dl)                            # worker 产出的 CPU 张量
        z = z.to(DEVICE, non_blocking=True); c = c.to(DEVICE, non_blocking=True); bd0 = bd0.to(DEVICE, non_blocking=True)
        t = torch.rand(z.shape[0], 1, 1, 1, 1, device=z.device)
        x_t, eps = self.path.sample_conditional_path(z, t)    # x_t = α·z + β·ε(返回它用的 ε)
        target = self.path.target_velocity(x_t, z, t, eps)    # z − ε
        x_t = inpaint_cond(x_t, c)
        
        mask = self._loss_mask(z)
        target = target * mask
        
        bd_t = boundary_from_current_theta(
            x_t, c, bd0, self.bd_updater
        )
        pred = self.net(x_t, t, c, bd=bd_t)
        pred = pred * mask
        
        return ((pred - target) ** 2).sum() / mask.sum().clamp(min=1)
    

# [PROD] Part 4 — ProductionTrainer

class ProductionTrainer(JellyfishFlowTrainer):
    """真实训练:JellyfishFlowTrainer(CFM loss) + EMA + MultiStepLR + checkpoint/resume。
    论文配置:Adam betas(0.9,0.99) lr1e-3,MultiStepLR ×0.1 @50k/150k,EMA 0.995。"""
    def __init__(self, net, data, bd_updater, lr=1e-3, betas=(0.9, 0.99),
                 milestones=(50000, 150000), sched_gamma=0.1, ema_decay=0.995,
                 out_dir="checkpoints/jellyfish_fm/joint_7ch", save_every=10000,
                 dim=64, dim_mults=(1, 2, 4)):
        super().__init__(net, data, bd_updater, lr=lr)                    # 拿 get_train_loss / path / data
        self.opt = torch.optim.Adam(net.parameters(), lr=lr, betas=betas) # 覆盖:论文 betas(0.9,0.99)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=list(milestones), gamma=sched_gamma)
        self.ema = EMA(net, decay=ema_decay)
        self.out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)  # 锚 repo 根,防 CWD 漂移
        self.save_every = save_every
        self.dim, self.dim_mults = dim, dim_mults     # 记下架构,供 save() 写进 ckpt 的 cfg(给采样器读)
        os.makedirs(self.out_dir, exist_ok=True)

    def save(self, step):                                                 # = Step 3 的 dict
        ck = {"model": self.net.state_dict(), "ema": self.ema.shadow,
              "opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "step": step,
              "cfg": {"dim": self.dim, "dim_mults": list(self.dim_mults),   # 给 jellyfish_fm.py --sample 读
                      "boundary_cond": True, "is_prior": False, "model_type": "joint"}}
        torch.save(ck, os.path.join(self.out_dir, f"joint_step{step+1}.pt"))
        torch.save(ck, os.path.join(self.out_dir, "joint_latest.pt"))

    def resume(self, path):                                               # 续跑:读回一切,返回起始 step
        ck = torch.load(path, map_location=DEVICE)
        self.net.load_state_dict(ck["model"]); self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"]); self.ema.shadow = ck["ema"]
        return ck["step"] + 1

    def train(self, num_steps, batch_size, start=0, print_every=200):     # = 基类循环 + sched + ema + save
        from tqdm.auto import tqdm
        self.net.train()
        t0 = time.time()
        # 重定向到日志(非 TTY)时关掉 tqdm,日志只留干净的普通 print;交互/tmux 直连时仍显示进度条
        pbar = tqdm(range(start, num_steps), desc="train", leave=True, disable=not sys.stdout.isatty())
        for step in pbar:
            self.opt.zero_grad()
            loss = self.get_train_loss(batch_size)
            if not torch.isfinite(loss):                                    # NaN/Inf 守卫:别用坏梯度毁掉权重+EMA+joint_latest
                print(f"[FATAL] non-finite loss at step {step}; saving & stopping", flush=True)
                self.save(step); break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)      # 官方同款梯度裁剪(稳定,尤其 lr=1e-3 无裁剪有风险)
            self.opt.step(); self.sched.step(); self.ema.update(self.net)  # ← 比基类多了 sched + ema
            self.loss_history.append(loss.item())
            if step % print_every == 0:
                avg = float(np.mean(self.loss_history[-print_every:]))
                sps = (step - start + 1) / (time.time() - t0)
                lr = self.opt.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.1e}"})
                print(f"step {step:6d}/{num_steps}  loss {loss.item():.4f}  avg{print_every} {avg:.4f}  "
                      f"lr {lr:.1e}  {sps:.2f} it/s", flush=True)   # plain print → tail -f 可监控
            if (step + 1) % self.save_every == 0 or step == num_steps - 1:
                self.save(step)
                print(f"  [saved] step {step+1} -> {self.out_dir}/joint_latest.pt", flush=True)
        return self.loss_history


def train(cfg, resume=None):
    ds = JellyfishDataset(split=cfg.split, n_sim=cfg.n_sim, device=DEVICE, load_bd=True)   # P1(7ch 要 bd)
    bd_updater = load_bd_updater(DEVICE)                                                   # P2
    net = JellyfishVectorField(dim=cfg.dim, dim_mults=cfg.dim_mults).to(DEVICE)            # P2(7ch)
    trainer = ProductionTrainer(net, 
                                ds, 
                                bd_updater, 
                                lr=cfg.lr, 
                                betas=cfg.betas,           # P4
                                milestones=cfg.milestones, 
                                ema_decay=cfg.ema_decay,
                                out_dir=cfg.out_dir,
                                save_every=cfg.save_every,
                                dim=cfg.dim,               # 供 save() 写进 ckpt cfg(给采样器读)
                                dim_mults=cfg.dim_mults
                                )
    start = trainer.resume(resume) if resume else 0
    trainer.train(num_steps=cfg.steps, batch_size=cfg.batch, start=start)
    return trainer

# [PROD] Part 5 — Config

@dataclass
class Config:
    # 数据
    split: str = "train_data"     # 真跑用 train_data(本地只有 test_data;代码不变)
    n_sim: int = 30000            # 全量 train_data(共30000条);629GB内存装得下(~135GB);首次加载~30-60min(一次性)。smoke 用 --n_sim 调小
    # 网络(论文 Table 6)
    dim: int = 64
    dim_mults: tuple = (1, 2, 4)
    # 训练(论文 F.4)
    batch: int = 16               # 论文 p31/Table7 = 16; 100GB GPU can use the paper batch size
    steps: int = 200000           # 论文 p31:200,000 iterations(⚠️官方代码写 400k = 遗留,不采用)
    lr: float = 1e-3              # 论文 p31/Table7:1e-3
    betas: tuple = (0.9, 0.99)    # 论文只说 Adam,未给 betas;沿用官方代码值
    milestones: tuple = (50000, 150000)   # 论文 p31:lr ×0.1 @ 50k & 150k(⚠️官方代码有第三个 300k = 遗留,不采用)
    ema_decay: float = 0.995
    save_every: int = 10000       # 每 10k 步存:200k→20个ckpt(~7.4GB);崩了最多丢~75min(batch4);50k/100k/150k 整点都有档
    out_dir: str = "checkpoints/jellyfish_fm/joint_7ch"



# [PROD] Part 6 · load_joint —— build the net from a checkpoint and load (EMA) weights for sampling
@torch.no_grad()
def load_joint(ckpt, use_ema=True):
    """Rebuild the joint vector field from a checkpoint and load its weights, ready to sample.

    The architecture (dim, dim_mults) is read from the `cfg` that save() stored inside the
    checkpoint, so the net always matches how it was trained.

    Args:
        ckpt:    path to a checkpoint written by save() ({model, ema, cfg, step, ...}).
        use_ema: True -> load the smoothed EMA weights (steadier samples, the default);
                 False -> load the raw training weights.
    Returns:
        the net in eval() mode.
    """
    ck = torch.load(ckpt, map_location=DEVICE)
    cfg = ck["cfg"]                                              # dim / dim_mults saved by save()
    net = JellyfishVectorField(dim=cfg["dim"], dim_mults=tuple(cfg["dim_mults"])).to(DEVICE)
    net.load_state_dict(ck["ema"] if use_ema else ck["model"])  # EMA = smoothed weights
    net.eval()
    print(f"loaded {ckpt} ({'EMA' if use_ema else 'raw'}), step {ck['step']+1}")
    return net

# [PROD] Part 6 · euler_sample —— Euler ODE 采样(joint-only, γ=1)
@torch.no_grad()
def euler_sample(
    net,
    c,
    bd0,
    bd_updater,
    n_steps=100,
    *,
    force_model=None,
    p_min=None,
    p_max=None,
    lambda0=0.3,
    zeta=1000.0,
):
    """Euler sample from noise to data, optionally with explicit J guidance.

    The vector field sees b_tau built from the current noisy theta.  Guidance is
    evaluated on the one-step clean estimate x_hat and follows

        v_guided = v_fm - lambda(tau) * grad_{x_hat} J(x_hat).

    The guidance strength is the paper's lambda_k = lambda0 * beta_k,
    using the official 1000-step sigmoid-beta schedule mapped onto FM progress
    from noise (tau=0) to data (tau=1). lambda0=0 disables guidance.
    """
    B = c.shape[0]

    if lambda0 > 0:
        missing = [
            name for name, value in (
                ("force_model", force_model),
                ("p_min", p_min),
                ("p_max", p_max),
            ) if value is None
        ]
        if missing:
            raise ValueError(
                "J guidance requires " + ", ".join(missing)
            )

    x = torch.randn(
        B, T_WIN, 4, H, W,
        device=c.device,
    )
    x = inpaint_cond(x, c)

    mask = torch.ones_like(x)
    mask[:, 0] = 0
    mask[:, -1, THETA_CH] = 0

    dt = 1.0 / n_steps
    paper_betas = None
    if lambda0 > 0:
        # Official jellyfish code uses coeff_ratio_J * betas.flip(0) while
        # iterating DDPM indices from K-1 down to 0.  In FM we iterate in the
        # opposite direction (noise -> data), so this is simply betas from
        # index 0 to K-1.  Keep K=1000 to preserve the published schedule even
        # when the FM Euler solver uses fewer steps.
        paper_betas = sigmoid_beta_schedule(1000).to(
            device=x.device, dtype=x.dtype
        )

    for k in range(n_steps):
        tau = torch.full(
            (B, 1, 1, 1, 1),
            k * dt,
            device=x.device,
        )

        # Same rule as training
        bd_t = boundary_from_current_theta(
            x, c, bd0, bd_updater
        )

        v_fm = net(x, tau, c, bd=bd_t) * mask

        if lambda0 > 0:
            tau_scalar = k * dt
            x_hat = inpaint_cond(
                x + (1.0 - tau_scalar) * v_fm,
                c,
            )

            # The outer sampler is no-grad; re-enable autograd only for J.
            # Both surrogate networks have frozen parameters, but gradients
            # still flow through their inputs back to x_obj.
            with torch.enable_grad():
                x_obj = x_hat.detach().requires_grad_(True)
                J = jellyfish_objective(
                    x_obj,
                    c,
                    bd0,
                    bd_updater,
                    force_model,
                    p_min,
                    p_max,
                    zeta=zeta,
                )
                grad_J = torch.autograd.grad(J.sum(), x_obj)[0]

            grad_J = grad_J * mask
            if not torch.isfinite(grad_J).all():
                raise FloatingPointError(
                    f"non-finite J gradient at Euler step {k}"
                )

            # Official DDPM code evaluates betas.flip(0)[t] while t runs from
            # K-1 (noise) to 0 (clean).  FM runs in the opposite direction, so
            # beta_idx increases from 0 to K-1.  This yields weak guidance near
            # noise (~0.0003*lambda0) and strong guidance near clean (~lambda0).
            progress = k / max(n_steps - 1, 1)
            beta_idx = round(progress * (paper_betas.numel() - 1))
            beta_k = float(paper_betas[beta_idx])

            # Minus sign: the paper minimizes J = -speed + zeta*R + periodicity.
            lambda_k = lambda0 * beta_k
            v = v_fm - lambda_k * grad_J.detach()
        else:
            v = v_fm

        x = x + dt * v
        x = inpaint_cond(x, c)

    return x

# [PROD] Part 6 · sample_test_thetas —— sample the test set, save theta sequences for LilyPad
@torch.no_grad()
def sample_test_thetas(
    net,
    bd_updater,
    n_sim=50,
    n_steps=100,
    out_dir="results/fm_thetas",
    *,
    lambda0=0.3,
    zeta=1000.0,
    force_model=None,
):
    """Generate a control (opening-angle) sequence for each test jellyfish and save it for LilyPad.

    For each of the first `n_sim` test sims:
      1. take its frame-0 state as the conditioning `c`,
      2. run `euler_sample` to generate a 20-frame trajectory,
      3. read off the per-frame opening angle theta (one scalar per frame),
      4. save it to `{out_dir}/{i}.npy` (the control sequence the LilyPad solver replays).
    Finally prints predicted-vs-ground-truth theta ranges as an OOD sanity check
    (predicted angles should sit near the GT range, not blow up).

    Args:
        net:        trained joint vector field (from load_joint).
        bd_updater: frozen boundary updater (from load_bd_updater).
        n_sim:      number of test sims to sample.
        n_steps:    number of Euler integration steps.
        out_dir:    directory to write the {i}.npy theta arrays.
    """
    ds = JellyfishDataset(split="test_data", n_sim=n_sim, load_bd=True)
    if lambda0 > 0 and force_model is None:
        force_model = load_force_model(DEVICE)
    os.makedirs(out_dir, exist_ok=True)
    preds, gts = [], []                       # collect predicted / ground-truth angle sequences

    for i in range(ds.n_sim):
        sid = ds.ids[i]                       # this sim's file id
        zf = ds._load_z(sid)                  # full GT sim: [40, 4, 64, 64]  (40 frames; 4ch = vx,vy,p,theta)

        # conditioning = frame 0 of the sim. 0:1 keeps the frame axis -> [1, 4, 64, 64]
        c = zf[0:1].to(DEVICE)

        # base boundary bd_0 = frame-0 wing geometry, repeated over all T_WIN frames;
        # euler_sample / boundary_from_current_theta deform THIS by Delta-theta each frame.
        #   _load_bd(sid)[0]            : [3, 64, 64]          (frame-0 boundary)
        #   .unsqueeze(0).unsqueeze(0)  : [1, 1, 3, 64, 64]    (add batch + frame axes)
        #   .expand(1, T_WIN, -1,-1,-1) : [1, 20, 3, 64, 64]   (same boundary on all 20 frames)
        bd0 = ds._load_bd(sid)[0].unsqueeze(0).unsqueeze(0).expand(1, T_WIN, -1, -1, -1).to(DEVICE)

        x = euler_sample(
            net,
            c,
            bd0,
            bd_updater,
            n_steps,
            force_model=force_model,
            p_min=ds.p_min,
            p_max=ds.p_max,
            lambda0=lambda0,
            zeta=zeta,
        )                                                   # [1,20,4,64,64]

        # predicted opening angle per frame = spatial mean of the theta channel -> [20] (radians).
        # (the model's theta plane is not perfectly uniform, so we average over the 64x64 grid)
        th = x[0, :, THETA_CH].mean(dim=(-1, -2)).cpu().numpy()
        np.save(os.path.join(out_dir, f"{i}.npy"), th)       # the control sequence LilyPad replays
        preds.append(th)

        # ground-truth angle: theta is stored as a UNIFORM plane, so pixel [0,0] IS the scalar;
        # [:T_WIN] = first 20 frames -> [20]
        gts.append(zf[:T_WIN, THETA_CH, 0, 0].numpy())

    preds, gts = np.array(preds), np.array(gts)              # both [n_sim, 20], radians
    print(f"saved {len(preds)} thetas -> {out_dir}")
    print(f"PRED theta [{np.degrees(preds).min():.1f}, {np.degrees(preds).max():.1f}] deg   "
          f"GT theta [{np.degrees(gts).min():.1f}, {np.degrees(gts).max():.1f}] deg   "
          f"(OOD check: PRED should sit near GT)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--sample", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--steps", type=int, default=None)   # 覆盖 Config.steps(smoke 用)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--split", default=None)             # 本地 smoke 用 test_data;AutoDL 真跑默认 train_data
    ap.add_argument("--n_sim", type=int, default=None)   # smoke 用小一点(本地 test_data 才 200 条)
    ap.add_argument(
        "--ckpt",
        default=os.path.join(
            ROOT, "checkpoints/jellyfish_fm/joint_7ch/joint_latest.pt"
        ),
        help="joint FM checkpoint used by --sample",
    )
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--lambda0", type=float, default=0.3)
    ap.add_argument("--zeta", type=float, default=1000.0)
    ap.add_argument(
        "--sample_out",
        default=os.path.join(ROOT, "results/fm_thetas_jgrad"),
    )
    ap.add_argument(
        "--raw",
        action="store_true",
        help="sample raw training weights instead of EMA weights",
    )
    a = ap.parse_args()

    if a.train:
        cfg = Config()
        if a.steps: cfg.steps = a.steps
        if a.batch: cfg.batch = a.batch
        if a.split: cfg.split = a.split
        if a.n_sim: cfg.n_sim = a.n_sim
        train(cfg, resume=a.resume)
    else:
        if not os.path.isfile(a.ckpt):
            ap.error(f"joint checkpoint not found: {a.ckpt}")
        net = load_joint(a.ckpt, use_ema=not a.raw)
        bd_updater = load_bd_updater(DEVICE)
        sample_test_thetas(
            net,
            bd_updater,
            n_sim=a.n_sim or 50,
            n_steps=a.n_steps,
            out_dir=a.sample_out,
            lambda0=a.lambda0,
            zeta=a.zeta,
        )
