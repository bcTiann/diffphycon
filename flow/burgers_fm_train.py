"""
burgers_fm_train.py — standalone Flow-Matching trainer for 1D Burgers control.

SELF-CONTAINED: does NOT import lab_four / lab_four_solved / any notebook-derived
code. Only imports stable repo modules (Unet2D, Burgers1D). Designed to run on a
cloud GPU (AutoDL / Modal) via CLI.

Trains one model at a time:
    --variant {vanilla, ot}     vanilla CFM  or  minibatch-OT-CFM
    --model   {joint, prior}    joint p(u,w|c)  or  prior p(w|c)  (u-channel zeroed)

Example (paper-scale FOPC joint):
    python flow/burgers_fm_train.py \
        --variant vanilla --model joint \
        --dim 128 --dim_mults 1 2 4 8 \
        --num_steps 200000 --batch_size 16 --lr 1e-4 \
        --dataset free_u_f_paper_fopc --device cuda \
        --save_path checkpoints/fm_vanilla_joint_paper.pt

Data layout convention (matches paper + lab_four):
    z = (b, 2, 16, 128)  channel 0 = u (state), channel 1 = w (control)
                         time axis 16 = 11 real steps (0..10) + 5 zero-pad
    c = (b, 2, 128)      (u_0, u_T*) = z[:,0,0,:] and z[:,0,T_IDX,:]
    t = (b, 1, 1, 1)     FM time τ ∈ [0,1]
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from ema_pytorch import EMA

# repo root importable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.burgers_1d.unet import Unet2D          # stable repo module
from dataset.data_1d import Burgers1D             # stable repo module

T_IDX = 10        # row of u_T* in the 16-step padded time axis (11 real steps 0..10)
RESCALER = 10.0   # dataset normalization (data divided by this)


# ============================================================================
# FM primitives (CondOT path: α_τ = τ, β_τ = 1 − τ)
# ============================================================================

class LinearAlpha:
    def __call__(self, t): return t
    def dt(self, t):       return torch.ones_like(t)

class LinearBeta:
    def __call__(self, t): return 1.0 - t
    def dt(self, t):       return -torch.ones_like(t)

class GaussianConditionalProbabilityPath:
    def __init__(self, alpha, beta):
        self.alpha, self.beta = alpha, beta

    def sample_conditional_path(self, z, t):
        eps = torch.randn_like(z)
        x_t = self.alpha(t) * z + self.beta(t) * eps
        return x_t, eps

    def target_velocity(self, x_t, z, t, eps):
        # Form A (ε form): α̇·z + β̇·ε  → for CondOT = z − ε
        return self.alpha.dt(t) * z + self.beta.dt(t) * eps


class BurgersVectorField(nn.Module):
    """Wraps Unet2D. Injects boundary c into x's u-channel rows 0 / T_IDX
    before the Unet (inpainting-style conditioning).

    Architecture hyperparams below match paper Table 5 (FO-PC) exactly.
    They are also Unet2D defaults, but we pass them explicitly so a future
    default change in Unet2D doesn't silently break paper-faithfulness.
    """
    def __init__(self, dim=128, dim_mults=(1, 2, 4),
                 resnet_block_groups=8,    # paper Table 5
                 attn_dim_head=32,          # paper Table 5
                 attn_heads=4):             # paper Table 5
        super().__init__()
        self.unet = Unet2D(
            dim=dim, dim_mults=dim_mults, channels=2,
            resnet_block_groups=resnet_block_groups,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
        )

    def forward(self, x, t, c):
        x_in = x.clone()
        x_in[:, 0, 0, :]     = c[:, 0]
        x_in[:, 0, T_IDX, :] = c[:, 1]
        t_flat = t.view(-1).to(x_in.dtype)
        return self.unet(x_in, t_flat)


# ============================================================================
# OT pairing (minibatch optimal transport)
# ============================================================================

def ot_pair(z: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Reorder eps along dim 0 so each z[k] is OT-matched to a nearby eps[k]
    (minimizes Σ||z_i − eps_match_i||²). Hungarian algorithm via scipy."""
    from scipy.optimize import linear_sum_assignment
    z_flat, eps_flat = z.flatten(1), eps.flatten(1)
    cost = torch.cdist(z_flat, eps_flat) ** 2            # (b, b) squared distances
    _, col = linear_sum_assignment(cost.detach().cpu().numpy())
    return eps[col]


# ============================================================================
# Dataset wrapper
# ============================================================================

def load_burgers(dataset: str, split: str, device: str) -> Burgers1D:
    return Burgers1D(
        dataset="burgers", input_steps=1, output_steps=10, time_interval=1,
        is_y_diff=False, split=split, transform=None, pre_transform=None,
        verbose=False, root_path=os.path.join(ROOT, "data", dataset),
        device=device, rescaler=RESCALER, stack_u_and_f=True,
        pad_for_2d_conv=True, partially_observed_fill_zero_unobserved=None, nt_total=11,
    )


class BurgersDataset:
    """Pre-stacks the dataset on CPU; sample(n) returns (z, c) on `device`."""
    def __init__(self, ds: Burgers1D, device: str = "cpu", is_prior: bool = False):
        self.all_z = torch.stack([ds[i] for i in range(len(ds))], dim=0)  # (N, 2, 16, 128) CPU
        self.N = self.all_z.shape[0]
        self.device = device
        self.is_prior = is_prior

    def sample(self, num_samples: int):
        idx = torch.randint(0, self.N, (num_samples,))
        z = self.all_z[idx].to(self.device)
        c = torch.stack([z[:, 0, 0, :], z[:, 0, T_IDX, :]], dim=1)  # (b, 2, 128)
        if self.is_prior:
            z = z.clone()
            z[:, 0] = 0      # prior: zero the u-channel (only learn p(w|c))
        return z, c


# ============================================================================
# Trainers
# ============================================================================

class FlowTrainer:
    """CFM trainer with paper-faithful additions (EMA, cosine LR, loss masking).

    Paper refs:
      - EMA: standard for DDPM Trainer (β=0.995 default).
      - Cosine annealing: Table 5 (UNet training config for FO-PC).
      - Loss masking: D.4 "u_0 and u_d ... outputs at corresponding locations
        are excluded from the loss." We mask u-channel boundary rows (0, T_IDX).
        For prior, we additionally mask the entire u-channel (it only learns p(w|c)).
    """
    def __init__(self, net, path, data, lr=1e-4, use_ot=False, is_prior=False,
                 use_ema=True, ema_decay=0.995, lr_scheduler=None, num_steps=None,
                 mask_boundary_loss=True):
        self.net = net
        self.path = path
        self.data = data
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self.use_ot = use_ot
        self.is_prior = is_prior
        self.mask_boundary_loss = mask_boundary_loss
        self.loss_history = []
        # EMA: slow-moving averaged weights; use these for eval, not raw net (paper-standard)
        self.ema = EMA(net, beta=ema_decay, update_every=10) if use_ema else None
        # Cosine LR annealing — needs total num_steps to know the schedule length (paper Table 5)
        self.scheduler = None
        if lr_scheduler == "cosine" and num_steps:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=num_steps
            )

    def get_train_loss(self, batch_size):
        z, c = self.data.sample(batch_size)
        eps = torch.randn_like(z)
        if self.use_ot:
            eps = ot_pair(z, eps)
        t = torch.rand(batch_size, 1, 1, 1, device=z.device)
        x_t = self.path.alpha(t) * z + self.path.beta(t) * eps
        u_target = self.path.target_velocity(x_t, z, t, eps)
        u_pred = self.net(x_t, t, c)

        # Per-element squared error
        sq = (u_pred - u_target) ** 2

        # Build loss mask: 1 = include in loss, 0 = exclude (paper D.4)
        mask = torch.ones_like(sq)
        if self.mask_boundary_loss:
            # u_0 (row 0) and u_T (row T_IDX) of u-channel are inpainted, NOT predicted
            mask[:, 0, 0, :]     = 0
            mask[:, 0, T_IDX, :] = 0
        if self.is_prior:
            # Prior learns only p(w | c) — exclude entire u-channel from loss
            mask[:, 0] = 0

        return (sq * mask).sum() / mask.sum().clamp(min=1)

    def train(self, num_steps, batch_size, print_every=500,
              ckpt_every=0, save_path=None, start_step=0):
        from tqdm.auto import tqdm
        self.net.train()
        pbar = tqdm(range(start_step, num_steps), desc="train",
                    initial=start_step, total=num_steps, leave=True)
        for step in pbar:
            self.opt.zero_grad()
            loss = self.get_train_loss(batch_size)
            loss.backward()
            self.opt.step()
            # EMA: update slow-moving weights (paper-standard for DDPM)
            if self.ema is not None:
                self.ema.update()
            # Cosine LR annealing: step every iteration
            if self.scheduler is not None:
                self.scheduler.step()
            self.loss_history.append(loss.item())
            if step % print_every == 0 or step == num_steps - 1:
                w = min(print_every, len(self.loss_history))
                avg = float(np.mean(self.loss_history[-w:]))
                lr_now = self.opt.param_groups[0]['lr']
                pbar.set_postfix({"loss": f"{loss.item():.5f}",
                                  f"avg{w}": f"{avg:.5f}",
                                  "lr": f"{lr_now:.2e}"})
            if ckpt_every and save_path and step > 0 and step % ckpt_every == 0:
                _save(self.net, self.loss_history, save_path, step=step,
                      ema=self.ema, optimizer=self.opt, scheduler=self.scheduler)
        return self.loss_history


def _save(net, loss_history, save_path, config=None, step=None,
          ema=None, optimizer=None, scheduler=None):
    """Save full training state for resume + eval.

    For RESUME (intermediate ckpt at step k):
      state_dict + ema_state_dict + optimizer + scheduler + step + loss_history
    For EVAL (final ckpt):
      state_dict + ema_state_dict (paper-faithful eval uses EMA weights)
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "loss_history": loss_history,
        "step": step if step is not None else len(loss_history),
    }
    if ema is not None:
        payload["ema_state_dict"] = ema.ema_model.state_dict()
        payload["ema_full"] = ema.state_dict()                # for resume (includes ema step counter)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()         # Adam momentum etc.
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()         # cosine progress
    if config is not None:
        payload["config"] = config
    path = save_path if step is None else save_path.replace(".pt", f"_step{step}.pt")
    torch.save(payload, path)
    return path


def _find_latest_intermediate(save_path):
    """Scan for intermediate ckpts matching save_path pattern; return path of latest by step."""
    import glob
    base = save_path.replace(".pt", "")
    pattern = f"{base}_step*.pt"
    matches = glob.glob(pattern)
    if not matches:
        return None, 0
    # parse step number from each filename, pick max
    def _step_of(p):
        try:
            return int(p.rsplit("_step", 1)[1].replace(".pt", ""))
        except (ValueError, IndexError):
            return -1
    matches.sort(key=_step_of)
    latest = matches[-1]
    return latest, _step_of(latest)


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Standalone FM trainer for 1D Burgers")
    p.add_argument("--variant", choices=["vanilla", "ot"], required=True)
    p.add_argument("--model", choices=["joint", "prior"], required=True)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--num_steps", type=int, default=190000)   # paper Table 5
    p.add_argument("--batch_size", type=int, default=16)      # paper Table 5
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dataset", type=str, default="free_u_f_paper_fopc")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--ckpt_every", type=int, default=20000, help="0 to disable intermediate ckpts")
    p.add_argument("--print_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    # Paper-faithful additions (paper Table 5 + D.4)
    p.add_argument("--ema_decay", type=float, default=0.995,
                   help="EMA decay; paper-standard 0.995. Use 0 to disable.")
    p.add_argument("--lr_scheduler", type=str, default="cosine",
                   choices=["cosine", "none"],
                   help="LR scheduler; paper uses 'cosine' annealing.")
    p.add_argument("--no_mask_boundary", dest="mask_boundary_loss",
                   action="store_false",
                   help="Disable boundary loss masking (NOT paper-faithful).")
    p.set_defaults(mask_boundary_loss=True)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[warn] cuda unavailable → falling back to {args.device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    is_prior = (args.model == "prior")
    use_ot = (args.variant == "ot")

    print(f"=== train {args.variant} {args.model} | dim={args.dim} steps={args.num_steps} "
          f"batch={args.batch_size} device={args.device} ===")
    ds_raw = load_burgers(args.dataset, split="train", device="cpu")
    ds = BurgersDataset(ds_raw, device=args.device, is_prior=is_prior)
    print(f"  dataset: {ds.N} train samples ({args.dataset})")

    path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
    net = BurgersVectorField(dim=args.dim, dim_mults=tuple(args.dim_mults)).to(args.device)
    print(f"  net params: {sum(p.numel() for p in net.parameters()):,}")

    trainer = FlowTrainer(
        net, path, ds,
        lr=args.lr, use_ot=use_ot, is_prior=is_prior,
        use_ema=(args.ema_decay > 0),
        ema_decay=args.ema_decay,
        lr_scheduler=args.lr_scheduler if args.lr_scheduler != "none" else None,
        num_steps=args.num_steps,
        mask_boundary_loss=args.mask_boundary_loss,
    )
    print(f"  EMA: {'ON β=' + str(args.ema_decay) if args.ema_decay > 0 else 'OFF'} | "
          f"LR scheduler: {args.lr_scheduler} | "
          f"boundary loss mask: {args.mask_boundary_loss}")

    # ---- Skip / resume logic ----
    start_step = 0
    if os.path.isfile(args.save_path):
        print(f"  ✅ FINAL ckpt already exists at {args.save_path} — skipping.")
        return
    latest, latest_step = _find_latest_intermediate(args.save_path)
    if latest is not None and latest_step > 0:
        print(f"  🔄 RESUME from {latest} (step {latest_step}/{args.num_steps})")
        ckpt = torch.load(latest, map_location=args.device, weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        if trainer.ema is not None and "ema_full" in ckpt:
            trainer.ema.load_state_dict(ckpt["ema_full"])
        if "optimizer" in ckpt:
            trainer.opt.load_state_dict(ckpt["optimizer"])
        if trainer.scheduler is not None and "scheduler" in ckpt:
            trainer.scheduler.load_state_dict(ckpt["scheduler"])
        trainer.loss_history = ckpt.get("loss_history", [])
        start_step = latest_step
        print(f"  ⚠️  RNG state not restored — minor non-determinism (see docstring).")
    else:
        print(f"  🆕 Fresh start from step 0.")

    trainer.train(args.num_steps, args.batch_size, print_every=args.print_every,
                  ckpt_every=args.ckpt_every, save_path=args.save_path,
                  start_step=start_step)

    config = vars(args)
    final = _save(net, trainer.loss_history, args.save_path, config=config,
                  ema=trainer.ema, optimizer=trainer.opt, scheduler=trainer.scheduler)
    print(f"  💾 saved final: {final}")


if __name__ == "__main__":
    main()
