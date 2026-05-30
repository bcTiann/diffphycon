"""
burgers_fm_eval_v2.py — paper-faithful eval for 1D Burgers FM models.

Loads 4 EMA-weighted checkpoints (vanilla joint, OT joint, vanilla prior, OT prior),
runs γ-sweep using reverse-sigmoid prior reweighting (paper Eq 21, D.4),
samples N_TEST=50 trajectories at 1000 Euler steps, computes J via the
ground-truth PDE solver (paper-faithful).

Output: <out_dir>/eval_table.csv with columns
    variant, gamma, J_mean, J_std, E_mean

Usage:
    python flow/burgers_fm_eval_v2.py \\
        --ckpt_dir checkpoints/paper_fopc_v2 \\
        --dataset free_u_f_paper_fopc \\
        --out_dir results/paper_fopc_v2
"""
from __future__ import annotations
import argparse
import os
import sys
import math

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Reuse the FM primitives & net + dataset wrapper from training script
from flow.burgers_fm_train import (
    LinearAlpha, LinearBeta, GaussianConditionalProbabilityPath,
    BurgersVectorField, load_burgers, BurgersDataset, T_IDX, RESCALER,
)

# -----------------------------------------------------------------------------
# Inference: γ-reweighted Euler sampler
# -----------------------------------------------------------------------------

def sigmoid_flip(t: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
    """Reverse-sigmoid schedule for prior reweighting (paper sigmoid_flip).

    Returns ~1.0 near t=0 (pure noise) and ~0.0 near t=1 (clean).
    Matches paper's β_{K-k} schedule where guidance is strong at early diffusion
    steps and weak near the end.
    """
    return 1.0 - torch.sigmoid(slope * (t - 0.5))


@torch.no_grad()
def euler_sample(joint, prior, c, n_steps: int, gamma: float,
                 shape=(2, 16, 128), device="cuda"):
    """Sample via Euler ODE with γ-prior-reweighting.

    For γ=1: v_eff = v_joint   (DiffPhyCon-lite)
    For γ≠1: v_eff = v_joint + (γ-1) * sigmoid_flip(τ) * v_prior_w
    where v_prior contributes only to w-channel (paper D.3.2).
    """
    b = c.shape[0]
    x = torch.randn(b, *shape, device=device)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_scalar = k * dt
        t = torch.full((b, 1, 1, 1), t_scalar, device=device)
        # Joint velocity
        v_joint = joint(x, t, c)
        if abs(gamma - 1.0) < 1e-8 or prior is None:
            v_eff = v_joint
        else:
            v_prior = prior(x, t, c)                                 # outputs both channels; w only
            sched = sigmoid_flip(torch.tensor(t_scalar, device=device))
            # Only mix into w-channel (channel 1); leave u-channel untouched
            v_eff = v_joint.clone()
            v_eff[:, 1] = v_joint[:, 1] + (gamma - 1.0) * sched * v_prior[:, 1]
        x = x + dt * v_eff
    return x


# -----------------------------------------------------------------------------
# J / E via PDE solver (matches our existing helper)
# -----------------------------------------------------------------------------

def compute_J_E(x_pred: torch.Tensor, u_target_full: torch.Tensor,
                rescaler: float = RESCALER):
    """J_actual via ground-truth Burgers solver (paper D.4 + D.2.2).

    Args:
        x_pred: (b, 2, 16, 128) model output (rescaled, pre-divided)
        u_target_full: (b, 11, 128) clean u trajectory from dataset (rescaled)

    burgers_metric handles `partial_control='front_rear_quarter'` masking
    internally (zeros central 50% of f before PDE solve).
    """
    from utils import burgers_metric
    x = x_pred.detach().cpu() * rescaler                # un-normalize
    w_pred = x[:, 1, :10, :]                            # (b, 10, 128)
    u_t = (u_target_full.detach().cpu() * rescaler)     # (b, 11, 128)
    J, E = burgers_metric(
        u_target=u_t,
        f=w_pred,
        target='final_u',
        partial_control='front_rear_quarter',
    )
    return J.numpy(), E.numpy()


# -----------------------------------------------------------------------------
# Load EMA weights from ckpt
# -----------------------------------------------------------------------------

def load_net(ckpt_path: str, device: str, dim: int, dim_mults: tuple):
    net = BurgersVectorField(dim=dim, dim_mults=dim_mults).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Prefer EMA weights for paper-faithful eval
    if "ema_state_dict" in ckpt:
        # ema_pytorch's ema_model.state_dict() keys are slightly different;
        # strip the "ema_model." or use raw if matches
        sd = ckpt["ema_state_dict"]
        # Try direct load; fall back to stripping prefix
        try:
            net.load_state_dict(sd)
            tag = "EMA"
        except RuntimeError:
            sd_stripped = {k.replace("ema_model.", ""): v for k, v in sd.items()}
            net.load_state_dict(sd_stripped, strict=False)
            tag = "EMA (prefix-stripped)"
    else:
        net.load_state_dict(ckpt["state_dict"])
        tag = "raw"
    net.eval()
    print(f"  loaded {os.path.basename(ckpt_path)}  [{tag}]")
    return net


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, help="dir containing vanilla/ot _joint/_prior .pt")
    p.add_argument("--dataset", default="free_u_f_paper_fopc")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_test", type=int, default=50, help="paper D.1 = 50")
    p.add_argument("--n_steps", type=int, default=1000, help="paper Table 5 = 1000")
    p.add_argument("--gammas", type=float, nargs="+",
                   default=[0.0, 0.3, 0.5, 0.7, 1.0],
                   help="paper Table 25 = [0.0, 0.3, 0.5, 0.7, 1.0]")
    p.add_argument("--variants", type=str, nargs="+", default=["vanilla", "ot"])
    p.add_argument("--joint_dim", type=int, default=128)
    p.add_argument("--joint_mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--prior_dim", type=int, default=32)
    p.add_argument("--prior_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build test batch (first n_test samples — paper-faithful deterministic)
    ds_raw = load_burgers(args.dataset, split="test", device="cpu")
    ds = BurgersDataset(ds_raw, device=args.device, is_prior=False)
    print(f"loaded test set: {ds.N} samples; using first {args.n_test}")
    zs = ds.all_z[:args.n_test].to(args.device)
    c_eval = torch.stack([zs[:, 0, 0, :], zs[:, 0, T_IDX, :]], dim=1)
    u_target_full = zs[:, 0, :11, :]   # (b, 11, 128) clean u for J_actual

    rows = []
    for variant in args.variants:
        joint_path = os.path.join(args.ckpt_dir, f"{variant}_joint.pt")
        prior_path = os.path.join(args.ckpt_dir, f"{variant}_prior.pt")
        if not os.path.isfile(joint_path):
            print(f"⚠️  missing {joint_path} — skipping {variant}"); continue
        print(f"\n=== {variant} ===")
        joint = load_net(joint_path, args.device, args.joint_dim, tuple(args.joint_mults))
        prior = None
        if os.path.isfile(prior_path):
            prior = load_net(prior_path, args.device, args.prior_dim, tuple(args.prior_mults))
        else:
            print(f"  ⚠️  no prior found → can only run γ=1.0")
        import time
        for gamma in args.gammas:
            if prior is None and abs(gamma - 1.0) > 1e-8:
                print(f"  γ={gamma}: skip (no prior)"); continue
            torch.manual_seed(args.seed)
            t_start = time.time()
            x_pred = euler_sample(joint, prior, c_eval,
                                  n_steps=args.n_steps, gamma=gamma,
                                  device=args.device)
            t_sample = time.time() - t_start
            Js, Es = compute_J_E(x_pred, u_target_full)
            J_mean, J_std, E_mean = Js.mean(), Js.std(), Es.mean()
            print(f"  γ={gamma:.1f}: J={J_mean:.5f} ± {J_std:.5f}   E={E_mean:.1f}   "
                  f"sampling_time={t_sample:.3f}s  (per_sample={t_sample/x_pred.shape[0]:.4f}s)")
            rows.append({
                "variant": variant, "gamma": gamma,
                "J_mean": J_mean, "J_std": J_std, "E_mean": E_mean,
                "sampling_time": t_sample,
                "per_sample_time": t_sample / x_pred.shape[0],
            })

    # Write CSV
    out_csv = os.path.join(args.out_dir, "eval_table.csv")
    with open(out_csv, "w") as f:
        f.write("variant,gamma,J_mean,J_std,E_mean,sampling_time,per_sample_time\n")
        for r in rows:
            f.write(f"{r['variant']},{r['gamma']},{r['J_mean']:.6f},{r['J_std']:.6f},"
                    f"{r['E_mean']:.3f},{r['sampling_time']:.4f},{r['per_sample_time']:.5f}\n")
    print(f"\n💾 wrote {out_csv}")


if __name__ == "__main__":
    main()
