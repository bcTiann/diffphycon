"""
plot_trajectories.py — visualize 5 test trajectories for vanilla+OT × γ-sweep.

For each (variant, γ) cell:
    [predicted u(t,x)] | [predicted w(t,x)] | [simulated u(t,x)] | [terminal match]

Usage:
    python flow/plot_trajectories.py \\
        --ckpt_dir checkpoints/paper_fopc_v2 \\
        --dataset free_u_f_paper_fopc \\
        --out_dir results/paper_fopc_v2 \\
        --n_samples 5 --gammas 0.0 1.0
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flow.burgers_fm_train import (
    load_burgers, BurgersDataset, T_IDX, RESCALER,
)
from flow.burgers_fm_eval_v2 import euler_sample, load_net


def simulate_pde(w_pred_batch: np.ndarray, u0_batch: np.ndarray) -> np.ndarray:
    """Run ground-truth Burgers solver on w to get u(T)."""
    from dataset.apps.generate_burgers import burgers_numeric_solve_free
    # Naive sample-by-sample; small enough (n_samples ≤ 10)
    u_T_sim = []
    for i in range(w_pred_batch.shape[0]):
        sol = burgers_numeric_solve_free(
            u0=u0_batch[i], f=w_pred_batch[i], nu=0.01, T=1.0, Nt_internal=10000,
        )
        u_T_sim.append(sol[-1])
    return np.stack(u_T_sim, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--dataset", default="free_u_f_paper_fopc")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_samples", type=int, default=5, help="paper Fig 7 = 5")
    p.add_argument("--n_steps", type=int, default=1000)
    p.add_argument("--gammas", type=float, nargs="+", default=[0.0, 1.0])
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

    # Test batch
    ds_raw = load_burgers(args.dataset, split="test", device="cpu")
    ds = BurgersDataset(ds_raw, device=args.device, is_prior=False)
    zs = ds.all_z[:args.n_samples].to(args.device)
    c_eval = torch.stack([zs[:, 0, 0, :], zs[:, 0, T_IDX, :]], dim=1)

    # One plot per (variant, gamma) combo
    for variant in args.variants:
        joint_path = os.path.join(args.ckpt_dir, f"{variant}_joint.pt")
        prior_path = os.path.join(args.ckpt_dir, f"{variant}_prior.pt")
        if not os.path.isfile(joint_path):
            print(f"⚠️  missing {joint_path}"); continue
        joint = load_net(joint_path, args.device, args.joint_dim, tuple(args.joint_mults))
        prior = load_net(prior_path, args.device, args.prior_dim, tuple(args.prior_mults)) \
                if os.path.isfile(prior_path) else None

        for gamma in args.gammas:
            if prior is None and abs(gamma - 1.0) > 1e-8:
                continue
            torch.manual_seed(args.seed)
            with torch.no_grad():
                x = euler_sample(joint, prior, c_eval,
                                 n_steps=args.n_steps, gamma=gamma,
                                 device=args.device)
            x_cpu = x.cpu() * RESCALER
            u_pred = x_cpu[:, 0, :11, :].numpy()
            w_pred = x_cpu[:, 1, :10, :].numpy()
            # mask w in central 50% for PDE (paper D.2.2)
            n_x = w_pred.shape[-1]
            w_pred_masked = w_pred.copy()
            w_pred_masked[:, :, n_x // 4 : 3 * n_x // 4] = 0.0
            u0 = c_eval[:, 0].cpu().numpy() * RESCALER
            uT_target = c_eval[:, 1].cpu().numpy() * RESCALER

            print(f"  simulating PDE for {variant} γ={gamma}...")
            uT_sim = simulate_pde(w_pred_masked, u0)

            # Plot: n_samples rows × 4 cols
            n = args.n_samples
            fig, axs = plt.subplots(n, 4, figsize=(16, 3.0 * n))
            if n == 1:
                axs = axs[np.newaxis, :]
            vmax_w = max(np.abs(w_pred).max(), 1e-3)
            for i in range(n):
                # u_pred
                axs[i, 0].imshow(u_pred[i], cmap="RdBu_r", aspect="auto",
                                 vmin=-3, vmax=3)
                axs[i, 0].set_title(f"sample {i}: pred u(t,x)" if i == 0 else "")
                axs[i, 0].set_ylabel(f"#{i}\nt")
                # w_pred
                axs[i, 1].imshow(w_pred[i], cmap="RdBu_r", aspect="auto",
                                 vmin=-vmax_w, vmax=vmax_w)
                axs[i, 1].set_title("pred w(t,x)" if i == 0 else "")
                # sim u (only shows u(T))
                axs[i, 2].plot(uT_target[i], "k-", label="target u_T*", lw=2)
                axs[i, 2].plot(uT_sim[i], "b--", label="sim u(T)", lw=1.5)
                axs[i, 2].plot(u0[i], color="grey", alpha=0.5, label="u_0", lw=0.8)
                axs[i, 2].set_title("terminal match" if i == 0 else "")
                axs[i, 2].legend(fontsize=7, loc="upper right")
                axs[i, 2].grid(alpha=0.3)
                # J metric per sample
                J_i = float(((uT_sim[i] - uT_target[i]) ** 2).mean())
                axs[i, 3].text(0.5, 0.5,
                               f"J = {J_i:.5f}\nE = {(w_pred[i] ** 2).sum():.1f}",
                               ha="center", va="center", transform=axs[i, 3].transAxes,
                               fontsize=12)
                axs[i, 3].axis("off")

            fig.suptitle(f"{variant} γ={gamma}  (5 test samples)", fontsize=13)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            out_png = os.path.join(args.out_dir,
                                   f"trajectories_{variant}_g{gamma:.1f}.png")
            plt.savefig(out_png, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  💾 saved {out_png}")


if __name__ == "__main__":
    main()
