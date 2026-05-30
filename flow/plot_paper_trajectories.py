"""
plot_paper_trajectories.py — render 4-panel plots from paper inference output (npz).

Paper inference (inference/inference_1d_burgers.py) saves to:
    outputs/trajectories/inference_trajectories{_tag}.npz
with keys:
    x_pred:  (B, 2, 11, 128) — predicted (u, f) by paper's DDPM/DDIM model
    x_gt:    (B, 11, 128)    — re-simulated u from predicted f via ground-truth solver
    target:  (B, 11, 128)    — full target u trajectory (with u_0 and u_T*)

Renders the same 4-panel layout as flow/plot_trajectories.py:
    [pred u(t,x)] [pred w(t,x)] [terminal match] [J / E text]

Usage:
    python flow/plot_paper_trajectories.py \\
        --npz outputs/trajectories/inference_trajectories.npz \\
        --out_png /tmp/paper_ddim_n8.png \\
        --title "Paper DDIM n_steps=8" \\
        --n_samples 5
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


T_IDX = 10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="path to paper inference_trajectories.npz")
    p.add_argument("--out_png", required=True)
    p.add_argument("--title", default="Paper inference")
    p.add_argument("--n_samples", type=int, default=5)
    args = p.parse_args()

    data = np.load(args.npz)
    x_pred = data['x_pred']      # (B, 2, 11, 128) — predicted (u, f)
    x_gt = data['x_gt']          # (B, 11, 128)    — re-simulated u from f
    target = data['target']      # (B, 11, 128)    — full target u

    n = min(args.n_samples, x_pred.shape[0])

    # Paper f shape: (B, 2, 11, 128). f-channel is index 1 (paper's representation)
    # paper layout: x_pred[:, 0] = u, x_pred[:, 1] = f (control)
    # f is over t=0..9 (10 steps, but stored as 11 with last possibly zero)
    u_pred_np = x_pred[:, 0, :, :]      # (B, 11, 128) — predicted u
    w_pred_np = x_pred[:, 1, :10, :]    # (B, 10, 128) — predicted w/f
    # mask central 50% of w (paper D.2.2)
    n_x = w_pred_np.shape[-1]
    w_pred_np = w_pred_np.copy()
    w_pred_np[:, :, n_x // 4 : 3 * n_x // 4] = 0.0

    u0_np = target[:, 0, :]            # (B, 128)
    uT_target_np = target[:, -1, :]    # (B, 128)
    uT_sim_np = x_gt[:, -1, :]         # (B, 128) — solver-simulated u(T) from predicted f

    # Overwrite boundary rows for visualization (same fix as our plot_trajectories.py)
    u_pred_np = u_pred_np.copy()
    u_pred_np[:, 0, :]     = u0_np
    u_pred_np[:, T_IDX, :] = uT_target_np

    fig, axs = plt.subplots(n, 4, figsize=(16, 3.0 * n))
    if n == 1:
        axs = axs[np.newaxis, :]
    vmax_w = max(np.abs(w_pred_np).max(), 1e-3)
    for i in range(n):
        axs[i, 0].imshow(u_pred_np[i], cmap="RdBu_r", aspect="auto", vmin=-3, vmax=3)
        axs[i, 0].set_title(f"sample {i}: pred u(t,x)" if i == 0 else "")
        axs[i, 0].set_ylabel(f"#{i}\nt")
        axs[i, 1].imshow(w_pred_np[i], cmap="RdBu_r", aspect="auto",
                         vmin=-vmax_w, vmax=vmax_w)
        axs[i, 1].set_title("pred w(t,x)" if i == 0 else "")
        axs[i, 2].plot(uT_target_np[i], "k-", label="target u_T*", lw=2)
        axs[i, 2].plot(uT_sim_np[i], "b--", label="sim u(T)", lw=1.5)
        axs[i, 2].plot(u0_np[i], color="grey", alpha=0.5, label="u_0", lw=0.8)
        axs[i, 2].set_title("terminal match" if i == 0 else "")
        axs[i, 2].legend(fontsize=7, loc="upper right")
        axs[i, 2].grid(alpha=0.3)
        J_i = float(((uT_sim_np[i] - uT_target_np[i]) ** 2).mean())
        E_i = float((w_pred_np[i] ** 2).sum())
        axs[i, 3].text(0.5, 0.5, f"J = {J_i:.5f}\nE = {E_i:.1f}",
                       ha="center", va="center", transform=axs[i, 3].transAxes,
                       fontsize=12)
        axs[i, 3].axis("off")

    fig.suptitle(f"{args.title}  ({n} test samples)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plt.savefig(args.out_png, dpi=120, bbox_inches="tight")
    print(f"💾 saved {args.out_png}")


if __name__ == "__main__":
    main()
