"""
plot_loss.py — render 4 loss curves (vanilla/OT × joint/prior) on log-y with sliding mean.

Usage:
    python flow/plot_loss.py \\
        --ckpt_dir checkpoints/paper_fopc_v2 \\
        --out results/paper_fopc_v2/loss_curves.png
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def smooth(x, w=200):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smooth", type=int, default=200, help="sliding window size")
    args = p.parse_args()

    models = [
        ("vanilla_joint", "vanilla joint"),
        ("ot_joint",      "OT joint"),
        ("vanilla_prior", "vanilla prior"),
        ("ot_prior",      "OT prior"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    axes = axes.flatten()

    for ax, (fname, label) in zip(axes, models):
        path = os.path.join(args.ckpt_dir, f"{fname}.pt")
        if not os.path.isfile(path):
            ax.text(0.5, 0.5, f"(missing)\n{fname}.pt",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        loss = np.array(ckpt.get("loss_history", []))
        if len(loss) == 0:
            ax.text(0.5, 0.5, "(no loss history)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        ax.plot(loss, color="lightgrey", alpha=0.5, lw=0.5, label="raw")
        sm = smooth(loss, args.smooth)
        ax.plot(np.arange(len(sm)) + args.smooth // 2, sm,
                color="C0", lw=1.5, label=f"avg{args.smooth}")
        ax.set_yscale("log")
        ax.set_title(f"{label}  (final avg{args.smooth} = {sm[-1]:.4f})")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"💾 saved {args.out}")


if __name__ == "__main__":
    main()
