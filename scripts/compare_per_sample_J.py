"""
compare_per_sample_J.py — compare per-sample J distribution: FM vs Paper DDPM/DDIM.

Loads:
  - FM ckpt → samples → per-sample J
  - Paper npz files → per-sample J (recomputed from saved x_gt, target)

For each method, prints distribution statistics + side-by-side comparison.

Usage:
    python scripts/compare_per_sample_J.py
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flow.burgers_fm_train import (
    LinearAlpha, LinearBeta, GaussianConditionalProbabilityPath,
    BurgersVectorField, load_burgers, BurgersDataset, T_IDX, RESCALER,
)
from flow.burgers_fm_eval_v2 import euler_sample, load_net, compute_J_E


FM_CKPT = '/tmp/fm_ckpts_170k/vanilla_joint.pt'
PAPER_NPZ_DIR = '/Users/baochen/Desktop/diffphycon_results/paper_trajectories_npz'
DATASET = 'free_u_f_paper_fopc'
N_TEST = 50
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def per_sample_J_paper(npz_path):
    d = np.load(npz_path)
    x_gt = d['x_gt']
    target = d['target']
    return ((x_gt[:, -1, :] - target[:, -1, :])**2).mean(axis=-1)  # (50,)


def per_sample_J_fm(ckpt_path, n_steps):
    joint = load_net(ckpt_path, DEVICE, 128, (1, 2, 4))
    ds_raw = load_burgers(DATASET, split='test', device='cpu')
    ds = BurgersDataset(ds_raw, device=DEVICE, is_prior=False)
    zs = ds.all_z[:N_TEST].to(DEVICE)
    c_eval = torch.stack([zs[:, 0, 0, :], zs[:, 0, T_IDX, :]], dim=1)
    u_target_full = zs[:, 0, :11, :]

    torch.manual_seed(42)
    x_pred = euler_sample(joint, None, c_eval, n_steps=n_steps, gamma=1.0, device=DEVICE)
    Js, _ = compute_J_E(x_pred, u_target_full)
    return np.asarray(Js)


def summarize(label, Js):
    print(f"\n=== {label} (n={len(Js)}) ===")
    print(f"  mean   = {Js.mean():.6f}")
    print(f"  median = {np.median(Js):.6f}")
    print(f"  std    = {Js.std():.6f}")
    print(f"  min    = {Js.min():.6f}")
    print(f"  max    = {Js.max():.6f}")
    top5 = np.sort(Js)[-5:]
    bot45_mean = np.sort(Js)[:-5].mean()
    print(f"  top 5  (worst): {' '.join(f'{j:.5f}' for j in top5[::-1])}")
    print(f"  mean WITHOUT top 5 outliers: {bot45_mean:.6f}")
    return {'mean': Js.mean(), 'median': np.median(Js), 'min': Js.min(),
            'max': Js.max(), 'top5_mean': top5.mean(), 'bot45_mean': bot45_mean}


def side_by_side(label_A, Js_A, label_B, Js_B):
    print(f"\n=== {label_A}  vs  {label_B}  (per-sample J, n=50) ===")
    print(f"{'rank':>5} | {label_A[:18]:>20} | {label_B[:18]:>20} | {'B better?':>10}")
    print("-" * 70)
    # Sort both by sample index (paired comparison)
    n_better = 0
    n_worse = 0
    for i in range(len(Js_A)):
        b_better = '✓' if Js_B[i] < Js_A[i] else ' '
        if Js_B[i] < Js_A[i]:
            n_better += 1
        else:
            n_worse += 1
        if i < 10 or i >= len(Js_A) - 5:
            print(f"{i:>5} | {Js_A[i]:>20.6f} | {Js_B[i]:>20.6f} | {b_better:>10}")
        elif i == 10:
            print(f"  ... ({len(Js_A)-15} more samples) ...")
    print()
    print(f"{label_B} better on {n_better}/{len(Js_A)} samples ({n_better/len(Js_A)*100:.0f}%)")


def main():
    # Paper DDPM
    paper_J = per_sample_J_paper(os.path.join(PAPER_NPZ_DIR, 'inference_trajectories_ddpm_1000.npz'))
    summarize('Paper DDPM 1000', paper_J)

    # Paper DDIM 8 (paper's fast variant)
    ddim8_J = per_sample_J_paper(os.path.join(PAPER_NPZ_DIR, 'inference_trajectories_ddim_8.npz'))
    summarize('Paper DDIM 8', ddim8_J)

    if not os.path.exists(FM_CKPT):
        print(f"\n⚠️  FM ckpt not found at {FM_CKPT}")
        print("    Run: bash /Users/baochen/diffphycon/pull_fm_ckpt.sh")
        return

    # FM at n=8 (the optimal step count)
    print(f"\nRunning FM @ n=8 on Mac MPS (a few seconds)...")
    fm8_J = per_sample_J_fm(FM_CKPT, n_steps=8)
    summarize('Our FM 170k, n=8', fm8_J)

    # FM at n=1 (extreme)
    print(f"\nRunning FM @ n=1 on Mac MPS (instant)...")
    fm1_J = per_sample_J_fm(FM_CKPT, n_steps=1)
    summarize('Our FM 170k, n=1', fm1_J)

    # Per-sample paired comparison: FM 8 vs Paper DDPM 1000
    side_by_side('Paper DDPM 1000', paper_J, 'Our FM n=8', fm8_J)

    # Same-sample win rate at every config
    print("\n=== Win rate (FM n=8 < Paper config) ===")
    for label, Js_other in [('Paper DDPM 1000', paper_J), ('Paper DDIM 8', ddim8_J)]:
        wins = np.sum(fm8_J < Js_other)
        print(f"  FM beats {label}: {wins}/{N_TEST} samples ({wins/N_TEST*100:.0f}%)")

    # Save combined CSV
    out = '/Users/baochen/Desktop/diffphycon_results/per_sample_J_comparison.csv'
    with open(out, 'w') as f:
        f.write("sample_idx,paper_ddpm_1000,paper_ddim_8,fm_n1,fm_n8\n")
        for i in range(N_TEST):
            f.write(f"{i},{paper_J[i]:.6f},{ddim8_J[i]:.6f},{fm1_J[i]:.6f},{fm8_J[i]:.6f}\n")
    print(f"\n💾 saved {out}")


if __name__ == '__main__':
    main()
