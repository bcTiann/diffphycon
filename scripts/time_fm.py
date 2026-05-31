"""
time_fm.py — measure FM euler_sample timing accurately (5-run average, warmup + CUDA sync).

Skips J/E computation — pure timing.

Usage:
    python scripts/time_fm.py \\
        --ckpt /root/autodl-tmp/diffphycon/flow/checkpoints/paper_fopc_v2/vanilla_joint.pt \\
        --dataset free_u_f_paper_fopc \\
        --n_steps_list 1 4 8 16 50 100 1000 \\
        --n_repeats 5 \\
        --device cuda \\
        --out /tmp/fm_timing.csv
"""
from __future__ import annotations
import argparse
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
from flow.burgers_fm_eval_v2 import euler_sample, load_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to vanilla_joint.pt")
    p.add_argument("--dataset", default="free_u_f_paper_fopc")
    p.add_argument("--n_test", type=int, default=50)
    p.add_argument("--n_steps_list", type=int, nargs="+",
                   default=[1, 4, 8, 16, 50, 100, 1000])
    p.add_argument("--n_repeats", type=int, default=5,
                   help="how many timed runs per n_steps (mean ± std)")
    p.add_argument("--joint_dim", type=int, default=128)
    p.add_argument("--joint_mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="output CSV")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    cuda = args.device.startswith("cuda")

    # Load model
    print(f"loading ckpt: {args.ckpt}")
    joint = load_net(args.ckpt, args.device, args.joint_dim, tuple(args.joint_mults))

    # Load test set + build c_eval
    ds_raw = load_burgers(args.dataset, split="test", device="cpu")
    ds = BurgersDataset(ds_raw, device=args.device, is_prior=False)
    zs = ds.all_z[:args.n_test].to(args.device)
    c_eval = torch.stack([zs[:, 0, 0, :], zs[:, 0, T_IDX, :]], dim=1)
    print(f"  using {c_eval.shape[0]} test samples")

    # CUDA warmup (run once outside timing loop)
    print(f"\nWarming up CUDA (1 dummy call)...")
    torch.manual_seed(args.seed)
    _ = euler_sample(joint, None, c_eval[:2], n_steps=max(args.n_steps_list),
                     gamma=1.0, device=args.device)
    if cuda:
        torch.cuda.synchronize()

    print(f"\nTiming (n_repeats={args.n_repeats}, 50 batched samples each, γ=1):\n")
    print(f"{'n_steps':>8} | {'mean_ms_total':>14} | {'std_ms_total':>14} | "
          f"{'mean_us_per_sample':>20} | {'throughput_samp/s':>20}")
    print("-" * 95)

    results = []
    for n_steps in args.n_steps_list:
        times = []
        for rep in range(args.n_repeats):
            torch.manual_seed(args.seed + rep)
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            x_pred = euler_sample(joint, None, c_eval,
                                  n_steps=n_steps, gamma=1.0,
                                  device=args.device)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        ts = np.array(times)
        mean_s = ts.mean()
        std_s  = ts.std()
        per_sample_us = mean_s / args.n_test * 1e6
        throughput   = args.n_test / mean_s
        print(f"{n_steps:>8} | {mean_s*1000:>14.3f} | {std_s*1000:>14.3f} | "
              f"{per_sample_us:>20.1f} | {throughput:>20.1f}")
        results.append({
            "n_steps": n_steps,
            "mean_total_s": mean_s,
            "std_total_s":  std_s,
            "per_sample_us": per_sample_us,
            "throughput_samp_per_s": throughput,
        })

    # Save CSV
    import csv
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n💾 saved {args.out}")


if __name__ == "__main__":
    main()
