"""
analyze_500fresh.py — full per-sample analysis on the 500-fresh sweep.

Inputs:
  /Users/baochen/Desktop/diffphycon_results_500fresh/
    sweep_500fresh/fm_n{1,4,8,100,500,1000}/per_sample_J_vanilla_g1.00_n*.npy
    paper_npz/inference_trajectories_ddim_{1,4,8,100,500}.npz

Outputs (saved next to inputs):
  per_sample_J_all.csv          — wide table: 500 rows × all methods
  plot_box.png                  — box plot of J distributions
  plot_J_vs_nsteps.png          — line plot, J vs n_steps log-log
  plot_paired_FMn8_vs_DDIM100.png — paired sample-by-sample comparison
  win_rates.txt                 — pairwise win rates + 95% bootstrap CI
  outlier_summary.txt           — max/p99/IQR per method

Usage:
  python scripts/analyze_500fresh.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from utils import burgers_metric

DATA_DIR = '/Users/baochen/Desktop/diffphycon_results_500fresh'
SWEEP = os.path.join(DATA_DIR, 'sweep_500fresh')
NPZ_DIR = os.path.join(DATA_DIR, 'paper_npz')
OUT = DATA_DIR  # save all outputs next to data

FM_STEPS = [1, 4, 8, 100, 500, 1000]
DDIM_STEPS = [1, 4, 8, 100, 500]


def load_fm_per_sample(n_steps):
    path = os.path.join(SWEEP, f'fm_n{n_steps}', f'per_sample_J_vanilla_g1.00_n{n_steps}.npy')
    return np.load(path)


def recompute_paper_per_sample(tag):
    """Use the same burgers_metric pipeline as the logged J_actual."""
    d = np.load(os.path.join(NPZ_DIR, f'inference_trajectories_{tag}.npz'))
    x_pred = torch.from_numpy(d['x_pred'])
    target = torch.from_numpy(d['target'])
    f = x_pred[:, 1, :10, :].clone()
    # Suppress the PDE solver tqdm bar
    import contextlib, io
    with contextlib.redirect_stderr(io.StringIO()):
        J, _ = burgers_metric(
            u_target=target, f=f,
            target='final_u', partial_control='front_rear_quarter',
        )
    return J.numpy()


def bootstrap_winrate_ci(a, b, n_boot=10000, alpha=0.05):
    """95% CI on P(a < b) via bootstrap resampling."""
    n = len(a)
    rng = np.random.default_rng(0)
    wins = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        wins.append(np.mean(a[idx] < b[idx]))
    lo = np.percentile(wins, 100 * alpha / 2)
    hi = np.percentile(wins, 100 * (1 - alpha / 2))
    return np.mean(a < b), lo, hi


def main():
    # ---------- Load all per-sample J ----------
    print("Loading FM per-sample J...")
    fm = {n: load_fm_per_sample(n) for n in FM_STEPS}
    print(f"  ✓ FM 6 configs loaded, shape={fm[8].shape}")

    print("\nRecomputing Paper per-sample J from npz (a few seconds each)...")
    paper = {}
    for s in DDIM_STEPS:
        print(f"  ddim_{s} ...", end='', flush=True)
        paper[s] = recompute_paper_per_sample(f'ddim_{s}')
        print(f" mean={paper[s].mean():.6f}  median={np.median(paper[s]):.6f}")

    # ---------- Wide CSV ----------
    csv_path = os.path.join(OUT, 'per_sample_J_all.csv')
    headers = ['sample_idx'] + [f'FM_n{n}' for n in FM_STEPS] + [f'DDIM_n{s}' for s in DDIM_STEPS]
    with open(csv_path, 'w') as fh:
        fh.write(','.join(headers) + '\n')
        for i in range(500):
            row = [str(i)]
            row += [f'{fm[n][i]:.6e}' for n in FM_STEPS]
            row += [f'{paper[s][i]:.6e}' for s in DDIM_STEPS]
            fh.write(','.join(row) + '\n')
    print(f"\n💾 saved {csv_path}")

    # ---------- Box plot ----------
    plt.figure(figsize=(12, 5))
    methods, data, colors = [], [], []
    for n in FM_STEPS:
        methods.append(f'FM\nn={n}')
        data.append(fm[n])
        colors.append('steelblue')
    for s in DDIM_STEPS:
        methods.append(f'DDIM\nn={s}')
        data.append(paper[s])
        colors.append('orange')

    # log-scale clip (avoid showing ddim_1's catastrophic value squishing rest)
    plt.boxplot(data, labels=methods, showfliers=True, widths=0.7,
                patch_artist=True,
                boxprops=dict(facecolor='lightgray', edgecolor='black'),
                medianprops=dict(color='red', lw=2))
    for box, col in zip(plt.gca().artists, colors):
        box.set_facecolor(col)
        box.set_alpha(0.5)
    plt.yscale('log')
    plt.ylabel('J per sample (log scale)')
    plt.title(f'Per-sample J distribution — 500 fresh held-out samples')
    plt.axhline(0.00037, ls='--', color='gray', alpha=0.5, label='paper Table 1 (J=0.00037)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'plot_box.png'), dpi=110)
    plt.close()
    print(f"💾 saved {OUT}/plot_box.png")

    # ---------- J vs n_steps line plot ----------
    plt.figure(figsize=(9, 5))
    fm_steps = sorted(fm.keys())
    fm_means = [fm[n].mean() for n in fm_steps]
    fm_meds = [np.median(fm[n]) for n in fm_steps]
    paper_steps = sorted(paper.keys())
    paper_means = [paper[s].mean() for s in paper_steps]
    paper_meds = [np.median(paper[s]) for s in paper_steps]

    plt.plot(fm_steps, fm_means, 'o-', color='steelblue', label='FM mean', lw=2)
    plt.plot(fm_steps, fm_meds, 'o--', color='steelblue', label='FM median', alpha=0.6)
    plt.plot(paper_steps, paper_means, 's-', color='orange', label='Paper DDIM mean', lw=2)
    plt.plot(paper_steps, paper_meds, 's--', color='orange', label='Paper DDIM median', alpha=0.6)
    plt.axhline(0.00037, ls='--', color='gray', alpha=0.7, label='paper Table 1')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('n_steps (log)')
    plt.ylabel('J (log)')
    plt.title('FM vs Paper DDIM: J vs n_steps (500-sample fresh test)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'plot_J_vs_nsteps.png'), dpi=110)
    plt.close()
    print(f"💾 saved {OUT}/plot_J_vs_nsteps.png")

    # ---------- Paired comparison: FM n=8 vs Paper DDIM 100 ----------
    fm8 = fm[8]
    ddim100 = paper[100]
    order = np.argsort(ddim100)
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(500), ddim100[order], 's-', color='orange', label='Paper DDIM 100', ms=4)
    plt.plot(np.arange(500), fm8[order], '^-', color='steelblue', label='FM n=8', ms=4)
    plt.fill_between(np.arange(500), fm8[order], ddim100[order],
                     where=fm8[order] < ddim100[order],
                     color='lightgreen', alpha=0.3, label='FM wins')
    plt.yscale('log')
    plt.xlabel('sample rank (sorted by Paper DDIM 100 difficulty)')
    plt.ylabel('J per sample (log)')
    win = (fm8 < ddim100).mean()
    plt.title(f'FM n=8 vs Paper DDIM 100 — FM wins {(fm8<ddim100).sum()}/500 ({win*100:.1f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'plot_paired_FMn8_vs_DDIM100.png'), dpi=110)
    plt.close()
    print(f"💾 saved {OUT}/plot_paired_FMn8_vs_DDIM100.png")

    # ---------- Win rate stats ----------
    win_path = os.path.join(OUT, 'win_rates.txt')
    pairs = [
        ('FM_n1',   fm[1],    'DDIM_n1',   paper[1]),
        ('FM_n4',   fm[4],    'DDIM_n4',   paper[4]),
        ('FM_n8',   fm[8],    'DDIM_n8',   paper[8]),
        ('FM_n100', fm[100],  'DDIM_n100', paper[100]),
        ('FM_n500', fm[500],  'DDIM_n500', paper[500]),
        ('FM_n8',   fm[8],    'DDIM_n100', paper[100]),
        ('FM_n8',   fm[8],    'DDIM_n500', paper[500]),
        ('FM_n1',   fm[1],    'DDIM_n100', paper[100]),
    ]
    with open(win_path, 'w') as fh:
        fh.write("Win rates (P(A < B)) on 500-sample fresh held-out\n")
        fh.write("=" * 70 + '\n')
        fh.write(f"{'A vs B':<30s} {'win%':>8s} {'95% CI (bootstrap)':>30s}\n")
        fh.write("-" * 70 + '\n')
        for name_a, a, name_b, b in pairs:
            wr, lo, hi = bootstrap_winrate_ci(a, b)
            line = (f"{name_a + ' < ' + name_b:<30s} "
                    f"{wr*100:>7.1f}%  [{lo*100:>5.1f}%, {hi*100:>5.1f}%]")
            fh.write(line + '\n')
            print('  ' + line)
    print(f"\n💾 saved {win_path}")

    # ---------- Outlier summary ----------
    out_path = os.path.join(OUT, 'outlier_summary.txt')
    with open(out_path, 'w') as fh:
        fh.write("Per-method distribution stats (J)\n")
        fh.write("=" * 90 + '\n')
        fh.write(f"{'method':<14s} {'mean':>12s} {'median':>12s} {'p99':>12s} "
                 f"{'max':>12s} {'std':>12s} {'IQR':>12s}\n")
        fh.write("-" * 90 + '\n')
        for n in FM_STEPS:
            x = fm[n]
            fh.write(f"{'FM n='+str(n):<14s} {x.mean():>12.6f} {np.median(x):>12.6f} "
                     f"{np.percentile(x, 99):>12.6f} {x.max():>12.6f} "
                     f"{x.std():>12.6f} "
                     f"{np.percentile(x,75)-np.percentile(x,25):>12.6f}\n")
        for s in DDIM_STEPS:
            x = paper[s]
            fh.write(f"{'DDIM n='+str(s):<14s} {x.mean():>12.6f} {np.median(x):>12.6f} "
                     f"{np.percentile(x, 99):>12.6f} {x.max():>12.6f} "
                     f"{x.std():>12.6f} "
                     f"{np.percentile(x,75)-np.percentile(x,25):>12.6f}\n")
    print(f"💾 saved {out_path}")

    print("\n" + "=" * 60)
    print("✅ ALL DONE")
    print("=" * 60)
    print(f"open {OUT}/plot_box.png")
    print(f"open {OUT}/plot_J_vs_nsteps.png")
    print(f"open {OUT}/plot_paired_FMn8_vs_DDIM100.png")
    print(f"cat {OUT}/win_rates.txt {OUT}/outlier_summary.txt")


if __name__ == '__main__':
    main()
