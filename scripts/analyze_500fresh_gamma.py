"""
analyze_500fresh_gamma.py — multi-γ analysis on 500-sample fresh sweep.

Inputs:
  ~/Desktop/diffphycon_results_500fresh/sweep_500fresh/fm_n{1,4,8,100,500,1000}/
    per_sample_J_vanilla_g{0.10,0.30,0.50,0.70,1.00,1.50,2.00,3.00}_n*.npy
  ~/Desktop/diffphycon_results_500fresh/paper_npz/inference_trajectories_ddim_*.npz

Outputs (saved to data dir):
  gamma_J_table.csv             — wide table: mean/median J × (n_steps, γ)
  plot_J_vs_gamma.png           — line plot, J vs γ for each n_steps
  plot_heatmap_gamma_nsteps.png — heatmap of mean J across (γ, n_steps) grid
  plot_paired_g1_vs_g0.5_n8.png — paired comparison FM γ=1.0 vs γ=0.5 (n=8)
  plot_paired_g1_vs_g0.1_n8.png — paired comparison FM γ=1.0 vs γ=0.1 (n=8) — most aggressive flatten
  outlier_gamma_tracking.txt    — top-10 worst @ γ=1.0, track their J across γ
  best_gamma_per_nsteps.txt     — best γ for each n_steps
"""
from __future__ import annotations
import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DATA_DIR = '/Users/baochen/Desktop/diffphycon_results_500fresh'
SWEEP = os.path.join(DATA_DIR, 'sweep_500fresh')
OUT = DATA_DIR

N_STEPS = [1, 4, 8, 100, 500, 1000]
GAMMAS = [0.10, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00, 3.00]


def load_fm(n_steps, gamma):
    path = os.path.join(SWEEP, f'fm_n{n_steps}',
                        f'per_sample_J_vanilla_g{gamma:.2f}_n{n_steps}.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def main():
    # Load all 48 npy
    print(f"Loading {len(N_STEPS)} × {len(GAMMAS)} = {len(N_STEPS)*len(GAMMAS)} npy files...")
    data = {}  # data[(n_steps, gamma)] = (500,) array
    for n in N_STEPS:
        for g in GAMMAS:
            data[(n, g)] = load_fm(n, g)
    print(f"  ✓ loaded, all (n_steps × γ) arrays have shape (500,)")

    # ---------- Table: mean / median J across (n_steps, γ) ----------
    csv_path = os.path.join(OUT, 'gamma_J_table.csv')
    with open(csv_path, 'w') as fh:
        fh.write('n_steps,stat,' + ','.join(f'g{g:.2f}' for g in GAMMAS) + '\n')
        for n in N_STEPS:
            row_mean = [f'{data[(n, g)].mean():.6f}' for g in GAMMAS]
            row_med = [f'{np.median(data[(n, g)]):.6f}' for g in GAMMAS]
            fh.write(f'{n},mean,' + ','.join(row_mean) + '\n')
            fh.write(f'{n},median,' + ','.join(row_med) + '\n')
    print(f"💾 {csv_path}")

    # Print table to stdout
    print("\n--- mean J across (n_steps, γ) ---")
    g_cols = ''.join(('γ=%.1f' % g).ljust(12) for g in GAMMAS)
    header = f"{'n_steps':<8}" + g_cols
    print(header)
    print('-' * len(header))
    for n in N_STEPS:
        row = f"{n:<8}" + ''.join(f'{data[(n,g)].mean():<12.6f}' for g in GAMMAS)
        print(row)

    print("\n--- median J across (n_steps, γ) ---")
    print(header)
    print('-' * len(header))
    for n in N_STEPS:
        row = f"{n:<8}" + ''.join(f'{np.median(data[(n,g)]):<12.6f}' for g in GAMMAS)
        print(row)

    # ---------- Plot 1: J vs γ (line per n_steps) ----------
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(N_STEPS)))
    for c, n in zip(colors, N_STEPS):
        means = [data[(n, g)].mean() for g in GAMMAS]
        medians = [np.median(data[(n, g)]) for g in GAMMAS]
        plt.plot(GAMMAS, means, 'o-', color=c, label=f'n={n} (mean)', lw=2)
        plt.plot(GAMMAS, medians, 's--', color=c, alpha=0.5, label=f'n={n} (median)')
    plt.axvline(1.0, ls=':', color='gray', alpha=0.5, label='γ=1.0 (no prior)')
    plt.axhline(0.00037, ls='--', color='red', alpha=0.5, label='paper Table 1 baseline')
    plt.xlabel('γ (prior reweighting; <1 flatten, >1 sharpen)')
    plt.ylabel('J (log)')
    plt.yscale('log')
    plt.title('FM J vs γ for each n_steps (500-sample held-out)')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'plot_J_vs_gamma.png'), dpi=110)
    plt.close()
    print(f"💾 {OUT}/plot_J_vs_gamma.png")

    # ---------- Plot 2: Heatmap ----------
    grid = np.array([[data[(n, g)].mean() for g in GAMMAS] for n in N_STEPS])
    plt.figure(figsize=(11, 5))
    im = plt.imshow(np.log10(grid), aspect='auto', cmap='RdYlGn_r')
    plt.colorbar(im, label='log10(mean J)')
    plt.xticks(range(len(GAMMAS)), [f'{g:.2f}' for g in GAMMAS])
    plt.yticks(range(len(N_STEPS)), [str(n) for n in N_STEPS])
    plt.xlabel('γ')
    plt.ylabel('n_steps')
    plt.title('mean J heatmap (log scale) — lower (greener) is better')
    # annotate cells with the actual J value
    for i, n in enumerate(N_STEPS):
        for j, g in enumerate(GAMMAS):
            v = grid[i, j]
            color = 'white' if np.log10(v) > -3.5 else 'black'
            plt.text(j, i, f'{v:.4f}', ha='center', va='center', fontsize=7, color=color)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'plot_heatmap_gamma_nsteps.png'), dpi=110)
    plt.close()
    print(f"💾 {OUT}/plot_heatmap_gamma_nsteps.png")

    # ---------- Plot 3 & 4: Paired comparison ----------
    def paired_plot(n_step, g_base, g_test, out_name):
        j_base = data[(n_step, g_base)]
        j_test = data[(n_step, g_test)]
        order = np.argsort(j_base)
        j_base_sorted = j_base[order]
        j_test_sorted = j_test[order]

        plt.figure(figsize=(12, 5))
        plt.plot(np.arange(500), j_base_sorted, 's-', color='orange',
                 label=f'FM γ={g_base:.2f} (baseline)', ms=4)
        plt.plot(np.arange(500), j_test_sorted, '^-', color='steelblue',
                 label=f'FM γ={g_test:.2f} (flatten)', ms=4)
        plt.fill_between(np.arange(500), j_test_sorted, j_base_sorted,
                         where=j_test_sorted < j_base_sorted,
                         color='lightgreen', alpha=0.3, label=f'γ={g_test} wins')
        plt.fill_between(np.arange(500), j_test_sorted, j_base_sorted,
                         where=j_test_sorted > j_base_sorted,
                         color='lightcoral', alpha=0.2, label=f'γ={g_test} loses')
        plt.yscale('log')
        plt.xlabel(f'sample rank (sorted by γ={g_base:.2f} difficulty)')
        plt.ylabel('J per sample (log)')
        win = (j_test < j_base).sum()
        plt.title(f'FM n={n_step}: γ={g_test:.2f} vs γ={g_base:.2f}  —  '
                  f'γ={g_test:.2f} wins {win}/500 ({win/5:.1f}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, out_name), dpi=110)
        plt.close()
        print(f"💾 {OUT}/{out_name}")
        return win

    w_05 = paired_plot(8, 1.00, 0.50, 'plot_paired_g1_vs_g0.5_n8.png')
    w_01 = paired_plot(8, 1.00, 0.10, 'plot_paired_g1_vs_g0.1_n8.png')

    # ---------- Outlier tracking at n=8 ----------
    n = 8
    j_g1 = data[(n, 1.00)]
    worst_idx = np.argsort(j_g1)[-10:][::-1]  # 10 worst at γ=1.0
    out_path = os.path.join(OUT, 'outlier_gamma_tracking.txt')
    with open(out_path, 'w') as fh:
        fh.write(f"Top-10 worst samples at FM n={n} γ=1.0, tracked across all γ\n")
        fh.write('=' * 100 + '\n')
        fh.write(f"{'sample':>7}  " + '  '.join(f'γ={g:.2f}'.rjust(10) for g in GAMMAS) + '\n')
        fh.write('-' * 100 + '\n')
        for idx in worst_idx:
            row = f"  {idx:>5}  " + '  '.join(f'{data[(n, g)][idx]:>10.6f}' for g in GAMMAS)
            fh.write(row + '\n')
            print(row)
        # Aggregate: how often does flatten help these outliers?
        fh.write('\n\nAggregate over top-10 worst (γ=1.0):\n')
        for g in GAMMAS:
            if abs(g - 1.0) < 1e-3: continue
            j_g = data[(n, g)][worst_idx]
            wins = (j_g < j_g1[worst_idx]).sum()
            ratio = j_g.mean() / j_g1[worst_idx].mean()
            fh.write(f"  γ={g:.2f}: outlier mean ratio = {ratio:.2f}× (win {wins}/10)\n")
    print(f"💾 {out_path}")

    # ---------- Best γ per n_steps ----------
    out_path = os.path.join(OUT, 'best_gamma_per_nsteps.txt')
    with open(out_path, 'w') as fh:
        fh.write("Best γ per n_steps (by mean J)\n")
        fh.write('=' * 80 + '\n')
        fh.write(f"{'n_steps':<8} {'γ@1.0 J':>12} {'best γ':>10} {'best J':>14} "
                 f"{'gain×':>8} {'gain %':>10}\n")
        fh.write('-' * 80 + '\n')
        for n in N_STEPS:
            j_g1 = data[(n, 1.00)].mean()
            best_g_idx = np.argmin([data[(n, g)].mean() for g in GAMMAS])
            best_g = GAMMAS[best_g_idx]
            best_j = data[(n, best_g)].mean()
            gain = j_g1 / best_j
            pct = (1 - best_j / j_g1) * 100
            line = f'{n:<8} {j_g1:>12.6f} {best_g:>10.2f} {best_j:>14.6f} {gain:>7.2f}× {pct:>9.1f}%'
            fh.write(line + '\n')
            print(line)
    print(f"💾 {out_path}")

    print("\n" + "=" * 60)
    print("✅ ALL DONE")
    print(f"open {OUT}/plot_J_vs_gamma.png")
    print(f"open {OUT}/plot_heatmap_gamma_nsteps.png")
    print(f"open {OUT}/plot_paired_g1_vs_g0.5_n8.png")
    print(f"open {OUT}/plot_paired_g1_vs_g0.1_n8.png")
    print(f"cat {OUT}/best_gamma_per_nsteps.txt {OUT}/outlier_gamma_tracking.txt")


if __name__ == '__main__':
    main()
