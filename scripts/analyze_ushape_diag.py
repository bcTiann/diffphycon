"""
analyze_ushape_diag.py — analyze the factorial U-shape diagnostic sweep.

Inputs (under ~/Desktop/diffphycon_results_500fresh/sweep_500fresh/):
  Baseline (already there): fm_n{N}/per_sample_J_vanilla_g1.00_n{N}.npy

Plus (from sweep_ushape_diag.sh, expected under sweep_ushape_diag/):
  cap_tau_{0.7,0.85,0.9,0.95}/fm_n{N}/...          (E1 only, 24)
  reinpaint/fm_n{N}/...                            (E3 only, 6)
  cap_tau_{0.7,0.85,0.9,0.95}_reinpaint/fm_n{N}/.. (E1+E3, 24)
  rk4/fm_n{N}/...                                  (E4 only, 6)

Outputs (saved next to inputs):
  ushape_diag_table.csv         — mean J for every cell × n_steps
  plot_ushape_J_vs_nsteps.png   — J vs n_steps, 1 line per condition
  plot_ushape_cap_tau_heatmap.png — heatmap (cap_τ, n_steps) with/without reinpaint side-by-side
  plot_ushape_factorial.png     — bar chart at each n_steps: baseline vs E1 vs E3 vs E1+E3 vs E4
  ushape_diag_report.txt        — judgment: which hypothesis won?
"""
from __future__ import annotations
import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = '/Users/baochen/Desktop/diffphycon_results_500fresh'
BASELINE_DIR = os.path.join(DATA_ROOT, 'sweep_500fresh')         # existing γ=1.0 baseline
DIAG_DIR     = os.path.join(DATA_ROOT, 'sweep_ushape_diag')      # new from sweep_ushape_diag.sh
OUT = DATA_ROOT

N_STEPS = [1, 4, 8, 100, 500, 1000]
CAP_TAUS = [0.7, 0.85, 0.9, 0.95]


def load_baseline(n_steps):
    """γ=1.0 from the existing sigmoid_flip sweep — our 'no fix' control."""
    path = os.path.join(BASELINE_DIR, f'fm_n{n_steps}',
                        f'per_sample_J_vanilla_g1.00_n{n_steps}.npy')
    return np.load(path)


def load_diag(cell_dir, n_steps):
    path = os.path.join(DIAG_DIR, cell_dir, f'fm_n{n_steps}',
                        f'per_sample_J_vanilla_g1.00_n{n_steps}.npy')
    if not os.path.exists(path):
        return None
    return np.load(path)


def main():
    # ─── Load everything ───
    print('Loading baseline (existing γ=1.0)...')
    baseline = {n: load_baseline(n) for n in N_STEPS}
    print(f'  ✓ baseline shape {baseline[8].shape}')

    print('\nLoading diagnostic cells...')
    cells = {'baseline': baseline}

    # E1 only: cap_τ variants
    for ct in CAP_TAUS:
        cells[f'capτ={ct}'] = {n: load_diag(f'cap_tau_{ct}', n) for n in N_STEPS}

    # E3 only: reinpaint
    cells['reinpaint'] = {n: load_diag('reinpaint', n) for n in N_STEPS}

    # E1+E3 combined
    for ct in CAP_TAUS:
        cells[f'capτ={ct}+rein'] = {n: load_diag(f'cap_tau_{ct}_reinpaint', n) for n in N_STEPS}

    # E4
    cells['rk4'] = {n: load_diag('rk4', n) for n in N_STEPS}

    # Report which cells loaded
    print()
    for name, data in cells.items():
        missing = [n for n, d in data.items() if d is None]
        loaded = sum(1 for d in data.values() if d is not None)
        msg = f'  {name:30s}: {loaded}/{len(N_STEPS)} loaded'
        if missing:
            msg += f'  (missing n={missing})'
        print(msg)

    # ─── Wide CSV ───
    csv = os.path.join(OUT, 'ushape_diag_table.csv')
    with open(csv, 'w') as fh:
        fh.write('condition,stat,' + ','.join(f'n={n}' for n in N_STEPS) + '\n')
        for name, data in cells.items():
            means = [f'{data[n].mean():.6f}' if data[n] is not None else 'NA' for n in N_STEPS]
            meds = [f'{np.median(data[n]):.6f}' if data[n] is not None else 'NA' for n in N_STEPS]
            fh.write(f'{name},mean,'   + ','.join(means) + '\n')
            fh.write(f'{name},median,' + ','.join(meds) + '\n')
    print(f'\n💾 {csv}')

    # ─── Print mean table ───
    print('\n=== mean J across (condition, n_steps) ===')
    width = 12
    hdr = f"{'condition':<25s}" + ''.join(f'n={n}'.rjust(width) for n in N_STEPS)
    print(hdr)
    print('─' * len(hdr))
    for name, data in cells.items():
        row = f"{name:<25s}"
        for n in N_STEPS:
            if data[n] is None:
                row += f"{'NA':>{width}}"
            else:
                row += f"{data[n].mean():>{width}.6f}"
        print(row)

    # ─── Plot 1: J vs n_steps line plot (1 line per condition) ───
    plt.figure(figsize=(11, 7))
    # baseline distinct: thick orange dashed
    plt.plot(N_STEPS, [baseline[n].mean() for n in N_STEPS],
             'o--', color='orange', lw=3, label='baseline (our v2, no fix)', zorder=10)

    # cap_τ alone: gradient blue
    cap_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(CAP_TAUS)))
    for c, ct in zip(cap_colors, CAP_TAUS):
        data = cells[f'capτ={ct}']
        vals = [data[n].mean() if data[n] is not None else None for n in N_STEPS]
        plt.plot(N_STEPS, vals, 's-', color=c, label=f'E1: cap_τ={ct}', alpha=0.85)

    # reinpaint alone: green
    data = cells['reinpaint']
    vals = [data[n].mean() if data[n] is not None else None for n in N_STEPS]
    plt.plot(N_STEPS, vals, '^-', color='forestgreen', lw=2.5, label='E3: reinpaint only', zorder=9)

    # E1+E3: gradient purple (showing whether combo helps beyond E3 alone)
    combo_colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(CAP_TAUS)))
    for c, ct in zip(combo_colors, CAP_TAUS):
        data = cells[f'capτ={ct}+rein']
        vals = [data[n].mean() if data[n] is not None else None for n in N_STEPS]
        plt.plot(N_STEPS, vals, 'd-', color=c, label=f'E1+E3: cap_τ={ct}+rein', alpha=0.85)

    # RK4
    data = cells['rk4']
    vals = [data[n].mean() if data[n] is not None else None for n in N_STEPS]
    plt.plot(N_STEPS, vals, 'P-', color='red', lw=2.5, label='E4: RK4 (Euler control)', zorder=9)

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('n_steps (Euler/RK4 iterations)')
    plt.ylabel('mean J')
    plt.title('FM U-shape diagnostic: J vs n_steps for each intervention\n'
              '(baseline U-shape = orange; flatter is better)')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'plot_ushape_J_vs_nsteps.png'), dpi=110)
    plt.close()
    print(f'💾 {OUT}/plot_ushape_J_vs_nsteps.png')

    # ─── Plot 2: cap_τ heatmap side-by-side (without/with reinpaint) ───
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cap_axis = [1.0] + CAP_TAUS  # row=cap_τ, col=n_steps
    # Sort cap_axis descending for natural reading (1.0 at top = no cap)
    cap_axis_sorted = sorted(cap_axis, reverse=True)

    # Left: without reinpaint (baseline + E1 cells)
    grid_no = np.array([
        [baseline[n].mean() if ct == 1.0 else
         (cells[f'capτ={ct}'][n].mean() if cells[f'capτ={ct}'][n] is not None else np.nan)
         for n in N_STEPS]
        for ct in cap_axis_sorted
    ])
    im1 = axes[0].imshow(np.log10(grid_no), aspect='auto', cmap='RdYlGn_r')
    axes[0].set_xticks(range(len(N_STEPS))); axes[0].set_xticklabels([str(n) for n in N_STEPS])
    axes[0].set_yticks(range(len(cap_axis_sorted))); axes[0].set_yticklabels([f'{c:.2f}' for c in cap_axis_sorted])
    axes[0].set_xlabel('n_steps'); axes[0].set_ylabel('cap_τ')
    axes[0].set_title('No reinpaint (E1 only)')
    for i, ct in enumerate(cap_axis_sorted):
        for j, n in enumerate(N_STEPS):
            v = grid_no[i, j]
            axes[0].text(j, i, f'{v:.5f}' if not np.isnan(v) else 'NA',
                         ha='center', va='center', fontsize=7,
                         color='white' if np.log10(v + 1e-10) > -3.5 else 'black')

    # Right: with reinpaint (E3 + E1+E3 cells)
    grid_yes = np.array([
        [cells['reinpaint'][n].mean() if ct == 1.0 and cells['reinpaint'][n] is not None else
         (cells[f'capτ={ct}+rein'][n].mean() if ct < 1.0 and cells[f'capτ={ct}+rein'][n] is not None else np.nan)
         for n in N_STEPS]
        for ct in cap_axis_sorted
    ])
    im2 = axes[1].imshow(np.log10(grid_yes), aspect='auto', cmap='RdYlGn_r',
                         vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])
    axes[1].set_xticks(range(len(N_STEPS))); axes[1].set_xticklabels([str(n) for n in N_STEPS])
    axes[1].set_xlabel('n_steps')
    axes[1].set_title('With reinpaint (E3 / E1+E3)')
    for i, ct in enumerate(cap_axis_sorted):
        for j, n in enumerate(N_STEPS):
            v = grid_yes[i, j]
            axes[1].text(j, i, f'{v:.5f}' if not np.isnan(v) else 'NA',
                         ha='center', va='center', fontsize=7,
                         color='white' if np.log10(v + 1e-10) > -3.5 else 'black')

    fig.colorbar(im2, ax=axes, label='log10(mean J)', shrink=0.8)
    fig.suptitle('Factorial heatmap: cap_τ × reinpaint × n_steps  (lower J = greener)',
                 y=1.02, fontsize=12)
    plt.savefig(os.path.join(OUT, 'plot_ushape_cap_tau_heatmap.png'), dpi=110, bbox_inches='tight')
    plt.close()
    print(f'💾 {OUT}/plot_ushape_cap_tau_heatmap.png')

    # ─── Per-sample paired: baseline vs best intervention at n=1000 ───
    # Standard pipeline: even if aggregates differ, per-sample tells you whether
    # an intervention helps the SAME samples that failed in baseline.
    def per_sample_paired(name_a, data_a_all, name_b, data_b_all, n_step, out_name):
        a = data_a_all[n_step]; b = data_b_all[n_step]
        if a is None or b is None:
            print(f'  skip paired {name_a} vs {name_b}: missing'); return
        order = np.argsort(a)
        a_s, b_s = a[order], b[order]
        plt.figure(figsize=(12, 5))
        plt.plot(np.arange(len(a)), a_s, 's-', color='orange', ms=4, label=name_a)
        plt.plot(np.arange(len(b)), b_s, '^-', color='steelblue', ms=4, label=name_b)
        plt.fill_between(np.arange(len(b)), b_s, a_s, where=b_s < a_s,
                         color='lightgreen', alpha=0.3, label=f'{name_b} wins')
        plt.fill_between(np.arange(len(b)), b_s, a_s, where=b_s > a_s,
                         color='lightcoral', alpha=0.2, label=f'{name_b} loses')
        plt.yscale('log')
        plt.xlabel(f'sample rank (sorted by {name_a} difficulty)')
        plt.ylabel('J per sample (log)')
        wins = int((b < a).sum())
        plt.title(f'n_steps={n_step}: {name_b} vs {name_a}  —  '
                  f'{name_b} wins {wins}/{len(a)} ({wins/len(a)*100:.1f}%)  |  '
                  f'mean ratio={b.mean()/a.mean():.3f}×')
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(OUT, out_name), dpi=110)
        plt.close()
        print(f'💾 {OUT}/{out_name}')

    print('\n── per-sample paired plots at n=1000 ──')
    # baseline vs each intervention
    if cells.get('reinpaint', {}).get(1000) is not None:
        per_sample_paired('baseline n=1000', baseline, 'reinpaint', cells['reinpaint'], 1000,
                          'plot_paired_ushape_reinpaint_vs_baseline_n1000.png')
    for ct in CAP_TAUS:
        key = f'capτ={ct}'
        if cells.get(key, {}).get(1000) is not None:
            per_sample_paired('baseline n=1000', baseline, f'cap_τ={ct}', cells[key], 1000,
                              f'plot_paired_ushape_capτ{ct}_vs_baseline_n1000.png')
    if cells.get('rk4', {}).get(1000) is not None:
        per_sample_paired('baseline n=1000', baseline, 'rk4', cells['rk4'], 1000,
                          'plot_paired_ushape_rk4_vs_baseline_n1000.png')

    # ─── Report: which hypothesis won? ───
    rpt = os.path.join(OUT, 'ushape_diag_report.txt')
    base_n8   = baseline[8].mean()
    base_n1k  = baseline[1000].mean()
    base_ratio = base_n1k / base_n8

    def cell_n1k(name):
        d = cells[name].get(1000)
        return d.mean() if d is not None else None

    with open(rpt, 'w') as fh:
        fh.write('FM U-shape diagnostic report\n')
        fh.write('============================\n\n')
        fh.write(f'Baseline (no fix):\n')
        fh.write(f'  J(n=8)    = {base_n8:.6f}\n')
        fh.write(f'  J(n=1000) = {base_n1k:.6f}\n')
        fh.write(f'  U-shape ratio (n=1000 / n=8) = {base_ratio:.2f}×\n\n')

        fh.write('Hypothesis tests — does cell flatten the U-shape (n=1000 down to n=8 level)?\n')
        fh.write('-' * 70 + '\n')

        # E1 alone
        fh.write('\nE1 — cap_τ only (τ-OOD hypothesis):\n')
        for ct in CAP_TAUS:
            v = cell_n1k(f'capτ={ct}')
            if v is None:
                fh.write(f'  cap_τ={ct}: NA\n'); continue
            improvement = (base_n1k - v) / base_n1k * 100
            ratio_to_n8 = v / base_n8
            fh.write(f'  cap_τ={ct}: J(n=1000) = {v:.6f}  '
                     f'({improvement:+.1f}% vs baseline, {ratio_to_n8:.2f}× of baseline n=8)\n')

        # E3 alone
        fh.write('\nE3 — reinpaint only (boundary drift bug):\n')
        v = cell_n1k('reinpaint')
        if v is not None:
            improvement = (base_n1k - v) / base_n1k * 100
            ratio_to_n8 = v / base_n8
            fh.write(f'  reinpaint=on: J(n=1000) = {v:.6f}  '
                     f'({improvement:+.1f}% vs baseline, {ratio_to_n8:.2f}× of baseline n=8)\n')

        # E1+E3 combined
        fh.write('\nE1+E3 — combined cap_τ + reinpaint:\n')
        for ct in CAP_TAUS:
            v = cell_n1k(f'capτ={ct}+rein')
            if v is None:
                fh.write(f'  cap_τ={ct}+rein: NA\n'); continue
            improvement = (base_n1k - v) / base_n1k * 100
            ratio_to_n8 = v / base_n8
            fh.write(f'  cap_τ={ct}+rein: J(n=1000) = {v:.6f}  '
                     f'({improvement:+.1f}% vs baseline, {ratio_to_n8:.2f}× of baseline n=8)\n')

        # E4
        fh.write('\nE4 — RK4 only (numerical accuracy):\n')
        v = cell_n1k('rk4')
        if v is not None:
            improvement = (base_n1k - v) / base_n1k * 100
            ratio_to_n8 = v / base_n8
            fh.write(f'  rk4 (Euler control): J(n=1000) = {v:.6f}  '
                     f'({improvement:+.1f}% vs baseline, {ratio_to_n8:.2f}× of baseline n=8)\n')

        # Verdict
        fh.write('\n' + '=' * 70 + '\n')
        fh.write('VERDICT — which intervention flattens the U-shape (n=1000 J near n=8 J)?\n')
        fh.write('=' * 70 + '\n\n')
        # Find best single intervention
        candidates = []
        for ct in CAP_TAUS:
            v = cell_n1k(f'capτ={ct}')
            if v is not None: candidates.append((f'cap_τ={ct}', v))
        v = cell_n1k('reinpaint')
        if v is not None: candidates.append(('reinpaint', v))
        for ct in CAP_TAUS:
            v = cell_n1k(f'capτ={ct}+rein')
            if v is not None: candidates.append((f'cap_τ={ct}+reinpaint', v))
        v = cell_n1k('rk4')
        if v is not None: candidates.append(('rk4', v))
        candidates.sort(key=lambda x: x[1])
        fh.write('Best 3 (lowest J(n=1000)):\n')
        for name, v in candidates[:3]:
            fh.write(f'  {name:30s}  J(n=1000) = {v:.6f}\n')

        # Decision logic
        fh.write('\nDiagnostic conclusion:\n')
        best_name, best_v = candidates[0]
        if best_v <= base_n8 * 1.1:
            fh.write(f'  ✓ FLATTENED — {best_name!r} brings J(n=1000) within 10% of baseline n=8.\n')
            fh.write(f'    Strong evidence this intervention addresses the root cause.\n')
        elif best_v <= base_n1k * 0.7:
            fh.write(f'  ~ PARTIAL  — {best_name!r} reduces J(n=1000) significantly but not to n=8 level.\n')
        else:
            fh.write(f'  ✗ FAILED   — No intervention reduced J(n=1000) by more than 30%.\n')
            fh.write(f'    U-shape may require model retraining (not just inference fix).\n')

    print(f'💾 {rpt}')

    # Print the report
    print('\n' + open(rpt).read())

    print('\n' + '=' * 60)
    print('✅ ALL DONE')
    print(f'open {OUT}/plot_ushape_J_vs_nsteps.png')
    print(f'open {OUT}/plot_ushape_cap_tau_heatmap.png')
    print(f'cat {OUT}/ushape_diag_report.txt')


if __name__ == '__main__':
    main()
