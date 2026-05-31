"""
analyze_jellyfish_vs_sigmoid.py — compare paper jellyfish β vs our sigmoid_flip.

Inputs:
  sigmoid_flip data:
    ~/Desktop/diffphycon_results_500fresh/sweep_500fresh/fm_n{N}/
      per_sample_J_vanilla_g{G}_n{N}.npy   for G ∈ 8 γs, N ∈ 6 steps = 48 npy
  jellyfish_beta data:
    ~/Desktop/diffphycon_results_500fresh/sweep_500fresh_jellyfish/fm_n{N}/
      per_sample_J_vanilla_g{G}_n{N}.npy   for G ∈ 7 γs, N ∈ 6 steps = 42 npy

Common γs (for direct comparison): {0.5, 0.7, 1.0, 1.5, 2.0, 3.0}

Outputs:
  plot_schedule_shapes.png        — sigmoid_flip vs jellyfish_beta curve vs τ
  plot_jellyfish_J_vs_gamma.png   — jellyfish-only J vs γ for each n_steps
  plot_compare_sigmoid_jellyfish.png — side-by-side at common γs
  jellyfish_vs_paper_l1.txt       — verdict: γ-effect under paper-faithful schedule?
"""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Inline paper β + sigmoid_flip (no import needed)
import torch

DATA_ROOT = '/Users/baochen/Desktop/diffphycon_results_500fresh'
SIGMOID_DIR = os.path.join(DATA_ROOT, 'sweep_500fresh')
JELLY_DIR   = os.path.join(DATA_ROOT, 'sweep_500fresh_jellyfish')
OUT = DATA_ROOT

N_STEPS  = [1, 4, 8, 100, 500, 1000]
SIG_GAMMAS  = [0.10, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00, 3.00]
JLY_GAMMAS  = [0.50, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
COMMON_GAMMAS = [0.50, 0.70, 1.00]   # both sweeps have these


def load_sig(n, g):
    p = os.path.join(SIGMOID_DIR, f'fm_n{n}', f'per_sample_J_vanilla_g{g:.2f}_n{n}.npy')
    return np.load(p) if os.path.exists(p) else None

def load_jly(n, g):
    p = os.path.join(JELLY_DIR, f'fm_n{n}', f'per_sample_J_vanilla_g{g:.2f}_n{n}.npy')
    return np.load(p) if os.path.exists(p) else None


def sigmoid_flip_sched(tau, slope=10.0):
    return float(1.0 - 1/(1 + np.exp(-slope*(tau - 0.5))))

def paper_sigmoid_beta(T=1000, start=-3, end=3, tau=1):
    steps = T + 1
    t = np.linspace(0, T, steps) / T
    v_start = 1/(1 + np.exp(-start/tau))
    v_end   = 1/(1 + np.exp(-end/tau))
    sig = 1/(1 + np.exp(-((t*(end-start)+start)/tau)))
    alphas_cumprod = (-sig + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

BETA_ARR = paper_sigmoid_beta(1000)
def jellyfish_sched(tau):
    idx = int(round(tau * 999))
    return float(BETA_ARR[max(0, min(999, idx))])


def main():
    # ─── Plot 1: schedule shapes ───
    plt.figure(figsize=(11, 5))
    taus = np.linspace(0, 1, 1000)
    sf_vals = [sigmoid_flip_sched(t) for t in taus]
    jb_vals = [jellyfish_sched(t) for t in taus]
    plt.plot(taus, sf_vals, lw=2, color='steelblue', label='sigmoid_flip (our original)')
    plt.plot(taus, jb_vals, lw=2, color='darkred', label='jellyfish_beta (paper)')
    # Mark FM step sample points for n=8
    for n in [8, 100, 1000]:
        ts = np.linspace(0, 1, n, endpoint=False)
        marker_sf = [sigmoid_flip_sched(t) for t in ts]
        marker_jb = [jellyfish_sched(t) for t in ts]
        plt.scatter(ts, marker_sf, s=8, color='steelblue', alpha=0.3 if n == 1000 else 0.6)
        plt.scatter(ts, marker_jb, s=8, color='darkred',   alpha=0.3 if n == 1000 else 0.6)
    plt.axvline(0.875, ls=':', color='gray', alpha=0.5, label='τ=0.875 (max sampled at n=8)')
    plt.xlabel('τ (FM time: 0=noise, 1=clean)')
    plt.ylabel('sched(τ)')
    plt.title('γ-reweighting schedule shapes:  sigmoid_flip (strong-at-noise) vs jellyfish_beta (strong-at-clean)')
    plt.legend(loc='center right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'plot_schedule_shapes.png'), dpi=110)
    plt.close()
    print(f'💾 {OUT}/plot_schedule_shapes.png')

    # ─── Plot 2: jellyfish-only J vs γ ───
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(N_STEPS)))
    for c, n in zip(colors, N_STEPS):
        means = [load_jly(n, g).mean() if load_jly(n, g) is not None else None for g in JLY_GAMMAS]
        plt.plot(JLY_GAMMAS, means, 'o-', color=c, lw=2, label=f'n={n}')
    plt.axvline(1.0, ls=':', color='gray', alpha=0.5)
    plt.xlabel('γ (paper ξ convention; γ=0.5 ≡ ξ=0.5)')
    plt.ylabel('mean J')
    plt.yscale('log')
    plt.title('JELLYFISH-β schedule:  J vs γ for each n_steps  (500-sample held-out)')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'plot_jellyfish_J_vs_gamma.png'), dpi=110)
    plt.close()
    print(f'💾 {OUT}/plot_jellyfish_J_vs_gamma.png')

    # ─── Plot 3: side-by-side sigmoid vs jellyfish at common γs ───
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, n in zip(axes.flat, N_STEPS):
        # Common γs for direct overlay
        sig_vals = [load_sig(n, g).mean() if load_sig(n, g) is not None else None for g in COMMON_GAMMAS]
        jly_vals = [load_jly(n, g).mean() if load_jly(n, g) is not None else None for g in COMMON_GAMMAS]
        x = np.arange(len(COMMON_GAMMAS))
        ax.bar(x - 0.2, sig_vals, width=0.4, color='steelblue', label='sigmoid_flip')
        ax.bar(x + 0.2, jly_vals, width=0.4, color='darkred',   label='jellyfish_beta')
        ax.set_xticks(x); ax.set_xticklabels([f'γ={g}' for g in COMMON_GAMMAS])
        ax.set_yscale('log')
        ax.set_title(f'n_steps = {n}')
        ax.grid(alpha=0.3, axis='y', which='both')
        if n == N_STEPS[0]:
            ax.legend(loc='upper left')
    fig.suptitle('sigmoid_flip vs jellyfish_beta — J at common γs (apples-to-apples)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'plot_compare_sigmoid_jellyfish.png'), dpi=110, bbox_inches='tight')
    plt.close()
    print(f'💾 {OUT}/plot_compare_sigmoid_jellyfish.png')

    # ─── Plot 4: per-sample paired comparison at n=8 (γ=1.0 vs γ=0.7) ───
    # Standard pipeline step: always check per-sample even if aggregates look identical.
    def paired_plot(schedule_data, gamma_base, gamma_test, n_step, sched_name, out_name):
        j_base = schedule_data(n_step, gamma_base)
        j_test = schedule_data(n_step, gamma_test)
        if j_base is None or j_test is None:
            print(f'  skip paired ({sched_name}): missing data'); return
        order = np.argsort(j_base)
        jb, jt = j_base[order], j_test[order]
        plt.figure(figsize=(12, 5))
        plt.plot(np.arange(len(jb)), jb, 's-', color='orange', ms=4, label=f'γ={gamma_base:.2f} (baseline)')
        plt.plot(np.arange(len(jt)), jt, '^-', color='steelblue', ms=4, label=f'γ={gamma_test:.2f}')
        plt.fill_between(np.arange(len(jt)), jt, jb, where=jt < jb,
                         color='lightgreen', alpha=0.3, label=f'γ={gamma_test:.2f} wins')
        plt.fill_between(np.arange(len(jt)), jt, jb, where=jt > jb,
                         color='lightcoral', alpha=0.2, label=f'γ={gamma_test:.2f} loses')
        plt.yscale('log')
        plt.xlabel(f'sample rank (sorted by γ={gamma_base:.2f} difficulty)')
        plt.ylabel('J per sample (log)')
        wins = int((j_test < j_base).sum())
        mean_ratio = j_test.mean() / j_base.mean()
        med_ratio = np.median(j_test) / np.median(j_base)
        plt.title(f'{sched_name}  n={n_step}:  γ={gamma_test:.2f} vs γ={gamma_base:.2f}  —  '
                  f'γ={gamma_test:.2f} wins {wins}/{len(j_base)} '
                  f'({wins/len(j_base)*100:.1f}%)  |  '
                  f'mean ratio={mean_ratio:.3f}×  median ratio={med_ratio:.3f}×')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, out_name), dpi=110)
        plt.close()
        print(f'💾 {OUT}/{out_name}  (wins={wins}, mean ratio={mean_ratio:.3f})')

    print('\n── per-sample paired plots ──')
    # Jellyfish: γ=0.7 vs γ=1.0 at FM sweet spot n=8
    paired_plot(load_jly, 1.0, 0.7, 8,    'jellyfish_beta', 'plot_paired_jly_g1_vs_g0.7_n8.png')
    paired_plot(load_jly, 1.0, 0.7, 1000, 'jellyfish_beta', 'plot_paired_jly_g1_vs_g0.7_n1000.png')
    paired_plot(load_jly, 1.0, 0.5, 8,    'jellyfish_beta', 'plot_paired_jly_g1_vs_g0.5_n8.png')
    # For reference: sigmoid_flip version at same γ
    paired_plot(load_sig, 1.0, 0.7, 8,    'sigmoid_flip',   'plot_paired_sig_g1_vs_g0.7_n8.png')

    # ─── Report: paper L.1 reproduction check ───
    rpt = os.path.join(OUT, 'jellyfish_vs_paper_l1.txt')
    with open(rpt, 'w') as fh:
        fh.write('Paper L.1 reproduction check — γ effect under jellyfish_beta schedule\n')
        fh.write('=' * 75 + '\n\n')
        fh.write('Paper claim (L.1): on 1D Burgers FOPC, γ has "near-zero effect" on J.\n')
        fh.write('Our test: with paper-faithful jellyfish_beta schedule + paper-ξ magnitude,\n')
        fh.write('          does sweeping γ ∈ {0.5..1.2} keep J within 5% of γ=1.0 baseline?\n\n')

        # First: per n_steps table
        fh.write(f"{'n_steps':<8}" + ''.join(f'γ={g:.2f}'.rjust(13) for g in JLY_GAMMAS) + '\n')
        fh.write('─' * 8 + '─' * (13 * len(JLY_GAMMAS)) + '\n')
        for n in N_STEPS:
            row = f'{n:<8}'
            for g in JLY_GAMMAS:
                j = load_jly(n, g)
                if j is None:
                    row += 'NA'.rjust(13)
                else:
                    row += f'{j.mean():.6f}'.rjust(13)
            fh.write(row + '\n')

        fh.write('\n--- Deviation from γ=1.0 baseline (per n_steps) ---\n')
        fh.write(f"{'n_steps':<8}" + ''.join(f'γ={g:.2f}'.rjust(13) for g in JLY_GAMMAS if g != 1.0) + '\n')
        fh.write('─' * 8 + '─' * (13 * (len(JLY_GAMMAS)-1)) + '\n')
        l1_ok_count = 0
        l1_total = 0
        for n in N_STEPS:
            row = f'{n:<8}'
            base = load_jly(n, 1.0).mean() if load_jly(n, 1.0) is not None else None
            for g in JLY_GAMMAS:
                if g == 1.0: continue
                j = load_jly(n, g)
                if j is None or base is None:
                    row += 'NA'.rjust(13); continue
                rel = (j.mean() - base) / base * 100
                row += f'{rel:+.1f}%'.rjust(13)
                l1_total += 1
                if abs(rel) <= 5.0:
                    l1_ok_count += 1
            fh.write(row + '\n')

        # Cross-schedule comparison
        fh.write('\n--- vs sigmoid_flip at COMMON γs (γ=0.5, 0.7, 1.0) ---\n')
        fh.write(f"{'n_steps':<8}{'γ':>6}  {'sigmoid_flip':>14}  {'jellyfish_beta':>16}  {'ratio (jly/sig)':>16}\n")
        fh.write('─' * 75 + '\n')
        for n in N_STEPS:
            for g in COMMON_GAMMAS:
                s = load_sig(n, g); j = load_jly(n, g)
                if s is None or j is None: continue
                ratio = j.mean() / s.mean()
                fh.write(f'{n:<8}{g:>6.2f}  {s.mean():>14.6f}  {j.mean():>16.6f}  {ratio:>15.3f}×\n')

        # Verdict
        fh.write('\n' + '=' * 75 + '\n')
        fh.write('VERDICT — does jellyfish_beta reproduce paper L.1 "γ near-zero effect"?\n')
        fh.write('=' * 75 + '\n\n')
        pct_within = l1_ok_count / l1_total * 100 if l1_total > 0 else 0
        fh.write(f'  Cells within ±5% of γ=1.0 baseline: {l1_ok_count}/{l1_total} ({pct_within:.0f}%)\n\n')
        if pct_within >= 80:
            fh.write('  ✓ paper L.1 CONFIRMED — under paper-faithful schedule, γ effect IS near zero.\n')
            fh.write('    Our earlier sigmoid_flip experiment that saw 9× J degradation was caused by\n')
            fh.write('    the more aggressive schedule shape + magnitude, NOT by prior model itself.\n')
        elif pct_within >= 50:
            fh.write('  ~ paper L.1 PARTIAL — some γ values within 5%, others deviate noticeably.\n')
        else:
            fh.write('  ✗ paper L.1 NOT REPRODUCED — γ still has substantial effect even with paper schedule.\n')
            fh.write('    Possible reasons: our prior model differs from paper, schedule semantics differ.\n')
    print(f'💾 {rpt}')

    # Print report to stdout
    print('\n' + open(rpt).read())

    print('\n✅ All done.')
    print(f'open {OUT}/plot_schedule_shapes.png')
    print(f'open {OUT}/plot_jellyfish_J_vs_gamma.png')
    print(f'open {OUT}/plot_compare_sigmoid_jellyfish.png')


if __name__ == '__main__':
    main()
