"""
Plot theta trajectory comparison: 8-step vs 1000-step sampling at γ=1.0.

Usage:
    # 1. download Modal volume results to local
    modal volume get jellyfish-data /results ./modal_results
    # 2. plot
    python plot_compare_steps.py
"""
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt

RESULTS_ROOT = "./inference_full"
if not os.path.exists(RESULTS_ROOT):
    print(f"ERROR: {RESULTS_ROOT} not found. Run `modal volume get jellyfish-data /results .` first.")
    sys.exit(1)

# find newest steps_8 and steps_1000 dirs
def latest(pattern):
    matches = sorted(glob.glob(os.path.join(RESULTS_ROOT, pattern)))
    if not matches:
        raise FileNotFoundError(f"no dir matching {pattern} in {RESULTS_ROOT}")
    return matches[-1]

dir_8 = latest("*_gamma_1.0_steps_8")
dir_1000 = latest("*_gamma_1.0_steps_1000")
print(f"8-step dir   : {os.path.basename(dir_8)}")
print(f"1000-step dir: {os.path.basename(dir_1000)}")

# each dir has thetas/{0,1,...}.npy — one .npy per sample
def load_thetas(d):
    files = sorted(glob.glob(f"{d}/thetas/*.npy"))
    return np.stack([np.load(f) for f in files])   # (n_samples, 20)

th8 = load_thetas(dir_8)        # (1, 20)  or (4, 20) depending on batch_size
th1000 = load_thetas(dir_1000)

# convert rad -> deg (paper Fig 4 uses degrees)
th8_deg = th8 * 180 / np.pi
th1000_deg = th1000 * 180 / np.pi

# train data reference (from previous analysis): θ ∈ [0.36, 0.87] rad ≈ [20.6, 49.8] deg
TRAIN_LO_DEG = 20.6
TRAIN_HI_DEG = 49.8

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, title, data_deg in [
    (axes[0], "8 steps (DDIM fast)", th8_deg),
    (axes[1], "1000 steps (DDPM paper)", th1000_deg),
]:
    t = np.arange(data_deg.shape[1])
    for i, traj in enumerate(data_deg):
        ax.plot(t, traj, marker='o', markersize=4, alpha=0.85, label=f'sample {i}')
    ax.axhspan(TRAIN_LO_DEG, TRAIN_HI_DEG, color='gray', alpha=0.18, label='train range')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('timestep t')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='best')

axes[0].set_ylabel('θ (degrees)')
fig.suptitle('γ = 1.0 — sampling steps comparison', fontsize=14)
plt.tight_layout()

out = "compare_steps.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()

# also print numeric summary
print(f"\n=== Numeric summary ===")
print(f"8-step    θ range = [{th8.min():+.3f}, {th8.max():+.3f}] rad   |  [{th8_deg.min():+.1f}, {th8_deg.max():+.1f}] deg")
print(f"1000-step θ range = [{th1000.min():+.3f}, {th1000.max():+.3f}] rad   |  [{th1000_deg.min():+.1f}, {th1000_deg.max():+.1f}] deg")
print(f"\nPaper train range (ref): [0.36, 0.87] rad  |  [20.6, 49.8] deg")
print(f"\nFigure saved: {os.path.abspath(out)}")
