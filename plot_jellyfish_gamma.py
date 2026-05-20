"""
Compare jellyfish theta (wing-opening) trajectories across different gamma values.
Each row = one test sample; each column = one gamma value.
Paper Fig 4 style: gamma<1 should produce "fast-close, slow-open" patterns
                   while gamma=1 (lite) should be more monotonic / cyclic.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Auto-discover result dirs in chronological order. We'll match gamma by reading
# the /tmp/jellyfish_gamma_results.txt log written by the sweep script.
LOG_FILE = "/tmp/jellyfish_gamma_results.txt"

# parse the log
gamma_dirs = []
if os.path.exists(LOG_FILE):
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if " -> " in line:
                gamma_part, dir_part = line.split(" -> ")
                gamma = float(gamma_part.replace("gamma=", ""))
                gamma_dirs.append((gamma, dir_part))
else:
    print(f"WARNING: {LOG_FILE} not found. Falling back to manual dir list.")
    # fallback
    base = "/Users/baochen/diffphycon/data/jellyfish/results/inference_full"
    gamma_dirs = []  # user can fill in manually

print(f"Loaded {len(gamma_dirs)} gamma runs:")
for g, d in gamma_dirs:
    print(f"  gamma={g}: {os.path.basename(d)}")

# Load theta arrays
N_SAMPLES = 4
all_thetas = {}   # gamma -> list of (sample_id, theta_array)
for g, d in gamma_dirs:
    thetas_dir = os.path.join(d, "thetas")
    arrs = []
    for sid in range(N_SAMPLES):
        path = os.path.join(thetas_dir, f"{sid}.npy")
        if os.path.exists(path):
            arrs.append((sid, np.load(path)))
    all_thetas[g] = arrs

# === plot: rows = samples, cols = gamma values ===
N_GAMMA = len(gamma_dirs)
fig, axes = plt.subplots(N_SAMPLES, N_GAMMA, figsize=(3.5 * N_GAMMA, 2.5 * N_SAMPLES), sharex=True)
if N_SAMPLES == 1:
    axes = axes[None, :]
if N_GAMMA == 1:
    axes = axes[:, None]

t = np.arange(20)  # 20 frames

for col, (g, d) in enumerate(gamma_dirs):
    arrs = all_thetas[g]
    for sid, theta in arrs:
        ax = axes[sid, col]
        ax.plot(t, theta, lw=2, color="C0", marker='o', markersize=3)
        # no ylim cap — let it autoscale so we see real theta range
        ax.grid(alpha=0.3)
        # show theta=0 (fully closed) and theta=1.0 (open) reference lines
        ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
        ax.axhline(1.0, color='gray', lw=0.5, ls='--', alpha=0.5)
        if sid == 0:
            ax.set_title(f"γ={g}", fontsize=14, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"sample #{sid}\nθ (wing angle)")
        if sid == N_SAMPLES - 1:
            ax.set_xlabel("time step")

plt.tight_layout()
os.makedirs("outputs/figures", exist_ok=True)
out = "outputs/figures/jellyfish_theta_gamma_sweep.png"
plt.savefig(out, dpi=110)
print(f"\nsaved {out}")
