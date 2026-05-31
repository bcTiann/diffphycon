"""
compute_paper_J_from_npz.py — read paper inference npz files and compute J/E.

Paper inference saves to outputs/trajectories/inference_trajectories_<tag>.npz with:
    x_pred:  (B, 2, 11, 128) — paper predicted (u, f)
    x_gt:    (B, 11, 128)    — re-simulated u via ground-truth solver
    target:  (B, 11, 128)    — ground-truth full u trajectory

J_actual = mean over batch of mean((x_gt[-1] - target[-1])**2 over space)
J_energy = mean over batch of sum(f**2 over time × space) where f is post-masked
"""
from pathlib import Path
import numpy as np

NPZ_DIR = Path('/Users/baochen/Desktop/diffphycon_results/paper_trajectories_npz')

# Match the tags we used in sweep
TAGS = [
    'ddpm_1000',
    'ddim_1', 'ddim_4', 'ddim_8', 'ddim_16', 'ddim_50', 'ddim_100', 'ddim_1000',
]

print(f"{'method':>15} | {'n_steps':>8} | {'J_mean':>10} | {'J_std':>10} | {'E_mean':>10}")
print("-" * 70)

results = []
for tag in TAGS:
    path = NPZ_DIR / f'inference_trajectories_{tag}.npz'
    if not path.exists():
        print(f"{'  '+tag:>15} | (missing)")
        continue
    data = np.load(path)
    x_pred = data['x_pred']      # (B, 2, 11, 128)
    x_gt   = data['x_gt']        # (B, 11, 128) — solver-simulated u from f
    target = data['target']      # (B, 11, 128) — ground-truth u trajectory

    # J_actual: MSE between simulated u(T) and target u_T over space, then mean over batch
    sim_uT    = x_gt[:, -1, :]                 # (B, 128)
    target_uT = target[:, -1, :]               # (B, 128)
    J_per_sample = ((sim_uT - target_uT) ** 2).mean(axis=-1)  # (B,)
    J_mean = J_per_sample.mean()
    J_std  = J_per_sample.std()

    # Energy: f is x_pred[:, 1] over time 0..9 (10 steps). Mask central 50% (paper D.2.2).
    f = x_pred[:, 1, :10, :].copy()             # (B, 10, 128)
    n_x = f.shape[-1]
    f[:, :, n_x // 4 : 3 * n_x // 4] = 0.0
    E_per_sample = (f ** 2).sum(axis=(-1, -2))  # (B,)
    E_mean = E_per_sample.mean()

    method = 'DDPM' if 'ddpm' in tag else 'DDIM'
    n_steps = int(tag.split('_')[-1])
    print(f"{method:>15} | {n_steps:>8} | {J_mean:>10.6f} | {J_std:>10.6f} | {E_mean:>10.2f}")
    results.append({'method': method, 'n_steps': n_steps,
                    'J_mean': J_mean, 'J_std': J_std, 'E_mean': E_mean})

# Save to CSV
import csv
csv_path = Path('/Users/baochen/Desktop/diffphycon_results') / 'paper_J_recomputed.csv'
with open(csv_path, 'w') as f:
    w = csv.DictWriter(f, fieldnames=['method', 'n_steps', 'J_mean', 'J_std', 'E_mean'])
    w.writeheader()
    for r in results:
        w.writerow(r)
print()
print(f"💾 saved {csv_path}")
