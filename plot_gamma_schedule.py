"""
Visualize γ_k schedules for our 10 ξ values.

DiffPhyCon uses sigmoid β schedule (paper L.1 + code diffusion_2d_jellyfish.py:513-526).
Formula: γ_k = 1 - ξ · β_{K-k}, k = 1..K=1000.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    """Exact replica of diffusion_2d_jellyfish.py:513-526."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).numpy()


K = 1000
betas = sigmoid_beta_schedule(K)
beta_flipped = betas[::-1]            # β_{K-k} for k=1..K (k=1 → β_{K-1}, k=K → β_0)

XI_VALUES = [0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# left panel: full schedule
k_axis = np.arange(1, K + 1)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(XI_VALUES)))
for xi, color in zip(XI_VALUES, colors):
    gamma_k = 1 - xi * beta_flipped     # length K
    gamma_1 = 1 - xi   # = gamma at k=1, paper's row label
    ax1.plot(k_axis, gamma_k, color=color, linewidth=2,
             label=f"ξ={xi:+.1f}  γ_1={gamma_1:.1f}")

ax1.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.set_xlabel("denoising step k  (k=1: most noisy → k=K: nearly clean)", fontsize=11)
ax1.set_ylabel("γ_k = 1 − ξ·β_{K−k}", fontsize=11)
ax1.set_title("γ schedule for each ξ — full denoising trajectory (K=1000)", fontsize=12)
ax1.legend(loc='center right', fontsize=9, ncol=1)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, K)

# right panel: zoom on first 100 steps (where β is non-trivial — sigmoid drops quickly toward k>~100)
zoom_K = 200
for xi, color in zip(XI_VALUES, colors):
    gamma_k = 1 - xi * beta_flipped
    ax2.plot(k_axis[:zoom_K], gamma_k[:zoom_K], color=color, linewidth=2,
             label=f"ξ={xi:+.1f}")
ax2.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.set_xlabel("denoising step k (first 200 steps zoomed)", fontsize=11)
ax2.set_ylabel("γ_k", fontsize=11)
ax2.set_title(f"Zoom: γ_k for k ∈ [1, {zoom_K}] (most reweighting happens here)", fontsize=12)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, zoom_K)

# annotate γ_1 endpoints for top curve (ξ=0.4) and bottom curve (ξ=-0.5)
for xi in [0.4, -0.5]:
    gamma_k = 1 - xi * beta_flipped
    ax2.annotate(f"γ_1={1-xi:.1f}", xy=(1, gamma_k[0]),
                 xytext=(15, gamma_k[0]), fontsize=10,
                 arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

fig.suptitle("Prior reweighting γ schedule — paper L.1 formula γ_k = 1 − ξ·β_{K−k}", fontsize=13)
plt.tight_layout()
out = "gamma_schedule.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}")

# print a few sample γ values for sanity check
print("\nSanity check (ξ=0.3, paper default):")
xi = 0.3
gamma_k = 1 - xi * beta_flipped
print(f"  γ_1    = {gamma_k[0]:.4f}   (should be ≈ 0.7)")
print(f"  γ_100  = {gamma_k[99]:.4f}")
print(f"  γ_500  = {gamma_k[499]:.4f}")
print(f"  γ_900  = {gamma_k[899]:.4f}")
print(f"  γ_1000 = {gamma_k[-1]:.4f}  (should be ≈ 1.0)")

print(f"\nβ schedule summary:")
print(f"  β_0    = {betas[0]:.6f}   (smallest, near-clean end)")
print(f"  β_500  = {betas[499]:.4f}")
print(f"  β_999  = {betas[-1]:.4f}   (largest, most-noisy end)")
