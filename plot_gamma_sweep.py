"""
Compare gamma values side-by-side (now supports both γ<1 flatten and γ>1 sharpen).
For ONE chosen test sample, show how (u, f, simulated u, terminal-state) change with γ.

Output: outputs/figures/gamma_sweep_<TAG_SUFFIX>_sample{IDX}.png
Rows sorted from γ small (top, flatten) to γ large (bottom, sharpen),
with γ=1 (baseline) marked in the middle.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# === config ===
SAMPLE_IDX = 0

# Tag format defines whether we're plotting old (gammaXX_*) or new (betaXX_*_paper) sweeps.
# The new paper-config sweeps use file names like inference_trajectories_beta15_FOPC_paper.npz.
# Below: (tag_token, γ value, display label). Sorted γ-ascending (flatten -> sharpen).
GAMMAS = [
    ('03', 0.3),
    ('05', 0.5),
    ('07', 0.7),
    ('09', 0.9),
    ('10', 1.0),
    ('15', 1.5),
    ('25', 2.5),
]

# Choose which sweep set to plot. Two options:
#   PREFIX="beta", TAG_SUFFIX="_FOPC_paper"  -> new FOPC paper-config sweep (7 γ values)
#   PREFIX="beta", TAG_SUFFIX="_POPC_paper"  -> new POPC paper-config sweep
#   PREFIX="gamma", TAG_SUFFIX="_10k"        -> old FOPC sweep (only γ <= 1)
#   PREFIX="gamma", TAG_SUFFIX="_POPC_10k"   -> old POPC sweep
PREFIX = "beta"
TAG_SUFFIX = "_POPC_paper"
NPZ_DIR = "outputs/trajectories"
# ==============

# Load whichever γ values have files. Quietly skip missing ones.
loaded = []
data = {}
for token, gval in GAMMAS:
    path = f"{NPZ_DIR}/inference_trajectories_{PREFIX}{token}{TAG_SUFFIX}.npz"
    if os.path.exists(path):
        data[token] = np.load(path)
        loaded.append((token, gval))
    else:
        print(f"  (skipping missing: {path})")

if not loaded:
    raise FileNotFoundError(f"No matching npz files found for prefix='{PREFIX}' suffix='{TAG_SUFFIX}'")

print(f"Loaded {len(loaded)} γ values: {[g for _,g in loaded]}")

nx = data[loaded[0][0]]["x_pred"].shape[-1]
x = np.linspace(0, 1, nx)

# === figure layout: N rows (γ values) x 4 cols (panels) ===
N = len(loaded)
fig, axes = plt.subplots(N, 4, figsize=(18, 3.0 * N))
if N == 1:
    axes = axes[None, :]  # keep 2D indexing

# common u color scale across all γ for visual comparability
all_u = np.stack([data[t]["x_pred"][SAMPLE_IDX, 0, :11, :] for t, _ in loaded] +
                 [data[t]["x_gt"][SAMPLE_IDX] for t, _ in loaded])
u_abs_max = np.abs(all_u).max()

for row, (token, gval) in enumerate(loaded):
    d = data[token]
    u_pred = d["x_pred"][SAMPLE_IDX, 0, :11, :]
    f_pred = d["x_pred"][SAMPLE_IDX, 1, :10, :]
    x_gt   = d["x_gt"][SAMPLE_IDX]
    target = d["target"][SAMPLE_IDX]

    # row label on the left — highlight γ=1 (baseline) and special γ values
    label = f"γ={gval}"
    if gval == 1.0:
        label += "\n(baseline)"
    elif gval < 1.0:
        label += "\n(flatten)"
    else:
        label += "\n(sharpen)"
    axes[row, 0].annotate(
        label, xy=(-0.22, 0.5), xycoords="axes fraction",
        fontsize=12, fontweight="bold", ha="center", va="center", rotation=90,
    )

    # ---- col 1: predicted u(t,x) ----
    ax = axes[row, 0]
    im = ax.imshow(u_pred, aspect="auto", origin="lower",
                   extent=[0, 1, 0, 1], cmap="RdBu_r",
                   vmin=-u_abs_max, vmax=u_abs_max)
    if row == 0: ax.set_title("predicted u(t,x)")
    ax.set_xlabel("x"); ax.set_ylabel("t")
    plt.colorbar(im, ax=ax, fraction=0.04)

    # ---- col 2: predicted f(t,x) ----
    ax = axes[row, 1]
    fmax = np.abs(f_pred).max()
    vmax = fmax if fmax > 0 else 1.0
    im = ax.imshow(f_pred, aspect="auto", origin="lower",
                   extent=[0, 1, 0, 1], cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    if row == 0: ax.set_title("predicted f(t,x)  (per-row scale)")
    ax.set_xlabel("x"); ax.set_ylabel("t")
    ax.axvline(0.25, color="k", lw=0.4, ls="--")
    ax.axvline(0.75, color="k", lw=0.4, ls="--")
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.text(0.5, 0.95, f"|f|max={fmax:.2f}",
            transform=ax.transAxes, ha="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.7))

    # ---- col 3: simulated u(t,x) ----
    ax = axes[row, 2]
    im = ax.imshow(x_gt, aspect="auto", origin="lower",
                   extent=[0, 1, 0, 1], cmap="RdBu_r",
                   vmin=-u_abs_max, vmax=u_abs_max)
    if row == 0: ax.set_title("simulated u(t,x) from f")
    ax.set_xlabel("x"); ax.set_ylabel("t")
    plt.colorbar(im, ax=ax, fraction=0.04)

    # ---- col 4: terminal-state comparison ----
    ax = axes[row, 3]
    ax.plot(x, target[0, :],   label="u_0",         color="gray", lw=0.8)
    ax.plot(x, target[10, :],  label="target u_T*", color="k",    lw=2)
    ax.plot(x, x_gt[-1, :],    label="simulated",   color="C0",   ls="--")
    ax.plot(x, u_pred[10, :],  label="predicted",   color="C1",   ls=":")
    if row == 0: ax.set_title("terminal-state comparison")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
os.makedirs("outputs/figures", exist_ok=True)
out = f"outputs/figures/gamma_sweep{TAG_SUFFIX}_sample{SAMPLE_IDX}.png"
plt.savefig(out, dpi=110)
print(f"saved {out}")
