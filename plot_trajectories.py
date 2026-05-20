"""
Visualize generated Burgers' training data.
Produces two figures:
  burgers_viz_1.png  — one trajectory, dissected (u heatmap, f heatmap, u snapshots)
  burgers_viz_2.png  — 4 random samples side-by-side
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data/free_u_f_1e5_front_rear_quarter/burgers_train.h5"

with h5py.File(DATA_PATH, "r") as f:
    u = f["train"]["pde_11-128"][:]      # (N, 11, 128)
    w = f["train"]["pde_11-128_f"][:]    # (N, 10, 128)
    nt, nx = f["train"]["pde_11-128"].attrs["nt"], f["train"]["pde_11-128"].attrs["nx"]
    tmin, tmax = f["train"]["pde_11-128"].attrs["tmin"], f["train"]["pde_11-128"].attrs["tmax"]

print(f"u shape = {u.shape},  w shape = {w.shape}")

x = np.linspace(0, 1, nx)
t_u = np.linspace(tmin, tmax, nt)        # 11 points: 0, 0.1, ..., 1.0
t_w = np.linspace(tmin, tmax, nt - 1)    # 10 points: midpoints of segments


# ============================================================
# Figure 1: dissect ONE trajectory
# ============================================================
SAMPLE_IDX = 0

fig, axes = plt.subplots(3, 1, figsize=(8, 9),
                         gridspec_kw={"height_ratios": [2, 2, 1.4]})

# Row 1: u(t, x)
ax = axes[0]
vmax = np.abs(u[SAMPLE_IDX]).max()
im = ax.imshow(u[SAMPLE_IDX], aspect="auto", origin="lower",
               extent=[0, 1, tmin, tmax], cmap="RdBu_r",
               vmin=-vmax, vmax=vmax)
ax.set_title(f"sample #{SAMPLE_IDX}: state u(t, x)")
ax.set_xlabel("x (space)"); ax.set_ylabel("t (time)")
plt.colorbar(im, ax=ax, label="u value")

# Row 2: f(t, x)  — partial control should be visible
ax = axes[1]
vmax = np.abs(w[SAMPLE_IDX]).max()
im = ax.imshow(w[SAMPLE_IDX], aspect="auto", origin="lower",
               extent=[0, 1, tmin, tmax], cmap="RdBu_r",
               vmin=-vmax, vmax=vmax)
ax.set_title(f"sample #{SAMPLE_IDX}: control force f(t, x)   (middle 1/2 should be white)")
ax.set_xlabel("x (space)"); ax.set_ylabel("t (time)")
ax.axvline(0.25, color="k", lw=0.8, ls="--")
ax.axvline(0.75, color="k", lw=0.8, ls="--")
plt.colorbar(im, ax=ax, label="f value")

# Row 3: u(x) snapshots at three time slices
ax = axes[2]
for ti in [0, 5, 10]:
    ax.plot(x, u[SAMPLE_IDX, ti], label=f"t = {t_u[ti]:.1f}")
ax.set_title(f"sample #{SAMPLE_IDX}: u(x) at three time slices")
ax.set_xlabel("x"); ax.set_ylabel("u")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
import os
os.makedirs("outputs/figures", exist_ok=True)
plt.savefig("outputs/figures/burgers_viz_1.png", dpi=110)
print("saved outputs/figures/burgers_viz_1.png")


# ============================================================
# Figure 2: 4 random samples
# ============================================================
np.random.seed(42)
idxs = np.random.choice(u.shape[0], size=4, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(14, 6))

for col, idx in enumerate(idxs):
    # u row
    ax = axes[0, col]
    vmax = np.abs(u[idx]).max()
    ax.imshow(u[idx], aspect="auto", origin="lower",
              extent=[0, 1, tmin, tmax], cmap="RdBu_r",
              vmin=-vmax, vmax=vmax)
    ax.set_title(f"u(t,x)   sample #{idx}")
    if col == 0:
        ax.set_ylabel("t")

    # f row
    ax = axes[1, col]
    vmax = np.abs(w[idx]).max()
    ax.imshow(w[idx], aspect="auto", origin="lower",
              extent=[0, 1, tmin, tmax], cmap="RdBu_r",
              vmin=-vmax, vmax=vmax)
    ax.set_title(f"f(t,x)   sample #{idx}")
    ax.axvline(0.25, color="k", lw=0.6, ls="--")
    ax.axvline(0.75, color="k", lw=0.6, ls="--")
    ax.set_xlabel("x")
    if col == 0:
        ax.set_ylabel("t")

plt.tight_layout()
plt.savefig("outputs/figures/burgers_viz_2.png", dpi=110)
print("saved outputs/figures/burgers_viz_2.png")
