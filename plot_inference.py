"""
Visualize DiffPhyCon inference results saved by inference_1d_burgers.py.
Per sample (first N_SHOW), 4 panels:
  1. predicted u(t,x)            — what the diffusion model thinks the state looks like
  2. predicted f(t,x)            — the control force suggested by the model
  3. simulated u(t,x) from f     — PDE-solved reality check
  4. terminal-state comparison   — u_0, target u_T*, simulated u(T), predicted u(T)
"""
import numpy as np
import matplotlib.pyplot as plt

import os
NPZ_PATH = "outputs/trajectories/inference_trajectories_gamma10_POPC_10k.npz"
N_SHOW = 4  # how many samples to visualize

data = np.load(NPZ_PATH)
x_pred = data["x_pred"]   # (B, 2, 11, 128)
x_gt   = data["x_gt"]     # (B, 11, 128)
target = data["target"]   # (B, 11, 128)

u_pred = x_pred[:, 0, :11, :]   # (B, 11, 128)
f_pred = x_pred[:, 1, :10, :]   # (B, 10, 128)

print(f"u_pred shape = {u_pred.shape}")
print(f"f_pred shape = {f_pred.shape}")
print(f"x_gt   shape = {x_gt.shape}")
print(f"target shape = {target.shape}")

nx = u_pred.shape[-1]
x  = np.linspace(0, 1, nx)

# ============================================================
# Figure: N_SHOW samples (rows) × 4 panels (cols)
# ============================================================
fig, axes = plt.subplots(N_SHOW, 4, figsize=(18, 3.2 * N_SHOW))

for row in range(N_SHOW):
    # ---- panel 1: predicted u(t, x) ----
    ax = axes[row, 0]
    vmax = np.abs(u_pred[row]).max()
    im = ax.imshow(u_pred[row], aspect="auto", origin="lower",
                   extent=[0, 1, 0, 1], cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f"#{row}: predicted u(t,x)")
    ax.set_xlabel("x"); ax.set_ylabel("t")
    plt.colorbar(im, ax=ax)

    # ---- panel 2: predicted f(t, x) ----
    ax = axes[row, 1]
    fmax = np.abs(f_pred[row]).max()
    vmax = fmax if fmax > 0 else 1.0
    im = ax.imshow(f_pred[row], aspect="auto", origin="lower",
                   extent=[0, 1, 0, 1], cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_title("predicted f(t,x)  (middle 1/2 should be 0)")
    ax.set_xlabel("x"); ax.set_ylabel("t")
    ax.axvline(0.25, color="k", lw=0.6, ls="--")
    ax.axvline(0.75, color="k", lw=0.6, ls="--")
    plt.colorbar(im, ax=ax)

    # ---- panel 3: simulated u(t, x) from f ----
    ax = axes[row, 2]
    vmax = np.abs(x_gt[row]).max()
    im = ax.imshow(x_gt[row], aspect="auto", origin="lower",
                   extent=[0, 1, 0, 1], cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_title("simulated u(t,x)  (PDE-solve with f)")
    ax.set_xlabel("x"); ax.set_ylabel("t")
    plt.colorbar(im, ax=ax)

    # ---- panel 4: terminal-state comparison ----
    ax = axes[row, 3]
    ax.plot(x, target[row,  0, :], label="initial u_0",     color="gray", lw=0.8)
    ax.plot(x, target[row, 10, :], label="target u_T*",     color="k",    lw=2)
    ax.plot(x, x_gt[row,   -1, :], label="simulated u(T)",  color="C0",   ls="--")
    ax.plot(x, u_pred[row, 10, :], label="predicted u(T)",  color="C1",   ls=":")
    ax.set_title("terminal-state comparison")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

os.makedirs("outputs/figures", exist_ok=True)
# derive output PNG name from the NPZ name (e.g. inference_trajectories_gamma05.npz -> inference_viz_gamma05.png)
_base = os.path.basename(NPZ_PATH).replace("inference_trajectories", "inference_viz").replace(".npz", ".png")
_png_path = f"outputs/figures/{_base}"
plt.tight_layout()
plt.savefig(_png_path, dpi=110)
print(f"saved {_png_path}")
