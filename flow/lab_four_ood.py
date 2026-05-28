# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Burgers FM — Out-of-Distribution (OOD) Test
#
# **Workstream Q3**: our FM model was trained only on a specific data distribution.
# What happens when we feed it initial conditions it has never seen?
#
# ## Training distribution (what the model saw)
#
# From `dataset/apps/generate_burgers.py:373-384`, every training `u_0` is:
#
# > **two Gaussians** — one positive peak (center ∈ [0.2, 0.4], amp ∈ [0, 2])
# > and one negative peak (center ∈ [0.6, 0.8], amp ∈ [-2, 0]).
#
# So the model has only ever seen smooth, **2-peak, one-positive-one-negative** patterns.
#
# ## The 4 OOD probes
#
# | OOD type | What's weird about it |
# |:---|:---|
# | **3-peak** | 3 Gaussians instead of 2 |
# | **jagged** | high-frequency `sin(8πx)` — training is all smooth |
# | **same-sign** | two *positive* peaks (no negative) |
# | **step** | hard discontinuity in the middle — maximally non-smooth |
#
# For each we set the control target `u_T* = 0.5 · u_0` (a "damp to half amplitude,
# keep the shape" objective) and ask the FM model to find a control `w` that achieves it.
#
# We measure **J** (how close the PDE-simulated terminal state gets to `u_T*`) and
# **Energy** (how much control effort `||w||²` it used). For in-distribution data,
# our FM gets J ≈ 0.003-0.01. We'll see how much worse OOD gets.
#
# **No training here** — pure inference on the existing `fm_joint_ema` / `fm_prior_ema`
# checkpoints.

# %%
# --- bootstrap: ensure repo root is importable as a package root ---
import os, sys
_cwd = os.path.abspath("")
_root = _cwd if os.path.basename(_cwd) != "flow" else os.path.dirname(_cwd)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
import matplotlib.pyplot as plt

from flow.lab_four_explore import (
    load_fm, infer, sweep, build_c_from_u0_uT,
    plot_trajectory_grid, savefig,
    make_eval_batch,
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# %%
# Load the trained FM models (EMA-finetuned joint + prior)
net_joint = load_fm("joint_ema", device=device)
net_prior = load_fm("prior_ema", device=device)

# %% [markdown]
# ## 1. Define the 4 OOD initial conditions
#
# All in PHYSICAL units (amplitude range ~[-2, 2], matching training `u_0` amplitude
# so we isolate the SHAPE-OOD effect, not an amplitude-OOD effect).

# %%
NX = 128
x = np.linspace(0, 1, NX)

def make_3peak_u0():
    """3 Gaussians: centers 0.2/0.5/0.8, signs +/-/+ (training only has 2 peaks)."""
    u = np.zeros(NX)
    for c, a, w in [(0.2, 1.5, 0.06), (0.5, -1.5, 0.05), (0.8, 1.5, 0.06)]:
        u += a * np.exp(-((x - c) ** 2) / (2 * w ** 2))
    return u.astype(np.float32)

def make_jagged_u0():
    """High-frequency sinusoid — training data is all smooth Gaussians."""
    return (1.5 * np.sin(8 * np.pi * x)).astype(np.float32)

def make_same_sign_u0():
    """Two POSITIVE peaks (training always has one +, one -)."""
    u = np.zeros(NX)
    for c, a, w in [(0.3, 1.5, 0.06), (0.7, 1.5, 0.06)]:
        u += a * np.exp(-((x - c) ** 2) / (2 * w ** 2))
    return u.astype(np.float32)

def make_step_u0():
    """Hard step discontinuity: +1.5 left half, -1.5 right half."""
    u = np.full(NX, 1.5, dtype=np.float32)
    u[NX // 2:] = -1.5
    return u

OOD_TYPES = {
    "3peak":     make_3peak_u0,
    "jagged":    make_jagged_u0,
    "same_sign": make_same_sign_u0,
    "step":      make_step_u0,
}

# %% [markdown]
# ### Visualize the 4 OOD `u_0` next to a real training `u_0`

# %%
# grab one real HELD-OUT TEST u_0 for visual comparison (NOT training data!)
c_ref_batch = make_eval_batch(n=8, split="test", device=device)   # first 8 test samples of 1e4
u0_train = (c_ref_batch[0, 0].cpu().numpy()) * 10.0   # un-normalize (it's an in-dist test sample)

fig, axes = plt.subplots(1, 5, figsize=(20, 3))
axes[0].plot(x, u0_train, "g-", lw=2)
axes[0].set_title("training u_0\n(2-peak +/-)", fontsize=10)
for ax, (name, fn) in zip(axes[1:], OOD_TYPES.items()):
    ax.plot(x, fn(), "r-", lw=2)
    ax.set_title(f"OOD: {name}", fontsize=10)
for ax in axes:
    ax.axhline(0, color="gray", lw=0.5); ax.grid(alpha=0.3); ax.set_ylim(-2.2, 2.2)
plt.tight_layout()
savefig(fig, "ood", "ood_u0_shapes")
plt.show()

# %% [markdown]
# ## 2. In-distribution baseline (reference anchor)
#
# Run inference on several real training `u_0` and average — gives a stable
# "this is what good looks like" number to compare the OOD results against.
# (A single sample can be unusually easy/hard, so we average 8.)

# %%
# c_ref_batch is the 8 held-out test samples built above (make_eval_batch)
indist_Js, indist_Es = [], []
for i in range(8):
    r = infer(net_joint, c_ref_batch[i:i+1], net_prior=net_prior, gamma=1.0, n_steps=100, seed=42)
    indist_Js.append(r["J"]); indist_Es.append(r["E"])
J_indist = float(np.mean(indist_Js))
E_indist = float(np.mean(indist_Es))
result_indist = {"J": J_indist, "E": E_indist}
print(f"in-distribution (γ=1, avg of 8):  J={J_indist:.5f} ± {np.std(indist_Js):.5f}   E={E_indist:.1f}")

# %% [markdown]
# ## 3. OOD inference — 4 types × {γ=1.0, γ=2.5}
#
# For each OOD `u_0`, build `c = (u_0, u_T*=0.5·u_0)` and run the FM sampler.
# We try both γ=1 (pure joint) and γ=2.5 (prior-reweighted, more aggressive control).

# %%
TARGET_SCALE = 0.5   # u_T* = TARGET_SCALE * u_0

results = []
titles = []
for name, fn in OOD_TYPES.items():
    u0 = fn()
    uT = TARGET_SCALE * u0
    c = build_c_from_u0_uT(u0, uT, device=device)
    for gamma in [1.0, 2.5]:
        r = infer(net_joint, c, net_prior=net_prior, gamma=gamma, n_steps=100, seed=42)
        r["ood_type"] = name
        results.append(r)
        titles.append(f"{name}  γ={gamma}")
        print(f"  {name:10s} γ={gamma}:  J={r['J']:.5f}  E={r['E']:.1f}")

# %% [markdown]
# ### Trajectory grid — all 8 OOD configs
#
# Each row: `pred u(t,x)` | `pred w(t,x)` | `sim u(t,x)` (real PDE solve from w) |
# terminal comparison (black=target u_T*, blue dashed=sim u(T), orange=pred u(T)).
#
# **The model signal to watch is the blue-dashed vs black gap** — how close the
# PDE-simulated terminal state gets to the target. (The orange `pred u(T)` is
# inpaint-forced to equal target, so it always overlaps — ignore it.)

# %%
fig = plot_trajectory_grid(results, titles)
savefig(fig, "ood", "ood_trajectory_grid")
plt.show()

# %% [markdown]
# ## 4. Summary table — OOD vs in-distribution

# %%
configs = []
for name, fn in OOD_TYPES.items():
    u0 = fn(); uT = TARGET_SCALE * u0
    c = build_c_from_u0_uT(u0, uT, device=device)
    for gamma in [1.0, 2.5]:
        configs.append({"ood_type": name, "gamma": gamma, "c": c})

df = sweep(
    configs,
    infer_fn=lambda ood_type, gamma, c: {**infer(net_joint, c, net_prior=net_prior, gamma=gamma, n_steps=100, seed=42), "ood_type": ood_type},
    verbose=False,
)
# add in-distribution reference row
import pandas as pd
ref_row = pd.DataFrame([{"ood_type": "IN-DIST", "gamma": 1.0,
                         "J": result_indist["J"], "E": result_indist["E"]}])
df_display = pd.concat([ref_row, df[["ood_type", "gamma", "J", "E"]]], ignore_index=True)
df_display["J / indist"] = (df_display["J"] / result_indist["J"]).round(1)
print(df_display.to_string(index=False))
df_display.to_csv("flow/results/ood/ood_summary.csv", index=False)
print("\n💾 saved flow/results/ood/ood_summary.csv")

# %% [markdown]
# ## 5. Interpretation
#
# (Fill this in after running — what to look for:)
#
# - **J / indist ratio**: how many times worse than in-distribution. < 5× = model
#   generalizes surprisingly well; > 50× = total failure.
# - **Which OOD types survive?** Expectation:
#   - `3peak` / `same_sign` — model may still partially control (smooth, just wrong peak count)
#   - `jagged` / `step` — model likely fails (high-freq / discontinuity never seen)
# - **Does γ=2.5 help or hurt on OOD?** Prior reweighting pushes toward training-typical
#   controls — on OOD that might either help (regularize) or hurt (force wrong pattern).
# - **Visual**: does the model "smooth away" the weird parts of u_0, or does the
#   simulated terminal state diverge wildly?
