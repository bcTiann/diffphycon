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
# # Burgers FM — J-gradient Guidance
#
# **Workstream Q1**: DiffPhyCon uses TWO separate inference-time mechanisms.
# We already have the first; this notebook adds the second.
#
# ## Two mechanisms
#
# | | Mechanism | Knob | Status |
# |:---|:---|:---|:---|
# | **(A)** | **Prior reweighting** $p_\gamma \propto p(w\mid c)^{\gamma-1}\,p(u,w\mid c)$ | `gamma` | ✅ already in `ReweightedVectorField` |
# | **(B)** | **J-gradient guidance** $p \propto p(u,w\mid c)\,e^{-\lambda J(u,w)}$ | `wfs` (energy), `wu` (state) | ⬅️ this notebook |
#
# ## Is our FM-vs-DDPM comparison still fair?
#
# **Yes.** Audit found that the paper's reported DDPM baseline numbers
# (`notes_baseline_summary.md`) were produced with `--wfs 0` — i.e. the baseline
# uses prior reweighting *only*, no J-gradient. Our FM also used prior-only. So the
# "FM beats DDPM 3-8×" result is apples-to-apples.
#
# **This notebook is an enhancement**, not a bug-fix: the paper's code *supports*
# J-gradient (`get_loss_fn_2dconv`, `--wfs`, `--J_scheduler`), and we're adding the
# equivalent to FM to see if it pushes J even lower.
#
# ## The FM math (how J-gradient enters the velocity field)
#
# For the CondOT path $\alpha_\tau=\tau,\ \beta_\tau=1-\tau$, given the network's
# velocity $v(x_\tau,\tau)\approx z-\varepsilon$, the **Tweedie** denoised estimate is
#
# $$\hat z = x_\tau + (1-\tau)\,v(x_\tau,\tau).$$
#
# We then perturb the velocity toward lower $J$:
#
# $$v_{\text{total}} = v(x_\tau,\tau) \;-\; \nabla_{x_\tau} J(\hat z)\cdot \text{sched}(\tau),$$
#
# where $J = w_u\cdot\|u_{\text{endpoint}}-u^*\|^2 + w_{fs}\cdot\|w\|^2$ is the paper's
# `ddpm_guidance_loss`, and $\text{sched}(\tau)$ ramps guidance up as $\tau\to1$ (clean end).
#
# This is implemented in `JGradEulerSampler` (in `lab_four_explore.py`). Setting
# `wfs=wu=0` makes it identical to the plain `BurgersEulerSampler` (sanity-checked below).

# %%
# --- bootstrap: ensure repo root is importable ---
import os, sys
_cwd = os.path.abspath("")
_root = _cwd if os.path.basename(_cwd) != "flow" else os.path.dirname(_cwd)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
import matplotlib.pyplot as plt

from flow.lab_four_explore import (
    load_fm, infer, sweep, savefig,
    make_eval_batch,
    JGradEulerSampler, BurgersEulerSampler,
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# %%
net_joint = load_fm("joint_ema", device=device)
net_prior = load_fm("prior_ema", device=device)

# one in-distribution batch to test on (8 samples for stable averages)
c_batch = make_eval_batch(n=8, split="test", device=device)   # held-out test, first 8 of 1e4
print("held-out test batch c:", tuple(c_batch.shape))

# %% [markdown]
# ## 1. Sanity: `wfs=0` must equal the plain sampler exactly
#
# If `JGradEulerSampler` with no guidance doesn't reproduce `BurgersEulerSampler`,
# something is wrong with the new code path.

# %%
# We must FORCE the JGradEulerSampler code path even at wfs=0 — otherwise infer()
# routes wfs=0 to the plain BurgersEulerSampler and we'd never exercise the new class.
# So we build both samplers explicitly and compare them on the SAME noise (seed=0).
from flow.lab_four_explore import BurgersEulerSampler, JGradEulerSampler, compute_J_and_energy

torch.manual_seed(0)
x_plain = BurgersEulerSampler(net_joint, n_steps=100).sample(c_batch)
torch.manual_seed(0)
x_jg0   = JGradEulerSampler(net_joint, n_steps=100, wfs=0.0).sample(c_batch)   # JGrad, guidance OFF

J_plain, E_plain = compute_J_and_energy(x_plain, c_batch)
J_jg0,   E_jg0   = compute_J_and_energy(x_jg0,   c_batch)
print(f"BurgersEulerSampler        J={J_plain:.6f}  E={E_plain:.2f}")
print(f"JGradEulerSampler(wfs=0)   J={J_jg0:.6f}  E={E_jg0:.2f}")

# the two trajectories should be bit-for-bit identical (wfs=0 → no guidance term)
max_abs_diff = (x_plain - x_jg0).abs().max().item()
assert max_abs_diff < 1e-5, f"FAIL: JGrad(wfs=0) should match plain, max|Δx|={max_abs_diff}"
print(f"✅ JGradEulerSampler(wfs=0) ≡ BurgersEulerSampler  (max|Δx| = {max_abs_diff:.2e})")

# %% [markdown]
# ## 2. Energy guidance sweep: `wfs` ∈ {0, 0.5, 1, 2, 5}
#
# `wfs` weights the **control energy** $\|w\|^2$ term in the guidance loss. Higher
# `wfs` pushes toward lower-energy (gentler) control. We sweep it at γ=1 and γ=2.5.
#
# **What actually happens (see §5):** as `wfs`↑, **Energy drops monotonically**
# (it's literally a penalty on `‖w‖²`) but **J rises** — gentler control drifts
# further from `u_T*`. So `wfs` is an *energy↔accuracy trade-off knob*, NOT a way to
# lower J. (This is on in-distribution data where the model's control is already
# near-optimal.)

# %%
WFS_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0]
configs = []
for gamma in [1.0, 2.5]:
    for wfs in WFS_VALUES:
        configs.append({"gamma": gamma, "wfs": wfs})

df_wfs = sweep(
    configs,
    infer_fn=lambda gamma, wfs: infer(
        net_joint, c_batch, net_prior=net_prior,
        gamma=gamma, wfs=wfs, wu=0.0, n_steps=100, seed=0,
        j_scheduler_name="cosine",
    ),
)
print()
print(df_wfs.to_string(index=False))

# %% [markdown]
# ### Plot: J and Energy vs `wfs`

# %%
fig, (axJ, axE) = plt.subplots(1, 2, figsize=(13, 4.5))
for gamma in [1.0, 2.5]:
    sub = df_wfs[df_wfs["gamma"] == gamma]
    axJ.plot(sub["wfs"], sub["J"], "o-", label=f"γ={gamma}")
    axE.plot(sub["wfs"], sub["E"], "s-", label=f"γ={gamma}")
axJ.set_xlabel("wfs (energy guidance weight)"); axJ.set_ylabel("J (terminal MSE)")
axJ.set_title("J vs wfs"); axJ.legend(); axJ.grid(alpha=0.3)
axE.set_xlabel("wfs (energy guidance weight)"); axE.set_ylabel("Energy ||w||²")
axE.set_title("Energy vs wfs"); axE.legend(); axE.grid(alpha=0.3)
plt.tight_layout()
savefig(fig, "jgrad", "wfs_sweep")
plt.show()

# %% [markdown]
# ## 3. State guidance sweep: `wu` ∈ {0, 0.5, 1, 2}
#
# `wu` weights the **state-match** term $\ (u_0-u_0^{gt})^2 + (u_T-u_T^*)^2\ $ in the
# guidance loss — guiding the model's u-channel endpoints toward the boundary values.
#
# **Hypothesis (before running):** `wu` should *lower J*, since J is a terminal-match
# metric. **Spoiler — it doesn't, and the result is instructive:** the inpainting trick
# already forces the model's u-endpoints to equal `u_0`/`u_T*` at every step, so
# `loss_u ≈ 0`, its gradient ≈ 0, and `wu` has **no effect** (J is flat across all `wu`).
# See §5 for why. We keep the sweep to demonstrate this concretely.

# %%
WU_VALUES = [0.0, 0.5, 1.0, 2.0]
configs_wu = [{"gamma": 1.0, "wu": wu} for wu in WU_VALUES]
df_wu = sweep(
    configs_wu,
    infer_fn=lambda gamma, wu: infer(
        net_joint, c_batch, net_prior=net_prior,
        gamma=gamma, wfs=0.0, wu=wu, n_steps=100, seed=0,
        j_scheduler_name="cosine",
    ),
)
print()
print(df_wu.to_string(index=False))

# %%
fig, (axJ, axE) = plt.subplots(1, 2, figsize=(13, 4.5))
axJ.plot(df_wu["wu"], df_wu["J"], "o-", color="C2")
axJ.set_xlabel("wu (state guidance weight)"); axJ.set_ylabel("J (terminal MSE)")
axJ.set_title("J vs wu  (γ=1)"); axJ.grid(alpha=0.3)
axE.plot(df_wu["wu"], df_wu["E"], "s-", color="C3")
axE.set_xlabel("wu (state guidance weight)"); axE.set_ylabel("Energy ||w||²")
axE.set_title("Energy vs wu  (γ=1)"); axE.grid(alpha=0.3)
plt.tight_layout()
savefig(fig, "jgrad", "wu_sweep")
plt.show()

# %% [markdown]
# ## 4. Summary — best J-grad config vs no-guidance baseline

# %%
import pandas as pd
baseline_J = df_wfs[(df_wfs.gamma == 1.0) & (df_wfs.wfs == 0.0)]["J"].iloc[0]
best_wfs = df_wfs.loc[df_wfs["J"].idxmin()]
best_wu = df_wu.loc[df_wu["J"].idxmin()]
print(f"no-guidance baseline (γ=1, wfs=0):     J={baseline_J:.5f}")
print(f"best energy-guided (wfs sweep):        J={best_wfs['J']:.5f}  @ γ={best_wfs['gamma']}, wfs={best_wfs['wfs']}  ({(1-best_wfs['J']/baseline_J)*100:+.0f}%)")
print(f"best state-guided  (wu sweep):         J={best_wu['J']:.5f}  @ wu={best_wu['wu']}  ({(1-best_wu['J']/baseline_J)*100:+.0f}%)")
df_wfs.to_csv("flow/results/jgrad/wfs_sweep.csv", index=False)
df_wu.to_csv("flow/results/jgrad/wu_sweep.csv", index=False)
print("\n💾 saved flow/results/jgrad/{wfs_sweep,wu_sweep}.csv")

# %% [markdown]
# ## 5. Interpretation (findings from the run)
#
# Two clear, somewhat surprising results:
#
# ### `wfs` (energy guidance) = an accuracy↔effort **tradeoff knob**, not a free J win
#
# As `wfs` increases (0 → 5):
# - **Energy drops monotonically** (1570 → 1236) — exactly what penalizing `‖w‖²` does.
# - **J rises** (0.0004 → 0.005) — gentler control means the simulated terminal drifts
#   further from `u_T*`.
#
# So energy guidance lets you **dial down control effort at the cost of accuracy** —
# a Pareto knob. It does **not** lower J on in-distribution data (the model's
# control is already near-optimal; using less of it can only hurt the match).
#
# ### `wu` (state guidance) is a **no-op** here — and that's expected
#
# J is identical (0.000389) for every `wu`. Reason: the **inpainting** trick forces
# `x[:,0,0,:]=u_0` and `x[:,0,T,:]=u_T*` at every Euler step, so the model's own
# u-channel endpoints *already equal* the target. The guidance loss term
# `(u_endpoint − u*)²` is therefore ≈ 0, its gradient ≈ 0, and `wu` changes nothing.
#
# The "real" J we report is the **PDE-simulated** terminal (running predicted `w`
# through the Burgers solver) — and we deliberately don't differentiate through that
# 10 000-step solver. So 2D-conv-proxy state guidance can't touch it.
#
# ### Takeaway
#
# For **this** problem (terminal match already enforced by inpainting), J-gradient
# guidance is best understood as an **energy regularizer** (`wfs`), not a J-reducer.
# Our headline result stands: the J win comes from prior reweighting (γ) + the FM
# model itself, and is **fair** vs the DDPM baseline (which also ran `wfs=0`).
#
# The mechanism would matter more in settings *without* terminal inpainting (e.g.
# pure objective-driven control with no `u_T*`), which is future work.
