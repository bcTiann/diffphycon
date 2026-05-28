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
# # Burgers FM — POPC (Partial Observation, Partial Control)
#
# **Workstream Q2**: we already did FOPC (Fully Observed, Partial Control).
# This notebook does the harder POPC variant and compares against the paper's
# DDPM POPC baseline.
#
# ## FOPC vs POPC
#
# | | control `w` | state `u` observation | difficulty |
# |:---|:---|:---|:---|
# | **FOPC** (done) | zeroed in middle 50% | **fully** observed (all 128 pts) | easier |
# | **POPC** (here) | zeroed in middle 50% | observed **only at front+rear quarter** ([0,1/4]∪[3/4,1]) | harder |
#
# In POPC the middle 50% of `u_0` and `u_T*` is **unobserved** — the model must
# *generate* it, not just copy it from the boundary condition. Only the observed
# quarters are inpainted.
#
# All the POPC-specific classes live in `flow/lab_four_popc_lib.py`:
# `BurgersPOPCDataset`, `BurgersPOPCFlowTrainer`, `BurgersPOPCVectorField`,
# `BurgersPOPCEulerSampler`, `inpaint_overwrite_popc`, plus prior variants.
#
# ## Target to beat (paper DDPM POPC, `notes_baseline_summary.md §3.2`)
#
# | γ | DDPM POPC J | DDPM POPC Energy |
# |:---:|:---:|:---:|
# | 1.0 | 0.0201 | 1409 |
# | 2.5 | 0.0173 | 1340 |
#
# (For reference, FOPC was much easier: J=0.0082 at γ=1.)

# %%
# --- bootstrap ---
import os, sys
_cwd = os.path.abspath("")
_root = _cwd if os.path.basename(_cwd) != "flow" else os.path.dirname(_cwd)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
import matplotlib.pyplot as plt

from flow.lab_four_explore import (
    GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta,
    ReweightedVectorField, EMA, finetune_with_ema,
    compute_J_and_energy, infer, sweep, savefig,
    load_burgers_train, make_eval_batch, EVAL_DATASET,
)
from flow.lab_four_popc_lib import (
    BurgersPOPCDataset, BurgersPOPCVectorField, BurgersPOPCEulerSampler,
    BurgersPOPCPriorDataset,
    LiveLossTrainerPOPC, LiveLossTrainerPOPCPrior,
    observed_mask,
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# ⚠️ Train on the SAME dataset the FOPC FM used (free_u_f_1e4 = 8k train / 2k test),
# NOT the tiny 1e5 default — so POPC numbers are comparable to FOPC + DDPM POPC baseline.
DS = EVAL_DATASET   # "free_u_f_1e4_front_rear_quarter"

CKPT_DIR = os.path.join(_root, "flow", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Sanity — POPC dataset masking + short train
#
# Verify the conditioning `c` has its middle 50% zeroed (unobserved) while the
# target `z` keeps the full trajectory (model must reconstruct the middle).

# %%
ds = BurgersPOPCDataset(load_burgers_train(device=device, dataset=DS), device=device)
z, c = ds.sample(4)
obs = observed_mask(128, device=c.device)
print(f"c middle (unobserved, should be ~0):  max|c[:,:,mid]| = {c[:, :, ~obs].abs().max().item():.5f}")
print(f"z middle (target, should be nonzero): max|z[:,0,:,mid]| = {z[:, 0, :, ~obs].abs().max().item():.5f}")

# %% [markdown]
# ## 2. Train the joint POPC model
#
# Paper-aligned config: dim=64, dim_mults=(1,2,4,8), lr=1e-4, batch=64, 25000 steps.
# On M4 Pro this is ~1-2 hr. The live plot below updates every 250 steps.
#
# **For a quick test first**, set `NUM_STEPS_JOINT = 3000` (~15 min) to confirm the
# pipeline, then bump to 25000 for the real run.

# %%
NUM_STEPS_JOINT = 25000   # set to 3000 for a quick smoke test

path = GaussianConditionalProbabilityPath(LinearAlpha(), LinearBeta())
net_joint = BurgersPOPCVectorField(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
trainer = LiveLossTrainerPOPC(net_joint, path, ds, lr=1e-4)
trainer.train(num_steps=NUM_STEPS_JOINT, batch_size=64, plot_every=250)

torch.save({"state_dict": net_joint.state_dict(), "loss_history": trainer.loss_history},
           os.path.join(CKPT_DIR, "fm_joint_popc.pt"))
print("💾 saved fm_joint_popc.pt")

# %% [markdown]
# ## 3. Train the prior POPC model
#
# Paper-aligned: 6250 steps (4× fewer than joint). ~15-30 min.

# %%
NUM_STEPS_PRIOR = 6250   # set to 1000 for quick smoke test

ds_prior = BurgersPOPCPriorDataset(load_burgers_train(device=device, dataset=DS), device=device)
net_prior = BurgersPOPCVectorField(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
trainer_prior = LiveLossTrainerPOPCPrior(net_prior, path, ds_prior, lr=1e-4)
trainer_prior.train(num_steps=NUM_STEPS_PRIOR, batch_size=64, plot_every=250)

torch.save({"state_dict": net_prior.state_dict(), "loss_history": trainer_prior.loss_history},
           os.path.join(CKPT_DIR, "fm_prior_popc.pt"))
print("💾 saved fm_prior_popc.pt")

# %% [markdown]
# ## 4. EMA fine-tune (smooths high-frequency noise in samples)
#
# Continue training ~2000 steps while accumulating an EMA of the weights, then
# swap in the EMA weights. Same trick we used for FOPC.

# %%
net_joint_ema = finetune_with_ema(
    net_joint, LiveLossTrainerPOPC, ds,
    num_steps=2000, ema_decay=0.995,
    save_path_ema=os.path.join(CKPT_DIR, "fm_joint_popc_ema.pt"),
)
net_prior_ema = finetune_with_ema(
    net_prior, LiveLossTrainerPOPCPrior, ds_prior,
    num_steps=2000, ema_decay=0.995,
    save_path_ema=os.path.join(CKPT_DIR, "fm_prior_popc_ema.pt"),
)

# %% [markdown]
# ## 5. γ sweep — POPC FM vs DDPM POPC baseline
#
# Sweep γ ∈ {0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.5} on the **held-out TEST split**
# (first 8 samples of 1e4, POPC-masked) — same convention as the DDPM POPC
# baseline (`notes_baseline_summary.md §3.2`). NOT training samples.

# %%
# held-out test eval batch with POPC masking (middle 50% of c unobserved)
c_eval = make_eval_batch(n=8, dataset=DS, split="test", device=device,
                         partially_observed="front_rear_quarter")

GAMMAS = [0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.5]
configs = [{"gamma": g} for g in GAMMAS]

df_popc = sweep(
    configs,
    infer_fn=lambda gamma: infer(
        net_joint_ema, c_eval, net_prior=net_prior_ema,
        gamma=gamma, n_steps=100, seed=42,
        sampler_cls=BurgersPOPCEulerSampler,
    ),
)

# DDPM POPC baseline numbers (notes §3.2)
ddpm_popc = {1.0: (0.0201, 1409), 2.5: (0.0173, 1340)}

print("\n=== POPC: FM vs DDPM baseline ===")
print(f"{'γ':>5} {'FM J':>10} {'FM E':>9} {'DDPM J':>10} {'DDPM E':>9}")
for _, row in df_popc.iterrows():
    g = row["gamma"]
    dd = ddpm_popc.get(g, (None, None))
    dd_j = f"{dd[0]:.4f}" if dd[0] else "   —"
    dd_e = f"{dd[1]:.0f}" if dd[1] else "   —"
    print(f"{g:>5} {row['J']:>10.5f} {row['E']:>9.1f} {dd_j:>10} {dd_e:>9}")

df_popc.to_csv("flow/results/popc/popc_gamma_sweep.csv", index=False)
print("\n💾 saved flow/results/popc/popc_gamma_sweep.csv")

# %% [markdown]
# ### Plot: γ vs J — POPC FM vs DDPM POPC

# %%
fig, (axJ, axE) = plt.subplots(1, 2, figsize=(13, 4.5))
axJ.plot(df_popc["gamma"], df_popc["J"], "o-", label="FM POPC (ours)")
axJ.scatter([1.0, 2.5], [0.0201, 0.0173], color="C1", marker="s", s=80, zorder=5, label="DDPM POPC (paper)")
axJ.set_xlabel("γ"); axJ.set_ylabel("J"); axJ.set_title("POPC: J vs γ"); axJ.legend(); axJ.grid(alpha=0.3)
axE.plot(df_popc["gamma"], df_popc["E"], "o-", label="FM POPC (ours)")
axE.scatter([1.0, 2.5], [1409, 1340], color="C1", marker="s", s=80, zorder=5, label="DDPM POPC (paper)")
axE.set_xlabel("γ"); axE.set_ylabel("Energy"); axE.set_title("POPC: Energy vs γ"); axE.legend(); axE.grid(alpha=0.3)
plt.tight_layout()
savefig(fig, "popc", "popc_gamma_sweep")
plt.show()

# %% [markdown]
# ## 6. Interpretation
#
# (Fill after running:)
#
# - **FM POPC J vs DDPM POPC J** at γ=1 and γ=2.5: is FM still lower? POPC is harder,
#   so the FM advantage may be smaller than the 3-8× we saw in FOPC.
# - **Does γ reweighting still help in POPC?** (J should drop as γ increases past 1.)
# - **Visual quality**: the unobserved middle 50% of `u` is generated — does the
#   simulated terminal still match `u_T*` at the observed quarters?
#
# To visualize a POPC sample, use:
# ```python
# from flow.lab_four_explore import plot_trajectory_grid
# r = infer(net_joint_ema, c_eval, net_prior=net_prior_ema, gamma=2.5,
#           n_steps=100, seed=42, sampler_cls=BurgersPOPCEulerSampler)
# plot_trajectory_grid([r], ["POPC γ=2.5"])
# ```
