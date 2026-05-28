"""
Plot LilyPad Phase 4 results: ours (n=5) vs paper Table 28 (n=50).

Three panels: v̄ (forward velocity), R(θ) (control magnitude), J (objective).
"""
import numpy as np
import matplotlib.pyplot as plt

# Paper Table 28 (p.47)
paper = {
    +0.4: {"gamma_1": 0.6, "v": 410.60, "R": 0.2581, "J": -152.51},
    +0.3: {"gamma_1": 0.7, "v": 279.87, "R": 0.2058, "J":  -74.11},
    +0.2: {"gamma_1": 0.8, "v": 197.18, "R": 0.1312, "J":  -65.99},
    +0.1: {"gamma_1": 0.9, "v":  76.97, "R": 0.0741, "J":   -2.84},
    +0.0: {"gamma_1": 1.0, "v":  95.04, "R": 0.0746, "J":  -20.47},
    -0.1: {"gamma_1": 1.1, "v":  81.41, "R": 0.0742, "J":   -7.21},
    -0.2: {"gamma_1": 1.2, "v":  84.56, "R": 0.0736, "J":  -10.93},
    -0.3: {"gamma_1": 1.3, "v":  65.12, "R": 0.0725, "J":    7.38},
    -0.4: {"gamma_1": 1.4, "v":  65.02, "R": 0.0734, "J":    8.43},
    -0.5: {"gamma_1": 1.5, "v":  64.07, "R": 0.0752, "J":   11.10},
}

# Our results (Phase 4 output, n=5 each)
ours = {
    +0.4: {"v_mean": 613.341, "v_std": 321.904, "R_mean": 0.3795, "R_std": 0.1999, "J_mean": -233.882, "J_std": 127.844},
    +0.3: {"v_mean": 504.405, "v_std": 247.536, "R_mean": 0.3070, "R_std": 0.1702, "J_mean": -197.403, "J_std":  78.649},
    +0.2: {"v_mean": 354.291, "v_std": 169.925, "R_mean": 0.1998, "R_std": 0.1170, "J_mean": -154.483, "J_std":  55.087},
    +0.1: {"v_mean": 235.386, "v_std": 109.863, "R_mean": 0.1187, "R_std": 0.0708, "J_mean": -116.680, "J_std":  40.063},
    +0.0: {"v_mean": 194.592, "v_std":  93.155, "R_mean": 0.1138, "R_std": 0.0636, "J_mean":  -80.830, "J_std":  30.405},
    -0.1: {"v_mean": 171.728, "v_std": 100.009, "R_mean": 0.1133, "R_std": 0.0664, "J_mean":  -58.431, "J_std":  35.167},
    -0.2: {"v_mean": 162.360, "v_std":  82.390, "R_mean": 0.1121, "R_std": 0.0648, "J_mean":  -50.304, "J_std":  18.638},
    -0.3: {"v_mean": 157.819, "v_std":  78.051, "R_mean": 0.1101, "R_std": 0.0625, "J_mean":  -47.711, "J_std":  16.993},
    -0.4: {"v_mean": 161.804, "v_std":  84.467, "R_mean": 0.1133, "R_std": 0.0630, "J_mean":  -48.458, "J_std":  24.003},
    -0.5: {"v_mean": 159.881, "v_std":  88.876, "R_mean": 0.1136, "R_std": 0.0629, "J_mean":  -46.249, "J_std":  26.134},
}

xis = sorted(paper.keys(), reverse=True)
paper_v = [paper[x]["v"] for x in xis]
paper_R = [paper[x]["R"] for x in xis]
paper_J = [paper[x]["J"] for x in xis]

ours_v  = [ours[x]["v_mean"]  for x in xis]
ours_vs = [ours[x]["v_std"]   for x in xis]
ours_R  = [ours[x]["R_mean"]  for x in xis]
ours_Rs = [ours[x]["R_std"]   for x in xis]
ours_J  = [ours[x]["J_mean"]  for x in xis]
ours_Js = [ours[x]["J_std"]   for x in xis]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

def add_no_gamma_marker(ax, xi_val=0.0, color="red"):
    """Vertical line marking ξ=0 = no γ reweighting (γ_k ≡ 1)."""
    ax.axvline(xi_val, color=color, linestyle=':', linewidth=1.2, alpha=0.7)
    ymin, ymax = ax.get_ylim()
    ax.text(xi_val + 0.01, ymin + 0.95*(ymax - ymin), "no γ\n(ξ=0)",
            fontsize=8, color=color, ha='left', va='top')

# ---- v̄ ----
ax = axes[0]
ax.errorbar(xis, ours_v, yerr=ours_vs, fmt='o-', color='steelblue',
            label='Ours (n=5, LilyPad)', capsize=4, linewidth=2, markersize=7)
ax.plot(xis, paper_v, 's--', color='darkorange', label='Paper Table 28 (n=50)',
        linewidth=2, markersize=7)
ax.set_xlabel("ξ = --coeff_ratio_w", fontsize=11)
ax.set_ylabel("v̄  (forward velocity, +x)", fontsize=11)
ax.set_title("v̄ — jellyfish forward velocity", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.invert_xaxis()
add_no_gamma_marker(ax)

# ---- R(θ) ----
ax = axes[1]
ax.errorbar(xis, ours_R, yerr=ours_Rs, fmt='o-', color='steelblue',
            label='Ours (n=5)', capsize=4, linewidth=2, markersize=7)
ax.plot(xis, paper_R, 's--', color='darkorange', label='Paper Table 28',
        linewidth=2, markersize=7)
ax.set_xlabel("ξ = --coeff_ratio_w", fontsize=11)
ax.set_ylabel("R(θ)  (mean squared θ)", fontsize=11)
ax.set_title("R(θ) — control magnitude (pure arithmetic)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.invert_xaxis()
add_no_gamma_marker(ax)

# ---- J ----
ax = axes[2]
ax.errorbar(xis, ours_J, yerr=ours_Js, fmt='o-', color='steelblue',
            label='Ours (n=5)', capsize=4, linewidth=2, markersize=7)
ax.plot(xis, paper_J, 's--', color='darkorange', label='Paper Table 28',
        linewidth=2, markersize=7)
ax.axhline(0, color='gray', linewidth=0.6, alpha=0.5)
ax.set_xlabel("ξ = --coeff_ratio_w", fontsize=11)
ax.set_ylabel("J = R(θ) − v̄  (objective, lower=better)", fontsize=11)
ax.set_title("J — objective (control cost − thrust)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.invert_xaxis()
add_no_gamma_marker(ax)

fig.suptitle("LilyPad Phase 4 Results: ours vs paper Table 28 across 10 ξ values\n"
             "(ξ=0 marks the 'no γ reweighting' baseline; ξ>0 = aggressive guidance, ξ<0 = anti-guidance)",
             fontsize=12)
plt.tight_layout()
out = "lilypad_results.png"
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f"Saved {out}")

# Print summary numbers for the no-γ vs best-γ comparison
print("\n=== ξ=0.4 (most aggressive γ) vs ξ=0.0 (no γ) — paper ===")
print(f"  v̄:   {paper[0.4]['v']:.1f}  vs  {paper[0.0]['v']:.1f}  →  {paper[0.4]['v']/paper[0.0]['v']:.2f}x")
print(f"  R:   {paper[0.4]['R']:.3f}  vs  {paper[0.0]['R']:.3f}  →  {paper[0.4]['R']/paper[0.0]['R']:.2f}x")
print(f"  J:   {paper[0.4]['J']:.1f}  vs  {paper[0.0]['J']:.1f}  (more negative is better)")

print("\n=== ξ=0.4 vs ξ=0.0 — ours ===")
print(f"  v̄:   {ours[0.4]['v_mean']:.1f}  vs  {ours[0.0]['v_mean']:.1f}  →  {ours[0.4]['v_mean']/ours[0.0]['v_mean']:.2f}x")
print(f"  R:   {ours[0.4]['R_mean']:.3f}  vs  {ours[0.0]['R_mean']:.3f}  →  {ours[0.4]['R_mean']/ours[0.0]['R_mean']:.2f}x")
print(f"  J:   {ours[0.4]['J_mean']:.1f}  vs  {ours[0.0]['J_mean']:.1f}")
