"""
Parse LilyPad force output → compute v̄, R(θ), J → compare to paper Table 28.

LilyPad force file format (one line per saved timestep, 40 lines total since sampling_step=10 in [200, 600]):
  (x-force, y-force): [ x1, y1, 0.0 ] , [ x2, y2, 0.0 ] ,  ;;

Paper Eq for v̄ (p.32, Section F.4):
    v̄ = v_0 + (1/T) Σ_{t=1}^{T-1} (T-t) · F_t
With v_0 = 0 (jellyfish starts at rest), T=20, F_t = horizontal thrust (paper's force convention).

Horizontal thrust direction:
    The jellyfish's two wings are symmetric about y=64 (centerline). x-force is the horizontal
    component on each wing. To get net horizontal force on the jellyfish body, SUM x-forces
    of both wings. By Newton's 3rd law, the fluid feels -F_total, so the jellyfish gets pushed
    by F_total. Sign of "forward" depends on geometry: jellyfish faces -x direction (based on
    paper Fig 1 + LilyPad setup), so positive thrust = negative x-force.

We test both sign conventions and report.
"""
import re, json, os, glob, sys
import numpy as np

FORCE_DIR = "/Users/baochen/diffphycon/lilypad_output/forces"
META_PATH = "/Users/baochen/diffphycon/lilypad_output/meta.json"

# regex to extract 6 floats per line: x1, y1, 0, x2, y2, 0
LINE_RE = re.compile(
    r"\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]"
)


def parse_force_file(path: str) -> np.ndarray:
    """Read LilyPad force_sim_N.txt → return array of shape (n_timesteps, 2, 3)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "x-force" not in line:
                continue
            matches = LINE_RE.findall(line)
            if len(matches) != 2:
                continue
            wing1 = [float(matches[0][0]), float(matches[0][1]), float(matches[0][2])]
            wing2 = [float(matches[1][0]), float(matches[1][1]), float(matches[1][2])]
            rows.append([wing1, wing2])
    return np.array(rows)


def compute_v_bar(forces: np.ndarray, T: int = 20, sign: int = +1) -> float:
    """forces: (n_timesteps, 2, 3) — wings × (x,y,z). Use only first T timesteps and x-component.

    sign: +1 if positive thrust = +x; -1 if jellyfish faces -x (positive thrust = -x).
    """
    # net horizontal force = sum of x-force from both wings
    F_x = forces[:T, :, 0].sum(axis=1)   # shape (T,)
    # weights = T-t for t=0..T-1
    weights = np.arange(T, 0, -1, dtype=np.float64)   # [T, T-1, ..., 1]
    v_bar = sign * (F_x * weights).mean()
    return float(v_bar)


def compute_R_theta(thetas: np.ndarray) -> float:
    """R(θ) = Σ (θ_{t+1} - θ_t)^2."""
    return float(((thetas[1:] - thetas[:-1]) ** 2).sum())


if not os.path.exists(META_PATH):
    print(f"Missing {META_PATH}. Run lilypad_prepare.py first.")
    sys.exit(1)
with open(META_PATH) as f:
    meta = json.load(f)

# Paper Table 28 reference
PAPER = {
    0.4: {"gamma_1": 0.6, "v_bar": 410.6, "R": 0.2581, "J": -152.51},
    0.3: {"gamma_1": 0.7, "v_bar": 279.87, "R": 0.2058, "J": -74.11},
    0.2: {"gamma_1": 0.8, "v_bar": 197.18, "R": 0.1312, "J": -65.99},
    0.1: {"gamma_1": 0.9, "v_bar": 76.97,  "R": 0.0741, "J": -2.84},
    0.0: {"gamma_1": 1.0, "v_bar": 95.04,  "R": 0.0746, "J": -20.47},
    -0.1: {"gamma_1": 1.1, "v_bar": 81.41,  "R": 0.0742, "J": -7.21},
    -0.2: {"gamma_1": 1.2, "v_bar": 84.56,  "R": 0.0736, "J": -10.93},
    -0.3: {"gamma_1": 1.3, "v_bar": 65.12,  "R": 0.0725, "J": 7.38},
    -0.4: {"gamma_1": 1.4, "v_bar": 65.02,  "R": 0.0734, "J": 8.43},
    -0.5: {"gamma_1": 1.5, "v_bar": 64.07,  "R": 0.0752, "J": 11.1},
}
LAMDA = 1000

# Need thetas to compute R(θ) — re-fetch from Modal volume
print("Re-fetching thetas from Modal volume to compute R(θ)...")
import modal
get_all_thetas = modal.Function.from_name("jellyfish-gamma-sweep", "get_all_thetas")
all_thetas = get_all_thetas.remote()
# Build a (xi, source_sample_id) → thetas lookup
theta_lookup = {}
for entry in meta:
    key = entry["source_dir"]
    if key not in all_thetas:
        continue
    th_arr = np.array(all_thetas[key])
    theta_lookup[entry["sim_index"]] = th_arr[entry["source_sample_id"]]

# Aggregate by ξ
per_xi = {}
n_processed = 0
n_missing = 0
for entry in meta:
    sim_idx = entry["sim_index"]
    xi = entry["xi"]
    force_path = f"{FORCE_DIR}/sim_{sim_idx}.txt"
    if not os.path.exists(force_path):
        n_missing += 1
        continue
    forces = parse_force_file(force_path)
    if forces.shape[0] < 20:
        print(f"  sim_{sim_idx}: force file has only {forces.shape[0]} rows, expected 40 → skip")
        continue
    thetas = theta_lookup.get(sim_idx)
    if thetas is None:
        print(f"  sim_{sim_idx}: missing thetas, skip")
        continue

    v_bar_pos = compute_v_bar(forces, T=20, sign=+1)
    v_bar_neg = compute_v_bar(forces, T=20, sign=-1)
    R = compute_R_theta(thetas)
    n_processed += 1
    per_xi.setdefault(xi, {"v_bar_pos": [], "v_bar_neg": [], "R": [], "J_pos": [], "J_neg": []})
    per_xi[xi]["v_bar_pos"].append(v_bar_pos)
    per_xi[xi]["v_bar_neg"].append(v_bar_neg)
    per_xi[xi]["R"].append(R)
    per_xi[xi]["J_pos"].append(-v_bar_pos + LAMDA * R)
    per_xi[xi]["J_neg"].append(-v_bar_neg + LAMDA * R)

print(f"\nParsed {n_processed} force files ({n_missing} missing). Aggregating by ξ...\n")

print("=" * 110)
print(f"{'ξ':>6} | {'γ_1':>4} | {'metric':>8} | {'ours (n samples)':>22} | {'paper Table 28':>14}")
print("-" * 110)
for xi in sorted(per_xi.keys(), reverse=True):
    s = per_xi[xi]
    n = len(s["R"])
    p = PAPER.get(xi)
    # try both sign conventions; pick the one closer to paper
    if p:
        diff_pos = abs(np.mean(s["v_bar_pos"]) - p["v_bar"])
        diff_neg = abs(np.mean(s["v_bar_neg"]) - p["v_bar"])
        chosen_sign = "+x" if diff_pos < diff_neg else "-x"
        vbars = s["v_bar_pos"] if diff_pos < diff_neg else s["v_bar_neg"]
        Js = s["J_pos"] if diff_pos < diff_neg else s["J_neg"]
    else:
        chosen_sign = "+x"
        vbars = s["v_bar_pos"]
        Js = s["J_pos"]
    Rs = s["R"]
    pkey = lambda k: f"{p[k]:+.4f}" if p else "—"
    print(f"{xi:>+6.2f} | {p['gamma_1'] if p else '?':>4} | {'v̄ ('+chosen_sign+')':>8} | {np.mean(vbars):+11.3f} ± {np.std(vbars):8.3f} (n={n})  | {pkey('v_bar'):>14}")
    print(f"{xi:>+6.2f} | {p['gamma_1'] if p else '?':>4} | {'R(θ)':>8} | {np.mean(Rs):+11.4f} ± {np.std(Rs):8.4f}        | {pkey('R'):>14}")
    print(f"{xi:>+6.2f} | {p['gamma_1'] if p else '?':>4} | {'J':>8} | {np.mean(Js):+11.3f} ± {np.std(Js):8.3f}        | {pkey('J'):>14}")
    print("-" * 110)

print("\nNote: 'sign' indicates which x-direction = positive thrust (auto-chosen to match paper sign).")
print("Sign should be consistent across all ξ.  R(θ) is pure arithmetic, should match paper closely.")
