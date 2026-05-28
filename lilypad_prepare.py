"""
Convert our predicted thetas from Modal volume to LilyPad angular velocity files.

LilyPad accumulates rotation: body.rotate(dt × angles[t]).
We need 200 angular velocities (one per integer sim step) per sample.
Our 20 thetas are sampled every 10 sim steps within a period of 200 → use step-function:
   angles[k*10 : (k+1)*10] = (theta[k+1] - theta[k]) / 10        for k = 0..18
   angles[190:200]         = (theta[0]   - theta[19]) / 10       (wrap to make periodic)

Initial body rotation in LilyPad: customsetup does body.rotate(thetaA + theta0).
For our samples, we set thetaA=0, theta0 = our predicted theta[0] (initial angle of our window).

Output files (under /Users/baochen/diffphycon/lilypad_output/):
   angles/sim_N.txt    — single comma-separated line of 200 angular velocities
   theta0/sim_N.txt    — single float (initial angle in radians)
   meta.json           — list of (sim_index, xi, source_sample_id) for parse_forces.py
"""
import os, json, subprocess, sys
import numpy as np


# ============================================================
# Step 1: fetch thetas dict from Modal (get_all_thetas function)
# ============================================================
print("[prepare] calling Modal get_all_thetas...")
# Pipe through modal CLI — runs the function defined in jellyfish_modal.py
out = subprocess.run(
    ["modal", "run", "--quiet", "jellyfish_modal.py::get_all_thetas"],
    capture_output=True, text=True, cwd="/Users/baochen/diffphycon",
)
if out.returncode != 0:
    print("modal run failed:", out.stderr)
    sys.exit(1)
# parse modal output: the function returns a dict, modal CLI prints it as repr
# easier alternative: use Modal's Python API directly
import modal
f = modal.Function.from_name("jellyfish-gamma-sweep", "get_all_thetas")
all_thetas = f.remote()
print(f"[prepare] fetched {len(all_thetas)} result dirs from Modal volume")


# ============================================================
# Step 2: collect xi → thetas mapping
# ============================================================
XI_VALUES = [0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5]


def latest_for(xi):
    suffix = f"_xi_{xi}_steps_1000"
    matches = sorted(k for k in all_thetas if k.endswith(suffix))
    return matches[-1] if matches else None


# ============================================================
# Step 3: convert each sample to LilyPad files
# ============================================================
OUT_DIR = "/Users/baochen/diffphycon/lilypad_output"
os.makedirs(f"{OUT_DIR}/angles", exist_ok=True)
os.makedirs(f"{OUT_DIR}/theta0", exist_ok=True)

PERIOD = 200            # sim steps per period (matches LilyPad's `period`)
SAMPLING_STEP = 10      # sim steps between saved samples (matches LilyPad's sampling_step)
T = 20                  # number of sampled thetas (matches inference)


def thetas_to_angle_velocities(thetas: np.ndarray) -> np.ndarray:
    """20 absolute angles (rad) → 200 angular velocities (rad per integer sim step).
    Step-function: constant velocity within each 10-step segment."""
    assert thetas.shape == (T,)
    vel = np.zeros(PERIOD, dtype=np.float64)
    for k in range(T):
        next_k = (k + 1) % T
        v = (thetas[next_k] - thetas[k]) / SAMPLING_STEP
        vel[k * SAMPLING_STEP : (k + 1) * SAMPLING_STEP] = v
    return vel


def total_rotation_per_period(vel: np.ndarray) -> float:
    """Sanity check: total rotation over one period.  Should be ~0 (periodic)."""
    return float(vel.sum())


meta = []
sim_index = 0
for xi in XI_VALUES:
    key = latest_for(xi)
    if key is None:
        print(f"[prepare] xi={xi}  NO RESULT, skip")
        continue
    thetas_arr = np.array(all_thetas[key])     # (n_samples, T)
    n = thetas_arr.shape[0]
    print(f"[prepare] xi={xi:+.2f}  n_samples={n}  dir={key[:60]}…")
    for sample_id in range(n):
        thetas = thetas_arr[sample_id]
        vel = thetas_to_angle_velocities(thetas)
        # save angle velocities
        with open(f"{OUT_DIR}/angles/sim_{sim_index}.txt", "w") as f:
            f.write(",".join(f"{v:.6f}" for v in vel))
        # save initial theta (body starts at thetas[0])
        with open(f"{OUT_DIR}/theta0/sim_{sim_index}.txt", "w") as f:
            f.write(f"{thetas[0]:.6f}")
        meta.append({
            "sim_index": sim_index,
            "xi": float(xi),
            "gamma_1": float(1 - xi),
            "source_sample_id": sample_id,
            "source_dir": key,
            "theta0": float(thetas[0]),
            "total_rotation_per_period": total_rotation_per_period(vel),
        })
        sim_index += 1

with open(f"{OUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n[prepare] wrote {sim_index} simulations to {OUT_DIR}/{{angles,theta0}}/")
print(f"[prepare] meta.json has {len(meta)} entries")
print(f"\nNext: update LilyPad.pde to read from these files, run, then parse forces.")
