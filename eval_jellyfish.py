"""
Offline jellyfish J evaluation using the force surrogate model.
Loads saved (states, thetas) from a result dir and computes
  v_bar = surrogate-predicted average velocity (= -J contribution)
  reg   = theta periodicity regularization (d(w_T, w_0))
  J     = -v_bar + lambda * reg

This is a surrogate-based J (no LilyPad), but consistent across gamma runs.
"""
import os, sys, glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from diffusion.diffusion_2d_jellyfish import ForceUnet

# ============================================================
# Auto-detect device
# ============================================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[device] {DEVICE}")

# ============================================================
# Config
# ============================================================
RESULT_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/baochen/diffphycon/data/jellyfish/results/inference_full/2026-05-20_19-02-29_coeff_ratio_w_0.3_J0.3_"
FORCE_CKPT = "/Users/baochen/diffphycon/data/jellyfish/checkpoints/force_surrogate_model/force_model_epoch_9.pth"
IMAGE_SIZE = 64
LAMDA = 100   # paper default
# ============================================================

# ============================================================
# Load force surrogate model
# ============================================================
print("Loading force surrogate model...")
force_model = ForceUnet(
    dim=IMAGE_SIZE,
    out_dim=1,
    dim_mults=(1, 2, 4, 8),
    channels=4,   # state(3) + theta(1)
)
force_model.load_state_dict(torch.load(FORCE_CKPT, map_location=DEVICE))
force_model.to(DEVICE)
force_model.eval()

# ============================================================
# Iterate over saved samples
# ============================================================
states_dir = os.path.join(RESULT_DIR, "states")
thetas_dir = os.path.join(RESULT_DIR, "thetas")
sample_ids = sorted(
    [int(os.path.splitext(f)[0]) for f in os.listdir(states_dir) if f.endswith(".npy")]
)
print(f"Found {len(sample_ids)} samples in {RESULT_DIR}")

results = []
with torch.no_grad():
    for sid in sample_ids:
        states = np.load(os.path.join(states_dir, f"{sid}.npy"))   # (20, 3, 64, 64)
        thetas = np.load(os.path.join(thetas_dir, f"{sid}.npy"))   # (20,)

        # Build input to force_model: per-frame (state, theta) → (4, 64, 64)
        # then batch over the 20 frames → (20, 4, 64, 64)
        states_t = torch.from_numpy(states).float().to(DEVICE)        # (20, 3, 64, 64)
        thetas_t = torch.from_numpy(thetas).float().to(DEVICE)        # (20,)
        theta_expand = thetas_t.view(20, 1, 1, 1).expand(20, 1, 64, 64)  # (20, 1, 64, 64)
        input_t = torch.cat([states_t, theta_expand], dim=1)          # (20, 4, 64, 64)

        # Forward
        force = force_model(input_t)                                   # (20, 1) typically; squeeze
        if force.dim() > 2:
            force = force.view(20, -1).mean(dim=-1)                    # collapse spatial
        force = force.view(-1)                                         # (20,)

        # Weighted average velocity (paper uses decreasing weights toward t=T)
        weights = torch.arange(force.shape[0], 0, -1, dtype=torch.float32, device=DEVICE)
        v_bar = (force * weights).mean().item()

        # Theta periodicity regularization d(w_T, w_0)
        reg = ((thetas_t[-1] - thetas_t[0]) ** 2).item()

        # J = -v_bar + LAMDA * reg
        J = -v_bar + LAMDA * reg

        results.append({"id": sid, "v_bar": v_bar, "reg": reg, "J": J})
        print(f"  sample {sid}: v_bar={v_bar:+.4f}  reg={reg:.5f}  J={J:+.4f}")

# Aggregate
v_bars = np.array([r["v_bar"] for r in results])
Js = np.array([r["J"] for r in results])
print(f"\n=== Summary ({len(results)} samples) ===")
print(f"  v_bar:  mean={v_bars.mean():+.4f}  std={v_bars.std():.4f}")
print(f"  J:      mean={Js.mean():+.4f}  std={Js.std():.4f}")
