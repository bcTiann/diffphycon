"""Read tensorboard events files and plot the training loss curve."""
import glob, os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# pick the most recent run dir
runs = sorted(glob.glob("tensorboard_runs/*"))
print(f"Found {len(runs)} runs. Using most recent: {runs[-1]}")

ea = EventAccumulator(runs[-1])
ea.Reload()

print(f"Available scalars: {ea.Tags()['scalars']}")

events = ea.Scalars('loss')
steps = [e.step for e in events]
values = [e.value for e in events]

print(f"  {len(events)} loss points, step {steps[0]} -> {steps[-1]}")
print(f"  loss range: {min(values):.5f} ~ {max(values):.5f}")
print(f"  latest loss (last 5 avg): {sum(values[-5:]) / 5:.5f}")

# plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(steps, values, lw=0.5, alpha=0.6, label="raw")
# smoothed (window=50)
import numpy as np
v = np.array(values)
w = min(50, len(v) // 10)
if w > 1:
    smooth = np.convolve(v, np.ones(w) / w, mode='valid')
    axes[0].plot(steps[w-1:], smooth, lw=1.5, color="C1", label=f"smoothed (w={w})")
axes[0].set_xlabel("step"); axes[0].set_ylabel("loss")
axes[0].set_title("Training loss (linear)")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].semilogy(steps, values, lw=0.5, alpha=0.6, label="raw")
if w > 1:
    axes[1].semilogy(steps[w-1:], smooth, lw=1.5, color="C1", label=f"smoothed (w={w})")
axes[1].set_xlabel("step"); axes[1].set_ylabel("loss (log)")
axes[1].set_title("Training loss (log)")
axes[1].legend(); axes[1].grid(alpha=0.3, which='both')

os.makedirs("outputs/figures", exist_ok=True)
# derive filename from tensorboard run dir's basename so each training gets its own file
_run_name = os.path.basename(runs[-1])
_out_path = f"outputs/figures/training_loss_{_run_name}.png"
plt.tight_layout()
plt.savefig(_out_path, dpi=110)
print(f"saved {_out_path}")
# also update a "latest" symlink/copy for easy refresh in image viewers
import shutil
shutil.copyfile(_out_path, "outputs/figures/training_loss_latest.png")
