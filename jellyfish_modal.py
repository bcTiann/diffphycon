"""
Jellyfish prior-reweighting (ξ) sweep on Modal.

The actual γ knob in paper's `design_guidance="standard-alpha"` mode is
`--coeff_ratio_w` (= paper ξ in γ_k = 1 − ξ·β_{K−k}), NOT `--w_prob_exp`
which is dead code in this branch (diffusion_2d_jellyfish.py:736).

Usage:
    # 1. (one-time) upload data to Modal volume — see fetch_from_drive function
    # 2. sanity test  (~2 min, ~$0.05)
    modal run jellyfish_modal.py --mode sanity
    # 3. full sweep  (10 ξ values from paper Table 28, ~6 min, ~$2)
    modal run jellyfish_modal.py --mode sweep
    # 4. eval (surrogate-based v̄ + paper-correct R(θ) + paper Table 28 comparison)
    modal run jellyfish_modal.py --mode eval_sweep
    # 5. plot (10-panel θ trajectory grid)
    modal run jellyfish_modal.py --mode plot_sweep
"""
import modal

app = modal.App("jellyfish-gamma-sweep")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "numpy<2",
        "h5py",
        "matplotlib",
        "tqdm",
        "einops",
        "einops_exts",
        "rotary_embedding_torch",
        "accelerate",
        "ema_pytorch",
        "tensorboardX",
        "scipy",
        "gdown",
        "ipython",
    )
    .run_commands(
        "git clone https://github.com/bcTiann/diffphycon.git /opt/diffphycon",
    )
)

# Google Drive file ID for jellyfish_all.tar (set this after upload)
DRIVE_FILE_ID = "15bmtPalm0AnOxBX24rwjryGAMPJwPJCy"

volume = modal.Volume.from_name("jellyfish-data", create_if_missing=True)
REPO = "/opt/diffphycon"
DATA_MOUNT = "/data"


@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    timeout=1800,
)
def fetch_from_drive():
    """One-time: download tarball from Google Drive and extract into volume."""
    import subprocess, os, shutil

    if DRIVE_FILE_ID == "PASTE_FILE_ID_HERE":
        raise RuntimeError("Set DRIVE_FILE_ID at top of jellyfish_modal.py to your Drive file ID first.")

    tar_path = "/tmp/jellyfish_all.tar"
    print(f"[fetch] downloading from Drive id={DRIVE_FILE_ID}")
    subprocess.run(
        ["gdown", f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", "-O", tar_path],
        check=True,
    )

    print(f"[fetch] downloaded {os.path.getsize(tar_path) / 1e9:.2f} GB, extracting to {DATA_MOUNT}/")
    os.makedirs(DATA_MOUNT, exist_ok=True)
    subprocess.run(["tar", "xf", tar_path, "-C", DATA_MOUNT], check=True)
    os.remove(tar_path)

    volume.commit()
    print(f"[fetch] done. Contents of {DATA_MOUNT}:")
    for name in sorted(os.listdir(DATA_MOUNT)):
        sub = os.path.join(DATA_MOUNT, name)
        if os.path.isdir(sub):
            n_files = sum(1 for _ in os.scandir(sub))
            print(f"  {name}/ ({n_files} entries)")
        else:
            print(f"  {name}")

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={DATA_MOUNT: volume},
    timeout=7200,
)
def run_gamma(
    xi: float,
    sampling_timesteps: int = 1000,
    batch_size: int = 5,
    num_batches: int = 1,
):
    """Run inference at a given prior-reweighting strength ξ (= --coeff_ratio_w).
    Paper schedule: γ_k = 1 − ξ·β_{K−k}, so γ_1 ≈ 1 − ξ.
    `--w_prob_exp` is dead code in default `design_guidance="standard-alpha"`.
    """
    import subprocess, os, shutil, glob

    # pull latest code (in case repo updated after image build)
    subprocess.run(["git", "-C", REPO, "pull"], check=False)

    # symlink volume data into the repo's expected data dir
    repo_data = f"{REPO}/data/jellyfish"
    os.makedirs(repo_data, exist_ok=True)
    for sub in ["checkpoints", "test_data", "train_data"]:
        src = f"{DATA_MOUNT}/{sub}"
        dst = f"{repo_data}/{sub}"
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
            print(f"[modal] symlinked {src} -> {dst}")

    # ensure logs / results dirs exist (the script doesn't create parents)
    os.makedirs(f"{repo_data}/logs/inference_full", exist_ok=True)
    os.makedirs(f"{repo_data}/results/inference_full", exist_ok=True)

    # overwrite filepath.py so the inference script reads from /opt/diffphycon, not the hardcoded /Users path
    filepath_content = f'''import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
JELLYFISH_DATA_PATH = "{REPO}/data/jellyfish/"
JELLYFISH_RESULTS_PATH = "{REPO}/data/jellyfish/"
SMOKE_DATA_PATH = "/data/smoke/"
SMOKE_RESULTS_PATH = "/data/smoke/"
'''
    with open(f"{REPO}/filepath.py", "w") as f:
        f.write(filepath_content)

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        "python", "-u", f"{REPO}/inference/inference_2d_jellyfish.py",
        "--num_batches", str(num_batches),
        "--batch_size", str(batch_size),
        "--coeff_ratio_w", str(xi),
        "--sampling_timesteps", str(sampling_timesteps),
    ]
    print(f"[modal ξ={xi}] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=REPO)
    if result.returncode != 0:
        raise RuntimeError(f"ξ={xi} inference failed with exit code {result.returncode}")

    # copy result dir into the persistent volume so it survives the run
    src_root = f"{repo_data}/results/inference_full"
    latest = sorted(d for d in os.listdir(src_root) if d.startswith("2"))[-1]
    dst_name = f"{latest}_xi_{xi}_steps_{sampling_timesteps}"
    dst_dir = f"{DATA_MOUNT}/results/inference_full/{dst_name}"
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    shutil.copytree(f"{src_root}/{latest}", dst_dir, dirs_exist_ok=True)
    volume.commit()

    print(f"[modal ξ={xi} steps={sampling_timesteps}] saved to volume: /results/inference_full/{dst_name}")
    return {"xi": xi, "sampling_timesteps": sampling_timesteps, "result_dir": dst_dir}


@app.function(
    image=image,
    gpu="T4",
    volumes={DATA_MOUNT: volume},
    timeout=600,
)
def evaluate_all():
    """Compute v̄, R(θ), J for each γ — replicates inference/inference_2d_jellyfish.py::force_fn pipeline.

    Force model input is (pressure_unnormalized, bd_updater(bd_0, theta)) → 4 channels,
    NOT (vx, vy, pressure, theta). R(θ) = Σ_t (θ_{t+1} - θ_t)^2 per reg_theta() in inference code.
    """
    import os, glob, sys, pickle
    import numpy as np
    import torch
    sys.path.insert(0, REPO)
    from diffusion.diffusion_2d_jellyfish import ForceUnet, Unet

    DEVICE = torch.device("cuda")
    LAMDA = 1000  # paper ζ = 1000 (Section F.4, p.32)
    IMAGE_SIZE = 64
    T = 20

    # pressure normalization stats
    normdict = pickle.load(open(f"{DATA_MOUNT}/train_data/normalization_max_min.pkl", "rb"))
    p_max, p_min = float(normdict["p_max"]), float(normdict["p_min"])
    print(f"[eval] p_max={p_max:.3f}, p_min={p_min:.3f}")

    # force surrogate model
    force_model = ForceUnet(dim=IMAGE_SIZE, out_dim=1, dim_mults=(1, 2, 4, 8), channels=4)
    force_model.load_state_dict(torch.load(
        f"{DATA_MOUNT}/checkpoints/force_surrogate_model/force_model_epoch_9.pth", map_location=DEVICE))
    force_model.to(DEVICE).eval()

    # boundary updater
    bd_updater = Unet(dim=IMAGE_SIZE, out_dim=3, dim_mults=(1, 2, 4, 8), channels=3)
    bd_updater.load_state_dict(torch.load(
        f"{DATA_MOUNT}/checkpoints/boundary_updater/boundary_updater_epoch_9.pth", map_location=DEVICE))
    bd_updater.to(DEVICE).eval()

    def load_bd_0(sim_id):
        """Load initial-frame boundary mask+offsets for given test sim_id, pad 62→64."""
        bd_full = np.load(f"{DATA_MOUNT}/test_data/bdry_merged_mask_offsets/sim_{sim_id:06d}.npz")["a"]  # [40, 62, 62, 3]
        bd_0 = np.transpose(bd_full[0], (2, 0, 1))  # [3, 62, 62]
        bd_0_pad = np.zeros((3, 64, 64), dtype=np.float32)
        bd_0_pad[:, 1:-1, 1:-1] = bd_0
        return torch.from_numpy(bd_0_pad)

    # Discover ξ values from result dirs (auto-find all _xi_*_steps_1000)
    all_dirs = sorted(glob.glob(f"{DATA_MOUNT}/results/inference_full/*_xi_*_steps_1000"))
    xi_to_dir = {}
    for d in all_dirs:
        name = os.path.basename(d)
        # name like "2026-...-..._xi_0.3_steps_1000" — parse xi value
        try:
            xi_str = name.split("_xi_")[1].split("_steps_")[0]
            xi = float(xi_str)
            xi_to_dir[xi] = d   # latest wins (sorted by date)
        except (IndexError, ValueError):
            continue
    print(f"[eval] discovered ξ values: {sorted(xi_to_dir.keys())}")

    summary = {}
    for xi in sorted(xi_to_dir.keys(), reverse=True):
        result_dir = xi_to_dir[xi]
        print(f"\n--- ξ={xi:+.2f} (γ_1={1-xi:.2f})  dir={os.path.basename(result_dir)} ---")
        states_dir = os.path.join(result_dir, "states")
        thetas_dir = os.path.join(result_dir, "thetas")
        sample_ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(states_dir) if f.endswith(".npy"))

        v_bars, regs, Js = [], [], []
        nan_warnings = 0
        with torch.no_grad():
            for sid in sample_ids:
                states = np.load(os.path.join(states_dir, f"{sid}.npy"))   # (20, 3, 64, 64)
                thetas = np.load(os.path.join(thetas_dir, f"{sid}.npy"))   # (20,)

                # NaN protection: replace NaN in inputs with 0 (matches dataset loader behavior, data_2d.py:78)
                if np.isnan(states).any():
                    print(f"    [warn] sample {sid}: NaN in states, replacing with 0")
                    states = np.nan_to_num(states, nan=0.0)
                    nan_warnings += 1
                if np.isnan(thetas).any():
                    print(f"    [warn] sample {sid}: NaN in thetas, replacing with 0")
                    thetas = np.nan_to_num(thetas, nan=0.0)
                    nan_warnings += 1

                states_t = torch.from_numpy(states).float().to(DEVICE)
                thetas_t = torch.from_numpy(thetas).float().to(DEVICE)

                # 1. pressure: extract channel 2, unnormalize from [-1,1] back to physical scale
                pressure = states_t[:, 2, :, :]   # [20, 64, 64]
                pressure_un = (0.5 * pressure + 0.5) * (p_max - p_min) + p_min   # [20, 64, 64]

                # 2. predicted boundaries via bd_updater(bd_0, theta) — matches force_fn exactly
                bd_0 = load_bd_0(sid).to(DEVICE)                           # [3, 64, 64]
                bd_0_expand = bd_0.unsqueeze(0).expand(T, -1, -1, -1)       # [20, 3, 64, 64]
                pred_bd = bd_updater(bd_0_expand, thetas_t)                # [20, 3, 64, 64]

                # 3. concat → (pressure, mask, offset_x, offset_y) → 4 channels
                input_t = torch.cat([pressure_un.unsqueeze(1), pred_bd], dim=1)  # [20, 4, 64, 64]

                # 4. force per frame
                force = force_model(input_t)
                if force.dim() > 2:
                    force = force.view(T, -1).mean(dim=-1)
                force = force.view(-1)                                     # [20]

                # NaN in force → skip sample (don't pollute mean)
                if torch.isnan(force).any():
                    nan_idx = torch.isnan(force).nonzero(as_tuple=True)[0].cpu().tolist()
                    print(f"    [warn] sample {sid}: force NaN at frames {nan_idx}, skipping")
                    nan_warnings += 1
                    continue

                # 5. weighted average velocity (paper Eq for v̄)
                weights = torch.arange(T, 0, -1, dtype=torch.float32, device=DEVICE)
                v_bar = (force * weights).mean().item()

                # 6. smoothness regularizer R(θ) = Σ (θ_{t+1} - θ_t)^2  (matches reg_theta())
                reg = ((thetas_t[1:] - thetas_t[:-1]) ** 2).sum().item()

                J = -v_bar + LAMDA * reg
                v_bars.append(v_bar); regs.append(reg); Js.append(J)
                print(f"  sample {sid}: v̄={v_bar:+.4f}  R(θ)={reg:.5f}  J={J:+.4f}")

        if not v_bars:
            print(f"  ! ALL samples for ξ={xi} skipped (NaN issues)")
            continue
        v_bars, regs, Js = np.array(v_bars), np.array(regs), np.array(Js)
        summary[xi] = {
            "n": len(Js), "nan_warnings": nan_warnings,
            "v_bar_mean": float(v_bars.mean()), "v_bar_std": float(v_bars.std()),
            "R_mean": float(regs.mean()),       "R_std": float(regs.std()),
            "J_mean": float(Js.mean()),         "J_std": float(Js.std()),
        }
        print(f"  >> ξ={xi}: n={len(Js)}  v̄={v_bars.mean():+.3f}±{v_bars.std():.3f}  R(θ)={regs.mean():.4f}±{regs.std():.4f}  J={Js.mean():+.3f}±{Js.std():.3f}")
    return summary


@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    timeout=300,
)
def get_all_thetas():
    """Return all saved theta arrays as a dict {result_dir_name: list-of-lists}."""
    import os, glob
    import numpy as np
    out = {}
    for d in sorted(glob.glob(f"{DATA_MOUNT}/results/inference_full/*")):
        name = os.path.basename(d)
        theta_files = sorted(glob.glob(f"{d}/thetas/*.npy"))
        if theta_files:
            thetas = np.stack([np.load(f) for f in theta_files])  # (n_samples, T)
            out[name] = thetas.tolist()
    return out


@app.local_entrypoint()
def main(mode: str = "sweep"):
    if mode == "sanity":
        print(">>> Sanity test: ξ=0.3 (paper default), sampling_timesteps=8, batch=1 (cheap, ~2 min)")
        out = run_gamma.remote(
            xi=0.3, sampling_timesteps=8, num_batches=1, batch_size=1
        )
        print(f"Sanity passed: {out}")
    elif mode == "sweep":
        print(">>> Paper Table 28 sweep: ξ ∈ {0.4, 0.3, ..., -0.5} (10 values, γ_1 ∈ {0.6, ..., 1.5})")
        print(">>> 1000 timesteps, batch=5, num_batches=1 → 5 samples per ξ, 10 A100 parallel, ~6 min, ~$2")
        xi_values = [0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5]
        configs = [(xi, 1000, 5, 1) for xi in xi_values]
        results = list(run_gamma.starmap(configs))
        print(f"\nSweep complete:")
        for r in results:
            print(f"  ξ={r['xi']:+.2f} (γ_1={1-r['xi']:.1f})  ->  {r['result_dir']}")
    elif mode == "compare":
        print(">>> Compare: ξ=0.3 (paper default) at 8 vs 1000 steps (parallel, ~6 min total)")
        configs = [(0.3, 8, 1, 1), (0.3, 1000, 1, 1)]
        results = list(run_gamma.starmap(configs))
        print("\nCompare complete:")
        for r in results:
            print(f"  steps={r['sampling_timesteps']}, ξ={r['xi']}  ->  {r['result_dir']}")
    elif mode == "plot":
        import numpy as np, matplotlib.pyplot as plt, os
        print(">>> Fetching theta arrays from Modal volume...")
        all_thetas = get_all_thetas.remote()

        # find newest dirs matching steps_8 / steps_1000 for γ=1.0
        def latest(suffix):
            matches = sorted(k for k in all_thetas if k.endswith(suffix))
            if not matches:
                raise FileNotFoundError(f"no result with suffix {suffix}")
            return matches[-1]

        key_8 = latest("_gamma_1.0_steps_8")
        key_1000 = latest("_gamma_1.0_steps_1000")
        print(f"  8-step    : {key_8}")
        print(f"  1000-step : {key_1000}")

        th8 = np.array(all_thetas[key_8])
        th1000 = np.array(all_thetas[key_1000])
        th8_deg, th1000_deg = th8 * 180/np.pi, th1000 * 180/np.pi

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, title, data in [
            (axes[0], "8 steps (DDIM fast)", th8_deg),
            (axes[1], "1000 steps (DDPM paper)", th1000_deg),
        ]:
            t = np.arange(data.shape[1])
            for i, traj in enumerate(data):
                ax.plot(t, traj, marker='o', markersize=4, alpha=0.85, label=f'sample {i}')
            ax.axhspan(20.6, 49.8, color='gray', alpha=0.18, label='train range')
            ax.set_title(title, fontsize=13)
            ax.set_xlabel('timestep t')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
        axes[0].set_ylabel('θ (degrees)')
        fig.suptitle('γ = 1.0 — sampling steps comparison', fontsize=14)
        plt.tight_layout()
        out = "compare_steps.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\n=== Numeric summary ===")
        print(f"8-step    θ rad=[{th8.min():+.3f}, {th8.max():+.3f}]   deg=[{th8_deg.min():+.1f}, {th8_deg.max():+.1f}]")
        print(f"1000-step θ rad=[{th1000.min():+.3f}, {th1000.max():+.3f}]   deg=[{th1000_deg.min():+.1f}, {th1000_deg.max():+.1f}]")
        print(f"train ref:  rad=[0.36, 0.87]   deg=[20.6, 49.8]")
        print(f"\nFigure saved: {os.path.abspath(out)}")
    elif mode == "plot_sweep":
        import numpy as np, matplotlib.pyplot as plt, os
        print(">>> Fetching theta arrays from Modal volume (all ξ)...")
        all_thetas = get_all_thetas.remote()

        # Paper Table 28 ξ values (ordered γ_1 from 0.6 to 1.5, i.e. ξ from 0.4 down to -0.5)
        XI_VALUES = [0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5]
        # paper R(w) for annotation
        PAPER_R = {0.4: 0.2581, 0.3: 0.2058, 0.2: 0.1312, 0.1: 0.0741, 0.0: 0.0746,
                   -0.1: 0.0742, -0.2: 0.0736, -0.3: 0.0725, -0.4: 0.0734, -0.5: 0.0752}

        def latest_for(xi):
            suffix = f"_xi_{xi}_steps_1000"
            matches = sorted(k for k in all_thetas if k.endswith(suffix))
            return matches[-1] if matches else None

        chosen = {xi: latest_for(xi) for xi in XI_VALUES}
        print("Chosen result dirs:")
        for xi, k in chosen.items():
            print(f"  ξ={xi:+.2f} (γ_1={1-xi:.1f})  ->  {k}")
        missing = [xi for xi, v in chosen.items() if v is None]
        if missing:
            print(f"WARN: missing 1000-step results for ξ={missing}, plotting available ones only")

        # 2x5 grid
        fig, axes = plt.subplots(2, 5, figsize=(25, 9), sharey=True, sharex=True)
        for ax, xi in zip(axes.flat, XI_VALUES):
            if chosen[xi] is None:
                ax.set_title(f'ξ={xi:+.2f}  (no data)', fontsize=11)
                ax.axis('off')
                continue
            data = np.array(all_thetas[chosen[xi]]) * 180/np.pi  # (n_samples, T) → deg
            t = np.arange(data.shape[1])
            for i, traj in enumerate(data):
                ax.plot(t, traj, marker='o', markersize=3, alpha=0.7, linewidth=1)
            # mean trajectory
            ax.plot(t, data.mean(axis=0), color='black', linewidth=2.2, label='mean')
            ax.axhspan(20.6, 49.8, color='gray', alpha=0.15, label='train range')
            gamma_1 = 1 - xi
            ax.set_title(f'ξ={xi:+.2f}  γ_1={gamma_1:.1f}  paper R={PAPER_R[xi]:.3f}', fontsize=10)
            ax.grid(alpha=0.3)
            if ax is axes.flat[0]:
                ax.legend(fontsize=8)

        for ax in axes[-1]:
            ax.set_xlabel('timestep t')
        for ax in axes[:, 0]:
            ax.set_ylabel('θ (degrees)')
        fig.suptitle('Jellyfish θ trajectory — full paper Table 28 ξ sweep (1000-step DDPM, 5 samples/ξ)', fontsize=14)
        plt.tight_layout()
        out = "sweep_xi.png"
        plt.savefig(out, dpi=130, bbox_inches='tight')

        print(f"\n=== Numeric summary ===")
        for xi in XI_VALUES:
            if chosen[xi] is None:
                continue
            data = np.array(all_thetas[chosen[xi]])
            data_deg = data * 180/np.pi
            print(f"  ξ={xi:+.2f} (γ_1={1-xi:.1f})  n={data.shape[0]}  θ rad=[{data.min():+.3f}, {data.max():+.3f}]  deg=[{data_deg.min():+.1f}, {data_deg.max():+.1f}]")
        print(f"  train ref:  rad=[0.36, 0.87]   deg=[20.6, 49.8]")
        print(f"\nFigure saved: {os.path.abspath(out)}")
    elif mode == "eval_sweep":
        print(">>> Computing v̄, R(θ), J for each ξ using force surrogate model (paper Table 28 metrics)...")
        summary = evaluate_all.remote()

        # Paper Table 28 reference (p.47): jellyfish full-observation, 1000-step DDPM, 50 samples
        # paper γ_1 = 1 - ξ
        paper = {
            0.4:  {"gamma_1": 0.6, "v_bar": 410.6,  "R": 0.2581, "J": -152.51},
            0.3:  {"gamma_1": 0.7, "v_bar": 279.87, "R": 0.2058, "J": -74.11},   # paper default
            0.2:  {"gamma_1": 0.8, "v_bar": 197.18, "R": 0.1312, "J": -65.99},
            0.1:  {"gamma_1": 0.9, "v_bar": 76.97,  "R": 0.0741, "J": -2.84},
            0.0:  {"gamma_1": 1.0, "v_bar": 95.04,  "R": 0.0746, "J": -20.47},   # no reweight = DiffPhyCon-lite
            -0.1: {"gamma_1": 1.1, "v_bar": 81.41,  "R": 0.0742, "J": -7.21},
            -0.2: {"gamma_1": 1.2, "v_bar": 84.56,  "R": 0.0736, "J": -10.93},
            -0.3: {"gamma_1": 1.3, "v_bar": 65.12,  "R": 0.0725, "J": 7.38},
            -0.4: {"gamma_1": 1.4, "v_bar": 65.02,  "R": 0.0734, "J": 8.43},
            -0.5: {"gamma_1": 1.5, "v_bar": 64.07,  "R": 0.0752, "J": 11.1},
        }
        print("\n" + "=" * 100)
        print(f"{'ξ':>6} | {'γ_1':>4} | {'metric':>8} | {'ours (5 samples)':>24} | {'paper (50)':>14}")
        print("-" * 100)
        for xi in sorted(paper.keys(), reverse=True):
            p = paper[xi]
            if xi not in summary:
                print(f"{xi:>+6.2f} | {p['gamma_1']:>4.1f} | (missing) — no result dir found for this ξ")
                continue
            s = summary[xi]
            for metric, ours_mean, ours_std, pkey in [
                ("v̄",   s["v_bar_mean"], s["v_bar_std"], "v_bar"),
                ("R(θ)", s["R_mean"],     s["R_std"],     "R"),
                ("J",    s["J_mean"],     s["J_std"],     "J"),
            ]:
                paper_val = f"{p[pkey]:+.4f}"
                print(f"{xi:>+6.2f} | {p['gamma_1']:>4.1f} | {metric:>8} | {ours_mean:+12.3f} ± {ours_std:7.3f} | {paper_val:>14}")
            print("-" * 100)
        print("\nNote: paper J uses ζ=1000.  R(θ) is pure arithmetic (no surrogate), so should match paper closely.")
        print("v̄ uses force surrogate (paper uses LilyPad simulator), absolute magnitudes will differ.")
    else:
        raise ValueError(f"unknown mode: {mode!r}  (use 'sanity', 'compare', 'sweep', 'plot', 'plot_sweep', or 'eval_sweep')")


# ============================================================
# How to upload data to Modal Volume (run these once in your Mac terminal):
# ============================================================
# modal volume create jellyfish-data
# modal volume put jellyfish-data /Users/baochen/diffphycon/data/jellyfish/checkpoints/joint_full /checkpoints/joint_full
# modal volume put jellyfish-data /Users/baochen/diffphycon/data/jellyfish/checkpoints/w_full /checkpoints/w_full
# modal volume put jellyfish-data /Users/baochen/diffphycon/data/jellyfish/checkpoints/force_surrogate_model /checkpoints/force_surrogate_model
# modal volume put jellyfish-data /Users/baochen/diffphycon/data/jellyfish/checkpoints/boundary_updater /checkpoints/boundary_updater
# modal volume put jellyfish-data /Users/baochen/diffphycon/data/jellyfish/test_data /test_data
# modal volume put jellyfish-data /Users/baochen/diffphycon/data/jellyfish/train_data /train_data
# modal volume ls jellyfish-data   # verify
