#!/usr/bin/env python3
"""Plot the best tested guided Jellyfish FM setting for the README."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "jellyfish"
FIGURE = ROOT / "figures" / "jellyfish_fm_gamma400_lambda400_vs_ddpm_best_first10.png"

FM_DIR = RESULTS / (
    "fm_refine8_relative_joint120k_prior70k_n100_seed42_"
    "lam400_gamma400_convcondot_schedpaper_beta_"
    "betacontinuous_taumin0.001_bdrelative_sim0-9"
)
DDPM_DIR = RESULTS / (
    "ddpm_diffphycon_best_reported_relativebd_lambda0.5_xi0.3_"
    "gamma0.7_zeta1000_seed42_batch4_sim0-49"
)

SIM_IDS = tuple(range(10))
FRAMES = np.arange(20)


def load_fm_theta(sim_id: int) -> np.ndarray:
    path = FM_DIR / f"{sim_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path), dtype=np.float64)


def load_gt_theta(sim_id: int) -> np.ndarray:
    path = FM_DIR / f"{sim_id}_joint.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as record:
        return np.asarray(record["gt_theta"], dtype=np.float64)


def load_ddpm_theta(sim_id: int) -> np.ndarray:
    path = DDPM_DIR / "thetas" / f"{sim_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path), dtype=np.float64)


def degrees(theta: np.ndarray) -> np.ndarray:
    return np.rad2deg(theta)


def main() -> None:
    gt = np.stack([load_gt_theta(sim_id) for sim_id in SIM_IDS])
    ddpm = np.stack([load_ddpm_theta(sim_id) for sim_id in SIM_IDS])
    fm = np.stack([load_fm_theta(sim_id) for sim_id in SIM_IDS])

    free = np.s_[:, 1:19]
    rmse_deg = float(np.sqrt(np.mean(degrees(fm[free] - ddpm[free]) ** 2)))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 6.3), sharex=True, sharey=True)

    for sim_id, ax in zip(SIM_IDS, axes.flat):
        ax.plot(
            FRAMES,
            degrees(gt[sim_id]),
            color="black",
            linestyle=":",
            linewidth=2.2,
            label="GT",
        )
        ax.plot(
            FRAMES,
            degrees(ddpm[sim_id]),
            color="#F28E2B",
            linewidth=2.2,
            label="DDPM best-reported",
        )
        ax.plot(
            FRAMES,
            degrees(fm[sim_id]),
            color="#7B2CBF",
            linewidth=2.4,
            marker="o",
            markersize=3.2,
            markevery=2,
            label=r"FM ($\gamma=400$, $\lambda=400$)",
        )
        ax.set_title(f"test sim {sim_id}")
        ax.set_xlim(0, 19)
        ax.set_ylim(0, 60)
        ax.set_xticks((0, 5, 10, 15, 19))
        ax.grid(alpha=0.22)
        if sim_id >= 5:
            ax.set_xlabel("frame")
        if sim_id % 5 == 0:
            ax.set_ylabel("theta (degree)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
    )
    fig.suptitle(
        "Guided Flow Matching and DDPM best-reported opening-angle trajectories\n"
        f"test sims 0–9 · free-frame RMSE = {rmse_deg:.2f}°",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.92))

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {FIGURE}")
    print(f"free-frame RMSE versus DDPM best-reported: {rmse_deg:.6f} degree")


if __name__ == "__main__":
    main()
