#!/usr/bin/env python3
"""Plot one Jellyfish simulation across all 20 trajectory frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CHANNELS = (
    (0, "vx", "coolwarm"),
    (1, "vy", "coolwarm"),
    (2, "pressure", "viridis"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fm-joint",
        type=Path,
        required=True,
        help="FM *_joint.npz file containing prediction and ground truth",
    )
    parser.add_argument(
        "--ddpm-state",
        type=Path,
        required=True,
        help="Plain-DDPM state .npy file",
    )
    parser.add_argument("--sim-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def color_limits(values: np.ndarray, channel_name: str) -> tuple[float, float]:
    if channel_name == "pressure":
        return tuple(np.percentile(values, (1, 99)))
    limit = float(np.percentile(np.abs(values), 99))
    return -limit, limit


def plot_channel(
    methods: list[tuple[str, np.ndarray]],
    channel: int,
    channel_name: str,
    cmap: str,
    sim_id: int,
    output_path: Path,
) -> None:
    frames_per_block = 5
    blocks = 4
    method_count = len(methods)
    all_values = np.concatenate(
        [state[:, channel].reshape(-1) for _, state in methods]
    )
    vmin, vmax = color_limits(all_values, channel_name)

    fig, axes = plt.subplots(
        blocks * method_count,
        frames_per_block,
        figsize=(13.5, 15.0),
        squeeze=False,
    )
    image = None
    for block in range(blocks):
        start = block * frames_per_block
        for method_index, (method_name, state) in enumerate(methods):
            row = block * method_count + method_index
            for column, frame in enumerate(range(start, start + frames_per_block)):
                ax = axes[row, column]
                image = ax.imshow(
                    state[frame, channel],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if method_index == 0:
                    ax.set_title(f"frame {frame}", fontsize=11)
                if column == 0:
                    ax.set_ylabel(
                        method_name,
                        rotation=0,
                        ha="right",
                        va="center",
                        fontsize=10,
                    )

        if block < blocks - 1:
            separator_row = (block + 1) * method_count - 1
            for ax in axes[separator_row]:
                ax.spines["bottom"].set_linewidth(2.0)
                ax.spines["bottom"].set_color("#666666")

    fig.suptitle(
        f"Normalized {channel_name} trajectory — test sim {sim_id}",
        fontsize=16,
        y=0.995,
    )
    assert image is not None
    colorbar_axis = fig.add_axes((0.925, 0.08, 0.012, 0.84))
    fig.colorbar(image, cax=colorbar_axis, label="normalized value")
    fig.subplots_adjust(
        left=0.13,
        right=0.90,
        top=0.965,
        bottom=0.015,
        hspace=0.22,
        wspace=0.04,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    with np.load(args.fm_joint, allow_pickle=False) as data:
        gt_state = np.asarray(data["gt_state_normalized"], dtype=np.float32)
        fm_state = np.asarray(data["pred_state_normalized"], dtype=np.float32)
    ddpm_state = np.asarray(np.load(args.ddpm_state), dtype=np.float32)

    expected_shape = (20, 3, 64, 64)
    for name, state in (
        ("GT", gt_state),
        ("FM plain", fm_state),
        ("DDPM plain", ddpm_state),
    ):
        if state.shape != expected_shape:
            raise ValueError(f"{name} has shape {state.shape}; expected {expected_shape}")

    methods = [
        ("GT", gt_state),
        ("FM plain", fm_state),
        ("DDPM plain", ddpm_state),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for channel, channel_name, cmap in CHANNELS:
        output_path = args.output_dir / (
            f"jellyfish_{channel_name}_sim{args.sim_id}_all_frames.png"
        )
        plot_channel(
            methods,
            channel,
            channel_name,
            cmap,
            args.sim_id,
            output_path,
        )
        print(output_path)


if __name__ == "__main__":
    main()
