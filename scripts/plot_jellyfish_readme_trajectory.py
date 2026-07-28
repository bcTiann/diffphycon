#!/usr/bin/env python3
"""Plot one Jellyfish simulation across all 20 trajectory frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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


def plot_storyboard(
    methods: list[tuple[str, np.ndarray]],
    channel: int,
    channel_name: str,
    cmap: str,
    sim_id: int,
    output_path: Path,
) -> None:
    rows, columns = 4, 5
    all_values = np.concatenate(
        [state[:, channel].reshape(-1) for _, state in methods]
    )
    vmin, vmax = color_limits(all_values, channel_name)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15.0, 7.8),
        squeeze=False,
    )
    image = None
    for frame, ax in enumerate(axes.flat):
        composite = np.concatenate(
            [state[frame, channel] for _, state in methods], axis=1
        )
        image = ax.imshow(
            composite,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.axvline(63.5, color="white", lw=1.4, alpha=0.9)
        ax.axvline(127.5, color="white", lw=1.4, alpha=0.9)
        ax.set_title(f"frame {frame}", fontsize=11, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Normalized {channel_name} trajectory — test sim {sim_id}\n"
        "Within each frame:  GT   |   FM plain   |   DDPM plain",
        fontsize=16,
        y=0.985,
    )
    assert image is not None
    colorbar_axis = fig.add_axes((0.925, 0.11, 0.012, 0.76))
    fig.colorbar(image, cax=colorbar_axis, label="normalized value")
    fig.subplots_adjust(
        left=0.035,
        right=0.90,
        top=0.88,
        bottom=0.035,
        hspace=0.33,
        wspace=0.08,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_animation(
    methods: list[tuple[str, np.ndarray]],
    channel: int,
    channel_name: str,
    cmap: str,
    sim_id: int,
    output_path: Path,
) -> None:
    all_values = np.concatenate(
        [state[:, channel].reshape(-1) for _, state in methods]
    )
    vmin, vmax = color_limits(all_values, channel_name)
    fig, axes = plt.subplots(1, len(methods), figsize=(9.6, 3.35))
    images = []
    for ax, (method_name, state) in zip(axes, methods):
        image = ax.imshow(
            state[0, channel],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        images.append(image)
        ax.set_title(method_name, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    frame_text = fig.text(0.5, 0.94, "frame 0", ha="center", va="center", fontsize=14)
    colorbar_axis = fig.add_axes((0.925, 0.16, 0.015, 0.66))
    fig.colorbar(images[0], cax=colorbar_axis, label="normalized value")
    fig.suptitle(
        f"Normalized {channel_name} trajectory — test sim {sim_id}",
        fontsize=15,
        y=1.06,
    )
    fig.subplots_adjust(left=0.025, right=0.90, top=0.84, bottom=0.04, wspace=0.08)

    def update(frame: int):
        for image, (_, state) in zip(images, methods):
            image.set_data(state[frame, channel])
        frame_text.set_text(f"frame {frame}")
        return [*images, frame_text]

    animation = FuncAnimation(
        fig,
        update,
        frames=20,
        interval=450,
        blit=False,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=2.2), dpi=120)
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
        storyboard_path = args.output_dir / (
            f"jellyfish_{channel_name}_sim{args.sim_id}_all_frames.png"
        )
        animation_path = args.output_dir / (
            f"jellyfish_{channel_name}_sim{args.sim_id}_trajectory.gif"
        )
        plot_storyboard(
            methods,
            channel,
            channel_name,
            cmap,
            args.sim_id,
            storyboard_path,
        )
        plot_animation(
            methods,
            channel,
            channel_name,
            cmap,
            args.sim_id,
            animation_path,
        )
        print(storyboard_path)
        print(animation_path)


if __name__ == "__main__":
    main()
