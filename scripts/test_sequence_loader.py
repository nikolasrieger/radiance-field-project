"""
Runs the sequence DataLoader twice and plots the first sequence from each
iteration side-by-side to verify the loader returns full sequences.

Usage (from project root):
    UCO3D_DATASET_ROOT=/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw \
        uv run python scripts/test_sequence_loader.py
"""
import os

os.environ.setdefault(
    "UCO3D_DATASET_ROOT",
    "/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw",
)

import math
import matplotlib.pyplot as plt
import torch

from radiance_field_project.data.datamodule import UCO3DDataModule


def plot_sequence(ax_row, batch, title_prefix: str) -> None:
    """Plot frames of a sequence batch across a row of axes."""
    images = batch.image_rgb  # (T, 3, H, W)
    T = min(images.shape[0], len(ax_row))
    for t in range(T):
        ax = ax_row[t]
        img = images[t].permute(1, 2, 0).cpu().float()
        img = img.clamp(0.0, 1.0)
        ax.imshow(img)
        ax.set_title(f"{title_prefix}\nframe {t}", fontsize=7)
        ax.axis("off")


MAX_DISPLAY_FRAMES = 16


def main() -> None:
    dm = UCO3DDataModule(
        image_height=200,
        image_width=200,
        load_gaussian_splats=False,
        num_workers=0,
        pin_memory=False,
        single_sequence=False,
    )
    loader = dm.train_loader()

    print(f"Loader has {len(loader)} sequences")

    # Grab the first sequence from two full passes over the loader.
    runs = []
    for run_idx in range(2):
        batch = next(iter(loader))
        T = batch.image_rgb.shape[0]
        seq = batch.sequence_name[0] if isinstance(batch.sequence_name, list) else batch.sequence_name
        print(f"  Run {run_idx + 1}: sequence '{seq}', {T} frames")
        runs.append((batch, seq, T))

    n_cols = min(MAX_DISPLAY_FRAMES, max(r[2] for r in runs))
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2, 5), squeeze=False)

    for row_idx, (batch, seq, T) in enumerate(runs):
        plot_sequence(axes[row_idx], batch, f"Run {row_idx + 1} — {seq}")
        for t in range(min(T, n_cols), n_cols):
            axes[row_idx][t].set_visible(False)

    fig.suptitle(
        f"Sequence loader — two passes (showing first {n_cols} of {runs[0][2]} frames)",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "sequence_loader_test.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
