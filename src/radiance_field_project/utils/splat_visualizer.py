"""Utilities for visualizing uco3d ``render_splats`` outputs in a notebook.

Example
-------
>>> from radiance_field_project.utils.splat_visualizer import SplatRenderVisualizer
>>> render_colors, render_alphas, render_info = render_splats(
...     cameras=frame_data.camera,
...     splats=frame_data.sequence_gaussian_splats,
...     render_size=[512, 512],
... )
>>> SplatRenderVisualizer.show(render_colors, render_alphas)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


class SplatRenderVisualizer:
    """Display the RGB image and alpha mask returned by ``uco3d.render_splats``.

    ``render_splats`` wraps ``gsplat.rasterization`` and returns:

    - ``render_colors`` of shape ``[B, H, W, 3]`` (or ``[B, H, W, 4]`` if depth
      is appended as the last channel),
    - ``render_alphas`` of shape ``[B, H, W, 1]``,

    both as ``torch.Tensor`` objects on the device used for rendering.
    """

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().float().numpy()

    @staticmethod
    def _select_batch(arr: np.ndarray, index: int) -> np.ndarray:
        """Pick a single image from a batched array, leaving a 3D array."""
        if arr.ndim == 4:
            if not (0 <= index < arr.shape[0]):
                raise IndexError(
                    f"Batch index {index} out of range for batch size {arr.shape[0]}"
                )
            return arr[index]
        return arr

    @classmethod
    def _prepare_rgb(cls, render_colors: torch.Tensor, batch_index: int) -> np.ndarray:
        arr = cls._select_batch(cls._to_numpy(render_colors), batch_index)
        # render_colors may carry depth as a 4th channel: keep only RGB.
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = arr[..., :3]
        return np.clip(arr, 0.0, 1.0)

    @classmethod
    def _prepare_alpha(
        cls, render_alphas: torch.Tensor, batch_index: int
    ) -> np.ndarray:
        arr = cls._select_batch(cls._to_numpy(render_alphas), batch_index)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def _composite_on_background(
        rgb: np.ndarray, alpha: np.ndarray, background: float
    ) -> np.ndarray:
        a = alpha[..., None]
        return rgb * a + background * (1.0 - a)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @classmethod
    def to_numpy(
        cls,
        render_colors: torch.Tensor,
        render_alphas: Optional[torch.Tensor] = None,
        batch_index: int = 0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert raw renderer tensors into displayable NumPy arrays."""
        rgb = cls._prepare_rgb(render_colors, batch_index)
        alpha = (
            cls._prepare_alpha(render_alphas, batch_index)
            if render_alphas is not None
            else None
        )
        return rgb, alpha

    @classmethod
    def show(
        cls,
        render_colors: torch.Tensor,
        render_alphas: Optional[torch.Tensor] = None,
        batch_index: int = 0,
        composite_background: Optional[float] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10.0, 5.0),
    ) -> plt.Figure:
        """Display the RGB image (and alpha mask) inline.

        Parameters
        ----------
        render_colors, render_alphas
            Tensors returned by ``render_splats``.
        batch_index
            Which item of the batch to show. Defaults to ``0``.
        composite_background
            If set (e.g. ``1.0`` for white, ``0.0`` for black), the RGB image is
            alpha-composited over a constant background instead of shown raw.
        title
            Optional figure title.
        figsize
            Figure size in inches.
        """
        rgb, alpha = cls.to_numpy(render_colors, render_alphas, batch_index)

        if composite_background is not None and alpha is not None:
            rgb = cls._composite_on_background(rgb, alpha, composite_background)

        ncols = 2 if alpha is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)

        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("RGB")
        axes[0, 0].axis("off")

        if alpha is not None:
            axes[0, 1].imshow(alpha, cmap="gray", vmin=0.0, vmax=1.0)
            axes[0, 1].set_title("Alpha")
            axes[0, 1].axis("off")

        if title is not None:
            fig.suptitle(title)

        fig.tight_layout()
        plt.show()
        return fig

    @classmethod
    def show_grid(
        cls,
        renders: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor]]],
        titles: Optional[Sequence[str]] = None,
        composite_background: Optional[float] = None,
        ncols: int = 4,
        figsize_per_cell: Tuple[float, float] = (3.0, 3.0),
    ) -> plt.Figure:
        """Show several renders (RGB only) in a grid — useful for sweeping views."""
        n = len(renders)
        if n == 0:
            raise ValueError("`renders` must contain at least one item")

        ncols = max(1, min(ncols, n))
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
            squeeze=False,
        )

        for idx in range(nrows * ncols):
            ax = axes[idx // ncols, idx % ncols]
            ax.axis("off")
            if idx >= n:
                continue

            colors, alphas = renders[idx]
            rgb, alpha = cls.to_numpy(colors, alphas)
            if composite_background is not None and alpha is not None:
                rgb = cls._composite_on_background(rgb, alpha, composite_background)
            ax.imshow(rgb)
            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx])

        fig.tight_layout()
        plt.show()
        return fig

    @classmethod
    def save(
        cls,
        path: str,
        render_colors: torch.Tensor,
        render_alphas: Optional[torch.Tensor] = None,
        batch_index: int = 0,
        composite_background: Optional[float] = None,
        labels: Optional[Sequence[str]] = None,
        columns: Optional[int] = None,
    ) -> None:
        """Save the RGB render to ``path`` as an 8-bit image (PNG/JPEG/...).

        If ``labels`` is provided the image is treated as a grid of equal
        panels and each label is drawn in the top-left corner of its panel.
        With ``columns=None`` the panels are assumed to lie in a single
        horizontal row; pass an explicit ``columns`` to lay out the labels
        on a ``len(labels)/columns``-row grid.
        """
        from PIL import Image  # local import keeps Pillow optional

        rgb, alpha = cls.to_numpy(render_colors, render_alphas, batch_index)
        if composite_background is not None and alpha is not None:
            rgb = cls._composite_on_background(rgb, alpha, composite_background)

        img = (rgb * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(img)

        if labels:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(pil)
            H, W = img.shape[:2]
            cols = columns or len(labels)
            rows = (len(labels) + cols - 1) // cols
            panel_w = W // cols
            panel_h = H // rows
            font_size = max(12, panel_w // 24)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
            except OSError:
                font = ImageFont.load_default()
            pad = max(6, font_size // 2)
            for i, label in enumerate(labels):
                r, c = divmod(i, cols)
                x, y = c * panel_w + pad, r * panel_h + pad
                # Black outline for legibility on bright backgrounds.
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx or dy:
                            draw.text((x + dx, y + dy), label,
                                      fill=(0, 0, 0), font=font)
                draw.text((x, y), label, fill=(255, 255, 255), font=font)

        pil.save(path)