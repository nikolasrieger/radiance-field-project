# src/data/dataset.py
"""Factory functions for constructing UCO3DDataset instances."""
from pathlib import Path
from typing import Sequence

from uco3d import UCO3DDataset, UCO3DFrameDataBuilder

from radiance_field_project.utils.paths import UCO3D_DATASET_ROOT


def make_uco3d_dataset(
    subsets: Sequence[str] = ("train",),
    subset_lists_filename: str = "set_lists_all-categories.sqlite",
    dataset_root: str | None = None,
    apply_alignment: bool = True,
    load_images: bool = True,
    load_depths: bool = False,
    load_masks: bool = False,
    load_point_clouds: bool = False,
    load_gaussian_splats: bool = True,
    box_crop: bool = True,
    image_height: int = 800,
    image_width: int = 800,
    pick_sequences: Sequence[str] | None = None,
) -> UCO3DDataset:
    """Build a UCO3DDataset with project-level defaults.

    Args:
        subsets: Which splits to include, e.g. ("train",) or ("train", "val").
        subset_lists_filename: SQLite file under set_lists/ defining the subset.
            "set_lists_all-categories.sqlite" for the full dataset.
        apply_alignment: Whether to align scenes to a canonical pose.
        load_*: Toggle which modalities to load. Disable what you don't need
            for faster iteration and lower memory.
        box_crop: Crop images around the object bounding box.
        image_height/width: Output resolution after cropping/resizing.
        pick_sequences: Restrict the dataset to these sequence names (useful
            for single/multi-sequence overfit runs). ``None`` or an empty
            sequence loads the full subset.
    """
    p = Path(subset_lists_filename)
    if p.is_absolute():
        subset_lists_file = p
    else:
        # Try the canonical layout first (UCO3D_DATASET_ROOT/set_lists/<name>),
        # then fall back to UCO3D_DATASET_ROOT/<name> directly (e.g. when the
        # sqlite is placed at the dataset/project root without a set_lists/ subdir).
        canonical = Path(UCO3D_DATASET_ROOT) / "set_lists" / subset_lists_filename
        flat      = Path(UCO3D_DATASET_ROOT) / subset_lists_filename
        subset_lists_file = canonical if canonical.exists() else flat
    if not subset_lists_file.exists():
        raise FileNotFoundError(
            f"Subset list not found: {subset_lists_file}. "
            f"Check UCO3D_DATASET_ROOT and that the dataset is downloaded."
        )

    builder_kwargs = dict(
        apply_alignment=apply_alignment,
        load_images=load_images,
        load_depths=load_depths,
        load_masks=load_masks,
        load_point_clouds=load_point_clouds,
        load_gaussian_splats=load_gaussian_splats,
        box_crop=box_crop,
        image_height=image_height,
        image_width=image_width,
    )
    if dataset_root:
        builder_kwargs["dataset_root"] = dataset_root
    frame_data_builder = UCO3DFrameDataBuilder(**builder_kwargs)

    kwargs = dict(
        subset_lists_file=str(subset_lists_file),
        subsets=list(subsets),
        frame_data_builder=frame_data_builder,
    )
    if pick_sequences:
        kwargs["pick_sequences"] = tuple(pick_sequences)

    return UCO3DDataset(**kwargs)
