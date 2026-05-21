"""DataLoader ``collate_fn`` for uCO3D batches."""
from uco3d.dataset_utils.frame_data import UCO3DFrameData


def uco3d_collate(batch):
    """Collate a list of ``UCO3DFrameData`` into a batched ``UCO3DFrameData``.

    Uses uCO3D's built-in collate, which knows how to stack tensors,
    handle variable-length point clouds, and preserve metadata.
    """
    return UCO3DFrameData.collate(batch)
