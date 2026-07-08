"""DataLoader construction for uCO3D."""
import hashlib
import json
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from torch.utils.data import DataLoader, Sampler
from uco3d import UCO3DDataset

from radiance_field_project.data.collate import uco3d_collate
from radiance_field_project.data.dataset import make_uco3d_dataset

_MANIFEST_CACHE_DIR = Path.home() / ".cache" / "radiance_field_project" / "splat_manifests"


def _manifest_path(dataset_root: str) -> Path:
    """Per-dataset-root sidecar file caching sequence -> Gaussian point count.

    Keyed by a hash of ``dataset_root`` (rather than living inside it) so it
    works even when the dataset directory is read-only, and is shared across
    every subset/pick_sequences/run that points at the same data.
    """
    key = hashlib.sha1(dataset_root.encode()).hexdigest()[:16]
    return _MANIFEST_CACHE_DIR / f"{key}.json"


class SequenceBatchSampler(Sampler[List[int]]):
    """Yields one batch per sequence, where each batch contains all frame indices of that sequence.

    This ensures that each training step sees a complete multi-view sequence
    rather than a random mix of frames from different sequences.
    """

    def __init__(
        self,
        dataset: UCO3DDataset,
        shuffle: bool = False,
        seed: int = 0,
        max_splats: Optional[int] = None,
    ) -> None:
        self._batches: List[List[int]] = []
        self._seq_names: List[str] = []
        skipped = 0

        # `max_splats` filtering needs each sequence's point count, which
        # otherwise means fully loading every sequence's splats just to read
        # `.means.shape[0]`. Cache counts to disk so repeat runs against the
        # same dataset_root skip that load entirely.
        manifest: dict = {}
        manifest_file: Optional[Path] = None
        manifest_dirty = False
        if max_splats is not None and getattr(dataset, "dataset_root", None):
            manifest_file = _manifest_path(str(dataset.dataset_root))
            if manifest_file.exists():
                try:
                    manifest = json.loads(manifest_file.read_text())
                except (json.JSONDecodeError, OSError):
                    manifest = {}

        for seq in dataset.sequence_names():
            indices = list(dataset.sequence_indices_in_order(seq))
            if max_splats is not None:
                if seq in manifest:
                    n_points = manifest[seq]
                else:
                    frame = dataset[indices[0]]
                    splats = frame.sequence_gaussian_splats
                    n_points = splats.means.shape[0] if splats is not None else 0
                    manifest[seq] = n_points
                    manifest_dirty = True
                if n_points >= max_splats:
                    skipped += 1
                    continue
            self._batches.append(indices)
            self._seq_names.append(seq)

        if manifest_file is not None and manifest_dirty:
            try:
                manifest_file.parent.mkdir(parents=True, exist_ok=True)
                manifest_file.write_text(json.dumps(manifest))
            except OSError:
                pass  # manifest caching is a speed-up, not correctness-critical

        if max_splats is not None:
            print(f"SequenceBatchSampler: kept {len(self._batches)} sequences, skipped {skipped} with >= {max_splats} splats")
            if not self._batches:
                raise RuntimeError(
                    f"All sequences were filtered out (max_splats={max_splats}). "
                    "Lower the threshold or check that gaussian splats are loaded."
                )
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        batches = list(self._batches)
        if self._shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
        self._epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches)


@dataclass
class UCO3DDataModule:
    """Builds train/val/test DataLoaders for uCO3D.

    Configuration values are kept on the dataclass so callers can override
    selectively without long arg lists at the call site.
    """
    subset_lists_filename: str = "set_lists_all-categories.sqlite"
    dataset_root: str = ""
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    image_height: int = 800
    image_width: int = 800
    load_gaussian_splats: bool = True
    load_depths: bool = False
    load_masks: bool = False
    pick_sequences: List[str] = field(default_factory=list)

    def _make_dataset(self, subsets: Sequence[str]):
        return make_uco3d_dataset(
            subsets=subsets,
            subset_lists_filename=self.subset_lists_filename,
            dataset_root=self.dataset_root or None,
            image_height=self.image_height,
            image_width=self.image_width,
            load_gaussian_splats=self.load_gaussian_splats,
            load_depths=self.load_depths,
            load_masks=self.load_masks,
            pick_sequences=self.pick_sequences or None,
        )

    def _make_loader(self, subsets: Sequence[str], shuffle: bool) -> DataLoader:
        dataset = self._make_dataset(subsets)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=uco3d_collate,
            persistent_workers=self.num_workers > 0,
            drop_last=shuffle,
        )

    def train_loader(self) -> DataLoader:
        return self._make_loader(subsets=("train",), shuffle=True)

    def val_loader(self) -> DataLoader:
        return self._make_loader(subsets=("val",), shuffle=False)

    def test_loader(self) -> DataLoader:
        return self._make_loader(subsets=("test",), shuffle=False)

    def splat_only_loader(self, max_splats: Optional[int] = None,
                          subsets: Sequence[str] = ("train",)) -> DataLoader:
        """DataLoader that skips image loading — useful for splat-only ablations.

        ``subsets`` selects the split(s), e.g. ``("train",)`` or ``("val",)``.
        """
        dataset = make_uco3d_dataset(
            subsets=subsets,
            subset_lists_filename=self.subset_lists_filename,
            dataset_root=self.dataset_root or None,
            load_images=False,
            load_depths=False,
            load_masks=False,
            load_gaussian_splats=True,
            image_height=self.image_height,
            image_width=self.image_width,
            pick_sequences=self.pick_sequences or None,
        )
        sampler = SequenceBatchSampler(dataset, shuffle=False, max_splats=max_splats)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=uco3d_collate,
        )

    def image_only_loader(self, subsets: Sequence[str] = ("train",),
                          shuffle: bool = True) -> DataLoader:
        """DataLoader that skips splat loading — useful for image-only ablations.

        ``subsets`` selects the split(s); ``shuffle`` controls sequence order
        (pass ``False`` for deterministic validation).
        """
        dataset = make_uco3d_dataset(
            subsets=subsets,
            subset_lists_filename=self.subset_lists_filename,
            dataset_root=self.dataset_root or None,
            load_images=True,
            load_depths=True,
            load_masks=self.load_masks,
            load_gaussian_splats=False,
            image_height=self.image_height,
            image_width=self.image_width,
            pick_sequences=self.pick_sequences or None,
        )
        sampler = SequenceBatchSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=uco3d_collate,
        )
