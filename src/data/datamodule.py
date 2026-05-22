"""DataLoader construction for uCO3D."""
import torch
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

from torch.utils.data import DataLoader, Sampler
from uco3d import UCO3DDataset

from src.data.collate import uco3d_collate
from src.data.dataset import make_uco3d_dataset


class SequenceBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: UCO3DDataset,
        shuffle: bool = False,
        seed: int = 0,
        max_splats: Optional[int] = None,
    ) -> None:
        self._batches: List[List[int]] = []
        skipped = 0
        for seq in dataset.sequence_names():
            indices = list(dataset.sequence_indices_in_order(seq))
            if max_splats is not None:
                frame = dataset[indices[0]]
                splats = frame.sequence_gaussian_splats
                if splats is not None and splats.means.shape[0] >= max_splats:
                    skipped += 1
                    continue
            self._batches.append(indices)
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
    subset_lists_filename: str = "/home/nrieger/radiance-field-project/set_lists_all-categories.sqlite"
    num_workers: int = 0
    pin_memory: bool = True
    image_height: int = 800
    image_width: int = 800
    load_gaussian_splats: bool = True
    load_depths: bool = False
    load_masks: bool = False
    single_sequence: bool = True

    def _make_dataset(self, subsets: Sequence[str]) -> UCO3DDataset:
        return make_uco3d_dataset(
            subsets=subsets,
            subset_lists_filename=self.subset_lists_filename,
            image_height=self.image_height,
            image_width=self.image_width,
            load_gaussian_splats=self.load_gaussian_splats,
            load_depths=self.load_depths,
            load_masks=self.load_masks,
            single_sequence=self.single_sequence,
        )

    def _make_loader(self, subsets: Sequence[str], shuffle: bool, max_splats: Optional[int] = None) -> DataLoader:
        dataset = self._make_dataset(subsets)
        sampler = SequenceBatchSampler(dataset, shuffle=shuffle, max_splats=max_splats)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=uco3d_collate,
            persistent_workers=self.num_workers > 0,
        )

    def train_loader(self, max_splats: Optional[int] = None) -> DataLoader:
        return self._make_loader(subsets=("train",), shuffle=True, max_splats=max_splats)

    def val_loader(self, max_splats: Optional[int] = None) -> DataLoader:
        return self._make_loader(subsets=("val",), shuffle=False, max_splats=max_splats)

    def test_loader(self, max_splats: Optional[int] = None) -> DataLoader:
        return self._make_loader(subsets=("test",), shuffle=False, max_splats=max_splats)

    def splat_only_loader(self, max_splats: Optional[int] = None) -> DataLoader:
        dataset = make_uco3d_dataset(
            subsets=("train",),
            subset_lists_filename=self.subset_lists_filename,
            load_images=False,
            load_depths=False,
            load_masks=False,
            load_gaussian_splats=True,
            image_height=self.image_height,
            image_width=self.image_width,
            single_sequence=self.single_sequence,
        )
        sampler = SequenceBatchSampler(dataset, shuffle=False, max_splats=max_splats)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=uco3d_collate,
        )

    def image_only_loader(self) -> DataLoader:
        dataset = make_uco3d_dataset(
            subsets=("train",),
            subset_lists_filename=self.subset_lists_filename,
            load_images=True,
            load_depths=True,
            load_masks=True,
            load_gaussian_splats=False,
            image_height=self.image_height,
            image_width=self.image_width,
            single_sequence=self.single_sequence,
        )
        sampler = SequenceBatchSampler(dataset, shuffle=True)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=uco3d_collate,
        )
