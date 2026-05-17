"""DataLoader construction for uCO3D."""
from dataclasses import dataclass
from typing import Sequence

from torch.utils.data import DataLoader

from src.data.collate import uco3d_collate
from src.data.dataset import make_uco3d_dataset


@dataclass
class UCO3DDataModule:
    """Builds train/val/test DataLoaders for uCO3D.

    Configuration values are kept on the dataclass so callers can override
    selectively without long arg lists at the call site.
    """
    subset_lists_filename: str = "/home/nrieger/radiance-field-project/set_lists_all-categories.sqlite"
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    image_height: int = 800
    image_width: int = 800
    load_gaussian_splats: bool = True
    load_depths: bool = False
    load_masks: bool = False

    def _make_dataset(self, subsets: Sequence[str]):
        return make_uco3d_dataset(
            subsets=subsets,
            subset_lists_filename=self.subset_lists_filename,
            image_height=self.image_height,
            image_width=self.image_width,
            load_gaussian_splats=self.load_gaussian_splats,
            load_depths=self.load_depths,
            load_masks=self.load_masks,
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