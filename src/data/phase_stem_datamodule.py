"""Lightning DataModule for phase classification from 4D-STEM diffraction patterns."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.phase_stem_dataset import create_dataset_from_phase_data

log = logging.getLogger(__name__)


class PhaseSTEMDataModule(LightningDataModule):
    """DataModule for phase classification from 4D-STEM data."""

    def __init__(
        self,
        data_root: str = "data/phase_stem/",
        phase_map_filename: Optional[str] = None,
        reliability_map_filename: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        augment_training: bool = True,
        pattern_size: int = 256,
        include_background: bool = True,
        background_class: int = 0,
        cache_dir: Optional[str] = "data/cache/",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_root = Path(data_root)
        self.phase_map_filename = phase_map_filename
        self.reliability_map_filename = reliability_map_filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_training = augment_training
        self.pattern_size = pattern_size
        self.include_background = include_background
        self.background_class = background_class
        self.cache_dir = cache_dir

        # Initialize datasets
        self.data_train = None
        self.data_val = None
        self.data_test = None

        # Build transforms
        self.train_transforms = self._build_train_transforms()
        self.eval_transforms = self._build_eval_transforms()

    def _build_train_transforms(self):
        """Build training transforms with augmentation."""
        transform_list = [
            transforms.Resize((self.pattern_size, self.pattern_size)),
        ]
        
        if self.augment_training:
            transform_list.extend([
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Add small random noise
                transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
            ])
        
        transform_list.extend([
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
        ])
        
        return transforms.Compose(transform_list)

    def _build_eval_transforms(self):
        """Build evaluation transforms without augmentation."""
        return transforms.Compose([
            transforms.Resize((self.pattern_size, self.pattern_size)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    @property
    def num_classes(self) -> int:
        """Get number of classes from training dataset."""
        if self.data_train is None:
            # Default assumption - update based on your data
            return 3
        
        # Count unique phase IDs in training data
        phase_ids = set()
        for sample in self.data_train.samples:
            phase_ids.add(sample["phase_id"])
        return len(phase_ids)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation, and testing."""
        if stage == "fit" or stage is None:
            if self.data_train is None:
                self.data_train = create_dataset_from_phase_data(
                    self.data_root,
                    split="train",
                    phase_map_filename=self.phase_map_filename,
                    include_background=self.include_background,
                    background_class=self.background_class,
                    cache_dir=self.cache_dir,
                    reliability_map_filename=self.reliability_map_filename,
                )
                self.data_train.transform = self.train_transforms
                
            if self.data_val is None:
                self.data_val = create_dataset_from_phase_data(
                    self.data_root,
                    split="val",
                    phase_map_filename=self.phase_map_filename,
                    include_background=self.include_background,
                    background_class=self.background_class,
                    cache_dir=self.cache_dir,
                    reliability_map_filename=self.reliability_map_filename,
                )
                self.data_val.transform = self.eval_transforms

        if stage == "test" or stage is None:
            if self.data_test is None:
                self.data_test = create_dataset_from_phase_data(
                    self.data_root,
                    split="test",
                    phase_map_filename=self.phase_map_filename,
                    include_background=self.include_background,
                    background_class=self.background_class,
                    cache_dir=self.cache_dir,
                    reliability_map_filename=self.reliability_map_filename,
                )
                self.data_test.transform = self.eval_transforms

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
