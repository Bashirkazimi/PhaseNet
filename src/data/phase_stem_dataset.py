"""Dataset for phase classification from 4D-STEM diffraction patterns and phase map."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.components.fourd_stem_utils import (
    load_diffraction_pattern,
    create_scan_position_mapping,
)

log = logging.getLogger(__name__)


class PhaseSTEMDataset(Dataset):
    """Dataset for phase classification from diffraction patterns."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        reliability_map: Optional[np.ndarray] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize dataset.
        
        Args:
            samples: List of sample dictionaries with keys: pattern_path, phase_id, etc.
            reliability_map: Optional reliability map for weighting samples
            transform: Optional transforms to apply to patterns
        """
        self.samples = samples
        self.reliability_map = reliability_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load diffraction pattern
        pattern = load_diffraction_pattern(sample["pattern_path"])
        
        # Apply transforms
        if self.transform:
            pattern = self.transform(pattern)
        
        # Get phase label
        phase_id = torch.tensor(sample["phase_id"], dtype=torch.long)
        
        result = {
            "pattern": pattern,
            "phase_id": phase_id,
        }
        
        # Add reliability if available
        if self.reliability_map is not None:
            row, col = sample["row"], sample["col"]
            reliability = self.reliability_map[row, col]
            result["reliability"] = torch.tensor(reliability, dtype=torch.float32)
        
        return result


def prepare_phase_splits(
    diffraction_dir: Union[str, Path],
    phase_map: np.ndarray,
    scan_mapping: Dict[int, Tuple[int, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    include_background: bool = True,
    background_class: int = 0,
    random_state: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare train/val/test splits for phase classification."""
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    diffraction_dir = Path(diffraction_dir)
    samples: List[Dict[str, Any]] = []
    
    for scan_idx, (r, c) in scan_mapping.items():
        pattern_path = diffraction_dir / f"Image-{scan_idx:05d}.tif"
        if not pattern_path.exists():
            continue
        if r >= phase_map.shape[0] or c >= phase_map.shape[1]:
            continue
            
        phase_id = int(phase_map[r, c])
        if (not include_background) and phase_id == background_class:
            continue
            
        samples.append({
            "pattern_path": str(pattern_path),
            "scan_idx": scan_idx,
            "row": r,
            "col": c,
            "phase_id": phase_id,
        })
    
    log.info(f"Collected {len(samples)} phase-labelled samples")
    if not samples:
        raise RuntimeError("No samples collected for phase classification. Check data layout.")
    
    # Stratified split
    train_samples, temp = train_test_split(
        samples, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state, 
        stratify=[s['phase_id'] for s in samples]
    )
    val_samples, test_samples = train_test_split(
        temp,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state,
        stratify=[s['phase_id'] for s in temp],
    )
    
    return {"train": train_samples, "val": val_samples, "test": test_samples}


def load_phase_map(path: Union[str, Path]) -> np.ndarray:
    """Load phase map from npy or tif file."""
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    else:  # tif fallback
        import tifffile as tiff
        arr = tiff.imread(path)
    return arr.astype(np.int32)


def create_dataset_from_phase_data(
    data_root: Union[str, Path],
    split: str = "train",
    phase_map_filename: Optional[str] = None,
    include_background: bool = True,
    background_class: int = 0,
    cache_dir: Optional[Union[str, Path]] = None,
    reliability_map_filename: Optional[str] = None,
) -> PhaseSTEMDataset:
    """Create phase classification dataset from 4D-STEM data."""
    data_root = Path(data_root)
    
    # Find phase map
    if phase_map_filename is None:
        for cand in ["phase_id_map.npy", "phase_id_map.tif"]:
            if (data_root / cand).exists():
                phase_map_filename = cand
                break
    if phase_map_filename is None:
        raise FileNotFoundError("Phase map file not found (looked for phase_id_map.(npy|tif)). Provide phase_map_filename.")
    
    phase_map_path = data_root / phase_map_filename
    phase_map = load_phase_map(phase_map_path)
    scan_shape = phase_map.shape
    
    # Check for diffraction patterns
    diffraction_dir = data_root / "DiffractionPatterns"
    if not diffraction_dir.exists():
        raise FileNotFoundError(f"DiffractionPatterns directory missing at {diffraction_dir}")
    
    mapping = create_scan_position_mapping(diffraction_dir, scan_shape)

    # Cache splits
    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"phase_dataset_splits_{scan_shape[0]}x{scan_shape[1]}.npz"

    splits = None
    if cache_file and cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        splits = data["splits"].item()
        
    if splits is None:
        splits = prepare_phase_splits(
            diffraction_dir,
            phase_map,
            mapping,
            include_background=include_background,
            background_class=background_class,
        )
        if cache_file:
            np.savez(cache_file, splits=splits)

    # Reliability map integration (optional)
    reliability_map = None
    if reliability_map_filename is not None:
        rel_path = data_root / reliability_map_filename
        if rel_path.exists():
            try:
                if rel_path.suffix == ".npy":
                    reliability_map = np.load(rel_path).astype(np.float32)
                else:
                    import tifffile as tiff
                    reliability_map = tiff.imread(rel_path).astype(np.float32)
                    if reliability_map.max() > 1.0:
                        reliability_map = reliability_map / 255.0  # Normalize if needed
                log.info(f"Loaded reliability map from {rel_path}")
            except Exception as e:
                log.warning(f"Failed to load reliability map: {e}")

    return PhaseSTEMDataset(
        samples=splits[split],
        reliability_map=reliability_map,
    )
