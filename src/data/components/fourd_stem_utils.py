"""Utilities for handling 4D-STEM data processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


def load_diffraction_pattern(pattern_path: Union[str, Path]) -> torch.Tensor:
    """Load a diffraction pattern from file and convert to tensor.
    
    Args:
        pattern_path: Path to the diffraction pattern image
        
    Returns:
        Tensor of shape (1, H, W) normalized to [0, 1]
    """
    pattern_path = Path(pattern_path)
    
    try:
        # Try PIL first (supports most formats)
        with Image.open(pattern_path) as img:
            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')
            pattern = np.array(img, dtype=np.float32)
    except Exception as e:
        log.warning(f"PIL failed to load {pattern_path}: {e}, trying tifffile")
        try:
            import tifffile as tiff
            pattern = tiff.imread(pattern_path).astype(np.float32)
        except ImportError:
            raise ImportError("Neither PIL nor tifffile could load the pattern. Install tifffile for TIFF support.")
        except Exception as e:
            raise RuntimeError(f"Failed to load pattern {pattern_path}: {e}")
    
    # Ensure 2D
    if pattern.ndim == 3:
        pattern = pattern.squeeze()
    if pattern.ndim != 2:
        raise ValueError(f"Expected 2D pattern, got shape {pattern.shape}")
    
    # Normalize to [0, 1]
    if pattern.max() > 1.0:
        pattern = pattern / pattern.max()
    
    # Convert to tensor and add channel dimension
    pattern_tensor = torch.from_numpy(pattern).unsqueeze(0)  # (1, H, W)
    
    return pattern_tensor


def create_scan_position_mapping(
    diffraction_dir: Union[str, Path],
    scan_shape: Tuple[int, int]
) -> Dict[int, Tuple[int, int]]:
    """Create mapping from scan index to (row, col) position.
    
    Args:
        diffraction_dir: Directory containing diffraction patterns
        scan_shape: Shape of the scan grid (rows, cols)
        
    Returns:
        Dictionary mapping scan_idx to (row, col) coordinates
    """
    diffraction_dir = Path(diffraction_dir)
    scan_height, scan_width = scan_shape
    
    # Find all pattern files
    pattern_files = list(diffraction_dir.glob("Image-*.tif"))
    if not pattern_files:
        # Try other common formats
        pattern_files = list(diffraction_dir.glob("*.tif"))
        if not pattern_files:
            pattern_files = list(diffraction_dir.glob("*.tiff"))
    
    if not pattern_files:
        raise FileNotFoundError(f"No diffraction patterns found in {diffraction_dir}")
    
    # Extract scan indices and create mapping
    mapping = {}
    for pattern_file in pattern_files:
        try:
            # Extract scan index from filename (e.g., Image-00001.tif -> 1)
            if pattern_file.name.startswith("Image-"):
                scan_idx_str = pattern_file.stem.split("-")[1]
                scan_idx = int(scan_idx_str)
            else:
                # Try to extract number from filename
                import re
                numbers = re.findall(r'\d+', pattern_file.stem)
                if numbers:
                    scan_idx = int(numbers[0])
                else:
                    continue
            
            # Convert scan index to row, col (assuming raster scan)
            row = scan_idx // scan_width
            col = scan_idx % scan_width
            
            # Check bounds
            if row < scan_height and col < scan_width:
                mapping[scan_idx] = (row, col)
                
        except (ValueError, IndexError) as e:
            log.warning(f"Could not parse scan index from {pattern_file.name}: {e}")
            continue
    
    log.info(f"Created mapping for {len(mapping)} scan positions from {len(pattern_files)} files")
    return mapping


def normalize_pattern(pattern: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize diffraction pattern.
    
    Args:
        pattern: Input pattern array
        method: Normalization method ("minmax", "zscore", "log")
        
    Returns:
        Normalized pattern
    """
    if method == "minmax":
        pmin, pmax = pattern.min(), pattern.max()
        if pmax > pmin:
            return (pattern - pmin) / (pmax - pmin)
        else:
            return pattern
    elif method == "zscore":
        mean, std = pattern.mean(), pattern.std()
        if std > 0:
            return (pattern - mean) / std
        else:
            return pattern - mean
    elif method == "log":
        # Log normalization (common for diffraction patterns)
        return np.log1p(pattern)  # log(1 + x) to handle zeros
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def apply_circular_mask(pattern: np.ndarray, center: Tuple[int, int] = None, radius: int = None) -> np.ndarray:
    """Apply circular mask to diffraction pattern.
    
    Args:
        pattern: Input pattern array
        center: Center of the circle (row, col). If None, uses image center
        radius: Radius of the circle. If None, uses min(height, width) / 2
        
    Returns:
        Masked pattern
    """
    h, w = pattern.shape[-2:]
    
    if center is None:
        center = (h // 2, w // 2)
    if radius is None:
        radius = min(h, w) // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    center_y, center_x = center
    
    # Create circular mask
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    
    # Apply mask
    if pattern.ndim == 2:
        return pattern * mask
    else:
        return pattern * mask[None, :, :]  # Broadcast for (C, H, W)
