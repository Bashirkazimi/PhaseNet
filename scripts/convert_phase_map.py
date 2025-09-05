"""
convert_phase_map.py

Convert Phase-PhaseMapDisplay.tif (RGB) into an integer label map:
  background (60,60,60) -> 0
  color1 (94,215,241)   -> 1
  color2 (246,143,30)   -> 2

Outputs:
  phase_id_map.npy  (H, W) int32
  phase_id_map.tif  (H, W) uint8
"""

import numpy as np
from PIL import Image
import argparse
import os


def main(infile, outprefix):
    img = np.array(Image.open(infile))  # (H,W,3), dtype=uint8
    H, W, _ = img.shape
    label_map = np.zeros((H, W), dtype=np.uint8)

    # Define color → class mapping
    color_to_class = {
        (60, 60, 60): 0,         # background/unindexed
        (94, 215, 241): 1,       # Ni
        (246, 143, 30): 2        # Al
    }

    # Flatten for speed
    flat = img.reshape(-1, 3)
    labels = np.zeros(flat.shape[0], dtype=np.uint8)

    for rgb, cls in color_to_class.items():
        mask = np.all(flat == np.array(rgb, dtype=np.uint8), axis=1)
        labels[mask] = cls

    label_map = labels.reshape(H, W)

    # Get directory of infile
    outdir = os.path.dirname(os.path.abspath(infile))
    
    # Save as numpy array (int32 for broader compatibility)
    npy_path = os.path.join(outdir, f"{outprefix}.npy")
    np.save(npy_path, label_map.astype(np.int32))
    print(f"Saved: {npy_path}")

    # Save as TIFF for visualization
    tif_path = os.path.join(outdir, f"{outprefix}.tif")
    try:
        import tifffile as tiff
        tiff.imwrite(tif_path, label_map)
        print(f"Saved: {tif_path}")
    except ImportError:
        # Fallback to PIL
        Image.fromarray(label_map, mode='L').save(tif_path)
        print(f"Saved: {tif_path} (using PIL)")

    # Print statistics
    unique, counts = np.unique(label_map, return_counts=True)
    print("\nPhase statistics:")
    for phase, count in zip(unique, counts):
        print(f"  Phase {phase}: {count} pixels")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Path to Phase-PhaseMapDisplay.tif")
    ap.add_argument("--outprefix", default="phase_id_map", help="Prefix for outputs")
    args = ap.parse_args()
    main(args.infile, args.outprefix)
