#!/usr/bin/env python3
"""Copy pre-trained centroid vectors from NadirClaw reference to models/centroids/.

Usage::

    python scripts/copy_centroids.py

This copies the pre-computed simple_centroid.npy and complex_centroid.npy
files from reference/NadirClaw/nadirclaw/ to models/centroids/ so the
CentroidRoutingStrategy can load them directly without needing to
regenerate from prototypes.
"""

import shutil
import sys
from pathlib import Path


def main() -> None:
    """Copy centroid files from NadirClaw reference to models/centroids/."""
    project_root = Path(__file__).resolve().parent.parent
    source_dir = project_root / "reference" / "NadirClaw" / "nadirclaw"
    target_dir = project_root / "models" / "centroids"

    files = ["simple_centroid.npy", "complex_centroid.npy"]

    # Validate source files exist
    for fname in files:
        src = source_dir / fname
        if not src.exists():
            print(f"ERROR: Source file not found: {src}", file=sys.stderr)
            print(
                "Ensure the NadirClaw submodule is initialized: "
                "git submodule update --init",
                file=sys.stderr,
            )
            sys.exit(1)

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    for fname in files:
        src = source_dir / fname
        dst = target_dir / fname
        shutil.copy2(str(src), str(dst))
        print(f"Copied: {src} → {dst}")

    print(f"\n✅ Centroid files copied to {target_dir}")

    # Verify with numpy if available
    try:
        import numpy as np

        for fname in files:
            arr = np.load(str(target_dir / fname))
            norm = float(np.linalg.norm(arr))
            print(f"  {fname}: shape={arr.shape}, norm={norm:.6f}")
    except ImportError:
        print("  (numpy not available, skipping verification)")


if __name__ == "__main__":
    main()
