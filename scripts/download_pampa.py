"""Backward-compatible dataset inspection/download script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import DATASET_REGISTRY, load_dataset_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect a TDC dataset.")
    parser.add_argument("--dataset", choices=sorted(DATASET_REGISTRY), default="PAMPA")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    splits, metadata = load_dataset_splits(args.dataset, seed=args.seed)
    print(metadata)
    for name, frame in splits.items():
        print(f"\n{name}: {len(frame)} rows")
        print(frame.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
