"""Detect (and optionally remove) the known duplicate local benchmark CSVs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

FILES = (Path("data/HIV.csv"), Path("data/PAMPA.csv"), Path("data/DAVIS.csv"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete-identical", action="store_true", help="Delete the three files only if all exist and have identical SHA-256 hashes.")
    args = parser.parse_args()
    existing = [path for path in FILES if path.exists()]
    if len(existing) < 2:
        print("No duplicate benchmark-file set found.")
        return
    hashes = {path: sha256(path) for path in existing}
    for path, digest in hashes.items():
        print(f"{path}: {digest}")
    identical = len(existing) == len(FILES) and len(set(hashes.values())) == 1
    if identical:
        print("WARNING: HIV.csv, PAMPA.csv and DAVIS.csv are byte-for-byte identical.")
        if args.delete_identical:
            for path in existing:
                path.unlink()
                print(f"Removed duplicate file: {path}")
    else:
        print("The files are not all identical; no files were removed.")


if __name__ == "__main__":
    main()
