"""Domain-specific molecular and protein featurization utilities.

ECFP4-style Morgan fingerprints use radius=2 and 2048 bits. Protein sequences
are represented with normalized Conjoint Triad (CT) descriptors (343 values).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048

# Standard seven amino-acid groups used by the Conjoint Triad descriptor.
_AA_GROUPS = (
    "AGV",      # aliphatic
    "ILFP",     # aliphatic, larger side chains
    "YMTS",     # aromatic / hydroxyl-containing
    "HNQW",     # polar
    "RK",       # positively charged
    "DE",       # negatively charged
    "C",        # sulphur-containing
)
_AA_TO_GROUP = {aa: group for group, residues in enumerate(_AA_GROUPS) for aa in residues}
_CT_DIMENSION = 7 ** 3


@lru_cache(maxsize=1)
def _morgan_generator():
    """Create the RDKit fingerprint generator once per process."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_N_BITS,
        includeChirality=False,
    )


def morgan_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """Return a 2048-bit radius-2 Morgan fingerprint, or None for invalid SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        return None
    fingerprint = _morgan_generator().GetFingerprintAsNumPy(molecule)
    return np.asarray(fingerprint, dtype=np.float32)


def conjoint_triad(sequence: str) -> Optional[np.ndarray]:
    """Return a normalized 343-dimensional Conjoint Triad protein descriptor.

    Every consecutive triplet is mapped to one of 7 x 7 x 7 amino-acid-group
    combinations. Counts are normalized to sum to 1. Windows containing an
    unknown amino-acid symbol are skipped; sequences with no valid triplets
    return None.
    """
    if not isinstance(sequence, str):
        return None

    groups = [_AA_TO_GROUP.get(residue) for residue in sequence.upper() if residue.isalpha()]
    if len(groups) < 3:
        return None

    counts = np.zeros(_CT_DIMENSION, dtype=np.float32)
    valid_triplets = 0
    for start in range(len(groups) - 2):
        first, second, third = groups[start : start + 3]
        if first is None or second is None or third is None:
            continue
        index = first * 49 + second * 7 + third
        counts[index] += 1.0
        valid_triplets += 1

    if valid_triplets == 0:
        return None
    return counts / float(valid_triplets)
