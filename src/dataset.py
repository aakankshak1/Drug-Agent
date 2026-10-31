"""Load paper-aligned DrugAgent datasets from Therapeutics Data Commons (TDC).

Datasets are downloaded/cached by PyTDC on first use. Large benchmark CSVs are
intentionally not committed to this repository.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATASET_REGISTRY = {
    "HIV": {
        "task": "HTS",
        "tdc_name": "HIV",
        "description": "Predict HIV replication-inhibition activity from a molecule SMILES string.",
        "split_method": "scaffold",
    },
    "PAMPA": {
        "task": "ADME",
        "tdc_name": "PAMPA_NCATS",
        "description": "Predict high versus low/moderate PAMPA membrane permeability from SMILES.",
        "split_method": "scaffold",
    },
    "DAVIS": {
        "task": "DTI",
        "tdc_name": "DAVIS",
        "description": "Predict drug-target interaction using drug SMILES and protein sequence.",
        "split_method": "cold_split",
        "davis_threshold_nm": 30.0,
    },
}


def _tdc_imports():
    """Support current PyTDC (`tdc_ml`) and legacy PyTDC (`tdc`) installations."""
    try:
        from tdc_ml.single_pred import ADME, HTS
        from tdc_ml.multi_pred import DTI
        return ADME, HTS, DTI
    except ImportError:
        try:
            from tdc.single_pred import ADME, HTS
            from tdc.multi_pred import DTI
            return ADME, HTS, DTI
        except ImportError as exc:
            raise RuntimeError(
                "PyTDC is not installed. Install dependencies with `pip install -r requirements.txt`."
            ) from exc


def normalize_split_keys(raw_splits: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Normalize TDC's `valid`/legacy `val` naming to train/valid/test."""
    valid = raw_splits.get("valid", raw_splits.get("val"))
    required = {"train": raw_splits.get("train"), "valid": valid, "test": raw_splits.get("test")}
    missing = [name for name, frame in required.items() if frame is None]
    if missing:
        raise ValueError(f"TDC did not return required splits: {', '.join(missing)}")
    normalized = {name: frame.copy().reset_index(drop=True) for name, frame in required.items()}
    if any(frame.empty for frame in normalized.values()):
        raise ValueError("TDC returned an empty train, validation, or test split.")
    return normalized


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise KeyError(f"Could not find any of {candidates} in dataset columns {list(frame.columns)}")


def _cold_drug_split(frame: pd.DataFrame, seed: int, fractions=(0.7, 0.1, 0.2)):
    """Create drug-cold train/validation/test splits for DAVIS.

    No drug compound can appear in more than one split. This is stricter than a
    random row split and reduces the risk of memorizing the same drug across
    training and evaluation.
    """
    drug_column = _find_column(frame, ("Drug", "smiles", "SMILES"))
    groups = frame[drug_column].astype(str).to_numpy()
    label_column = _find_column(frame, ("Y", "label", "Label", "affinity"))
    labels = frame[label_column].astype(int).to_numpy()
    indices = np.arange(len(frame))

    # Five folds create an approximately 20% held-out test group. A second
    # eight-fold split of the remaining 80% creates an approximately 10%
    # validation set while keeping drugs disjoint and preserving class balance.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_valid_idx, test_idx = next(outer.split(indices, labels, groups=groups))

    train_valid = frame.iloc[train_valid_idx].reset_index(drop=True)
    train_valid_labels = train_valid[label_column].astype(int).to_numpy()
    train_valid_groups = train_valid[drug_column].astype(str).to_numpy()
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + 1)
    train_idx, valid_idx = next(
        inner.split(
            np.arange(len(train_valid)),
            train_valid_labels,
            groups=train_valid_groups,
        )
    )

    return {
        "train": train_valid.iloc[train_idx].reset_index(drop=True),
        "valid": train_valid.iloc[valid_idx].reset_index(drop=True),
        "test": frame.iloc[test_idx].reset_index(drop=True),
    }


def load_dataset_splits(dataset_name: str, seed: int = 42):
    """Download/load one of the paper's datasets and return split DataFrames.

    HIV and PAMPA use TDC scaffold splits. DAVIS is binarized at 30 nM (lower
    affinity values indicate stronger binding) and split by drug identity.
    """
    name = dataset_name.upper()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset {dataset_name!r}. Choose from {', '.join(DATASET_REGISTRY)}.")

    ADME, HTS, DTI = _tdc_imports()
    config = DATASET_REGISTRY[name]
    if name == "HIV":
        dataset = HTS(name=config["tdc_name"])
    elif name == "PAMPA":
        dataset = ADME(name=config["tdc_name"])
    else:
        dataset = DTI(name=config["tdc_name"])

    if name == "DAVIS":
        frame = dataset.get_data().copy()
        label_column = _find_column(frame, ("Y", "label", "Label", "affinity"))
        drug_column = _find_column(frame, ("Drug", "smiles", "SMILES"))
        numeric_labels = pd.to_numeric(frame[label_column], errors="coerce")
        frame = frame.loc[numeric_labels.notna()].copy()
        numeric_labels = numeric_labels.loc[numeric_labels.notna()]

        # DAVIS labels are binding affinity values in nM. The paper/TDC setup
        # treats values below 30 nM as positive binders.
        unique_labels = set(numeric_labels.unique().tolist())
        if unique_labels.issubset({0, 1, 0.0, 1.0}):
            frame[label_column] = numeric_labels.astype(int)
            label_rule = "Dataset labels already binary (0/1)."
        else:
            frame[label_column] = (numeric_labels < config["davis_threshold_nm"]).astype(int)
            label_rule = "Positive if binding affinity < 30 nM; negative otherwise."

        splits = _cold_drug_split(frame, seed=seed)
        metadata = {
            "dataset": name,
            "tdc_dataset": config["tdc_name"],
            "task": config["task"],
            "description": config["description"],
            "split_method": "drug-cold StratifiedGroupKFold (70/10/20 approx.)",
            "label_rule": label_rule,
            "drug_column": drug_column,
            "label_column": label_column,
            "protein_column": _find_column(frame, ("Target", "Protein", "Sequence", "protein_sequence")),
        }
    else:
        raw_splits = dataset.get_split(
            method="scaffold",
            seed=seed,
            frac=[0.7, 0.1, 0.2],
        )
        splits = normalize_split_keys(raw_splits)
        metadata = {
            "dataset": name,
            "tdc_dataset": config["tdc_name"],
            "task": config["task"],
            "description": config["description"],
            "split_method": "TDC scaffold split (70/10/20)",
            "label_rule": "TDC-provided binary labels.",
        }

    metadata["split_sizes"] = {key: int(len(value)) for key, value in splits.items()}
    metadata["columns"] = list(splits["train"].columns)
    return splits, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and inspect a DrugAgent TDC dataset.")
    parser.add_argument("--dataset", choices=sorted(DATASET_REGISTRY), default="HIV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_frames, details = load_dataset_splits(args.dataset, seed=args.seed)
    print(details)
    for split_name, frame in split_frames.items():
        print(f"\n{split_name}: {len(frame)} rows")
        print(frame.head(3).to_string(index=False))
