import unittest

import pandas as pd

from src.dataset import normalize_split_keys, _cold_drug_split


class DatasetTests(unittest.TestCase):
    def test_normalize_split_keys_accepts_valid(self):
        raw = {
            "train": pd.DataFrame({"x": [1]}),
            "valid": pd.DataFrame({"x": [2]}),
            "test": pd.DataFrame({"x": [3]}),
        }
        splits = normalize_split_keys(raw)
        self.assertEqual(list(splits), ["train", "valid", "test"])
        self.assertEqual(len(splits["valid"]), 1)

    def test_normalize_split_keys_accepts_legacy_val(self):
        raw = {
            "train": pd.DataFrame({"x": [1]}),
            "val": pd.DataFrame({"x": [2]}),
            "test": pd.DataFrame({"x": [3]}),
        }
        splits = normalize_split_keys(raw)
        self.assertEqual(len(splits["valid"]), 1)

    def test_davis_cold_split_has_no_drug_overlap(self):
        rows = []
        for index in range(60):
            drug = f"drug-{index}"
            for target in ("protein-a", "protein-b"):
                rows.append({"Drug": drug, "Target": target, "Y": index % 2})
        frame = pd.DataFrame(rows)
        splits = _cold_drug_split(frame, seed=7)
        drug_sets = {name: set(part["Drug"]) for name, part in splits.items()}
        self.assertFalse(drug_sets["train"] & drug_sets["valid"])
        self.assertFalse(drug_sets["train"] & drug_sets["test"])
        self.assertFalse(drug_sets["valid"] & drug_sets["test"])


if __name__ == "__main__":
    unittest.main()
