import unittest

import pandas as pd

from src.llm_instructor import LLMInstructor


class InstructorTests(unittest.TestCase):
    def setUp(self):
        self.instructor = LLMInstructor(client=object())

    @staticmethod
    def _rows(smiles, start=0):
        return pd.DataFrame({
            "Drug": smiles,
            "Y": [(i + start) % 2 for i in range(len(smiles))],
        })

    def test_candidate_uses_validation_split(self):
        smiles = [
            "C", "CC", "CCC", "CCCC", "CCO", "CCCO", "CCN", "CCCN",
            "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1",
            "CC(=O)O", "CCC(=O)O", "CCOC", "CCSC", "COC", "CCN(CC)CC",
            "C1CCCCC1", "C1CCCCC1O", "CCCl", "CCBr", "CCF", "CC=O",
        ]
        splits = {
            "train": self._rows(smiles),
            "valid": self._rows(smiles[:12], start=1),
            "test": self._rows(smiles[12:], start=1),
        }
        idea = {
            "model": "Logistic Regression",
            "featurization": "ECFP4_Morgan",
            "hyperparameters": {"C": 1.0, "class_weight": "balanced"},
        }
        report = self.instructor.evaluate_candidate("HIV", splits, idea)
        self.assertTrue(report["success"], report)
        self.assertIsNotNone(report["validation_roc_auc"])
        self.assertEqual(report["feature_count"], 2048)

    def test_final_evaluation_runs_on_test_after_selection(self):
        smiles = [
            "C", "CC", "CCC", "CCCC", "CCO", "CCCO", "CCN", "CCCN",
            "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1",
            "CC(=O)O", "CCC(=O)O", "CCOC", "CCSC", "COC", "CCN(CC)CC",
            "C1CCCCC1", "C1CCCCC1O", "CCCl", "CCBr", "CCF", "CC=O",
        ]
        splits = {
            "train": self._rows(smiles[:16]),
            "valid": self._rows(smiles[16:20], start=1),
            "test": self._rows(smiles[20:], start=1),
        }
        idea = {
            "model": "Random Forest",
            "featurization": "ECFP4_Morgan",
            "hyperparameters": {
                "n_estimators": 20,
                "max_features": "sqrt",
                "min_samples_leaf": 1,
                "class_weight": "balanced",
            },
        }
        report = self.instructor.final_evaluate("HIV", splits, idea)
        self.assertEqual(report["feature_count"], 2048)
        self.assertEqual(report["test_rows_used"], 4)
        self.assertIn("roc_auc", report["test_metrics"])

    def test_davis_feature_vector_combines_drug_and_protein(self):
        frame = pd.DataFrame({
            "Drug": ["CCO", "CCN", "c1ccccc1", "CC(=O)O"],
            "Target": ["AGVILFPYMTSHNQW", "AGVILFPYMTSHNQW", "RKDECGAVILFPY", "AGVILFPYMTSHNQW"],
            "Y": [0, 1, 0, 1],
        })
        features, labels, cleaning = self.instructor._featurize_frame(frame, "DAVIS", "ECFP4_plus_CT")
        self.assertEqual(features.shape, (4, 2391))
        self.assertEqual(labels.tolist(), [0, 1, 0, 1])
        self.assertEqual(cleaning["invalid_smiles"], 0)


if __name__ == "__main__":
    unittest.main()
