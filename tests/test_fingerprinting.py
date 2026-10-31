import unittest

import numpy as np

from src.fingerprinting import conjoint_triad, morgan_fingerprint


class FingerprintingTests(unittest.TestCase):
    def test_morgan_fingerprint_shape(self):
        vector = morgan_fingerprint("CCO")
        self.assertIsNotNone(vector)
        self.assertEqual(vector.shape, (2048,))
        self.assertTrue(set(np.unique(vector)).issubset({0.0, 1.0}))

    def test_invalid_smiles_is_skipped(self):
        self.assertIsNone(morgan_fingerprint("not-a-smiles"))

    def test_conjoint_triad_shape_and_normalization(self):
        vector = conjoint_triad("AGVILFP")
        self.assertIsNotNone(vector)
        self.assertEqual(vector.shape, (343,))
        self.assertAlmostEqual(float(vector.sum()), 1.0, places=6)

    def test_conjoint_triad_rejects_too_short_sequence(self):
        self.assertIsNone(conjoint_triad("AG"))


if __name__ == "__main__":
    unittest.main()
