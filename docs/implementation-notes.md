# Implementation notes

## Dataset integrity finding

The original GitHub tree reported exactly the same Git blob SHA and byte size for
`data/HIV.csv`, `data/PAMPA.csv`, and `data/DAVIS.csv`. Git blob hashes uniquely
identify file contents; identical hashes indicate the three tracked paths held
identical bytes. The updated loader therefore downloads the named datasets from
TDC instead of trusting these local files.

## DTI / DAVIS feature representation

The DAVIS task includes a drug and a protein. The implementation concatenates:

- 2048-bit radius-2 Morgan fingerprint for the drug SMILES;
- 343-dimensional normalized Conjoint Triad descriptor for the protein sequence.

This gives a 2391-dimensional baseline feature vector per drug-target pair.

## Validation and test isolation

Every candidate is fit only on `train` and compared on `valid`. The planner's
choice is based on validation ROC-AUC. The selected hyperparameter configuration
is then fit on `train + valid`, and `test` is evaluated once. Do not use the test
score to revise the candidate space.

## Honest scope boundary

The Instructor performs real domain-specific operations (schema inspection,
chemical parsing, molecular/protein featurization, training, metrics, and
failure reporting) but does not dynamically write Python source code. That is a
concrete and testable executor, not a full reproduction of the paper's arbitrary
code-generation and dynamic domain-tool-construction capabilities.
