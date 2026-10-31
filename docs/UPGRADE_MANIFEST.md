# Upgrade manifest

This is an overlay archive for the existing `aakankshak1/Drug-Agent` repository.
Extract it into the repository root; unchanged files are intentionally omitted.

## Replaced

- `README.md`
- `requirements.txt`
- `main.py`
- `src/dataset.py`
- `src/fingerprinting.py`
- `src/llm_planner.py`
- `src/llm_instructor.py`
- `src/run_agent.py`
- `src/drug_agent.py`
- `scripts/download_pampa.py`

## Added

- `INSTALL_UPGRADE.md`
- `data/README.md`
- `docs/implementation-notes.md`
- `scripts/check_dataset_files.py`
- `src/__init__.py`
- `tests/test_dataset.py`
- `tests/test_fingerprinting.py`
- `tests/test_instructor.py`
- `tests/test_planner.py`
- `results/.gitkeep`
- `.gitignore`

## Important migration note

The original repository snapshot had identical Git blob hashes and sizes for
`data/HIV.csv`, `data/PAMPA.csv`, and `data/DAVIS.csv`. The new implementation
loads named datasets through PyTDC instead. The included cleanup script can
remove those three files from an existing clone only after verifying that all
three are byte-for-byte identical.

## Validation performed for this archive

- Python compilation: passed.
- Unit tests: 14 passed.
- Synthetic ML fit/evaluation smoke tests: passed.
- Live TDC downloads and live Gemini API calls: not run in this environment.
