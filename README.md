# Drug-Agent

Drug-Agent is an LLM-guided machine-learning experimentation pipeline for drug discovery. It uses a **Planner–Instructor workflow** to explore supported model configurations, run molecular or drug–target prediction experiments, and compare candidates using validation results.

The project is inspired by the research paper *DrugAgent: Automating AI-aided Drug Discovery Programming through LLM Multi-Agent Collaboration* (Liu et al., 2025). This repository is a lightweight, research-oriented implementation of selected ideas from that work. It is **not a full reproduction of the paper**, and it is not intended for clinical use or production drug-discovery decisions.

## What it does

For a selected benchmark task, Drug-Agent:

1. Loads the benchmark dataset through Therapeutics Data Commons (TDC) using PyTDC.
2. Creates task-appropriate training, validation, and test splits.
3. Uses Gemini to suggest candidate machine-learning configurations from a defined, supported search space.
4. Performs dataset and domain checks before running an experiment.
5. Converts molecular SMILES into 2048-bit, radius-2 Morgan fingerprints using RDKit.
6. For DAVIS, also converts protein sequences into 343-dimensional Conjoint Triad descriptors.
7. Trains and evaluates supported scikit-learn classifiers.
8. Feeds experiment outcomes back to the Planner for another exploration round.
9. Selects a candidate using validation ROC-AUC, evaluates the selected configuration on the held-out test set once, and saves a JSON report.

## Project Structure

```text
Drug-Agent/
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── data/
│   └── README.md                # Dataset source and download notes
├── docs/
│   └── implementation-notes.md  # Design and evaluation notes
├── results/                     # Generated experiment reports
├── scripts/
│   ├── check_dataset_files.py   # Checks legacy duplicate CSV files
│   └── download_pampa.py        # Downloads and inspects a TDC dataset
├── src/
│   ├── dataset.py               # TDC loading, labels, and data splits
│   ├── fingerprinting.py        # Molecular and protein featurization
│   ├── llm_planner.py           # Gemini-based experiment planning
│   ├── llm_instructor.py        # Domain checks, model execution, evaluation
│   ├── run_agent.py             # Planner–Instructor orchestration
│   └── drug_agent.py            # Backward-compatible entry point
└── tests/                       # Unit tests
```

Earlier experiment folders and prototypes from the original repository may also remain in the project for reference. They are not used by the current main pipeline.

## Setup

The following instructions are for macOS and Linux.

1. Clone the repository:

   ```bash
   git clone https://github.com/aakankshak1/Drug-Agent.git
   cd Drug-Agent
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

The first dataset load requires an internet connection. PyTDC downloads and caches the selected benchmark data; the benchmark CSV files are not bundled with this repository.

## Configuration

Drug-Agent uses the Gemini API for Planner generation and feedback. Set an API key in the terminal session before running an experiment:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

You can optionally select a Gemini model through the `DRUGAGENT_LLM_MODEL` environment variable:

```bash
export DRUGAGENT_LLM_MODEL="your-supported-gemini-model"
```

Keep API keys private. Do not commit them to GitHub or place them directly in source files.

## Usage

Run an experiment using the main entry point:

```bash
python main.py --dataset HIV --k 3 --rounds 2
```

The supported datasets are `HIV`, `PAMPA`, and `DAVIS`:

```bash
python main.py --dataset HIV --k 3 --rounds 2
python main.py --dataset PAMPA --k 3 --rounds 2
python main.py --dataset DAVIS --k 3 --rounds 2
```

You can also run the orchestration module directly:

```bash
python src/run_agent.py --dataset HIV --k 3 --rounds 2
```

### Command-line options

| Option | Description | Default |
|---|---|---:|
| `--dataset` | Benchmark to run: `HIV`, `PAMPA`, or `DAVIS` | `HIV` |
| `--k` | Number of new candidate configurations requested per round | `3` |
| `--rounds` | Number of Planner–Instructor exploration rounds | `2` |
| `--seed` | Random seed used for data splitting | `42` |

The Planner is restricted to implemented models and validated hyperparameter values. It cannot introduce arbitrary algorithms or generate and execute unrestricted Python code.

## Datasets

| CLI name | TDC dataset | Task | Prediction target |
|---|---|---|---|
| `HIV` | `HIV` | High-throughput screening (HTS) | HIV replication-inhibition activity from molecular SMILES |
| `PAMPA` | `PAMPA_NCATS` | ADME | High vs. low/moderate membrane permeability from molecular SMILES |
| `DAVIS` | `DAVIS` | Drug–target interaction (DTI) | Binding interaction from drug SMILES and protein sequence |

### Data splitting and labels

- **HIV and PAMPA:** use TDC scaffold splits with approximately 70% training, 10% validation, and 20% test data.
- **DAVIS:** uses drug-grouped, stratified splits so the same drug is not shared across training, validation, and test partitions. When the source labels are binding affinities in nM, values below 30 nM are treated as positive binders.

The pipeline retrieves datasets from TDC rather than relying on the legacy CSV files previously committed under `data/`.

## Architecture

The current pipeline has two main components:

- **Planner (`src/llm_planner.py`):** asks Gemini to propose candidate model and hyperparameter configurations, validates the suggestions against the supported search space, and uses experiment feedback to propose further candidates.
- **Instructor (`src/llm_instructor.py`):** inspects dataset structure, performs domain-oriented checks, creates features, trains supported models, evaluates candidates, and returns structured reports.

The orchestration layer (`src/run_agent.py`) runs multiple rounds, tracks evaluated candidates, selects the best successful candidate using validation ROC-AUC, and performs the final test evaluation.

## Models and featurization

The supported classifiers are:

- Random Forest
- Logistic Regression

Feature representations:

- **HIV and PAMPA:** 2048-bit radius-2 Morgan fingerprints (ECFP4-style) generated from SMILES using RDKit.
- **DAVIS:** the drug Morgan fingerprint is concatenated with a 343-dimensional normalized Conjoint Triad descriptor generated from the protein sequence.

The available models and their hyperparameters are deliberately bounded and validated before execution.

## Evaluation

Candidate configurations are trained on the training split and compared using **validation ROC-AUC**. The test split is not used to select or revise candidates.

After selection, the chosen configuration is retrained on the combined training and validation data and evaluated on the held-out test set once. Reports include ROC-AUC, average precision, accuracy, balanced accuracy, precision, recall, F1, row counts, and counts of invalid input rows where applicable.

Results are saved as timestamped JSON files in `results/`. These are experimental benchmark results, not evidence of clinical efficacy, drug safety, or readiness for real-world deployment.

## Tests

Run the local unit tests with:

```bash
python -m unittest discover -s tests -v
```

The unit tests cover local components such as data handling, feature generation, planning validation, and model evaluation. They do not replace an end-to-end run with live Gemini API access and TDC dataset downloads.

## Legacy dataset-file check

The original repository snapshot contained `HIV.csv`, `PAMPA.csv`, and `DAVIS.csv` files with identical contents. The current pipeline loads the named benchmarks through TDC and does not use those local CSVs.

To check whether an existing clone still contains identical copies:

```bash
python scripts/check_dataset_files.py
```

The optional cleanup command removes the three named files only if the script verifies that all three exist and have identical SHA-256 hashes:

```bash
python scripts/check_dataset_files.py --delete-identical
```

Review the check output before running the cleanup command.

## Background

The original DrugAgent paper presents a multi-agent approach to automating AI-aided drug-discovery programming. A Planner proposes and refines experiment ideas, while an Instructor uses domain knowledge and tools to implement and evaluate them.

This project adapts that high-level Planner–Instructor pattern into a smaller, constrained ML experimentation workflow. Gemini proposes configurations, while the Python Instructor executes a fixed set of implemented operations. The current version does not dynamically construct domain tools, retrieve documentation through RAG, or generate and execute arbitrary code. It also uses a narrower model and featurization set than the research system.

**Reference**

Liu, S., Lu, Y., Chen, S., Hu, X., Zhao, J., Lu, Y., & Zhao, Y. (2025). *DrugAgent: Automating AI-aided Drug Discovery Programming through LLM Multi-Agent Collaboration.* arXiv:2411.15692v2.

- Paper: https://arxiv.org/abs/2411.15692
- Therapeutics Data Commons: https://tdcommons.ai/
- PyTDC: https://pytdc.arcell.ai/
- RDKit documentation: https://www.rdkit.org/docs/

## Notes

Drug-Agent is an educational and research-oriented prototype. Benchmark performance depends on dataset versions, split composition, feature handling, model configuration, and runtime environment. Validate results carefully and keep a human in the loop for any scientific interpretation.
