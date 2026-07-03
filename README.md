# Drug-Agent

Drug-Agent is a small machine learning pipeline for predicting molecular properties
(e.g. HIV inhibition activity, membrane permeability) directly from SMILES strings.
It converts molecules into Morgan fingerprints and trains classical ML models
(Random Forest, Logistic Regression) to make predictions, with a simple planner
component that selects a model based on the dataset provided.

## Project Structure

```
Drug-Agent/
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── data/                    # Datasets (DAVIS, HIV, PAMPA, sample)
├── src/                     # Core pipeline
│   ├── fingerprinting.py    # SMILES -> Morgan fingerprint conversion
│   ├── dataset.py           # Dataset loading + featurization
│   ├── planner_agent.py     # Chooses a model based on dataset name
│   ├── instructor.py        # Trains and evaluates a model
│   └── drug_agent.py        # Combines planner + instructor into one agent
├── scripts/
│   └── download_pampa.py    # Downloads the PAMPA dataset via TDC
└── experiments/             # Earlier prototypes, kept for reference
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/aakankshak1/Drug-Agent.git
   cd Drug-Agent
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

Run the full agent, which will ask which dataset to use and pick a model
accordingly (Random Forest or Logistic Regression):

```bash
python src/drug_agent.py
```

To re-download the PAMPA dataset:

```bash
python scripts/download_pampa.py
```

Earlier prototypes and one-off experiments used during development are kept
in `experiments/` for reference and are not part of the main pipeline.

## Datasets

| File          | Description                                  |
|---------------|-----------------------------------------------|
| `HIV.csv`     | Molecules labeled for HIV inhibition activity |
| `PAMPA.csv`   | Membrane permeability assay data              |
| `DAVIS.csv`   | Drug-target binding affinity data             |
| `sample.csv`  | Small sample dataset for quick testing        |

## Background

This project is a personal recreation of the **DrugAgent** paper, which
proposes a multi-agent LLM framework for automating ML programming in drug
discovery. The paper pairs an LLM **Planner** (generates and refines solution
ideas) with an LLM **Instructor** (implements those ideas with domain-specific
knowledge) to build ADMET, HTS, and DTI prediction pipelines automatically.

This repo follows the same Planner/Instructor structure
(`src/planner_agent.py`, `src/instructor.py`) and the same three tasks/datasets
(ADMET/PAMPA, HTS/HIV, DTI/DAVIS) as the paper, but reimplements them with
classical ML — fingerprinting + Random Forest/Logistic Regression selected
per dataset — rather than the LLM-driven idea generation and domain-tool
retrieval described in the original work. It's a learning project, not an
attempt to reproduce the paper's benchmark results.

> Liu, S., Lu, Y., Chen, S., Hu, X., Zhao, J., Lu, Y., & Zhao, Y. (2025).
> *DrugAgent: Automating AI-aided Drug Discovery Programming through LLM
> Multi-Agent Collaboration.* arXiv:2411.15692.

## Notes

This project is a work in progress. See the "Known Issues" list shared
alongside this README for a few things worth cleaning up in `src/`.
