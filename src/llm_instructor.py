"""
LLM Instructor: domain-specific knowledge check + execution.

Mirrors DrugAgent's Instructor role (Section 2, "LLM Instructor:
Domain-specific Knowledge and Tool Preparation"): before running an
idea, it asks the LLM to flag domain-specific pitfalls for this
dataset/featurization combo, then executes the actual training code
and returns a success/failure report the Planner can use to revise
the idea space.

The model-fitting itself intentionally stays as real, deterministic
sklearn/RDKit code rather than LLM-generated code -- the LLM's job
here is domain reasoning and reporting, not writing the training loop.
"""

import os

import numpy as np
import pandas as pd
from anthropic import Anthropic
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DEFAULT_MODEL = os.environ.get("DRUGAGENT_LLM_MODEL", "claude-sonnet-5")

# Candidate column names, since HIV/PAMPA/DAVIS don't share a schema.
SMILES_COLUMNS = ["smiles", "SMILES", "Drug"]
LABEL_COLUMNS = ["HIV_active", "Y", "label", "Label"]


class LLMInstructor:
    def __init__(self, model: str = DEFAULT_MODEL, client: Anthropic | None = None):
        self.model = model
        self.client = client or Anthropic()

    def _call(self, prompt: str, max_tokens: int = 512) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    def check_domain_knowledge(self, dataset_name: str, idea: dict) -> str:
        """
        Ask the LLM to flag anything domain-specific that could break this
        idea before we spend time running it (e.g. DAVIS needing protein
        features, not just drug SMILES).
        """
        prompt = f"""You are the Instructor agent in a drug-discovery ML pipeline,
about to implement this idea:

Dataset: {dataset_name}
Model: {idea['model']}
Featurization: {idea['featurization']}

In 1-3 sentences, flag any domain-specific risk with this plan (e.g.
missing protein/target features for a drug-target interaction dataset,
invalid SMILES handling, class imbalance). If there is no significant
risk, say so briefly.
"""
        return self._call(prompt, max_tokens=200).strip()

    def implement_and_run(self, dataset_name: str, idea: dict) -> dict:
        """
        Execute the idea and return a success/failure report:
          {"success": bool, "accuracy": float | None, "message": str}
        """
        try:
            df = self._load_dataset(dataset_name)
            smiles_col = self._find_column(df, SMILES_COLUMNS)
            label_col = self._find_column(df, LABEL_COLUMNS)

            X, y = self._featurize(df, smiles_col, label_col, idea["featurization"])

            if len(set(y)) < 2:
                return {
                    "success": False,
                    "accuracy": None,
                    "message": "Only one class present after featurization; cannot train.",
                }

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model = self._build_model(idea["model"])
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)

            return {
                "success": True,
                "accuracy": auc,
                "message": f"Trained {idea['model']} on {len(X)} valid molecules "
                f"from {dataset_name} using {smiles_col}/{label_col}.",
            }

        except Exception as exc:  # noqa: BLE001 - report failures back to the Planner
            return {"success": False, "accuracy": None, "message": f"{type(exc).__name__}: {exc}"}

    # -- helpers -----------------------------------------------------

    @staticmethod
    def _load_dataset(dataset_name: str) -> pd.DataFrame:
        path = f"data/{dataset_name}.csv"
        return pd.read_csv(path)

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")

    @staticmethod
    def _featurize(df: pd.DataFrame, smiles_col: str, label_col: str, method: str):
        if method != "ECFP4_morgan":
            raise ValueError(f"Unsupported featurization: {method}")

        X, y = [], []
        for smiles, label in zip(df[smiles_col], df[label_col]):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            X.append(np.array(fp))
            y.append(label)

        return np.array(X), np.array(y)

    @staticmethod
    def _build_model(name: str):
        if name == "Random Forest":
            return RandomForestClassifier(random_state=42)
        if name == "Logistic Regression":
            return LogisticRegression(max_iter=1000)
        raise ValueError(f"Unsupported model: {name}")
