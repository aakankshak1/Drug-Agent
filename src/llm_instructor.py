"""Domain-aware executor for DrugAgent's bounded ML experiments.

Gemini performs a domain-risk review. Python/RDKit/scikit-learn perform all
featurization, training, and evaluation deterministically; the LLM does not
write executable training code.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
try:
    from google import genai
except ImportError:  # allows local unit tests without the optional LLM SDK
    genai = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .fingerprinting import conjoint_triad, morgan_fingerprint
    from .llm_planner import DEFAULT_MODEL
except ImportError:  # supports direct script execution from src/
    from fingerprinting import conjoint_triad, morgan_fingerprint
    from llm_planner import DEFAULT_MODEL

SMILES_COLUMNS = ("Drug", "SMILES", "smiles", "drug", "compound")
PROTEIN_COLUMNS = ("Target", "Protein", "Sequence", "protein_sequence", "target_sequence")
LABEL_COLUMNS = ("Y", "label", "Label", "HIV_active", "activity")


class LLMInstructor:
    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        self.model = model
        if client is not None:
            self.client = client
        elif genai is not None:
            self.client = genai.Client()
        else:
            raise RuntimeError("google-genai is not installed. Run `pip install -r requirements.txt`.")

    def _call(self, prompt: str, max_tokens: int = 400) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"max_output_tokens": max_tokens, "temperature": 0.1},
        )
        return (response.text or "").strip()

    @staticmethod
    def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
        lookup = {str(column).strip().lower(): column for column in frame.columns}
        for candidate in candidates:
            if candidate.lower() in lookup:
                return lookup[candidate.lower()]
        raise KeyError(f"None of {candidates} found in columns {list(frame.columns)}")

    def inspect_dataset(self, dataset_name: str, splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Run deterministic schema and label checks before model training."""
        train = splits["train"]
        smiles_column = self._find_column(train, SMILES_COLUMNS)
        label_column = self._find_column(train, LABEL_COLUMNS)
        metadata = {
            "dataset": dataset_name,
            "split_rows": {key: int(len(value)) for key, value in splits.items()},
            "smiles_column": str(smiles_column),
            "label_column": str(label_column),
            "label_distribution": train[label_column].astype(str).value_counts().to_dict(),
        }
        if dataset_name.upper() == "DAVIS":
            protein_column = self._find_column(train, PROTEIN_COLUMNS)
            metadata["protein_column"] = str(protein_column)
        return metadata

    def check_domain_knowledge(
        self,
        dataset_name: str,
        idea: dict[str, Any],
        inspection: dict[str, Any],
    ) -> str:
        task_context = {
            "HIV": "HTS binary classification: molecule SMILES to HIV replication-inhibition activity.",
            "PAMPA": "ADME binary classification: molecule SMILES to high vs low/moderate membrane permeability.",
            "DAVIS": "DTI binary classification: drug SMILES plus protein sequence to binding interaction; positive label uses the 30 nM cutoff.",
        }.get(dataset_name.upper(), dataset_name)
        prompt = f"""You are the domain-review component of a drug-discovery ML system.
Task: {task_context}
Candidate: {idea}
Observed dataset metadata: {inspection}

Flag concrete chemistry/biology/data risks in no more than 3 sentences. For
DAVIS, verify that both drug and protein features are included conceptually.
Do not claim to have executed code, inspected external literature, or measured
performance. If no additional concern is evident, say so.
"""
        try:
            return self._call(prompt, max_tokens=300)
        except Exception as exc:
            return f"LLM domain review unavailable ({type(exc).__name__}); deterministic schema checks still apply."

    @staticmethod
    def _featurize_frame(
        frame: pd.DataFrame,
        dataset_name: str,
        featurization: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        smiles_column = LLMInstructor._find_column(frame, SMILES_COLUMNS)
        label_column = LLMInstructor._find_column(frame, LABEL_COLUMNS)
        is_dti = dataset_name.upper() == "DAVIS"
        if is_dti and featurization != "ECFP4_plus_CT":
            raise ValueError("DAVIS requires ECFP4_plus_CT (drug Morgan + protein Conjoint Triad).")
        if not is_dti and featurization != "ECFP4_Morgan":
            raise ValueError(f"Unsupported featurization for {dataset_name}: {featurization}")
        protein_column = LLMInstructor._find_column(frame, PROTEIN_COLUMNS) if is_dti else None

        features: list[np.ndarray] = []
        labels: list[int] = []
        invalid_smiles = 0
        invalid_proteins = 0
        invalid_labels = 0

        for _, row in frame.iterrows():
            label_value = pd.to_numeric(pd.Series([row[label_column]]), errors="coerce").iloc[0]
            if pd.isna(label_value) or label_value not in (0, 1):
                invalid_labels += 1
                continue

            drug_features = morgan_fingerprint(str(row[smiles_column]))
            if drug_features is None:
                invalid_smiles += 1
                continue

            if is_dti:
                protein_features = conjoint_triad(str(row[protein_column]))
                if protein_features is None:
                    invalid_proteins += 1
                    continue
                vector = np.concatenate((drug_features, protein_features)).astype(np.float32, copy=False)
            else:
                vector = drug_features

            features.append(vector)
            labels.append(int(label_value))

        if not features:
            raise ValueError("No valid rows remained after molecular/protein featurization.")
        return np.vstack(features), np.asarray(labels, dtype=np.int64), {
            "invalid_smiles": invalid_smiles,
            "invalid_proteins": invalid_proteins,
            "invalid_labels": invalid_labels,
        }

    @staticmethod
    def _build_model(idea: dict[str, Any]):
        model_name = idea["model"]
        params = idea.get("hyperparameters", {})
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 200)),
                max_features=params.get("max_features", "sqrt"),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                class_weight=params.get("class_weight", "balanced"),
                random_state=42,
                n_jobs=-1,
            )
        if model_name == "Logistic Regression":
            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                class_weight=params.get("class_weight", "balanced"),
                max_iter=2000,
                solver="liblinear",
                random_state=42,
            )
        raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def _positive_probabilities(model, features: np.ndarray) -> np.ndarray:
        probabilities = model.predict_proba(features)
        positive_indices = np.flatnonzero(model.classes_ == 1)
        if len(positive_indices) != 1:
            raise ValueError("Trained classifier does not expose a positive class (1).")
        return probabilities[:, int(positive_indices[0])]

    @staticmethod
    def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
        predictions = (probabilities >= 0.5).astype(int)
        has_both_classes = len(np.unique(labels)) == 2
        return {
            "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both_classes else None,
            "pr_auc_average_precision": float(average_precision_score(labels, probabilities)),
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
        }

    def evaluate_candidate(
        self,
        dataset_name: str,
        splits: dict[str, pd.DataFrame],
        idea: dict[str, Any],
    ) -> dict[str, Any]:
        """Train on train only and score on validation only (never on test)."""
        try:
            x_train, y_train, train_cleaning = self._featurize_frame(
                splits["train"], dataset_name, idea["featurization"]
            )
            x_valid, y_valid, valid_cleaning = self._featurize_frame(
                splits["valid"], dataset_name, idea["featurization"]
            )
            if len(np.unique(y_train)) < 2:
                raise ValueError("Training split contains only one class after featurization.")
            if len(np.unique(y_valid)) < 2:
                raise ValueError("Validation split contains only one class; ROC-AUC cannot be computed.")

            model = self._build_model(idea)
            model.fit(x_train, y_train)
            probabilities = self._positive_probabilities(model, x_valid)
            metrics = self._metrics(y_valid, probabilities)
            return {
                "success": True,
                "validation_roc_auc": metrics["roc_auc"],
                "validation_metrics": metrics,
                "train_rows_used": int(len(y_train)),
                "validation_rows_used": int(len(y_valid)),
                "feature_count": int(x_train.shape[1]),
                "cleaning": {"train": train_cleaning, "valid": valid_cleaning},
                "message": "Trained on train split and evaluated on validation split.",
            }
        except Exception as exc:
            return {
                "success": False,
                "validation_roc_auc": None,
                "validation_metrics": None,
                "message": f"{type(exc).__name__}: {exc}",
            }

    def final_evaluate(
        self,
        dataset_name: str,
        splits: dict[str, pd.DataFrame],
        chosen_idea: dict[str, Any],
    ) -> dict[str, Any]:
        """Refit selected configuration on train+validation; evaluate test once."""
        combined_train = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
        x_train, y_train, train_cleaning = self._featurize_frame(
            combined_train, dataset_name, chosen_idea["featurization"]
        )
        x_test, y_test, test_cleaning = self._featurize_frame(
            splits["test"], dataset_name, chosen_idea["featurization"]
        )
        if len(np.unique(y_train)) < 2:
            raise ValueError("Combined training data contains only one class.")

        model = self._build_model(chosen_idea)
        model.fit(x_train, y_train)
        probabilities = self._positive_probabilities(model, x_test)
        metrics = self._metrics(y_test, probabilities)
        return {
            "test_metrics": metrics,
            "train_rows_used": int(len(y_train)),
            "test_rows_used": int(len(y_test)),
            "feature_count": int(x_train.shape[1]),
            "cleaning": {"train_plus_valid": train_cleaning, "test": test_cleaning},
            "note": "Test split evaluated once after configuration selection on validation data.",
        }
