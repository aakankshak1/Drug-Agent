"""Gemini-backed Planner for bounded, iterative ML experiment search."""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from google import genai
except ImportError:  # allows local unit tests without the optional LLM SDK
    genai = None

DEFAULT_MODEL = os.environ.get("DRUGAGENT_LLM_MODEL", "gemini-3.8-flash")
SUPPORTED_MODELS = ("Random Forest", "Logistic Regression")

# These are the only parameter values the Instructor is allowed to execute.
# The LLM proposes combinations; Python validates them before any training.
MODEL_PARAMETER_SPACE = {
    "Random Forest": {
        "n_estimators": (200, 400),
        "max_features": ("sqrt", 0.5),
        "min_samples_leaf": (1, 2, 4),
        "class_weight": ("balanced", "balanced_subsample"),
    },
    "Logistic Regression": {
        "C": (0.1, 1.0, 10.0),
        "class_weight": ("balanced", None),
    },
}

DEFAULT_PARAMETERS = {
    "Random Forest": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": "balanced",
    },
    "Logistic Regression": {"C": 1.0, "class_weight": "balanced"},
}


class LLMPlanner:
    """Generate constrained experiment ideas and refine the search from feedback."""

    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        self.model = model
        if client is not None:
            self.client = client
        elif genai is not None:
            self.client = genai.Client()
        else:
            raise RuntimeError("google-genai is not installed. Run `pip install -r requirements.txt`.")

    def _call(self, prompt: str, max_tokens: int = 2048) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"max_output_tokens": max_tokens, "temperature": 0.2},
        )
        return (response.text or "").strip()

    @staticmethod
    def _extract_json_array(raw: str) -> list[dict[str, Any]]:
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end < start:
            raise ValueError("Planner response did not contain a JSON array.")
        parsed = json.loads(raw[start : end + 1])
        if not isinstance(parsed, list):
            raise ValueError("Planner JSON response must be an array.")
        return parsed

    @staticmethod
    def _extract_json_object(raw: str) -> dict[str, Any]:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Planner response did not contain a JSON object.")
        parsed = json.loads(raw[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Planner JSON response must be an object.")
        return parsed

    @staticmethod
    def _allowed_featurization(dataset_name: str) -> str:
        return "ECFP4_plus_CT" if dataset_name.upper() == "DAVIS" else "ECFP4_Morgan"

    @staticmethod
    def _validate_idea(raw_idea: dict[str, Any], dataset_name: str) -> dict[str, Any] | None:
        if not isinstance(raw_idea, dict):
            return None
        model = raw_idea.get("model")
        if model not in SUPPORTED_MODELS:
            return None

        raw_params = raw_idea.get("hyperparameters", {})
        if not isinstance(raw_params, dict):
            return None
        allowed = MODEL_PARAMETER_SPACE[model]
        params = dict(DEFAULT_PARAMETERS[model])
        for key, value in raw_params.items():
            if key not in allowed or value not in allowed[key]:
                return None
            params[key] = value

        return {
            "model": model,
            "featurization": LLMPlanner._allowed_featurization(dataset_name),
            "hyperparameters": params,
            "rationale": str(raw_idea.get("rationale", "No rationale supplied."))[:500],
        }

    @staticmethod
    def signature(idea: dict[str, Any]) -> str:
        return json.dumps(
            {"model": idea["model"], "featurization": idea["featurization"], "hyperparameters": idea["hyperparameters"]},
            sort_keys=True,
        )

    def _fallback_ideas(self, dataset_name: str, excluded: set[str], limit: int) -> list[dict[str, Any]]:
        defaults = [
            {"model": "Random Forest", "hyperparameters": {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 1, "class_weight": "balanced"}},
            {"model": "Random Forest", "hyperparameters": {"n_estimators": 400, "max_features": "sqrt", "min_samples_leaf": 2, "class_weight": "balanced_subsample"}},
            {"model": "Random Forest", "hyperparameters": {"n_estimators": 200, "max_features": 0.5, "min_samples_leaf": 4, "class_weight": "balanced"}},
            {"model": "Logistic Regression", "hyperparameters": {"C": 0.1, "class_weight": "balanced"}},
            {"model": "Logistic Regression", "hyperparameters": {"C": 1.0, "class_weight": "balanced"}},
            {"model": "Logistic Regression", "hyperparameters": {"C": 10.0, "class_weight": None}},
        ]
        results = []
        for item in defaults:
            idea = self._validate_idea(item, dataset_name)
            if idea and self.signature(idea) not in excluded:
                idea["rationale"] = "Deterministic fallback candidate from the supported experiment space."
                results.append(idea)
            if len(results) >= limit:
                break
        return results

    def generate_ideas(
        self,
        dataset_name: str,
        k: int = 3,
        feedback: list[dict[str, Any]] | None = None,
        excluded_signatures: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate new candidates; use validated fallback candidates if needed."""
        excluded = set(excluded_signatures or set())
        model_space = json.dumps(MODEL_PARAMETER_SPACE, indent=2)
        feedback_text = json.dumps(feedback or [], indent=2, default=str)[:7000]
        featurization = self._allowed_featurization(dataset_name)
        prompt = f"""You are the Planner in a drug-discovery ML experimentation system.

Dataset: {dataset_name}
Task: {dataset_name} benchmark task. The Instructor uses {featurization} featurization.
Supported model/parameter space (do not invent any value):
{model_space}

Previously tested candidates to avoid:
{json.dumps(sorted(excluded), indent=2)}

Experimental feedback from previous rounds:
{feedback_text}

Generate up to {k} DISTINCT new candidate configurations. Use the feedback to
explore alternatives to weak candidates and refine promising ones. Return ONLY
valid JSON, with this exact structure:
[
  {{"model":"Random Forest","hyperparameters":{{"n_estimators":200,"max_features":"sqrt","min_samples_leaf":1,"class_weight":"balanced"}},"rationale":"..."}}
]
Every parameter value must be selected from the supported space. Do not generate
code, make up metrics, or suggest unimplemented featurization methods.
"""
        candidates = []
        try:
            raw = self._call(prompt, max_tokens=2048)
            raw_ideas = self._extract_json_array(raw)
        except Exception:
            raw_ideas = []

        seen = set(excluded)
        for raw_idea in raw_ideas:
            idea = self._validate_idea(raw_idea, dataset_name)
            if not idea:
                continue
            key = self.signature(idea)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(idea)
            if len(candidates) >= k:
                break

        if len(candidates) < k:
            candidates.extend(self._fallback_ideas(dataset_name, seen, k - len(candidates)))
        return candidates

    def revise_ideas(self, explored: list[dict[str, Any]]) -> dict[str, Any]:
        """Choose from validation results and ask Gemini for a concise explanation.

        The numeric selection is guarded: an LLM response cannot select a failed
        candidate or override a measurably better validation ROC-AUC.
        """
        valid = [
            item for item in explored
            if item.get("report", {}).get("success")
            and item["report"].get("validation_roc_auc") is not None
        ]
        if not valid:
            raise RuntimeError("No candidate completed validation successfully.")
        best = max(valid, key=lambda item: item["report"]["validation_roc_auc"])
        summary = [
            {
                "index": index + 1,
                "model": item["idea"]["model"],
                "hyperparameters": item["idea"]["hyperparameters"],
                "validation_roc_auc": item["report"]["validation_roc_auc"],
                "success": item["report"]["success"],
                "message": item["report"].get("message", ""),
            }
            for index, item in enumerate(explored)
        ]
        prompt = f"""Review these drug-discovery ML experiments:
{json.dumps(summary, indent=2)}

The candidate with the highest validation ROC-AUC is index
{explored.index(best) + 1}. Explain in 1-3 concise sentences why that candidate
is the strongest validation choice. Do not invent results or refer to test data.
Return JSON only: {{"reasoning":"..."}}.
"""
        try:
            reasoning = self._extract_json_object(self._call(prompt, max_tokens=512)).get("reasoning", "")
        except Exception:
            reasoning = "Selected the successful candidate with the highest validation ROC-AUC."
        return {"chosen": best, "reasoning": str(reasoning)[:1000]}
