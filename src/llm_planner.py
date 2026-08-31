"""
LLM Planner: manages the idea space for a drug discovery task.

This replaces the old hardcoded if/elif PlannerAgent with a real
LLM-driven planner, following DrugAgent's two-phase design
(arXiv:2411.15692, Section 2 - "LLM Planner: Idea Space Management"):

  1. Idea Generation: from a task description, derive K candidate ideas.
  2. Exploration: after each idea is tried by the Instructor, revise the
     idea set using the success/failure report.

Requires a GEMINI_API_KEY environment variable (free tier available at
https://aistudio.google.com/apikey -- Flash and Flash-Lite models are
free, no credit card required).
"""

import json
import os

from google import genai

# Flash is free-tier eligible and plenty capable for these short,
# structured prompts. Swap to "gemini-2.5-flash-lite" for an even
# cheaper/faster free-tier option if you hit rate limits.
DEFAULT_MODEL = os.environ.get("DRUGAGENT_LLM_MODEL", "gemini-2.5-flash")

# What the Instructor actually knows how to execute right now.
# Kept explicit so the Planner can't propose something the codebase
# can't run yet.
SUPPORTED_FEATURIZATIONS = ["ECFP4_morgan"]
SUPPORTED_MODELS = ["Random Forest", "Logistic Regression"]

TASK_DESCRIPTIONS = {
    "HIV": (
        "High-throughput screening (HTS) task: predict whether a small "
        "molecule (given as a SMILES string) is active against HIV "
        "replication inhibition. Binary classification."
    ),
    "PAMPA": (
        "ADMET task: predict membrane permeability (PAMPA assay) of a "
        "small molecule from its SMILES string. Binary classification "
        "(high vs. low/moderate permeability)."
    ),
    "DAVIS": (
        "Drug-target interaction (DTI) task: predict binding between a "
        "drug (SMILES) and a protein (amino acid sequence). Binary "
        "classification on binding affinity threshold."
    ),
}


class LLMPlanner:
    """LLM-backed idea space manager."""

    def __init__(self, model: str = DEFAULT_MODEL, client: "genai.Client | None" = None):
        self.model = model
        # genai.Client() reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the
        # environment automatically if api_key isn't passed explicitly.
        self.client = client or genai.Client()

    def _call(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"max_output_tokens": max_tokens},
        )
        return response.text or ""

    def generate_ideas(self, dataset_name: str, k: int = 3) -> list[dict]:
        """Idea Generation phase: derive K candidate solution ideas."""

        task_description = TASK_DESCRIPTIONS.get(
            dataset_name, f"Binary classification task on the {dataset_name} dataset."
        )

        prompt = f"""You are the Planner agent in a drug-discovery ML pipeline.

Task: {task_description}

Constraints: you may only choose a model from {SUPPORTED_MODELS} and a
featurization from {SUPPORTED_FEATURIZATIONS} (Morgan/ECFP4 fingerprints
on the SMILES string). Do not propose anything outside these lists.

Generate {k} distinct candidate ideas for solving this task. Vary the
model choice and briefly justify each idea using domain knowledge about
the task (e.g. why a tree-based model might suit a sparse fingerprint
space, or why linear models suit smaller datasets).

Respond with ONLY a JSON array, no other text, in this exact format:
[
  {{"model": "<one of {SUPPORTED_MODELS}>", "featurization": "<one of {SUPPORTED_FEATURIZATIONS}>", "rationale": "<1-2 sentences>"}}
]
"""
        raw = self._call(prompt,max_tokens=2048)
        ideas = self._extract_json_array(raw)

        # Guard against the LLM proposing something the Instructor can't run.
        validated = []
        for idea in ideas:
            if idea.get("model") in SUPPORTED_MODELS and idea.get(
                "featurization"
            ) in SUPPORTED_FEATURIZATIONS:
                validated.append(idea)

        if not validated:
            raise ValueError(f"Planner returned no valid ideas. Raw response: {raw!r}")

        return validated[:k]

    def revise_ideas(self, dataset_name: str, explored: list[dict]) -> dict:
        """
        Exploration phase: given ideas that have each been tried and
        returned a success/failure report, pick the best one and explain
        why, the way the paper's Planner selects a final submission.

        `explored` is a list of dicts like:
          {"idea": {...}, "report": {"success": bool, "accuracy": float|None, "message": str}}
        """
        summary_lines = []
        for i, item in enumerate(explored):
            idea, report = item["idea"], item["report"]
            status = "SUCCESS" if report["success"] else "FAILURE"
            acc = f"{report['accuracy']:.4f}" if report.get("accuracy") is not None else "N/A"
            summary_lines.append(
                f"{i+1}. model={idea['model']}, featurization={idea['featurization']} "
                f"-> {status}, accuracy={acc}, note: {report['message']}"
            )
        summary = "\n".join(summary_lines)

        prompt = f"""You are the Planner agent reviewing exploration results for the
{dataset_name} task.

Results from this round:
{summary}

Pick the single best idea to submit as the final answer. Respond with
ONLY a JSON object, no other text, in this exact format:
{{"chosen_index": <1-based index from the list above>, "reasoning": "<1-2 sentences>"}}
"""
        raw = self._call(prompt, max_tokens=1024)
        decision = self._extract_json_object(raw)

        idx = decision.get("chosen_index", 1) - 1
        idx = max(0, min(idx, len(explored) - 1))

        return {
            "chosen": explored[idx],
            "reasoning": decision.get("reasoning", ""),
        }

    @staticmethod
    def _extract_json_array(text: str) -> list[dict]:
        text = text.strip()
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON array found in Planner response: {text!r}")
        return json.loads(text[start : end + 1])

    @staticmethod
    def _extract_json_object(text: str) -> dict:
        text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in Planner response: {text!r}")
        return json.loads(text[start : end + 1])
