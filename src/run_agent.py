"""
End-to-end DrugAgent loop: LLM Planner <-> LLM Instructor.

This is the real agent loop the old experiment scripts (04/06/07) and
src/drug_agent.py were standing in for with a hardcoded if/elif table.
It now actually:

  1. Asks the LLM Planner to generate K ideas for the chosen dataset.
  2. Has the LLM Instructor flag domain-specific risks, then executes
     each idea and produces a success/failure report.
  3. Sends the reports back to the Planner, which picks the best idea.

Usage:
    export GEMINI_API_KEY=...   # free key: https://aistudio.google.com/apikey
    python src/run_agent.py --dataset HIV --k 3

Note: DAVIS is a drug-target interaction dataset (drug + protein pair)
per the paper. The current Instructor only featurizes the SMILES
column, so DAVIS runs will report a domain-knowledge risk about the
missing protein features -- that's the Instructor doing its job, not
a bug. Wiring in real protein featurization (e.g. PyBioMed CT
descriptors, as in the paper) is the next step, not part of this fix.
"""

import argparse
import os
import sys

from llm_instructor import LLMInstructor
from llm_planner import LLMPlanner


def run(dataset_name: str, k: int) -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey, then export it:\n"
            "  export GEMINI_API_KEY=..."
        )

    planner = LLMPlanner()
    instructor = LLMInstructor()

    print(f"\n===== Planner: generating {k} ideas for {dataset_name} =====")
    ideas = planner.generate_ideas(dataset_name, k=k)
    for i, idea in enumerate(ideas, 1):
        print(f"  Idea {i}: {idea['model']} + {idea['featurization']}")
        print(f"    rationale: {idea['rationale']}")

    explored = []
    for i, idea in enumerate(ideas, 1):
        print(f"\n===== Instructor: exploring idea {i}/{len(ideas)} =====")

        risk_note = instructor.check_domain_knowledge(dataset_name, idea)
        print(f"  Domain check: {risk_note}")

        report = instructor.implement_and_run(dataset_name, idea)
        status = "SUCCESS" if report["success"] else "FAILURE"
        acc = f"{report['accuracy']:.4f}" if report.get("accuracy") is not None else "N/A"
        print(f"  Result: {status} (ROC-AUC={acc}) - {report['message']}")

        explored.append({"idea": idea, "report": report})

    print("\n===== Planner: revising idea space =====")
    decision = planner.revise_ideas(dataset_name, explored)
    chosen = decision["chosen"]

    print(f"  Reasoning: {decision['reasoning']}")
    print("\n===== FINAL SUBMISSION =====")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model: {chosen['idea']['model']}")
    print(f"  Featurization: {chosen['idea']['featurization']}")
    acc = chosen["report"].get("accuracy")
    print(f"  ROC-AUC: {acc:.4f}" if acc is not None else "  ROC-AUC: N/A")
    print("=============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DrugAgent Planner/Instructor loop.")
    parser.add_argument(
        "--dataset", choices=["HIV", "PAMPA", "DAVIS"], default="HIV",
        help="Which dataset to run the agent on.",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of ideas the Planner generates.")
    args = parser.parse_args()

    run(args.dataset, args.k)
