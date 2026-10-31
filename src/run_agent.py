"""Run the multi-round DrugAgent Planner <-> Instructor workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .dataset import DATASET_REGISTRY, load_dataset_splits
    from .llm_instructor import LLMInstructor
    from .llm_planner import LLMPlanner
else:  # supports `python src/run_agent.py`
    from dataset import DATASET_REGISTRY, load_dataset_splits
    from llm_instructor import LLMInstructor
    from llm_planner import LLMPlanner


def run(dataset_name: str, k: int = 3, rounds: int = 2, seed: int = 42) -> dict:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create a key in Google AI Studio and set it in your shell:\n"
            "  export GEMINI_API_KEY='your-key'"
        )
    if rounds < 1 or k < 1:
        raise ValueError("Both --rounds and --k must be positive integers.")

    dataset_name = dataset_name.upper()
    splits, dataset_metadata = load_dataset_splits(dataset_name, seed=seed)
    planner = LLMPlanner()
    instructor = LLMInstructor()
    inspection = instructor.inspect_dataset(dataset_name, splits)

    print(f"\n=== DrugAgent: {dataset_name} ===")
    print(json.dumps(dataset_metadata, indent=2, default=str))
    print("\n=== Instructor: deterministic schema/domain inspection ===")
    print(json.dumps(inspection, indent=2, default=str))

    explored: list[dict] = []
    excluded_signatures: set[str] = set()

    for round_number in range(1, rounds + 1):
        print(f"\n=== Planner: round {round_number}/{rounds} ===")
        feedback = [
            {
                "model": item["idea"]["model"],
                "hyperparameters": item["idea"]["hyperparameters"],
                "success": item["report"]["success"],
                "validation_roc_auc": item["report"].get("validation_roc_auc"),
                "message": item["report"].get("message"),
            }
            for item in explored
        ]
        ideas = planner.generate_ideas(
            dataset_name,
            k=k,
            feedback=feedback,
            excluded_signatures=excluded_signatures,
        )
        if not ideas:
            print("No new supported candidates remain; stopping exploration.")
            break

        for idea_number, idea in enumerate(ideas, start=1):
            signature = planner.signature(idea)
            excluded_signatures.add(signature)
            print(
                f"\n--- Candidate {idea_number}/{len(ideas)}: {idea['model']} "
                f"{idea['hyperparameters']} ---"
            )
            print(f"Rationale: {idea['rationale']}")
            risk_note = instructor.check_domain_knowledge(dataset_name, idea, inspection)
            print(f"Domain review: {risk_note}")
            report = instructor.evaluate_candidate(dataset_name, splits, idea)
            if report["success"]:
                print(f"Validation ROC-AUC: {report['validation_roc_auc']:.4f}")
            else:
                print(f"Candidate failed: {report['message']}")
            explored.append({
                "round": round_number,
                "idea": idea,
                "domain_review": risk_note,
                "report": report,
            })

        if not any(item["report"]["success"] for item in explored):
            continue
        round_decision = planner.revise_ideas(explored)
        best_round = round_decision["chosen"]
        print("\nPlanner feedback:", round_decision["reasoning"])
        print(
            "Best validation candidate so far:",
            best_round["idea"]["model"],
            best_round["report"]["validation_roc_auc"],
        )

    if not explored or not any(item["report"]["success"] for item in explored):
        raise RuntimeError("All candidate experiments failed. Review the failure reports above.")

    decision = planner.revise_ideas(explored)
    chosen = decision["chosen"]
    print("\n=== Final selection (based on validation only) ===")
    print("Model:", chosen["idea"]["model"])
    print("Hyperparameters:", chosen["idea"]["hyperparameters"])
    print("Validation ROC-AUC:", f"{chosen['report']['validation_roc_auc']:.4f}")
    print("Planner reasoning:", decision["reasoning"])

    print("\n=== Final test evaluation ===")
    final_report = instructor.final_evaluate(dataset_name, splits, chosen["idea"])
    print(json.dumps(final_report, indent=2, default=str))

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_metadata": dataset_metadata,
        "seed": seed,
        "rounds_requested": rounds,
        "candidates_requested_per_round": k,
        "candidates_executed": len(explored),
        "chosen_idea": chosen["idea"],
        "planner_reasoning": decision["reasoning"],
        "chosen_validation_report": chosen["report"],
        "final_test_report": final_report,
        "experiments": explored,
        "evaluation_protocol": "Candidate selection on validation ROC-AUC; selected model refit on train+valid; test evaluated once.",
    }

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{dataset_name.lower()}_{timestamp}.json"
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved experiment report: {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-guided ML experimentation for drug discovery.")
    parser.add_argument("--dataset", choices=sorted(DATASET_REGISTRY), default="HIV")
    parser.add_argument("--k", type=int, default=3, help="New candidate configurations per round.")
    parser.add_argument("--rounds", type=int, default=2, help="Number of Planner/Instructor exploration rounds.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        run(args.dataset, k=args.k, rounds=args.rounds, seed=args.seed)
    except Exception as exc:
        print(f"\nDrugAgent stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
