"""Backward-compatible entry point for the current DrugAgent workflow."""

try:
    from .run_agent import main
except ImportError:  # supports `python src/drug_agent.py`
    from run_agent import main


if __name__ == "__main__":
    main()
