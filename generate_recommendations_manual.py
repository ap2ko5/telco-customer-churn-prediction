"""Compatibility wrapper for scripts/recommendations/generate_recommendations_manual.py."""
from runpy import run_path
from pathlib import Path

if __name__ == "__main__":
    run_path(str(Path(__file__).parent / "scripts" / "recommendations" / "generate_recommendations_manual.py"), run_name="__main__")
