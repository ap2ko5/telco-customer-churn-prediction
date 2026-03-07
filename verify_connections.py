"""Compatibility wrapper for scripts/maintenance/verify_connections.py."""
from runpy import run_path
from pathlib import Path

if __name__ == "__main__":
    run_path(str(Path(__file__).parent / "scripts" / "maintenance" / "verify_connections.py"), run_name="__main__")
