#!/usr/bin/env python
"""
bi_auto_update.py
=================
Watch dataset files and automatically refresh the prediction output + open Power BI.

Behavior:
1. Monitor data CSV files for mtime changes.
2. On change, run run_demo.py with the changed file as --data.
3. Optionally force-close Power BI/Excel first (prevents CSV lock failures).
4. Reopen PBIX after a successful run.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_WATCH_FILES = [
    "data/customer_churn.csv",
    "data/test_dataset.csv",
    "data/validation_dataset.csv",
    "data/manual_test_data.csv",
]


def _safe_mtime(path: Path) -> float:
    if not path.exists():
        return -1.0
    return path.stat().st_mtime


def _kill_lock_holders() -> None:
    # Avoid CSV lock conflicts before writing new outputs.
    subprocess.run(["taskkill", "/IM", "PBIDesktop.exe", "/F"], capture_output=True, text=True)
    subprocess.run(["taskkill", "/IM", "EXCEL.EXE", "/F"], capture_output=True, text=True)


def _open_pbix(pbix_path: Path) -> None:
    if not pbix_path.exists():
        print(f"[auto-bi] WARNING: PBIX not found: {pbix_path}")
        return

    try:
        os.startfile(str(pbix_path))  # type: ignore[attr-defined]
        print(f"[auto-bi] Opened Power BI: {pbix_path}")
    except Exception as exc:
        print(f"[auto-bi] WARNING: Could not open PBIX: {exc}")


def _run_pipeline(project_root: Path, python_exe: Path, data_path: Path, skip_ai: bool) -> bool:
    cmd = [str(python_exe), str(project_root / "run_demo.py"), "--data", str(data_path)]
    if skip_ai:
        cmd.append("--skip-ai")

    print(f"[auto-bi] Running pipeline for changed file: {data_path}")
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        print(f"[auto-bi] ERROR: Pipeline failed with exit code {result.returncode}")
        return False

    print("[auto-bi] Pipeline completed successfully.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-refresh BI when data changes.")
    parser.add_argument("--interval", type=float, default=3.0, help="Polling interval in seconds.")
    parser.add_argument("--skip-ai", action="store_true", help="Run pipeline in fast mode without Gemini.")
    parser.add_argument("--close-lock-holders", action="store_true", help="Force-close Power BI/Excel before pipeline run.")
    parser.add_argument("--run-once", action="store_true", help="Run once immediately and exit (for testing).")
    parser.add_argument(
        "--pbix-path",
        default=r"C:\Users\ABEL ABRAHAM\Downloads\dash1.pbix",
        help="Path to Power BI PBIX file.",
    )
    parser.add_argument(
        "--watch-files",
        nargs="*",
        default=DEFAULT_WATCH_FILES,
        help="Relative paths of CSV files to watch.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    python_exe = project_root / ".venv" / "Scripts" / "python.exe"
    pbix_path = Path(args.pbix_path)

    if not python_exe.exists():
        print(f"[auto-bi] ERROR: Python executable not found: {python_exe}")
        return 1

    watch_paths = [project_root / p for p in args.watch_files]
    if not watch_paths:
        print("[auto-bi] ERROR: No watch files configured.")
        return 1

    known_mtimes = {p: _safe_mtime(p) for p in watch_paths}

    print("[auto-bi] Watching for data changes...")
    for p in watch_paths:
        print(f"  - {p}")

    if args.run_once:
        # Use the first existing file as input for a one-time test run.
        candidate = next((p for p in watch_paths if p.exists()), None)
        if candidate is None:
            print("[auto-bi] ERROR: None of the watch files exist.")
            return 1
        if args.close_lock_holders:
            _kill_lock_holders()
        ok = _run_pipeline(project_root, python_exe, candidate, args.skip_ai)
        if ok:
            _open_pbix(pbix_path)
            return 0
        return 1

    while True:
        time.sleep(max(args.interval, 0.5))
        changed: Path | None = None

        for path in watch_paths:
            now_mtime = _safe_mtime(path)
            old_mtime = known_mtimes.get(path, -1.0)
            if now_mtime != old_mtime:
                known_mtimes[path] = now_mtime
                if now_mtime >= 0:
                    changed = path
                    break

        if changed is None:
            continue

        if args.close_lock_holders:
            _kill_lock_holders()

        ok = _run_pipeline(project_root, python_exe, changed, args.skip_ai)
        if ok:
            _open_pbix(pbix_path)


if __name__ == "__main__":
    raise SystemExit(main())
