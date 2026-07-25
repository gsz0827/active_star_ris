from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n>", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    print("Active STAR-RIS complete baseline")
    print(f"Python: {sys.version.split()[0]}")
    run([sys.executable, "-m", "pytest", "-q"])
    run([sys.executable, "scripts/run_main_experiment.py", "--quick"])
    print("\nALL CHECKS: PASS")


if __name__ == "__main__":
    main()
