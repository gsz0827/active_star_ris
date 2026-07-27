from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n>", " ".join(command))
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source_path if not existing else source_path + os.pathsep + existing
    )
    subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    run([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_full_scheme_v2.py"])
    run(
        [
            sys.executable,
            "scripts/train_full_scheme_v2.py",
            "--smoke",
            "--steps",
            "10",
            "--output-dir",
            "results/full_scheme_v2/smoke",
        ]
    )
    run(
        [
            sys.executable,
            "scripts/run_full_scheme_experiments_v2.py",
            "--quick",
            "--output-dir",
            "results/full_scheme_v2/quick_experiments",
        ]
    )
    print("\nAll full_scheme_v2 checks passed.")


if __name__ == "__main__":
    main()
