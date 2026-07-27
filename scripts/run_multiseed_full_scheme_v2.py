from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full_scheme_v2 over multiple seeds.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_scheme_v2.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("results/full_scheme_v2"))
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--evaluation-samples", type=int, default=2048)
    parser.add_argument("--evaluation-objective-samples", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        output_dir = args.output_root / f"seed_{seed}"
        checkpoint = output_dir / "td3_final.pt"
        if args.skip_existing and checkpoint.exists():
            print(f"[skip] seed={seed}: {checkpoint}")
            continue
        command = [
            sys.executable,
            "scripts/train_full_scheme_v2.py",
            "--config",
            str(args.config),
            "--output-dir",
            str(output_dir),
            "--steps",
            str(args.steps),
            "--seed",
            str(seed),
            "--evaluation-episodes",
            str(args.evaluation_episodes),
            "--evaluation-samples",
            str(args.evaluation_samples),
            "--evaluation-objective-samples",
            str(args.evaluation_objective_samples),
        ]
        if args.device:
            command.extend(["--device", args.device])
        print(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
