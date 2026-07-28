from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from active_star_ris.full_scheme_v2.config import load_config
from active_star_ris.full_scheme_v2.experiments import train_td3


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 full_scheme_v2 鲁棒 TD3。")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/full_scheme_v2_paper.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--architecture", choices=[
        "passive",
        "partially_active_fixed",
        "partially_active_dynamic",
        "fully_active_fixed",
    ], default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/full_scheme_v2/train")
    parser.add_argument("--smoke", action="store_true", help="运行极小规模烟雾测试。")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.architecture:
        config = replace(config, environment=replace(config.environment, architecture=args.architecture))
    steps = args.steps if args.steps is not None else config.experiment.training_steps
    if args.smoke:
        steps = min(steps, 20)
        config = replace(
            config,
            robust=replace(config.robust, objective_samples=2),
            probing=replace(config.probing, samples_per_step=32),
            td3=replace(config.td3, batch_size=8, warmup_steps=4, hidden_dimensions=[32, 32]),
            environment=replace(config.environment, episode_length=10),
        )
    train_td3(config, steps=steps, output_dir=args.output_dir, seed=args.seed)
    print(f"训练完成：{args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
