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
from active_star_ris.full_scheme_v2.experiments import run_architecture_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 full_scheme_v2 多架构、多种子论文实验。")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/full_scheme_v2_paper.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/full_scheme_v2/paper")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--objective-samples", type=int, default=32)
    parser.add_argument(
        "--final-probing-samples",
        type=int,
        default=1024,
        help="正式协议评价使用的探测块长度；训练仍使用配置文件中的长度。",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    steps = args.steps or config.experiment.training_steps
    episodes = args.episodes or config.experiment.evaluation_episodes
    seeds = args.seeds or config.experiment.seeds
    objective_samples = args.objective_samples
    if args.smoke:
        steps = min(steps, 10)
        episodes = min(episodes, 1)
        seeds = seeds[:1]
        objective_samples = 2
        args.final_probing_samples = 32
        config = replace(
            config,
            experiment=replace(config.experiment, architectures=["passive", "partially_active_fixed"]),
            robust=replace(config.robust, objective_samples=2),
            probing=replace(config.probing, samples_per_step=24),
            td3=replace(config.td3, batch_size=4, warmup_steps=2, hidden_dimensions=[16, 16]),
            environment=replace(config.environment, episode_length=5),
        )
    run_architecture_suite(
        config,
        output_dir=args.output_dir,
        training_steps=steps,
        evaluation_episodes=episodes,
        seeds=seeds,
        objective_samples=objective_samples,
        evaluation_probing_samples=args.final_probing_samples,
    )
    print(f"实验完成：{args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
