from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from active_star_ris.full_scheme_v2 import (
    RobustFullSchemeEnvironment,
    TD3Agent,
    TD3Config,
    TrainingConfig,
    evaluate_agent,
    load_environment_config,
    train_td3,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练鲁棒TD3联合优化部分有源STAR-RIS完整密钥生成方案。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full_scheme_v2.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/full_scheme_v2"))
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument("--evaluation-samples", type=int, default=2048)
    parser.add_argument("--evaluation-objective-samples", type=int, default=128)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="使用很小的样本和步数检查代码链路。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_environment_config(args.config)

    steps = args.steps
    evaluation_samples = args.evaluation_samples
    evaluation_objective_samples = args.evaluation_objective_samples
    evaluation_episodes = args.evaluation_episodes

    if args.smoke:
        steps = min(steps, 20)
        evaluation_samples = min(evaluation_samples, 128)
        evaluation_objective_samples = 4
        evaluation_episodes = 1
        config = replace(
            config,
            probing=replace(config.probing, samples_per_step=64),
            robust=replace(
                config.robust,
                objective_samples=4,
                cvar_alpha=1.0,
                minimum_tail_samples=4,
            ),
            max_episode_steps=5,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_environment = RobustFullSchemeEnvironment(config, seed=args.seed)
    evaluation_environment = training_environment.evaluation_copy(
        samples_per_step=evaluation_samples,
        objective_samples=evaluation_objective_samples,
        cvar_alpha=(1.0 if evaluation_objective_samples < 10 else 0.10),
        seed=args.seed + 10_000,
    )

    td3_config = TD3Config()
    agent = TD3Agent(
        training_environment.state_dim,
        training_environment.action_dim,
        td3_config,
        device=args.device,
        seed=args.seed,
    )

    training_config = TrainingConfig(
        total_environment_steps=steps,
        replay_capacity=max(10_000, steps * 2),
        random_action_steps=min(5_000, max(1, steps // 10)),
        learning_starts=min(5_000, max(1, steps // 10)),
        batch_size=8 if args.smoke else 256,
        evaluation_interval=max(1, steps // 5),
        evaluation_episodes=evaluation_episodes,
        seed=args.seed,
    )

    def progress(step: int, history) -> None:
        interval = max(1, steps // 20)
        if step % interval != 0 and step != steps:
            return
        latest_return = (
            history.episode_returns[-1]
            if history.episode_returns
            else float("nan")
        )
        print(f"step={step:8d} latest_episode_return={latest_return:12.6f}")

    history = train_td3(
        training_environment,
        agent,
        training_config,
        evaluation_environment=evaluation_environment,
        progress_callback=progress,
    )

    if args.smoke and agent.update_count <= 0:
        raise RuntimeError(
            "Smoke test completed without any TD3 gradient update."
        )

    if args.smoke:
        print(
            f"Smoke TD3 updates: {agent.update_count}"
        )

    checkpoint_path = args.output_dir / "td3_final.pt"
    agent.save(
        checkpoint_path,
        extra={
            "seed": args.seed,
            "steps": steps,
            "config_path": str(args.config),
        },
    )

    final_summary = evaluate_agent(
        evaluation_environment,
        agent,
        evaluation_episodes,
        seed=args.seed + 100_000,
    )

    with (args.output_dir / "training_history.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(asdict(history), file, ensure_ascii=False, indent=2)

    with (args.output_dir / "final_evaluation.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(asdict(final_summary), file, ensure_ascii=False, indent=2)

    print(f"checkpoint: {checkpoint_path}")
    print(json.dumps(asdict(final_summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
