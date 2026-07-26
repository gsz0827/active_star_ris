from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from active_star_ris.experiment_pipeline import (
    run_full_experiment_suite,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate passive, partially active, "
            "fully active STAR-RIS baselines and ablations."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/default.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/full_experiment_suite"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a shortened smoke experiment. "
            "Do not use quick results in a paper."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Override training environment steps "
            "for every run."
        ),
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    with arguments.config.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    experiment = config.get(
        "experiment_suite",
        {},
    )

    seeds = (
        arguments.seeds
        if arguments.seeds is not None
        else experiment.get(
            "seeds",
            [0, 1, 2],
        )
    )

    evaluation_seed = int(
        experiment.get(
            "evaluation_seed",
            500_000,
        )
    )

    final_evaluation_episodes = int(
        experiment.get(
            "final_evaluation_episodes",
            30,
        )
    )

    if arguments.quick:
        final_evaluation_episodes = min(
            final_evaluation_episodes,
            2,
        )

    if arguments.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = arguments.device

    print(
        f"Device: {device}"
    )
    print(
        f"Seeds: {seeds}"
    )
    print(
        f"Output: {arguments.output}"
    )

    manifest = (
        run_full_experiment_suite(
            config,
            output_directory=(
                arguments.output
            ),
            seeds=seeds,
            evaluation_seed=(
                evaluation_seed
            ),
            final_evaluation_episodes=(
                final_evaluation_episodes
            ),
            device=device,
            quick=arguments.quick,
            total_steps_override=(
                arguments.steps
            ),
            run_baselines=(
                not arguments
                .skip_baselines
            ),
            run_ablations=(
                not arguments
                .skip_ablations
            ),
        )
    )

    print(
        "Experiment suite completed."
    )
    print(
        f"Baseline runs: "
        f"{manifest['baseline_runs']}"
    )
    print(
        f"Ablation runs: "
        f"{manifest['ablation_runs']}"
    )


if __name__ == "__main__":
    main()