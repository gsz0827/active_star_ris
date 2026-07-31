from __future__ import annotations

import argparse
import csv
import os

# Limit nested numerical-library thread pools before NumPy/PyTorch are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from active_star_ris.full_scheme_v2.config import load_config
from active_star_ris.full_scheme_v2.experiments import (
    evaluate_policy,
    summarize_rows,
    train_td3,
    write_csv,
    write_json,
)

ARCHITECTURES = [
    "passive",
    "partially_active_fixed",
    "partially_active_dynamic",
    "fully_active_fixed",
]


@dataclass(frozen=True)
class Task:
    config_path: str
    output_dir: str
    architecture: str
    seed: int
    training_steps: int
    evaluation_episodes: int
    objective_samples: int
    final_probing_samples: int
    skip_completed: bool


def _read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            converted: dict[str, float | str] = {}
            for key, value in row.items():
                if key == "architecture":
                    converted[key] = value
                else:
                    try:
                        converted[key] = float(value)
                    except (TypeError, ValueError):
                        converted[key] = value
            rows.append(converted)
    return rows


def _run_task(task: Task) -> dict[str, float | str]:
    config = load_config(task.config_path)
    config = replace(
        config,
        environment=replace(
            config.environment,
            architecture=task.architecture,
            seed=task.seed,
        ),
    )
    run_dir = Path(task.output_dir) / task.architecture / f"seed_{task.seed}"
    evaluation_path = run_dir / "evaluation.csv"

    if task.skip_completed and evaluation_path.exists():
        rows = _read_rows(evaluation_path)
        if rows:
            summary = summarize_rows(rows)
            summary["seed"] = float(task.seed)
            print(
                f"[跳过已完成] architecture={task.architecture}, seed={task.seed}",
                flush=True,
            )
            return summary

    agent, _ = train_td3(
        config,
        steps=task.training_steps,
        output_dir=run_dir,
        seed=task.seed,
    )
    evaluation_config = replace(
        config,
        probing=replace(
            config.probing,
            samples_per_step=task.final_probing_samples,
        ),
    )
    rows = evaluate_policy(
        evaluation_config,
        lambda state: agent.act(state, 0.0),
        episodes=task.evaluation_episodes,
        seed=task.seed + 100_000,
        full_protocol=True,
        objective_samples=task.objective_samples,
    )
    write_csv(evaluation_path, rows)
    summary = summarize_rows(rows)
    summary["seed"] = float(task.seed)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="并行运行彼此独立的 full_scheme_v2 架构-种子任务。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/full_scheme_v2_paper_corrected.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/full_scheme_v2/paper_parallel",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--objective-samples", type=int, default=32)
    parser.add_argument("--final-probing-samples", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--architectures",
        nargs="*",
        choices=ARCHITECTURES,
        default=None,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="建议单机先使用 2；CPU/内存充足后再增加。",
    )
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    steps = args.steps if args.steps is not None else base.experiment.training_steps
    episodes = (
        args.episodes
        if args.episodes is not None
        else base.experiment.evaluation_episodes
    )
    seeds = args.seeds if args.seeds else list(base.experiment.seeds)
    architectures = (
        args.architectures
        if args.architectures
        else list(base.experiment.architectures)
    )
    objective_samples = args.objective_samples
    final_probing_samples = args.final_probing_samples

    if args.smoke:
        steps = min(steps, 10)
        episodes = min(episodes, 1)
        seeds = seeds[:1]
        architectures = architectures[:2]
        objective_samples = 2
        final_probing_samples = 32

    if args.max_workers < 1:
        parser.error("--max-workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        Task(
            config_path=str(args.config),
            output_dir=str(args.output_dir),
            architecture=architecture,
            seed=seed,
            training_steps=steps,
            evaluation_episodes=episodes,
            objective_samples=objective_samples,
            final_probing_samples=final_probing_samples,
            skip_completed=args.skip_completed,
        )
        for architecture in architectures
        for seed in seeds
    ]

    summaries: list[dict[str, float | str]] = []
    print(
        f"[并行套件] tasks={len(tasks)}, max_workers={args.max_workers}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                summary = future.result()
            except Exception as error:
                print(
                    f"[任务失败] architecture={task.architecture}, "
                    f"seed={task.seed}: {error}",
                    flush=True,
                )
                raise
            summaries.append(summary)
            write_csv(args.output_dir / "all_seed_summaries.partial.csv", summaries)

    architecture_order = {name: index for index, name in enumerate(architectures)}
    summaries.sort(
        key=lambda row: (
            architecture_order.get(str(row.get("architecture", "")), 999),
            float(row.get("seed", 0.0)),
        )
    )
    write_csv(args.output_dir / "all_seed_summaries.csv", summaries)
    write_json(args.output_dir / "config_snapshot.json", asdict(base))
    print(f"实验完成：{args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
