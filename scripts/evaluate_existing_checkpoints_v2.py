from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from active_star_ris.full_scheme_v2.config import load_config
from active_star_ris.full_scheme_v2.environment import ActiveStarRisKeyEnvironment
from active_star_ris.full_scheme_v2.experiments import (
    evaluate_policy,
    summarize_rows,
    write_csv,
    write_json,
)
from active_star_ris.full_scheme_v2.td3 import TD3Agent

ARCHITECTURES = [
    "passive",
    "partially_active_fixed",
    "partially_active_dynamic",
    "fully_active_fixed",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "加载已有 td3_checkpoint.pt，只重新执行最终完整协议评价，"
            "不重新训练。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/full_scheme_v2_paper_corrected.yaml",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "results/full_scheme_v2/pilot_parallel",
        help="包含 architecture/seed_x/td3_checkpoint.pt 的目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/full_scheme_v2/pilot_eval_1024",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--objective-samples", type=int, default=32)
    parser.add_argument("--final-probing-samples", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=ARCHITECTURES,
    )
    args = parser.parse_args()

    base = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float | str]] = []

    for architecture in args.architectures:
        for seed in args.seeds:
            checkpoint = (
                args.checkpoint_root
                / architecture
                / f"seed_{seed}"
                / "td3_checkpoint.pt"
            )
            if not checkpoint.exists():
                raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")

            config = replace(
                base,
                environment=replace(
                    base.environment,
                    architecture=architecture,
                    seed=seed,
                ),
                probing=replace(
                    base.probing,
                    samples_per_step=args.final_probing_samples,
                ),
            )

            probe_env = ActiveStarRisKeyEnvironment(config)
            agent = TD3Agent(
                probe_env.state_dimension,
                probe_env.action_dimension,
                config.td3,
                seed=seed,
            )
            agent.load(checkpoint)

            print(
                f"\n[重新评价] architecture={architecture}, seed={seed}, "
                f"samples={args.final_probing_samples}, "
                f"objective_samples={args.objective_samples}, "
                f"episodes={args.episodes}",
                flush=True,
            )

            rows = evaluate_policy(
                config,
                lambda state, current_agent=agent: current_agent.act(state, 0.0),
                episodes=args.episodes,
                seed=seed + 200_000,
                full_protocol=True,
                objective_samples=args.objective_samples,
            )

            run_dir = args.output_dir / architecture / f"seed_{seed}"
            write_csv(run_dir / "evaluation.csv", rows)

            summary = summarize_rows(rows)
            summary["seed"] = float(seed)
            summaries.append(summary)

            write_csv(
                args.output_dir / "all_seed_summaries.partial.csv",
                summaries,
            )

    order = {name: index for index, name in enumerate(args.architectures)}
    summaries.sort(
        key=lambda row: (
            order.get(str(row.get("architecture", "")), 999),
            float(row.get("seed", 0.0)),
        )
    )
    write_csv(args.output_dir / "all_seed_summaries.csv", summaries)
    write_json(args.output_dir / "config_snapshot.json", asdict(base))

    print(f"\n评价完成：{args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
