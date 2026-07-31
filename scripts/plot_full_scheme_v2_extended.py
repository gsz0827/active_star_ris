from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARCHITECTURE_LABELS = {
    "passive": "Passive",
    "partially_active_fixed": "Partially active (fixed)",
    "partially_active_dynamic": "Partially active (dynamic)",
    "fully_active_fixed": "Fully active",
}


def read_numeric_csv(path: Path) -> list[dict[str, float | str]]:
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


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 4000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    boot = np.mean(values[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(boot, [alpha, 1.0 - alpha])
    return mean, float(lower), float(upper)


def save_figure(figure: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def discover_histories(root: Path) -> dict[str, list[list[dict[str, float | str]]]]:
    histories: dict[str, list[list[dict[str, float | str]]]] = defaultdict(list)
    for path in sorted(root.glob("*/seed_*/training_history.csv")):
        architecture = path.parent.parent.name
        rows = read_numeric_csv(path)
        if rows:
            histories[architecture].append(rows)
    return histories


def plot_training_metric(
    histories: dict[str, list[list[dict[str, float | str]]]],
    metric: str,
    ylabel: str,
    output_base: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    plotted = False
    for architecture, runs in histories.items():
        valid_runs = [run for run in runs if metric in run[0]]
        if not valid_runs:
            continue
        common_steps = np.asarray(
            sorted({float(row["step"]) for run in valid_runs for row in run}),
            dtype=np.float64,
        )
        matrix = []
        for run in valid_runs:
            steps = np.asarray([float(row["step"]) for row in run], dtype=np.float64)
            values = np.asarray([float(row[metric]) for row in run], dtype=np.float64)
            finite = np.isfinite(steps) & np.isfinite(values)
            if np.count_nonzero(finite) < 2:
                continue
            matrix.append(np.interp(common_steps, steps[finite], values[finite]))
        if not matrix:
            continue
        array = np.asarray(matrix, dtype=np.float64)
        mean = np.mean(array, axis=0)
        if array.shape[0] > 1:
            standard_error = np.std(array, axis=0, ddof=1) / np.sqrt(array.shape[0])
            t95 = {
                2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
                12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
                17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
                22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
                27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
            }.get(array.shape[0], 1.96)
            half_width = t95 * standard_error
        else:
            half_width = np.zeros_like(mean)
        label = ARCHITECTURE_LABELS.get(architecture, architecture)
        axis.plot(common_steps, mean, label=label)
        axis.fill_between(common_steps, mean - half_width, mean + half_width, alpha=0.18)
        plotted = True
    axis.set_xlabel("Training step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend()
    figure.tight_layout()
    save_figure(figure, output_base)


def plot_final_metric(
    rows: list[dict[str, float | str]],
    column: str,
    ylabel: str,
    output_base: Path,
) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        architecture = str(row["architecture"])
        value = row.get(column)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            grouped[architecture].append(float(value))
    architectures = [name for name in ARCHITECTURE_LABELS if name in grouped]
    means = []
    lower_errors = []
    upper_errors = []
    for index, architecture in enumerate(architectures):
        mean, lower, upper = bootstrap_mean_ci(
            np.asarray(grouped[architecture]),
            seed=2026 + index,
        )
        means.append(mean)
        lower_errors.append(mean - lower)
        upper_errors.append(upper - mean)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    positions = np.arange(len(architectures))
    axis.bar(
        positions,
        means,
        yerr=np.asarray([lower_errors, upper_errors]),
        capsize=4,
    )
    axis.set_xticks(
        positions,
        [ARCHITECTURE_LABELS.get(name, name) for name in architectures],
        rotation=12,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output_base)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 full_scheme_v2 论文训练与最终结果图。")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/full_scheme_v2/paper_parallel"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_dir / "figures_extended")
    histories = discover_histories(args.results_dir)
    training_specs = [
        ("reward", "Robust reward", "training_reward"),
        ("mean_secure_key_rate_bps", "Secure key rate (bit/s)", "training_key_rate"),
        ("mean_raw_kdr", "Raw KDR", "training_raw_kdr"),
        ("mean_reciprocity", "Reciprocity", "training_reciprocity"),
        ("mean_surface_power_watt", "Surface DC power (W)", "training_power"),
        ("critic_loss", "Critic loss", "training_critic_loss"),
        ("actor_loss", "Actor loss", "training_actor_loss"),
    ]
    for metric, ylabel, filename in training_specs:
        plot_training_metric(histories, metric, ylabel, output_dir / filename)

    summary_path = args.results_dir / "all_seed_summaries.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"未找到 {summary_path}")
    summaries = read_numeric_csv(summary_path)
    final_specs = [
        ("mean_secure_key_rate_bps_mean", "Secure key rate (bit/s)", "final_key_rate"),
        ("mean_raw_kdr_mean", "Raw KDR", "final_raw_kdr"),
        (
            "mean_post_reconciliation_kdr_mean",
            "Post-reconciliation KDR",
            "final_post_kdr",
        ),
        ("mean_reciprocity_mean", "Reciprocity", "final_reciprocity"),
        ("mean_surface_power_watt_mean", "Surface DC power (W)", "final_power"),
        (
            "power_violation_probability_mean",
            "Constraint violation probability",
            "final_violation_probability",
        ),
        ("mean_active_elements_mean", "Active elements", "final_active_elements"),
    ]
    for column, ylabel, filename in final_specs:
        if summaries and column in summaries[0]:
            plot_final_metric(summaries, column, ylabel, output_dir / filename)

    print(f"图已保存到：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
