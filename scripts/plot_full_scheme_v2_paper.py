from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "passive_star_ris": "Passive STAR-RIS",
    "random_partially_active": "Random PA STAR-RIS",
    "phase_aligned_partially_active": "Phase-aligned PA STAR-RIS",
    "robust_td3": "Robust TD3",
    "complete_model": "Complete model",
    "without_internal_noise": "No amplifier noise",
    "perfect_control_csi": "Perfect CSI",
    "without_delay_mismatch": "No probing delay",
    "without_hardware_mismatch": "No hardware mismatch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for full_scheme_v2."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/full_scheme_v2"),
        help="Directory containing seed_*/training_history.json.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("results/full_scheme_v2/experiments"),
        help="Directory containing experiment CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/full_scheme_v2/figures"),
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def preferred_final_rate_key(rows: list[dict[str, str]]) -> str:
    if rows and any(
        np.isfinite(as_float(row, "mean_system_final_key_rate_bps"))
        for row in rows
    ):
        return "mean_system_final_key_rate_bps"
    return "mean_final_key_rate_bps"


def common_steps(histories: list[dict]) -> np.ndarray:
    step_sets = [set(map(int, h.get("evaluation_steps", []))) for h in histories]
    if not step_sets:
        return np.empty(0, dtype=int)
    shared = set.intersection(*step_sets)
    return np.asarray(sorted(shared), dtype=int)


def history_series(history: dict, key: str, steps: np.ndarray) -> np.ndarray:
    all_steps = list(map(int, history.get("evaluation_steps", [])))
    values = history.get(key, [])
    lookup = {step: float(value) for step, value in zip(all_steps, values)}
    return np.asarray([lookup[int(step)] for step in steps], dtype=np.float64)


def mean_ci95(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0)
    if matrix.shape[0] < 2:
        return mean, np.zeros_like(mean)
    std = np.std(matrix, axis=0, ddof=1)
    ci = 1.96 * std / np.sqrt(matrix.shape[0])
    return mean, ci


def find_training_histories(results_root: Path) -> list[Path]:
    paths = sorted(results_root.glob("seed_*/training_history.json"))
    if not paths:
        direct = results_root / "training_history.json"
        if direct.exists():
            paths = [direct]
    return paths


def plot_convergence(results_root: Path, output_dir: Path) -> None:
    paths = find_training_histories(results_root)
    if not paths:
        print(f"[skip] No training_history.json found under {results_root}")
        return

    histories = [read_json(path) for path in paths]
    steps = common_steps(histories)
    if steps.size == 0:
        print("[skip] Training histories have no common evaluation steps")
        return

    candidates = [
        ("evaluation_returns", "Evaluation return", "convergence_return", False),
        (
            "evaluation_training_key_rates_bps",
            "Training-bound KGR (bit/s)",
            "convergence_training_kgr",
            True,
        ),
        (
            "evaluation_final_key_rates_bps",
            "Final key rate (bit/s)",
            "convergence_final_kgr",
            True,
        ),
        (
            "evaluation_system_final_key_rates_bps",
            "System final key rate (bit/s)",
            "convergence_system_final_kgr",
            True,
        ),
        ("evaluation_raw_kdr", "Raw KDR", "convergence_raw_kdr", False),
        ("evaluation_reciprocity", "Observation reciprocity", "convergence_reciprocity", False),
        ("evaluation_surface_power", "Surface DC power (W)", "convergence_power", False),
        ("evaluation_feasibility", "Feasibility rate", "convergence_feasibility", False),
    ]

    # Backward compatibility with the repository's original history field.
    if not all("evaluation_final_key_rates_bps" in h for h in histories):
        for history in histories:
            if "evaluation_key_rates_bps" in history:
                history["evaluation_final_key_rates_bps"] = history[
                    "evaluation_key_rates_bps"
                ]

    for key, ylabel, stem, log_y in candidates:
        if not all(key in history and history[key] for history in histories):
            print(f"[skip] Missing history field: {key}")
            continue
        matrix = np.stack([history_series(h, key, steps) for h in histories])
        mean, ci = mean_ci95(matrix)
        fig, ax = plt.subplots(figsize=(5.2, 3.5))
        ax.plot(steps, mean, marker="o", label=f"Mean over {len(histories)} seeds")
        ax.fill_between(steps, mean - ci, mean + ci, alpha=0.2, label="95% CI")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        if log_y and np.any(mean > 0.0):
            ax.set_yscale("symlog", linthresh=1.0)
        ax.legend(frameon=False)
        save_figure(fig, output_dir, stem)


def labels_for_rows(rows: Iterable[dict[str, str]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        method = row.get("method", "unknown")
        result.append(METHOD_LABELS.get(method, method.replace("_", " ")))
    return result


def bar_plot(
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
    ci_metric: str | None = None,
) -> None:
    if not rows or all(not np.isfinite(as_float(row, metric)) for row in rows):
        return
    labels = labels_for_rows(rows)
    values = np.asarray([as_float(row, metric, 0.0) for row in rows])
    errors = None
    if ci_metric is not None and any(np.isfinite(as_float(row, ci_metric)) for row in rows):
        errors = np.asarray([as_float(row, ci_metric, 0.0) for row in rows])
    fig, ax = plt.subplots(figsize=(max(5.4, 1.25 * len(rows)), 3.7))
    positions = np.arange(len(rows))
    ax.bar(positions, values, yerr=errors, capsize=3 if errors is not None else 0)
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    save_figure(fig, output_dir, stem)


def plot_baselines(experiments_dir: Path, output_dir: Path) -> None:
    path = experiments_dir / "baseline_comparison.csv"
    if not path.exists():
        print(f"[skip] Missing {path}")
        return
    rows = read_csv(path)
    final_rate_key = preferred_final_rate_key(rows)
    final_rate_ci_key = final_rate_key.replace("mean_", "ci95_")
    bar_plot(
        rows,
        final_rate_key,
        "System final key rate (bit/s)",
        output_dir,
        "baseline_final_kgr",
        final_rate_ci_key,
    )
    bar_plot(
        rows,
        "mean_raw_kdr",
        "Raw KDR",
        output_dir,
        "baseline_raw_kdr",
        "ci95_raw_kdr",
    )
    bar_plot(
        rows,
        "mean_reciprocity",
        "Observation reciprocity",
        output_dir,
        "baseline_reciprocity",
        "ci95_reciprocity",
    )
    bar_plot(
        rows,
        "mean_surface_dc_power",
        "Surface DC power (W)",
        output_dir,
        "baseline_power",
        "ci95_surface_dc_power",
    )

    # Power-rate Pareto view.
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    for row, label in zip(rows, labels_for_rows(rows)):
        x = as_float(row, "mean_surface_dc_power")
        y = as_float(row, preferred_final_rate_key(rows))
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter([x], [y], s=45)
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Surface DC power (W)")
    ax.set_ylabel("Final key rate (bit/s)")
    save_figure(fig, output_dir, "pareto_power_vs_final_kgr")


def line_sweep_plot(
    csv_path: Path,
    x_key: str,
    x_label: str,
    output_dir: Path,
    stem_prefix: str,
    x_scale: float = 1.0,
) -> None:
    if not csv_path.exists():
        print(f"[skip] Missing {csv_path}")
        return
    rows = read_csv(csv_path)
    rows = [row for row in rows if np.isfinite(as_float(row, x_key))]
    rows.sort(key=lambda row: as_float(row, x_key))
    if not rows:
        return
    x = np.asarray([as_float(row, x_key) * x_scale for row in rows])
    metrics = [
        (preferred_final_rate_key(rows), "System final key rate (bit/s)", "final_kgr"),
        ("mean_raw_kdr", "Raw KDR", "raw_kdr"),
        ("mean_reciprocity", "Observation reciprocity", "reciprocity"),
        ("mean_surface_dc_power", "Surface DC power (W)", "power"),
        ("mean_feasibility_rate", "Feasibility rate", "feasibility"),
    ]
    for metric, ylabel, suffix in metrics:
        y = np.asarray([as_float(row, metric) for row in rows])
        if not np.any(np.isfinite(y)):
            continue
        ci_key = metric.replace("mean_", "ci95_")
        ci = np.asarray([as_float(row, ci_key, 0.0) for row in rows])
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        ax.errorbar(x, y, yerr=ci, marker="o", capsize=3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        save_figure(fig, output_dir, f"{stem_prefix}_{suffix}")


def plot_sweeps(experiments_dir: Path, output_dir: Path) -> None:
    line_sweep_plot(
        experiments_dir / "active_ratio_sweep.csv",
        "active_ratio",
        "Active-element ratio",
        output_dir,
        "active_ratio",
    )
    line_sweep_plot(
        experiments_dir / "nmse_sweep.csv",
        "nmse_db",
        "Control-CSI NMSE (dB)",
        output_dir,
        "nmse",
    )
    line_sweep_plot(
        experiments_dir / "delay_sweep.csv",
        "delay_seconds",
        "Forward-reverse delay (ms)",
        output_dir,
        "delay",
        x_scale=1.0e3,
    )
    line_sweep_plot(
        experiments_dir / "rf_power_budget_sweep.csv",
        "rf_budget",
        "RF power budget (W)",
        output_dir,
        "rf_budget",
    )
    line_sweep_plot(
        experiments_dir / "dc_power_budget_sweep.csv",
        "dc_budget",
        "DC power budget (W)",
        output_dir,
        "dc_budget",
    )


def plot_ablation(experiments_dir: Path, output_dir: Path) -> None:
    path = experiments_dir / "impairment_sensitivity.csv"
    if not path.exists():
        print(f"[skip] Missing {path}")
        return
    rows = read_csv(path)
    final_rate_key = preferred_final_rate_key(rows)
    bar_plot(
        rows,
        final_rate_key,
        "System final key rate (bit/s)",
        output_dir,
        "ablation_final_kgr",
        final_rate_key.replace("mean_", "ci95_"),
    )
    bar_plot(
        rows,
        "mean_raw_kdr",
        "Raw KDR",
        output_dir,
        "ablation_raw_kdr",
        "ci95_raw_kdr",
    )


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    plot_convergence(args.results_root, args.output_dir)
    plot_baselines(args.experiments_dir, args.output_dir)
    plot_sweeps(args.experiments_dir, args.output_dir)
    plot_ablation(args.experiments_dir, args.output_dir)
    print(f"Figures written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
