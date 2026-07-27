from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate full_scheme_v2 experiment CSV files across independently "
            "trained TD3 seeds."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/full_scheme_v2/experiments"),
        help="Directory containing seed_*/<experiment>.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/full_scheme_v2/experiments_aggregated"),
        help="Directory for aggregated CSV files.",
    )
    parser.add_argument(
        "--seed-pattern",
        type=str,
        default="seed_*",
        help="Glob pattern used to find per-seed experiment directories.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def statistics(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty metric")

    mean = float(np.mean(array))
    if array.size < 2:
        return mean, 0.0, 0.0

    std = float(np.std(array, ddof=1))
    ci95 = float(1.96 * std / np.sqrt(array.size))
    return mean, std, ci95


def metric_mean_columns(rows: list[dict[str, str]]) -> list[str]:
    result: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith("mean_") and parse_float(value) is not None:
                result.add(key)
    return sorted(result)


def grouping_columns(
    rows: list[dict[str, str]],
    metric_columns: list[str],
) -> list[str]:
    ignored = {
        "episodes",
        "training_seed",
        "training_seed_count",
        "total_evaluation_episodes",
    }
    ignored.update(metric_columns)

    for metric in metric_columns:
        suffix = metric[len("mean_") :]
        ignored.add(f"std_{suffix}")
        ignored.add(f"ci95_{suffix}")

    columns: set[str] = set()
    for row in rows:
        columns.update(row.keys())

    return sorted(column for column in columns if column not in ignored)


def aggregate_file(
    seed_directories: list[Path],
    filename: str,
    output_dir: Path,
) -> None:
    per_seed_rows: list[dict[str, str]] = []

    for seed_directory in seed_directories:
        path = seed_directory / filename
        if not path.exists():
            print(f"[warn] Missing {path}; this seed is skipped for {filename}")
            continue

        for row in read_csv(path):
            enriched = dict(row)
            enriched["training_seed"] = seed_directory.name
            per_seed_rows.append(enriched)

    if not per_seed_rows:
        print(f"[skip] No rows found for {filename}")
        return

    metric_columns = metric_mean_columns(per_seed_rows)
    group_columns = grouping_columns(per_seed_rows, metric_columns)

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in per_seed_rows:
        key = tuple(row.get(column, "") for column in group_columns)
        grouped[key].append(row)

    aggregated_rows: list[dict[str, object]] = []

    for key in sorted(grouped):
        rows = grouped[key]
        output: dict[str, object] = {
            column: value
            for column, value in zip(group_columns, key, strict=True)
        }

        seed_names = sorted({row["training_seed"] for row in rows})
        output["training_seed_count"] = len(seed_names)

        episode_counts = [
            parsed
            for row in rows
            if (parsed := parse_float(row.get("episodes"))) is not None
        ]
        output["total_evaluation_episodes"] = int(sum(episode_counts))

        for mean_column in metric_columns:
            values = [
                parsed
                for row in rows
                if (parsed := parse_float(row.get(mean_column))) is not None
            ]
            if not values:
                continue

            mean, std, ci95 = statistics(values)
            suffix = mean_column[len("mean_") :]

            output[mean_column] = mean
            output[f"std_{suffix}"] = std
            output[f"ci95_{suffix}"] = ci95

        aggregated_rows.append(output)

    write_csv(
        output_dir / filename,
        aggregated_rows,
    )

    per_seed_output: list[dict[str, object]] = [
        dict(row) for row in per_seed_rows
    ]
    write_csv(
        output_dir / f"{Path(filename).stem}_per_seed.csv",
        per_seed_output,
    )

    print(
        f"[ok] {filename}: {len(per_seed_rows)} per-seed rows -> "
        f"{len(aggregated_rows)} aggregated rows"
    )


def main() -> None:
    args = parse_args()

    seed_directories = sorted(
        path
        for path in args.input_root.glob(args.seed_pattern)
        if path.is_dir()
    )
    if not seed_directories:
        raise FileNotFoundError(
            f"No seed directories matched {args.input_root / args.seed_pattern}"
        )

    filenames = sorted(
        {
            path.name
            for seed_directory in seed_directories
            for path in seed_directory.glob("*.csv")
        }
    )
    if not filenames:
        raise FileNotFoundError(
            f"No CSV files found under {args.input_root / args.seed_pattern}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        aggregate_file(
            seed_directories,
            filename,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
