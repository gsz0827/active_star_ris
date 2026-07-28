from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description="绘制 full_scheme_v2 论文汇总图。")
    parser.add_argument("--input", type=Path, required=True, help="all_seed_summaries.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit("输入文件没有数据")
    architectures = sorted({row["architecture"] for row in rows})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("mean_secure_key_rate_bps_mean", "Secure key rate (bit/s)", "secure_key_rate.png"),
        ("mean_raw_kdr_mean", "Raw KDR", "raw_kdr.png"),
        ("mean_surface_power_watt_mean", "Surface power (W)", "surface_power.png"),
        ("power_violation_probability_mean", "Violation probability", "violation_probability.png"),
    ]
    for key, ylabel, filename in metrics:
        means = []
        errors = []
        for architecture in architectures:
            values = np.asarray([float(row[key]) for row in rows if row["architecture"] == architecture])
            means.append(float(np.mean(values)))
            errors.append(float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0)
        plt.figure(figsize=(8, 4.5))
        plt.bar(np.arange(len(architectures)), means, yerr=errors, capsize=4)
        plt.xticks(np.arange(len(architectures)), architectures, rotation=18, ha="right")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(args.output_dir / filename, dpi=200)
        plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
