from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from active_star_ris.simulation import (  # noqa: E402
    load_config,
    monte_carlo,
    summarize_results,
)


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    simulation_cfg = config["simulation"]
    seed = int(config["seed"])
    num_trials = max(
        100,
        int(simulation_cfg["monte_carlo_trials"]) // 5,
    )
    element_values = [int(value) for value in simulation_cfg["sweep_num_elements"]]
    ratios = [float(value) for value in simulation_cfg["sweep_active_ratios"]]

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sweep_results.csv"

    rows: list[dict[str, float]] = []
    for n in element_values:
        for ratio in ratios:
            k_active = min(n, int(round(n * ratio)))
            results = monte_carlo(
                config,
                num_trials=num_trials,
                seed=seed + 1000 * n + k_active,
                num_elements=n,
                num_active_elements=k_active,
            )
            summary = summarize_results(results)
            optimized = summary["partial_active_optimized_beta"]
            rows.append(
                {
                    "num_elements": float(n),
                    "active_ratio": ratio,
                    "num_active_elements": float(k_active),
                    "mean_weighted_sum_rate": optimized[
                        "mean_weighted_sum_rate"
                    ],
                    "mean_transmission_rate": optimized[
                        "mean_transmission_rate"
                    ],
                    "mean_reflection_rate": optimized[
                        "mean_reflection_rate"
                    ],
                    "mean_active_amplitude": optimized[
                        "mean_active_amplitude"
                    ],
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(8.5, 5.2))
    for ratio in ratios:
        selected = [row for row in rows if row["active_ratio"] == ratio]
        x = np.asarray([row["num_elements"] for row in selected], dtype=float)
        y = np.asarray(
            [row["mean_weighted_sum_rate"] for row in selected],
            dtype=float,
        )
        plt.plot(x, y, marker="o", label=f"active ratio={ratio:g}")
    plt.xlabel("Number of STAR-RIS elements")
    plt.ylabel("Average weighted sum rate (bit/s/Hz)")
    plt.title("Element-count and active-ratio sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_sum_rate.png", dpi=180)
    plt.close()

    print(f"Sweep rows: {len(rows)}")
    print(f"Saved: {csv_path}")
    print("SWEEP: PASS")


if __name__ == "__main__":
    main()
