from __future__ import annotations

import argparse
import json
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


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short validation experiment.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    trials = (
        60
        if args.quick
        else int(config["simulation"]["monte_carlo_trials"])
    )
    seed = int(config["seed"])

    results = monte_carlo(config, trials, seed)
    summary = summarize_results(results)

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "main_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    names = list(summary)
    rate_t = [summary[name]["mean_transmission_rate"] for name in names]
    rate_r = [summary[name]["mean_reflection_rate"] for name in names]
    snr_t_db = [
        10.0 * np.log10(summary[name]["mean_transmission_snr"])
        for name in names
    ]
    snr_r_db = [
        10.0 * np.log10(summary[name]["mean_reflection_snr"])
        for name in names
    ]

    x = np.arange(len(names), dtype=float)
    width = 0.36

    plt.figure(figsize=(10.0, 5.2))
    plt.bar(x - width / 2, rate_t, width, label="Transmission user")
    plt.bar(x + width / 2, rate_r, width, label="Reflection user")
    plt.xticks(x, names, rotation=18)
    plt.ylabel("Average rate (bit/s/Hz)")
    plt.title("Average rates under different STAR-RIS designs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "main_average_rates.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10.0, 5.2))
    plt.bar(x - width / 2, snr_t_db, width, label="Transmission user")
    plt.bar(x + width / 2, snr_r_db, width, label="Reflection user")
    plt.xticks(x, names, rotation=18)
    plt.ylabel("Average SNR (dB)")
    plt.title("Average SNR under different STAR-RIS designs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "main_average_snrs.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    for name, entries in results.items():
        values = np.asarray(
            [entry.weighted_sum_rate for entry in entries],
            dtype=float,
        )
        cdf_x, cdf_y = empirical_cdf(values)
        plt.plot(cdf_x, cdf_y, label=name)
    plt.xlabel("Weighted sum rate (bit/s/Hz)")
    plt.ylabel("Empirical CDF")
    plt.title("Weighted-sum-rate distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "main_rate_cdf.png", dpi=180)
    plt.close()

    print(f"Monte Carlo trials: {trials}")
    for name in names:
        item = summary[name]
        print(
            f"{name:30s} "
            f"R_T={item['mean_transmission_rate']:.4f}, "
            f"R_R={item['mean_reflection_rate']:.4f}, "
            f"WSR={item['mean_weighted_sum_rate']:.4f}, "
            f"P_RIS={item['mean_ris_output_power']:.4f}"
        )
    print("MAIN EXPERIMENT: PASS")


if __name__ == "__main__":
    main()
