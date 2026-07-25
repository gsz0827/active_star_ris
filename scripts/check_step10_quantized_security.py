from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from active_star_ris.parameter_sweep import (  # noqa: E402
    SweepExperimentConfig,
    SweepPoint,
    build_one_factor_sweep,
    run_parameter_sweep,
)
from active_star_ris.sweep_statistics import (  # noqa: E402
    aggregate_sweep_results,
)


def main() -> None:
    config = SweepExperimentConfig(
        num_samples=3000,
        num_elements=16,
        parameter_estimation_fraction=0.1,
        active_noise_variance=0.002,
        authentication_leakage_bits=128.0,
        maximum_final_key_bits=128,
        channel_seed=20260719,
        coefficient_seed=20260720,
        observation_seed=20260721,
        practical_seed=20260722,
    )

    baseline = SweepPoint(
        active_fraction=0.25,
        active_gain=1.5,
        directional_gain_error_std_db=0.0,
        directional_phase_error_std_rad=0.0,
        eve_channel_scale=1.0,
        eve_channel_correlation=0.0,
        legitimate_receiver_noise_variance=0.01,
        eve_receiver_noise_variance=0.001,
        guard_band_sigma=0.05,
        selection_policy="alice",
    )

    points = build_one_factor_sweep(
        baseline,
        "eve_channel_correlation",
        [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ],
    )

    results = run_parameter_sweep(
        config,
        points,
        num_repetitions=3,
    )

    summaries = aggregate_sweep_results(
        results
    )

    print(
        "Step 10 quantized-domain security check"
    )

    print(
        f"Parameter points = {len(points)}"
    )

    print(
        "Repetitions      = 3"
    )

    print()

    grouped_results = {}

    for result in results:
        grouped_results.setdefault(
            result.scenario,
            [],
        ).append(
            result
        )

    mean_min_entropy = {}
    mean_operational_rate = {}

    for scenario, group in grouped_results.items():
        min_entropy = sum(
            item
            .weighted_quantized_conditional_min_entropy_bits_per_retained_bit
            for item in group
        ) / len(group)

        quantized_eve_mi = sum(
            item
            .weighted_quantized_eve_mutual_information_bits_per_retained_bit
            for item in group
        ) / len(group)

        operational_rate = sum(
            item
            .weighted_operational_bound_bits_per_sample
            for item in group
        ) / len(group)

        success_rate = sum(
            float(
                item.dual_side_success
            )
            for item in group
        ) / len(group)

        mean_min_entropy[
            scenario
        ] = min_entropy

        mean_operational_rate[
            scenario
        ] = operational_rate

        print(
            f"{scenario:35s} "
            f"Q-Eve-MI={quantized_eve_mi:7.4f} "
            f"Hmin={min_entropy:7.4f} "
            f"Operational={operational_rate:7.4f} "
            f"Success={success_rate:5.2f}"
        )

    independent_name = (
        "eve_channel_correlation=0.0"
    )

    correlated_name = (
        "eve_channel_correlation=1.0"
    )

    assert (
        mean_min_entropy[
            correlated_name
        ]
        < mean_min_entropy[
            independent_name
        ]
    )

    assert (
        mean_operational_rate[
            correlated_name
        ]
        < mean_operational_rate[
            independent_name
        ]
    )

    print()
    print(
        "STEP 10 MANUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()