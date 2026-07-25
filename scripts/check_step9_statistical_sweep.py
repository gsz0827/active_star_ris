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
    write_sweep_results_csv,
)
from active_star_ris.sweep_statistics import (  # noqa: E402
    aggregate_sweep_results,
    write_sweep_summary_csv,
)


def main() -> None:
    config = SweepExperimentConfig(
        num_samples=1500,
        num_elements=16,
        parameter_estimation_fraction=0.1,
        beta_transmission=0.65,
        active_noise_variance=0.002,
        reconciliation_efficiency=0.95,
        authentication_leakage_bits=128.0,
        implementation_margin_bits_per_sample=0.01,
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
        eve_channel_scale=0.2,
        eve_channel_correlation=0.0,
        legitimate_receiver_noise_variance=0.01,
        eve_receiver_noise_variance=0.5,
        guard_band_sigma=0.05,
        selection_policy="alice",
    )

    points = []

    points.extend(
        build_one_factor_sweep(
            baseline,
            "directional_phase_error_std_rad",
            [
                0.0,
                0.1,
                0.2,
                0.5,
            ],
        )
    )

    points.extend(
        build_one_factor_sweep(
            baseline,
            "eve_channel_correlation",
            [
                0.0,
                0.5,
                1.0,
            ],
        )
    )

    points.extend(
        build_one_factor_sweep(
            baseline,
            "guard_band_sigma",
            [
                0.0,
                0.05,
                0.15,
                0.25,
            ],
        )
    )

    num_repetitions = 3

    print(
        "Step 9 statistical sweep check"
    )

    print(
        f"Parameter points = {len(points)}"
    )

    print(
        f"Repetitions      = {num_repetitions}"
    )

    print(
        f"Total runs       = "
        f"{len(points) * num_repetitions}"
    )

    raw_results = run_parameter_sweep(
        config,
        points,
        num_repetitions=(
            num_repetitions
        ),
    )

    summaries = aggregate_sweep_results(
        raw_results
    )

    raw_output_path = (
        PROJECT_ROOT
        / "results"
        / "step9_statistical_sweep_raw.csv"
    )

    summary_output_path = (
        PROJECT_ROOT
        / "results"
        / "step9_statistical_sweep_summary.csv"
    )

    write_sweep_results_csv(
        raw_results,
        raw_output_path,
    )

    write_sweep_summary_csv(
        summaries,
        summary_output_path,
    )

    print()
    print(
        f"Raw CSV     = {raw_output_path}"
    )

    print(
        f"Summary CSV = {summary_output_path}"
    )

    print()
    print(
        "Mean ± 95% CI summary"
    )

    for summary in summaries:
        print(
            f"{summary.sweep_name:42s} "
            f"{summary.scenario:46s} "
            f"MI="
            f"{summary.legitimate_mi_mean:7.4f}"
            f"±{summary.legitimate_mi_ci95:6.4f} "
            f"Eve="
            f"{summary.eve_leakage_mean:7.4f}"
            f"±{summary.eve_leakage_ci95:6.4f} "
            f"Operational="
            f"{summary.operational_bound_rate_mean:7.4f}"
            f"±{summary.operational_bound_rate_ci95:6.4f} "
            f"Success="
            f"{summary.dual_side_success_rate:5.2f}"
        )

    ideal_phase = next(
        item
        for item in summaries
        if (
            item.sweep_name
            == "directional_phase_error_std_rad"
            and item.scenario
            == "directional_phase_error_std_rad=0.0"
        )
    )

    strong_phase = next(
        item
        for item in summaries
        if (
            item.sweep_name
            == "directional_phase_error_std_rad"
            and item.scenario
            == "directional_phase_error_std_rad=0.5"
        )
    )

    independent_eve = next(
        item
        for item in summaries
        if (
            item.sweep_name
            == "eve_channel_correlation"
            and item.scenario
            == "eve_channel_correlation=0.0"
        )
    )

    correlated_eve = next(
        item
        for item in summaries
        if (
            item.sweep_name
            == "eve_channel_correlation"
            and item.scenario
            == "eve_channel_correlation=1.0"
        )
    )

    assert len(raw_results) == (
        len(points)
        * num_repetitions
    )

    assert len(summaries) == len(
        points
    )

    assert (
        strong_phase.legitimate_mi_mean
        < ideal_phase.legitimate_mi_mean
    )

    assert (
        correlated_eve.eve_leakage_mean
        > independent_eve.eve_leakage_mean
    )

    assert raw_output_path.exists()
    assert summary_output_path.exists()

    print()
    print(
        "STEP 9 MANUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()