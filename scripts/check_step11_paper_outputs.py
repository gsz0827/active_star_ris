from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT
        / "src"
    ),
)

from active_star_ris.paper_plots import (  # noqa: E402
    generate_paper_figures,
    write_sweep_split_csvs,
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
        # 功能验证配置。
        # 正式论文实验建议使用10000以上样本。
        num_samples=2000,
        num_elements=16,
        parameter_estimation_fraction=0.1,
        active_noise_variance=0.002,
        authentication_leakage_bits=128.0,
        privacy_margin_bits=64,
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

    points = []

    points.extend(
        build_one_factor_sweep(
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
    )

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
            "guard_band_sigma",
            [
                0.0,
                0.05,
                0.10,
                0.15,
                0.25,
            ],
        )
    )

    num_repetitions = 3

    print(
        "Step 11 paper-output check"
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

    results_directory = (
        PROJECT_ROOT
        / "results"
        / "step11"
    )

    raw_csv_path = (
        results_directory
        / "step11_raw_results.csv"
    )

    summary_csv_path = (
        results_directory
        / "step11_summary_results.csv"
    )

    split_csv_directory = (
        results_directory
        / "plotting_tables"
    )

    figure_directory = (
        results_directory
        / "figures"
    )

    write_sweep_results_csv(
        raw_results,
        raw_csv_path,
    )

    write_sweep_summary_csv(
        summaries,
        summary_csv_path,
    )

    split_csv_paths = (
        write_sweep_split_csvs(
            summaries,
            split_csv_directory,
        )
    )

    figure_paths = generate_paper_figures(
        summaries,
        figure_directory,
    )

    print()
    print(
        f"Raw CSV     = {raw_csv_path}"
    )

    print(
        f"Summary CSV = {summary_csv_path}"
    )

    print(
        f"Split CSVs  = {len(split_csv_paths)}"
    )

    print(
        f"Figure files = {len(figure_paths)}"
    )

    print()
    print(
        "Quantized-domain summary"
    )

    eve_summaries = sorted(
        [
            item
            for item in summaries
            if (
                item.sweep_name
                == "eve_channel_correlation"
            )
        ],
        key=lambda item: (
            item.eve_channel_correlation
        ),
    )

    for item in eve_summaries:
        print(
            f"rho={item.eve_channel_correlation:4.2f} "
            f"Q-Eve-MI="
            f"{item.quantized_eve_mi_mean:7.4f}"
            f"±{item.quantized_eve_mi_ci95:6.4f} "
            f"Hmin="
            f"{item.quantized_min_entropy_mean:7.4f}"
            f"±{item.quantized_min_entropy_ci95:6.4f} "
            f"Operational="
            f"{item.operational_bound_rate_mean:7.4f}"
            f"±{item.operational_bound_rate_ci95:6.4f} "
            f"Success="
            f"{item.dual_side_success_rate:5.2f}"
        )

    independent_eve = eve_summaries[0]
    correlated_eve = eve_summaries[-1]

    phase_summaries = sorted(
        [
            item
            for item in summaries
            if (
                item.sweep_name
                == "directional_phase_error_std_rad"
            )
        ],
        key=lambda item: (
            item
            .directional_phase_error_std_rad
        ),
    )

    assert (
        correlated_eve.quantized_eve_mi_mean
        > independent_eve.quantized_eve_mi_mean
    )

    assert (
        correlated_eve.quantized_min_entropy_mean
        < independent_eve.quantized_min_entropy_mean
    )

    assert (
        correlated_eve.operational_bound_rate_mean
        < independent_eve.operational_bound_rate_mean
    )

    assert (
        phase_summaries[-1].legitimate_mi_mean
        < phase_summaries[0].legitimate_mi_mean
    )

    assert raw_csv_path.exists()
    assert summary_csv_path.exists()

    assert all(
        path.exists()
        for path in split_csv_paths
    )

    assert all(
        path.exists()
        for path in figure_paths
    )

    print()
    print(
        "STEP 11 MANUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()