from __future__ import annotations

import csv
from dataclasses import replace
from math import sqrt

from active_star_ris.paper_plots import (
    generate_paper_figures,
    write_sweep_split_csvs,
)
from active_star_ris.parameter_sweep import (
    SweepExperimentConfig,
    SweepPoint,
    run_parameter_point,
)
from active_star_ris.sweep_statistics import (
    aggregate_sweep_results,
    write_sweep_summary_csv,
)


def _small_result():
    config = SweepExperimentConfig(
        num_samples=700,
        num_elements=8,
        parameter_estimation_fraction=0.1,
        authentication_leakage_bits=32.0,
        privacy_margin_bits=16,
        maximum_final_key_bits=32,
        channel_seed=121,
        coefficient_seed=122,
        observation_seed=123,
        practical_seed=124,
    )

    point = SweepPoint(
        sweep_name="test",
        scenario="test-point",
        active_fraction=0.25,
        active_gain=1.5,
        eve_channel_scale=0.2,
        eve_channel_correlation=0.0,
        legitimate_receiver_noise_variance=0.01,
        eve_receiver_noise_variance=0.5,
        guard_band_sigma=0.05,
        selection_policy="alice",
    )

    return run_parameter_point(
        config,
        point,
        repetition=0,
    )


def test_three_repetitions_use_student_t_interval() -> None:
    first = _small_result()

    second = replace(
        first,
        repetition=1,
        weighted_legitimate_mi_bits_per_sample=(
            first
            .weighted_legitimate_mi_bits_per_sample
            + 1.0
        ),
    )

    third = replace(
        first,
        repetition=2,
        weighted_legitimate_mi_bits_per_sample=(
            first
            .weighted_legitimate_mi_bits_per_sample
            + 2.0
        ),
    )

    summary = aggregate_sweep_results(
        [
            first,
            second,
            third,
        ]
    )[0]

    normal_ci = (
        1.96
        * summary.legitimate_mi_std
        / sqrt(3.0)
    )

    assert (
        summary.legitimate_mi_ci95
        > normal_ci
    )


def test_quantized_metrics_are_aggregated() -> None:
    first = _small_result()

    second = replace(
        first,
        repetition=1,
        weighted_quantized_eve_mutual_information_bits_per_retained_bit=(
            first
            .weighted_quantized_eve_mutual_information_bits_per_retained_bit
            + 0.2
        ),
        weighted_quantized_conditional_min_entropy_bits_per_retained_bit=(
            max(
                0.0,
                first
                .weighted_quantized_conditional_min_entropy_bits_per_retained_bit
                - 0.2,
            )
        ),
    )

    summary = aggregate_sweep_results(
        [
            first,
            second,
        ]
    )[0]

    expected_eve_mi = (
        first
        .weighted_quantized_eve_mutual_information_bits_per_retained_bit
        + 0.1
    )

    assert (
        summary.quantized_eve_mi_mean
        == expected_eve_mi
    )

    assert (
        0.0
        <= summary.quantized_min_entropy_mean
        <= 1.0
    )


def test_summary_csv_contains_quantized_fields(
    tmp_path,
) -> None:
    first = _small_result()

    summary = aggregate_sweep_results(
        [first]
    )

    output_path = (
        tmp_path
        / "summary.csv"
    )

    write_sweep_summary_csv(
        summary,
        output_path,
    )

    with output_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 1

    assert (
        "quantized_eve_mi_mean"
        in rows[0]
    )

    assert (
        "quantized_min_entropy_mean"
        in rows[0]
    )

    assert (
        "operational_bound_rate_ci95"
        in rows[0]
    )


def test_split_csv_export(
    tmp_path,
) -> None:
    first = _small_result()

    second = replace(
        first,
        sweep_name="other_sweep",
        scenario="other-point",
    )

    summaries = aggregate_sweep_results(
        [
            first,
            second,
        ]
    )

    output_paths = write_sweep_split_csvs(
        summaries,
        tmp_path,
    )

    assert len(output_paths) == 2

    assert all(
        path.exists()
        for path in output_paths
    )


def test_paper_figures_are_generated(
    tmp_path,
) -> None:
    first_result = _small_result()

    base_summary = aggregate_sweep_results(
        [first_result]
    )[0]

    summaries = []

    for correlation in [
        0.0,
        0.5,
        1.0,
    ]:
        summaries.append(
            replace(
                base_summary,
                sweep_name=(
                    "eve_channel_correlation"
                ),
                scenario=(
                    f"eve_channel_correlation="
                    f"{correlation}"
                ),
                eve_channel_correlation=(
                    correlation
                ),
                quantized_eve_mi_mean=(
                    0.7 * correlation
                ),
                quantized_min_entropy_mean=(
                    0.8
                    * (
                        1.0
                        - correlation
                    )
                ),
                operational_bound_rate_mean=(
                    0.4
                    * (
                        1.0
                        - correlation
                    )
                ),
                dual_side_success_rate=(
                    1.0
                    if correlation < 0.5
                    else 0.0
                ),
            )
        )

    for phase_error in [
        0.0,
        0.2,
        0.5,
    ]:
        summaries.append(
            replace(
                base_summary,
                sweep_name=(
                    "directional_phase_error_std_rad"
                ),
                scenario=(
                    "directional_phase_error_std_rad="
                    f"{phase_error}"
                ),
                directional_phase_error_std_rad=(
                    phase_error
                ),
                legitimate_mi_mean=(
                    8.0
                    - 8.0
                    * phase_error
                ),
                operational_bound_rate_mean=(
                    max(
                        0.0,
                        0.5
                        - phase_error,
                    )
                ),
            )
        )

    for guard_band in [
        0.0,
        0.1,
        0.25,
    ]:
        summaries.append(
            replace(
                base_summary,
                sweep_name="guard_band_sigma",
                scenario=(
                    f"guard_band_sigma={guard_band}"
                ),
                guard_band_sigma=guard_band,
                transmission_raw_kdr_mean=(
                    max(
                        0.0,
                        0.02
                        - 0.05
                        * guard_band,
                    )
                ),
                reflection_raw_kdr_mean=(
                    max(
                        0.0,
                        0.03
                        - 0.05
                        * guard_band,
                    )
                ),
                transmission_retention_ratio_mean=(
                    1.0
                    - 0.5
                    * guard_band
                ),
                reflection_retention_ratio_mean=(
                    1.0
                    - 0.55
                    * guard_band
                ),
            )
        )

    figure_paths = generate_paper_figures(
        summaries,
        tmp_path,
    )

    assert len(figure_paths) >= 10

    assert all(
        path.exists()
        for path in figure_paths
    )