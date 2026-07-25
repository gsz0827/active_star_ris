from __future__ import annotations

import csv
from dataclasses import replace

import pytest

from active_star_ris.parameter_sweep import (
    SweepExperimentConfig,
    SweepPoint,
    build_one_factor_sweep,
    run_parameter_point,
    run_parameter_sweep,
    write_sweep_results_csv,
)


def _config(
    num_samples: int = 1200,
) -> SweepExperimentConfig:
    return SweepExperimentConfig(
        num_samples=num_samples,
        num_elements=12,
        parameter_estimation_fraction=0.1,
        active_noise_variance=0.001,
        maximum_final_key_bits=64,
        channel_seed=81,
        coefficient_seed=82,
        observation_seed=83,
        practical_seed=84,
    )


def _baseline_point() -> SweepPoint:
    return SweepPoint(
        active_fraction=0.25,
        active_gain=1.5,
        eve_channel_scale=0.3,
        eve_channel_correlation=0.0,
        legitimate_receiver_noise_variance=0.01,
        eve_receiver_noise_variance=0.2,
        guard_band_sigma=0.05,
        selection_policy="alice",
    )


def test_one_factor_builder_changes_only_target() -> None:
    base = _baseline_point()

    points = build_one_factor_sweep(
        base,
        "active_gain",
        [1.0, 1.5, 2.0],
    )

    assert len(points) == 3

    assert [
        point.active_gain
        for point in points
    ] == [1.0, 1.5, 2.0]

    assert all(
        point.active_fraction
        == base.active_fraction
        for point in points
    )

    assert all(
        point.eve_channel_correlation
        == base.eve_channel_correlation
        for point in points
    )


def test_single_point_is_reproducible() -> None:
    config = _config()
    point = _baseline_point()

    first = run_parameter_point(
        config,
        point,
        repetition=0,
    )

    second = run_parameter_point(
        config,
        point,
        repetition=0,
    )

    assert first == second

    assert (
        0.0
        <= first.transmission_raw_kdr
        <= 1.0
    )

    assert (
        0.0
        <= first.reflection_raw_kdr
        <= 1.0
    )

    assert (
        first.aggregate_final_key_bits
        >= 0
    )


def test_strong_directional_mismatch_reduces_mi() -> None:
    config = _config(
        num_samples=2500
    )

    ideal = replace(
        _baseline_point(),
        directional_gain_error_std_db=0.0,
        directional_phase_error_std_rad=0.0,
    )

    impaired = replace(
        ideal,
        directional_gain_error_std_db=3.0,
        directional_phase_error_std_rad=1.0,
    )

    ideal_result = run_parameter_point(
        config,
        ideal,
    )

    impaired_result = run_parameter_point(
        config,
        impaired,
    )

    assert (
        impaired_result
        .weighted_legitimate_mi_bits_per_sample
        < ideal_result
        .weighted_legitimate_mi_bits_per_sample
    )


def test_correlated_eve_increases_leakage() -> None:
    config = _config(
        num_samples=2500
    )

    independent_eve = replace(
        _baseline_point(),
        eve_channel_scale=1.0,
        eve_channel_correlation=0.0,
        eve_receiver_noise_variance=0.001,
    )

    colocated_eve = replace(
        independent_eve,
        eve_channel_correlation=1.0,
    )

    independent_result = run_parameter_point(
        config,
        independent_eve,
    )

    colocated_result = run_parameter_point(
        config,
        colocated_eve,
    )

    assert (
        colocated_result
        .weighted_eve_leakage_bits_per_sample
        > independent_result
        .weighted_eve_leakage_bits_per_sample
    )

    assert (
        colocated_result
        .weighted_asymptotic_secret_rate_bits_per_sample
        <= independent_result
        .weighted_asymptotic_secret_rate_bits_per_sample
    )


def test_repeated_sweep_and_csv_export(
    tmp_path,
) -> None:
    config = _config(
        num_samples=700
    )

    points = build_one_factor_sweep(
        _baseline_point(),
        "guard_band_sigma",
        [0.0, 0.1],
    )

    results = run_parameter_sweep(
        config,
        points,
        num_repetitions=2,
    )

    assert len(results) == 4

    output_path = (
        tmp_path
        / "parameter_sweep.csv"
    )

    written_path = (
        write_sweep_results_csv(
            results,
            output_path,
        )
    )

    assert written_path.exists()

    with written_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(
                file
            )
        )

    assert len(rows) == 4

    assert (
        "weighted_finite_length_rate_bits_per_sample"
        in rows[0]
    )

    assert (
        "aggregate_final_key_bits"
        in rows[0]
    )