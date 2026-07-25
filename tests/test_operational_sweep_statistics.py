from __future__ import annotations

import csv
from dataclasses import replace
from math import floor

import pytest

from active_star_ris.finite_length_security import (
    FiniteLengthSecurityParameters,
    evaluate_finite_length_branch,
)
from active_star_ris.operational_security import (
    evaluate_operational_pre_reconciliation_bound,
)
from active_star_ris.parameter_sweep import (
    SweepExperimentConfig,
    SweepPoint,
    run_parameter_point,
)
from active_star_ris.secure_key_generation import (
    BranchSecrecyMetrics,
)
from active_star_ris.sweep_statistics import (
    aggregate_sweep_results,
    write_sweep_summary_csv,
)


def _branch_metrics() -> BranchSecrecyMetrics:
    return BranchSecrecyMetrics(
        legitimate_mutual_information_bits_per_sample=8.0,
        leakage_from_a_bits_per_sample=0.1,
        leakage_from_b_bits_per_sample=0.2,
        eve_leakage_bits_per_sample=0.2,
        public_leakage_bits_per_sample=0.05,
        raw_secret_key_margin_bits_per_sample=7.75,
        secret_key_rate_bits_per_sample=7.75,
    )


def _finite_metrics():
    return evaluate_finite_length_branch(
        _branch_metrics(),
        FiniteLengthSecurityParameters(
            block_length=1000,
            parameter_estimation_samples=100,
            reconciliation_efficiency=0.5,
            authentication_leakage_bits=128.0,
            implementation_margin_bits_per_sample=0.01,
        ),
    )


def test_operational_bound_uses_unreconciled_information() -> None:
    finite = _finite_metrics()

    result = (
        evaluate_operational_pre_reconciliation_bound(
            _branch_metrics(),
            finite,
            authentication_leakage_bits=128.0,
        )
    )

    expected_rate = (
        8.0
        - 0.2
        - 0.05
        - finite.aep_penalty_bits_per_sample
        - finite.parameter_estimation_penalty_bits_per_sample
        - 0.01
    )

    expected_bits = max(
        0,
        floor(
            900
            * max(
                0.0,
                expected_rate,
            )
        )
        - 128,
    )

    assert (
        result
        .pre_reconciliation_entropy_bound_bits
        == expected_bits
    )

    # 实际协议上界不使用beta=0.5，
    # 因此应高于已扣除协调效率的代理值。
    assert (
        result
        .pre_reconciliation_entropy_bound_bits
        > finite.extractable_secret_bits
    )


def test_authentication_leakage_reduces_bound() -> None:
    finite = _finite_metrics()

    without_authentication = (
        evaluate_operational_pre_reconciliation_bound(
            _branch_metrics(),
            finite,
            authentication_leakage_bits=0.0,
        )
    )

    with_authentication = (
        evaluate_operational_pre_reconciliation_bound(
            _branch_metrics(),
            finite,
            authentication_leakage_bits=128.0,
        )
    )

    assert (
        without_authentication
        .pre_reconciliation_entropy_bound_bits
        - with_authentication
        .pre_reconciliation_entropy_bound_bits
        == 128
    )


def _small_result():
    config = SweepExperimentConfig(
        num_samples=700,
        num_elements=8,
        parameter_estimation_fraction=0.1,
        maximum_final_key_bits=32,
        channel_seed=91,
        coefficient_seed=92,
        observation_seed=93,
        practical_seed=94,
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


def test_aggregation_calculates_mean_and_success_rate() -> None:
    first = _small_result()

    second = replace(
        first,
        repetition=1,
        weighted_legitimate_mi_bits_per_sample=(
            first
            .weighted_legitimate_mi_bits_per_sample
            + 2.0
        ),
        dual_side_success=False,
    )

    summaries = aggregate_sweep_results(
        [
            first,
            second,
        ]
    )

    assert len(summaries) == 1

    summary = summaries[0]

    assert summary.num_repetitions == 2

    assert (
        summary.legitimate_mi_mean
        == pytest.approx(
            first
            .weighted_legitimate_mi_bits_per_sample
            + 1.0
        )
    )

    expected_success_rate = (
        float(first.dual_side_success)
        + 0.0
    ) / 2.0

    assert (
        summary.dual_side_success_rate
        == pytest.approx(
            expected_success_rate
        )
    )


def test_identical_repetitions_have_zero_ci() -> None:
    first = _small_result()

    second = replace(
        first,
        repetition=1,
    )

    summary = aggregate_sweep_results(
        [
            first,
            second,
        ]
    )[0]

    assert (
        summary.legitimate_mi_std
        == pytest.approx(0.0)
    )

    assert (
        summary.legitimate_mi_ci95
        == pytest.approx(0.0)
    )


def test_summary_csv_export(
    tmp_path,
) -> None:
    first = _small_result()

    second = replace(
        first,
        repetition=1,
    )

    summary = aggregate_sweep_results(
        [
            first,
            second,
        ]
    )

    output_path = (
        tmp_path
        / "sweep_summary.csv"
    )

    written_path = write_sweep_summary_csv(
        summary,
        output_path,
    )

    assert written_path.exists()

    with written_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 1

    assert (
        "operational_bound_rate_mean"
        in rows[0]
    )

    assert (
        "dual_side_success_rate"
        in rows[0]
    )