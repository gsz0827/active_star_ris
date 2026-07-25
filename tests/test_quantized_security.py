from __future__ import annotations

import numpy as np
import pytest

from active_star_ris.parameter_sweep import (
    SweepExperimentConfig,
    SweepPoint,
    run_parameter_point,
)
from active_star_ris.practical_key_generation import (
    generate_end_to_end_key_from_quantization,
    quantize_with_guard_band,
)
from active_star_ris.quantized_security import (
    evaluate_quantized_eve_security,
    evaluate_quantized_pre_reconciliation_bound,
)


def _complex_gaussian(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    scale: float = 1.0,
) -> np.ndarray:
    return (
        scale
        * (
            rng.normal(size=shape)
            + 1j * rng.normal(size=shape)
        )
        / np.sqrt(2.0)
    )


def _quantization_fixture():
    rng = np.random.default_rng(101)

    num_samples = 5000

    source = _complex_gaussian(
        rng,
        (num_samples,),
    )

    observation_a = (
        source
        + _complex_gaussian(
            rng,
            (num_samples,),
            scale=0.05,
        )
    )

    observation_b = (
        source
        + _complex_gaussian(
            rng,
            (num_samples,),
            scale=0.05,
        )
    )

    quantization = quantize_with_guard_band(
        observation_a,
        observation_b,
        feature="real",
        guard_band_sigma=0.05,
        selection_policy="alice",
    )

    return (
        rng,
        observation_a,
        quantization,
    )


def test_independent_eve_has_high_conditional_entropy() -> None:
    (
        rng,
        observation_a,
        quantization,
    ) = _quantization_fixture()

    eve_forward = _complex_gaussian(
        rng,
        observation_a.shape,
    )

    eve_reverse = _complex_gaussian(
        rng,
        observation_a.shape,
    )

    metrics = evaluate_quantized_eve_security(
        quantization,
        eve_forward,
        eve_reverse,
        estimation_failure_probability=1.0e-6,
        rng=np.random.default_rng(102),
    )

    assert (
        metrics
        .conditional_min_entropy_lower_bound_bits_per_retained_bit
        > 0.6
    )

    assert (
        metrics
        .eve_guessing_probability_upper_bound
        < 0.7
    )


def test_informative_eve_reduces_conditional_entropy() -> None:
    (
        rng,
        observation_a,
        quantization,
    ) = _quantization_fixture()

    independent_metrics = (
        evaluate_quantized_eve_security(
            quantization,
            _complex_gaussian(
                rng,
                observation_a.shape,
            ),
            _complex_gaussian(
                rng,
                observation_a.shape,
            ),
            rng=np.random.default_rng(103),
        )
    )

    informative_eve_forward = (
        observation_a
        + _complex_gaussian(
            rng,
            observation_a.shape,
            scale=0.01,
        )
    )

    informative_eve_reverse = (
        observation_a
        + _complex_gaussian(
            rng,
            observation_a.shape,
            scale=0.02,
        )
    )

    informative_metrics = (
        evaluate_quantized_eve_security(
            quantization,
            informative_eve_forward,
            informative_eve_reverse,
            rng=np.random.default_rng(104),
        )
    )

    assert (
        informative_metrics
        .eve_guessing_probability_upper_bound
        > independent_metrics
        .eve_guessing_probability_upper_bound
    )

    assert (
        informative_metrics
        .conditional_min_entropy_lower_bound_bits_per_retained_bit
        < independent_metrics
        .conditional_min_entropy_lower_bound_bits_per_retained_bit
    )


def test_authentication_leakage_is_subtracted() -> None:
    (
        rng,
        observation_a,
        quantization,
    ) = _quantization_fixture()

    metrics = evaluate_quantized_eve_security(
        quantization,
        _complex_gaussian(
            rng,
            observation_a.shape,
        ),
        _complex_gaussian(
            rng,
            observation_a.shape,
        ),
        rng=np.random.default_rng(105),
    )

    without_authentication = (
        evaluate_quantized_pre_reconciliation_bound(
            metrics,
            authentication_leakage_bits=0.0,
        )
    )

    with_authentication = (
        evaluate_quantized_pre_reconciliation_bound(
            metrics,
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


def test_precomputed_quantization_generates_matching_key() -> None:
    (
        rng,
        observation_a,
        quantization,
    ) = _quantization_fixture()

    metrics = evaluate_quantized_eve_security(
        quantization,
        _complex_gaussian(
            rng,
            observation_a.shape,
        ),
        _complex_gaussian(
            rng,
            observation_a.shape,
        ),
        rng=np.random.default_rng(106),
    )

    bound = evaluate_quantized_pre_reconciliation_bound(
        metrics,
        authentication_leakage_bits=64.0,
    )

    result = generate_end_to_end_key_from_quantization(
        quantization,
        pre_reconciliation_entropy_bound_bits=(
            bound
            .pre_reconciliation_entropy_bound_bits
        ),
        maximum_final_key_bits=128,
        rng=np.random.default_rng(107),
    )

    assert result.success

    assert (
        result.final_key_length_bits
        == 128
    )

    assert (
        result.reconciliation
        .post_reconciliation_kdr
        == pytest.approx(0.0)
    )


def test_parameter_point_exposes_quantized_metrics() -> None:
    config = SweepExperimentConfig(
        num_samples=1500,
        num_elements=10,
        parameter_estimation_fraction=0.1,
        authentication_leakage_bits=64.0,
        maximum_final_key_bits=64,
        channel_seed=111,
        coefficient_seed=112,
        observation_seed=113,
        practical_seed=114,
    )

    point = SweepPoint(
        eve_channel_scale=0.5,
        eve_channel_correlation=0.5,
        eve_receiver_noise_variance=0.05,
        guard_band_sigma=0.05,
        selection_policy="alice",
    )

    result = run_parameter_point(
        config,
        point,
    )

    assert (
        0.0
        <= result
        .weighted_quantized_eve_mutual_information_bits_per_retained_bit
        <= 1.0
    )

    assert (
        0.0
        <= result
        .weighted_quantized_conditional_min_entropy_bits_per_retained_bit
        <= 1.0
    )

    assert (
        result
        .aggregate_operational_secret_bit_bound
        >= 0
    )