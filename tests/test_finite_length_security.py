from __future__ import annotations

import pytest

from active_star_ris.finite_length_security import (
    FiniteLengthSecurityParameters,
    evaluate_dual_side_finite_length_security,
    evaluate_finite_length_branch,
)
from active_star_ris.secure_key_generation import (
    BranchSecrecyMetrics,
)


def _branch_metrics(
    legitimate_information: float = 8.0,
    eve_leakage: float = 0.2,
    public_leakage: float = 0.05,
) -> BranchSecrecyMetrics:
    raw_margin = (
        legitimate_information
        - eve_leakage
        - public_leakage
    )

    return BranchSecrecyMetrics(
        legitimate_mutual_information_bits_per_sample=(
            legitimate_information
        ),
        leakage_from_a_bits_per_sample=(
            0.9 * eve_leakage
        ),
        leakage_from_b_bits_per_sample=(
            eve_leakage
        ),
        eve_leakage_bits_per_sample=(
            eve_leakage
        ),
        public_leakage_bits_per_sample=(
            public_leakage
        ),
        raw_secret_key_margin_bits_per_sample=(
            raw_margin
        ),
        secret_key_rate_bits_per_sample=max(
            0.0,
            raw_margin,
        ),
    )


def test_invalid_block_configuration_raises_error() -> None:
    with pytest.raises(ValueError):
        evaluate_finite_length_branch(
            _branch_metrics(),
            FiniteLengthSecurityParameters(
                block_length=100,
                parameter_estimation_samples=100,
            ),
        )

    with pytest.raises(ValueError):
        evaluate_finite_length_branch(
            _branch_metrics(),
            FiniteLengthSecurityParameters(
                block_length=100,
                reconciliation_efficiency=1.1,
            ),
        )


def test_longer_block_increases_finite_length_rate() -> None:
    secrecy = _branch_metrics()

    short_block = evaluate_finite_length_branch(
        secrecy,
        FiniteLengthSecurityParameters(
            block_length=1000,
            parameter_estimation_samples=100,
            reconciliation_efficiency=0.95,
        ),
    )

    long_block = evaluate_finite_length_branch(
        secrecy,
        FiniteLengthSecurityParameters(
            block_length=100000,
            parameter_estimation_samples=10000,
            reconciliation_efficiency=0.95,
        ),
    )

    assert (
        long_block
        .finite_length_rate_bits_per_sample
        > short_block
        .finite_length_rate_bits_per_sample
    )

    assert (
        long_block.extractable_secret_bits
        > short_block.extractable_secret_bits
    )


def test_lower_reconciliation_efficiency_reduces_rate() -> None:
    secrecy = _branch_metrics()

    efficient = evaluate_finite_length_branch(
        secrecy,
        FiniteLengthSecurityParameters(
            block_length=10000,
            parameter_estimation_samples=1000,
            reconciliation_efficiency=0.98,
        ),
    )

    inefficient = evaluate_finite_length_branch(
        secrecy,
        FiniteLengthSecurityParameters(
            block_length=10000,
            parameter_estimation_samples=1000,
            reconciliation_efficiency=0.80,
        ),
    )

    assert (
        efficient
        .finite_length_rate_bits_per_sample
        > inefficient
        .finite_length_rate_bits_per_sample
    )

    assert (
        inefficient
        .reconciliation_loss_bits_per_sample
        > efficient
        .reconciliation_loss_bits_per_sample
    )


def test_stricter_security_probability_reduces_rate() -> None:
    secrecy = _branch_metrics()

    relaxed = evaluate_finite_length_branch(
        secrecy,
        FiniteLengthSecurityParameters(
            block_length=5000,
            parameter_estimation_samples=500,
            epsilon_smoothing=1.0e-4,
            epsilon_parameter_estimation=1.0e-4,
            epsilon_privacy_amplification=1.0e-4,
        ),
    )

    strict = evaluate_finite_length_branch(
        secrecy,
        FiniteLengthSecurityParameters(
            block_length=5000,
            parameter_estimation_samples=500,
            epsilon_smoothing=1.0e-12,
            epsilon_parameter_estimation=1.0e-12,
            epsilon_privacy_amplification=1.0e-12,
        ),
    )

    assert (
        strict
        .total_finite_length_penalty_bits_per_sample
        > relaxed
        .total_finite_length_penalty_bits_per_sample
    )

    assert (
        strict
        .finite_length_rate_bits_per_sample
        < relaxed
        .finite_length_rate_bits_per_sample
    )


def test_zero_clamping_and_dual_side_weighting() -> None:
    transmission_secrecy = _branch_metrics(
        legitimate_information=8.0,
        eve_leakage=0.2,
        public_leakage=0.05,
    )

    reflection_secrecy = _branch_metrics(
        legitimate_information=0.5,
        eve_leakage=1.0,
        public_leakage=0.1,
    )

    parameters = FiniteLengthSecurityParameters(
        block_length=10000,
        parameter_estimation_samples=1000,
        reconciliation_efficiency=0.95,
    )

    result = (
        evaluate_dual_side_finite_length_security(
            transmission_secrecy=(
                transmission_secrecy
            ),
            reflection_secrecy=(
                reflection_secrecy
            ),
            transmission_parameters=parameters,
            reflection_parameters=parameters,
            transmission_weight=0.7,
            reflection_weight=0.3,
        )
    )

    assert (
        result
        .reflection
        .finite_length_rate_bits_per_sample
        == pytest.approx(0.0)
    )

    branch_rates = [
        result
        .transmission
        .finite_length_rate_bits_per_sample,
        result
        .reflection
        .finite_length_rate_bits_per_sample,
    ]

    assert (
        min(branch_rates)
        <= result
        .weighted_finite_length_rate_bits_per_sample
        <= max(branch_rates)
    )

    expected_weighted_rate = (
        0.7
        * result
        .transmission
        .finite_length_rate_bits_per_sample
        + 0.3
        * result
        .reflection
        .finite_length_rate_bits_per_sample
    )

    assert (
        result
        .weighted_finite_length_rate_bits_per_sample
        == pytest.approx(
            expected_weighted_rate
        )
    )