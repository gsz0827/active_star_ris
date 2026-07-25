from __future__ import annotations

import numpy as np
import pytest

from active_star_ris.star_key_system import (
    build_star_coefficients,
    energy_splitting_residual,
    simulate_dual_side_key_generation,
)


def test_star_coefficients_satisfy_energy_splitting() -> None:
    amplitudes = np.array(
        [1.0, 1.5, 2.0]
    )

    beta_t = np.array(
        [0.2, 0.5, 0.8]
    )

    beta_r = 1.0 - beta_t

    coefficients = build_star_coefficients(
        amplitudes=amplitudes,
        beta_transmission=beta_t,
        beta_reflection=beta_r,
        phase_transmission=np.zeros(3),
        phase_reflection=np.zeros(3),
    )

    np.testing.assert_allclose(
        energy_splitting_residual(
            coefficients
        ),
        0.0,
        atol=1.0e-12,
    )


def test_invalid_energy_splitting_raises_error() -> None:
    with pytest.raises(ValueError):
        build_star_coefficients(
            amplitudes=np.ones(3),
            beta_transmission=np.array(
                [0.6, 0.6, 0.6]
            ),
            beta_reflection=np.array(
                [0.6, 0.4, 0.4]
            ),
            phase_transmission=np.zeros(3),
            phase_reflection=np.zeros(3),
        )


def test_zero_noise_gives_perfect_reciprocity() -> None:
    rng = np.random.default_rng(21)

    num_samples = 300
    num_elements = 8

    g = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    )

    h_t = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    )

    h_r = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    )

    coefficients = build_star_coefficients(
        amplitudes=np.ones(
            num_elements
        ),
        beta_transmission=np.full(
            num_elements,
            0.5,
        ),
        beta_reflection=np.full(
            num_elements,
            0.5,
        ),
        phase_transmission=np.zeros(
            num_elements
        ),
        phase_reflection=np.zeros(
            num_elements
        ),
    )

    result = simulate_dual_side_key_generation(
        channel_controller_to_ris=g,
        channel_ris_to_transmission_user=h_t,
        channel_ris_to_reflection_user=h_r,
        coefficients=coefficients,
        active_mask=np.zeros(
            num_elements,
            dtype=bool,
        ),
        active_noise_variance=0.0,
        receiver_noise_variance_controller=0.0,
        receiver_noise_variance_transmission_user=0.0,
        receiver_noise_variance_reflection_user=0.0,
        rng=rng,
    )

    assert (
        result.transmission
        .metrics
        .correlation_magnitude
        == pytest.approx(1.0)
    )

    assert (
        result.reflection
        .metrics
        .correlation_magnitude
        == pytest.approx(1.0)
    )

    assert (
        result.transmission
        .metrics
        .key_disagreement_rate
        == pytest.approx(0.0)
    )

    assert (
        result.reflection
        .metrics
        .key_disagreement_rate
        == pytest.approx(0.0)
    )


def test_transmission_and_reflection_are_distinct() -> None:
    rng = np.random.default_rng(22)

    num_samples = 200
    num_elements = 6

    g = np.ones(
        (num_samples, num_elements),
        dtype=np.complex128,
    )

    h_t = np.ones(
        (num_samples, num_elements),
        dtype=np.complex128,
    )

    h_r = np.full(
        (num_samples, num_elements),
        2.0 + 0j,
        dtype=np.complex128,
    )

    coefficients = build_star_coefficients(
        amplitudes=np.ones(
            num_elements
        ),
        beta_transmission=np.full(
            num_elements,
            0.7,
        ),
        beta_reflection=np.full(
            num_elements,
            0.3,
        ),
        phase_transmission=np.zeros(
            num_elements
        ),
        phase_reflection=np.zeros(
            num_elements
        ),
    )

    result = simulate_dual_side_key_generation(
        channel_controller_to_ris=g,
        channel_ris_to_transmission_user=h_t,
        channel_ris_to_reflection_user=h_r,
        coefficients=coefficients,
        active_mask=np.zeros(
            num_elements,
            dtype=bool,
        ),
        rng=rng,
    )

    assert not np.allclose(
        result.transmission
        .probing
        .effective_channel,
        result.reflection
        .probing
        .effective_channel,
    )


def test_weighted_metrics_lie_between_branch_values() -> None:
    rng = np.random.default_rng(23)

    num_samples = 2000
    num_elements = 10

    g = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    h_t = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    h_r = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    coefficients = build_star_coefficients(
        amplitudes=np.ones(
            num_elements
        ),
        beta_transmission=np.full(
            num_elements,
            0.6,
        ),
        beta_reflection=np.full(
            num_elements,
            0.4,
        ),
        phase_transmission=np.zeros(
            num_elements
        ),
        phase_reflection=np.zeros(
            num_elements
        ),
    )

    result = simulate_dual_side_key_generation(
        channel_controller_to_ris=g,
        channel_ris_to_transmission_user=h_t,
        channel_ris_to_reflection_user=h_r,
        coefficients=coefficients,
        active_mask=np.zeros(
            num_elements,
            dtype=bool,
        ),
        receiver_noise_variance_controller=0.01,
        receiver_noise_variance_transmission_user=0.01,
        receiver_noise_variance_reflection_user=0.02,
        transmission_weight=0.7,
        reflection_weight=0.3,
        rng=rng,
    )

    branch_correlations = [
        result.transmission
        .metrics
        .correlation_magnitude,
        result.reflection
        .metrics
        .correlation_magnitude,
    ]

    assert (
        min(branch_correlations)
        <= result.weighted_correlation
        <= max(branch_correlations)
    )

    branch_kdrs = [
        result.transmission
        .metrics
        .key_disagreement_rate,
        result.reflection
        .metrics
        .key_disagreement_rate,
    ]

    assert (
        min(branch_kdrs)
        <= (
            result
            .weighted_key_disagreement_rate
        )
        <= max(branch_kdrs)
    )