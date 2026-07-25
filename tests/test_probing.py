from __future__ import annotations

import numpy as np
import pytest

from active_star_ris.key_generation import (
    evaluate_key_generation,
)
from active_star_ris.probing import (
    effective_star_channel,
    simulate_bidirectional_probing,
)


def test_effective_star_channel_value() -> None:
    g = np.array(
        [1 + 0j, 2 + 0j]
    )
    h = np.array(
        [3 + 0j, 4 + 0j]
    )
    phi = np.array(
        [1 + 0j, 1 + 0j]
    )

    channel = effective_star_channel(
        g,
        h,
        phi,
        direct_channel=2 + 0j,
    )

    expected = (
        2
        + 1 * 3
        + 2 * 4
    )

    assert channel.shape == (1,)
    assert channel[0] == pytest.approx(
        expected
    )


def test_no_noise_gives_identical_observations() -> None:
    rng = np.random.default_rng(10)

    num_samples = 100
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

    h = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    )

    phi = np.ones(
        num_elements,
        dtype=np.complex128,
    )

    result = simulate_bidirectional_probing(
        g,
        h,
        phi,
        active_mask=np.zeros(
            num_elements,
            dtype=bool,
        ),
        active_noise_variance=0.0,
        receiver_noise_variance_a=0.0,
        receiver_noise_variance_b=0.0,
        rng=rng,
    )

    np.testing.assert_allclose(
        result.observation_at_a,
        result.effective_channel,
    )

    np.testing.assert_allclose(
        result.observation_at_b,
        result.effective_channel,
    )

    np.testing.assert_allclose(
        result.observation_at_a,
        result.observation_at_b,
    )


def test_passive_elements_do_not_generate_active_noise() -> None:
    rng = np.random.default_rng(11)

    num_samples = 200
    num_elements = 6

    g = np.ones(
        (num_samples, num_elements),
        dtype=np.complex128,
    )

    h = np.ones(
        (num_samples, num_elements),
        dtype=np.complex128,
    )

    phi = np.ones(
        num_elements,
        dtype=np.complex128,
    )

    result = simulate_bidirectional_probing(
        g,
        h,
        phi,
        active_mask=np.zeros(
            num_elements,
            dtype=bool,
        ),
        active_noise_variance=10.0,
        receiver_noise_variance_a=0.0,
        receiver_noise_variance_b=0.0,
        rng=rng,
    )

    np.testing.assert_allclose(
        result.forwarded_active_noise_at_a,
        0.0,
    )

    np.testing.assert_allclose(
        result.forwarded_active_noise_at_b,
        0.0,
    )


def test_active_noise_is_direction_dependent() -> None:
    rng = np.random.default_rng(12)

    num_samples = 500
    num_elements = 8

    g = np.ones(
        (num_samples, num_elements),
        dtype=np.complex128,
    )

    h = np.ones(
        (num_samples, num_elements),
        dtype=np.complex128,
    )

    phi = np.full(
        num_elements,
        1.5 + 0j,
        dtype=np.complex128,
    )

    result = simulate_bidirectional_probing(
        g,
        h,
        phi,
        active_mask=np.ones(
            num_elements,
            dtype=bool,
        ),
        active_noise_variance=0.1,
        receiver_noise_variance_a=0.0,
        receiver_noise_variance_b=0.0,
        rng=rng,
    )

    assert not np.allclose(
        result.forwarded_active_noise_at_a,
        result.forwarded_active_noise_at_b,
    )

    assert not np.allclose(
        result.observation_at_a,
        result.observation_at_b,
    )


def test_probing_results_can_compute_key_metrics() -> None:
    rng = np.random.default_rng(13)

    num_samples = 4000
    num_elements = 12

    g = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    h = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    phases = rng.uniform(
        -np.pi,
        np.pi,
        size=num_elements,
    )

    phi = np.exp(
        1j * phases
    )

    active_mask = np.zeros(
        num_elements,
        dtype=bool,
    )
    active_mask[:3] = True

    result = simulate_bidirectional_probing(
        g,
        h,
        phi,
        active_mask=active_mask,
        pilot_power_a=1.0,
        pilot_power_b=1.0,
        active_noise_variance=0.002,
        receiver_noise_variance_a=0.01,
        receiver_noise_variance_b=0.01,
        rng=rng,
    )

    metrics = evaluate_key_generation(
        result.observation_at_a,
        result.observation_at_b,
    )

    assert (
        metrics.correlation_magnitude
        > 0.9
    )

    assert (
        metrics.key_disagreement_rate
        < 0.2
    )


def test_invalid_active_mask_raises_error() -> None:
    with pytest.raises(ValueError):
        simulate_bidirectional_probing(
            channel_a_to_ris=np.ones(4),
            channel_ris_to_b=np.ones(4),
            surface_coefficients=np.ones(4),
            active_mask=np.ones(
                3,
                dtype=bool,
            ),
        )