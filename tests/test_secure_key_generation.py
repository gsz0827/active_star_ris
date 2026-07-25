from __future__ import annotations

import numpy as np
import pytest

from active_star_ris.secure_key_generation import (
    complex_gaussian_mutual_information,
    evaluate_branch_secrecy,
    simulate_dual_side_secure_key_generation,
)
from active_star_ris.star_key_system import (
    build_star_coefficients,
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


def _coefficients(
    num_elements: int,
):
    amplitudes = np.ones(
        num_elements
    )

    amplitudes[:3] = 1.4

    return build_star_coefficients(
        amplitudes=amplitudes,
        beta_transmission=np.full(
            num_elements,
            0.6,
        ),
        beta_reflection=np.full(
            num_elements,
            0.4,
        ),
        phase_transmission=np.linspace(
            -1.0,
            1.0,
            num_elements,
        ),
        phase_reflection=np.linspace(
            1.0,
            -1.0,
            num_elements,
        ),
    )


def test_independent_observations_have_small_mi() -> None:
    rng = np.random.default_rng(41)

    num_samples = 6000

    source = _complex_gaussian(
        rng,
        (num_samples,),
    )

    observations = np.column_stack(
        (
            _complex_gaussian(
                rng,
                (num_samples,),
            ),
            _complex_gaussian(
                rng,
                (num_samples,),
            ),
        )
    )

    mutual_information = (
        complex_gaussian_mutual_information(
            source,
            observations,
        )
    )

    assert mutual_information < 0.02


def test_identical_observation_has_large_mi() -> None:
    rng = np.random.default_rng(42)

    num_samples = 3000

    source = _complex_gaussian(
        rng,
        (num_samples,),
    )

    independent_noise = _complex_gaussian(
        rng,
        (num_samples,),
    )

    observations = np.column_stack(
        (
            source,
            independent_noise,
        )
    )

    mutual_information = (
        complex_gaussian_mutual_information(
            source,
            observations,
        )
    )

    assert mutual_information > 20.0


def test_noisy_independent_eve_allows_positive_rate() -> None:
    rng = np.random.default_rng(43)

    num_samples = 3000
    num_elements = 10

    g = _complex_gaussian(
        rng,
        (num_samples, num_elements),
    )

    h_t = _complex_gaussian(
        rng,
        (num_samples, num_elements),
    )

    h_r = _complex_gaussian(
        rng,
        (num_samples, num_elements),
    )

    h_e_t = _complex_gaussian(
        rng,
        (num_samples, num_elements),
        scale=0.2,
    )

    h_e_r = _complex_gaussian(
        rng,
        (num_samples, num_elements),
        scale=0.2,
    )

    active_mask = np.zeros(
        num_elements,
        dtype=bool,
    )
    active_mask[:3] = True

    result = (
        simulate_dual_side_secure_key_generation(
            channel_controller_to_ris=g,
            channel_ris_to_transmission_user=h_t,
            channel_ris_to_reflection_user=h_r,
            channel_ris_to_eve_transmission=h_e_t,
            channel_ris_to_eve_reflection=h_e_r,
            coefficients=_coefficients(
                num_elements
            ),
            active_mask=active_mask,
            active_noise_variance=0.002,
            receiver_noise_variance_controller=0.01,
            receiver_noise_variance_transmission_user=0.01,
            receiver_noise_variance_reflection_user=0.01,
            receiver_noise_variance_eve_transmission=0.5,
            receiver_noise_variance_eve_reflection=0.5,
            public_leakage_transmission_bits_per_sample=0.05,
            public_leakage_reflection_bits_per_sample=0.05,
            transmission_weight=0.6,
            reflection_weight=0.4,
            rng=np.random.default_rng(44),
        )
    )

    assert (
        result.transmission_secrecy
        .secret_key_rate_bits_per_sample
        > 0.0
    )

    assert (
        result.reflection_secrecy
        .secret_key_rate_bits_per_sample
        > 0.0
    )

    branch_rates = [
        result.transmission_secrecy
        .secret_key_rate_bits_per_sample,
        result.reflection_secrecy
        .secret_key_rate_bits_per_sample,
    ]

    assert (
        min(branch_rates)
        <= result
        .weighted_secret_key_rate_bits_per_sample
        <= max(branch_rates)
    )


def test_colocated_clean_eve_clamps_rate_to_zero() -> None:
    rng = np.random.default_rng(45)

    num_samples = 3000
    num_elements = 10

    g = _complex_gaussian(
        rng,
        (num_samples, num_elements),
    )

    h_t = _complex_gaussian(
        rng,
        (num_samples, num_elements),
    )

    h_r = _complex_gaussian(
        rng,
        (num_samples, num_elements),
    )

    active_mask = np.zeros(
        num_elements,
        dtype=bool,
    )
    active_mask[:3] = True

    result = (
        simulate_dual_side_secure_key_generation(
            channel_controller_to_ris=g,
            channel_ris_to_transmission_user=h_t,
            channel_ris_to_reflection_user=h_r,
            # Eve与对应合法用户共址。
            channel_ris_to_eve_transmission=h_t,
            channel_ris_to_eve_reflection=h_r,
            coefficients=_coefficients(
                num_elements
            ),
            active_mask=active_mask,
            active_noise_variance=0.002,
            receiver_noise_variance_controller=0.01,
            receiver_noise_variance_transmission_user=0.01,
            receiver_noise_variance_reflection_user=0.01,
            receiver_noise_variance_eve_transmission=0.0,
            receiver_noise_variance_eve_reflection=0.0,
            rng=np.random.default_rng(46),
        )
    )

    assert (
        result.transmission_secrecy
        .secret_key_rate_bits_per_sample
        == pytest.approx(0.0)
    )

    assert (
        result.reflection_secrecy
        .secret_key_rate_bits_per_sample
        == pytest.approx(0.0)
    )

    assert (
        result
        .weighted_secret_key_rate_bits_per_sample
        == pytest.approx(0.0)
    )


def test_public_leakage_is_subtracted_and_clamped() -> None:
    rng = np.random.default_rng(47)

    num_samples = 4000

    common_source = _complex_gaussian(
        rng,
        (num_samples,),
    )

    observation_a = (
        common_source
        + _complex_gaussian(
            rng,
            (num_samples,),
            scale=0.1,
        )
    )

    observation_b = (
        common_source
        + _complex_gaussian(
            rng,
            (num_samples,),
            scale=0.1,
        )
    )

    eve_forward = _complex_gaussian(
        rng,
        (num_samples,),
    )

    eve_reverse = _complex_gaussian(
        rng,
        (num_samples,),
    )

    low_public_leakage = evaluate_branch_secrecy(
        observation_at_a=observation_a,
        observation_at_b=observation_b,
        eve_observation_forward=eve_forward,
        eve_observation_reverse=eve_reverse,
        public_leakage_bits_per_sample=0.1,
    )

    high_public_leakage = evaluate_branch_secrecy(
        observation_at_a=observation_a,
        observation_at_b=observation_b,
        eve_observation_forward=eve_forward,
        eve_observation_reverse=eve_reverse,
        public_leakage_bits_per_sample=100.0,
    )

    assert (
        low_public_leakage
        .secret_key_rate_bits_per_sample
        > 0.0
    )

    assert (
        high_public_leakage
        .secret_key_rate_bits_per_sample
        == pytest.approx(0.0)
    )

    assert (
        high_public_leakage
        .raw_secret_key_margin_bits_per_sample
        < 0.0
    )