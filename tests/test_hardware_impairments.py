from __future__ import annotations

import numpy as np

from active_star_ris.hardware_impairments import (
    HardwareMismatchParameters,
    apply_hardware_mismatch,
)
from active_star_ris.star_key_system import (
    build_star_coefficients,
    simulate_dual_side_key_generation,
)


def _ideal_coefficients(
    num_elements: int,
):
    return build_star_coefficients(
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


def test_zero_mismatch_returns_ideal_coefficients() -> None:
    num_elements = 8

    ideal = _ideal_coefficients(
        num_elements
    )

    realization = apply_hardware_mismatch(
        ideal_coefficients=ideal,
        active_mask=np.zeros(
            num_elements,
            dtype=bool,
        ),
        parameters=(
            HardwareMismatchParameters()
        ),
        rng=np.random.default_rng(31),
    )

    np.testing.assert_allclose(
        realization
        .forward_coefficients
        .transmission,
        ideal.transmission,
    )

    np.testing.assert_allclose(
        realization
        .reverse_coefficients
        .reflection,
        ideal.reflection,
    )


def test_static_mismatch_is_common_to_both_directions() -> None:
    num_elements = 10

    ideal = _ideal_coefficients(
        num_elements
    )

    realization = apply_hardware_mismatch(
        ideal_coefficients=ideal,
        active_mask=np.ones(
            num_elements,
            dtype=bool,
        ),
        parameters=HardwareMismatchParameters(
            static_gain_error_std_db=1.0,
            static_phase_error_std_rad=0.2,
            directional_gain_error_std_db=0.0,
            directional_phase_error_std_rad=0.0,
        ),
        rng=np.random.default_rng(32),
    )

    np.testing.assert_allclose(
        realization
        .forward_coefficients
        .transmission,
        realization
        .reverse_coefficients
        .transmission,
    )

    assert not np.allclose(
        realization
        .forward_coefficients
        .transmission,
        ideal.transmission,
    )


def test_directional_mismatch_separates_forward_reverse() -> None:
    num_elements = 10

    ideal = _ideal_coefficients(
        num_elements
    )

    realization = apply_hardware_mismatch(
        ideal_coefficients=ideal,
        active_mask=np.ones(
            num_elements,
            dtype=bool,
        ),
        parameters=HardwareMismatchParameters(
            directional_gain_error_std_db=1.0,
            directional_phase_error_std_rad=0.3,
        ),
        rng=np.random.default_rng(33),
    )

    assert not np.allclose(
        realization
        .forward_coefficients
        .transmission,
        realization
        .reverse_coefficients
        .transmission,
    )

    assert not np.allclose(
        realization
        .forward_coefficients
        .reflection,
        realization
        .reverse_coefficients
        .reflection,
    )


def test_passive_elements_are_not_amplified() -> None:
    num_elements = 12

    ideal = _ideal_coefficients(
        num_elements
    )

    active_mask = np.zeros(
        num_elements,
        dtype=bool,
    )
    active_mask[:4] = True

    realization = apply_hardware_mismatch(
        ideal_coefficients=ideal,
        active_mask=active_mask,
        parameters=HardwareMismatchParameters(
            static_gain_error_std_db=5.0,
            directional_gain_error_std_db=5.0,
        ),
        rng=np.random.default_rng(34),
    )

    passive_mask = ~active_mask

    assert np.all(
        realization
        .forward_coefficients
        .amplitudes[passive_mask]
        <= 1.0 + 1.0e-12
    )

    assert np.all(
        realization
        .reverse_coefficients
        .amplitudes[passive_mask]
        <= 1.0 + 1.0e-12
    )


def test_directional_mismatch_reduces_reciprocity() -> None:
    rng = np.random.default_rng(35)

    num_samples = 3000
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

    ideal = _ideal_coefficients(
        num_elements
    )

    active_mask = np.zeros(
        num_elements,
        dtype=bool,
    )
    active_mask[:4] = True

    realization = apply_hardware_mismatch(
        ideal_coefficients=ideal,
        active_mask=active_mask,
        parameters=HardwareMismatchParameters(
            directional_gain_error_std_db=1.5,
            directional_phase_error_std_rad=0.5,
        ),
        rng=np.random.default_rng(36),
    )

    result = simulate_dual_side_key_generation(
        channel_controller_to_ris=g,
        channel_ris_to_transmission_user=h_t,
        channel_ris_to_reflection_user=h_r,
        coefficients=(
            realization.forward_coefficients
        ),
        reverse_coefficients=(
            realization.reverse_coefficients
        ),
        active_mask=active_mask,
        rng=np.random.default_rng(37),
    )

    assert not np.allclose(
        result.transmission
        .probing
        .effective_channel_forward,
        result.transmission
        .probing
        .effective_channel_reverse,
    )

    assert (
        result.transmission
        .metrics
        .correlation_magnitude
        < 1.0 - 1.0e-6
    )