from __future__ import annotations

import numpy as np
import pytest

from active_star_ris.surface import (
    EnergySplit,
    build_surface_coefficients,
)
from active_star_ris.system import (
    evaluate_two_user_system,
    forwarded_active_noise_variance,
    ris_output_power,
)


def make_surface(active: bool):
    n = 4
    split = EnergySplit.from_transmission(0.5, n)
    mask = np.full(n, active)
    gain = np.full(n, 2.0 if active else 1.0)
    return build_surface_coefficients(
        split,
        np.zeros(n),
        np.zeros(n),
        gain,
        mask,
    )


def test_forwarded_noise_is_zero_for_passive_surface() -> None:
    surface = make_surface(False)
    variance = forwarded_active_noise_variance(
        np.ones(4),
        surface.phi_transmission,
        surface.active_mask,
        0.1,
    )
    assert variance == pytest.approx(0.0)


def test_forwarded_noise_is_positive_for_active_surface() -> None:
    surface = make_surface(True)
    variance = forwarded_active_noise_variance(
        np.ones(4),
        surface.phi_transmission,
        surface.active_mask,
        0.1,
    )
    assert variance > 0.0


def test_ris_output_power_matches_manual_value() -> None:
    surface = make_surface(True)
    power = ris_output_power(
        np.ones(4),
        surface,
        transmit_power=1.0,
        ris_internal_noise_variance=0.1,
    )
    assert power == pytest.approx(4 * 4.0 * 1.1)


def test_two_user_metrics_are_finite() -> None:
    surface = make_surface(False)
    metrics = evaluate_two_user_system(
        np.ones(4),
        np.ones(4),
        np.ones(4),
        surface,
        transmit_power=1.0,
        user_noise_variance=0.1,
        ris_internal_noise_variance=0.01,
        ris_output_power_budget=10.0,
    )
    assert np.isfinite(metrics.weighted_sum_rate)
    assert metrics.transmission.snr_linear > 0
    assert metrics.reflection.snr_linear > 0
