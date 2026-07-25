from __future__ import annotations

import numpy as np
import pytest

from active_star_ris.surface import (
    EnergySplit,
    build_surface_coefficients,
)


def test_energy_split_constraint() -> None:
    split = EnergySplit.from_transmission(0.35, 8)
    assert np.allclose(split.beta_transmission, 0.35)
    assert np.allclose(split.beta_reflection, 0.65)
    assert split.maximum_constraint_error() == pytest.approx(0.0)


def test_invalid_energy_split_is_rejected() -> None:
    with pytest.raises(ValueError):
        EnergySplit.from_transmission(1.1, 4)


def test_passive_surface_energy() -> None:
    n = 6
    split = EnergySplit.from_transmission(0.4, n)
    surface = build_surface_coefficients(
        split,
        np.zeros(n),
        np.ones(n),
    )
    assert np.allclose(np.abs(surface.phi_transmission) ** 2, 0.4)
    assert np.allclose(np.abs(surface.phi_reflection) ** 2, 0.6)
    assert surface.maximum_energy_error() < 1e-12


def test_passive_elements_are_forced_to_unit_gain() -> None:
    n = 5
    split = EnergySplit.from_transmission(0.5, n)
    mask = np.array([True, False, False, True, False])
    surface = build_surface_coefficients(
        split,
        np.zeros(n),
        np.zeros(n),
        amplitude_gain=np.full(n, 2.0),
        active_mask=mask,
    )
    assert np.allclose(surface.amplitude_gain[mask], 2.0)
    assert np.allclose(surface.amplitude_gain[~mask], 1.0)
