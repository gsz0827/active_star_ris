from __future__ import annotations

import numpy as np
import pytest

from active_star_ris.channels import (
    ChannelConfig,
    gauss_markov_sequence,
    generate_channel,
    rayleigh_channel,
    rician_channel,
)


def test_rayleigh_average_power() -> None:
    rng = np.random.default_rng(1)
    channel = rayleigh_channel(200000, rng)
    assert np.mean(np.abs(channel) ** 2) == pytest.approx(1.0, abs=0.02)


def test_rician_average_power() -> None:
    rng = np.random.default_rng(2)
    channel = rician_channel(200000, rng, k_factor_db=5.0)
    assert np.mean(np.abs(channel) ** 2) == pytest.approx(1.0, abs=0.02)


def test_generate_channel_shape() -> None:
    rng = np.random.default_rng(3)
    config = ChannelConfig(model="rician", k_factor_db=3.0)
    assert generate_channel(17, rng, config).shape == (17,)


def test_gauss_markov_identity_at_unit_correlation() -> None:
    rng = np.random.default_rng(4)
    initial = np.array([1 + 2j, 3 - 1j])
    sequence = gauss_markov_sequence(initial, 5, 1.0, rng)
    assert np.allclose(sequence, initial)
