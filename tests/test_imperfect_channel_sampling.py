from pathlib import Path

import numpy as np
import pytest

from active_star_ris.csi_estimation import (
    calculate_empirical_nmse,
)
from active_star_ris.simulation import (
    load_config,
    sample_channels_with_imperfect_csi,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "default.yaml"
)


def test_default_config_contains_csi_section():
    config = load_config(
        CONFIG_PATH
    )

    assert "csi" in config

    assert (
        config["csi"]["nmse_db_min"]
        == -25.0
    )

    assert (
        config["csi"]["nmse_db_max"]
        == -10.0
    )

    assert (
        config["csi"]["default_nmse_db"]
        == -15.0
    )


def test_imperfect_channel_snapshot_has_correct_shapes():
    config = load_config(
        CONFIG_PATH
    )

    rng = np.random.default_rng(
        101
    )

    result = (
        sample_channels_with_imperfect_csi(
            config=config,
            rng=rng,
            num_elements=32,
            nmse_db=-15.0,
        )
    )

    assert result.nmse_db == -15.0

    assert (
        result.true.alice_to_ris.shape
        == (32,)
    )

    assert (
        result.estimated.alice_to_ris.shape
        == (32,)
    )

    assert (
        result.true.ris_to_transmission_user.shape
        == (32,)
    )

    assert (
        result.estimated.ris_to_transmission_user.shape
        == (32,)
    )

    assert (
        result.true.ris_to_reflection_user.shape
        == (32,)
    )

    assert (
        result.estimated.ris_to_reflection_user.shape
        == (32,)
    )

    assert isinstance(
        result.true.direct_transmission,
        complex,
    )

    assert isinstance(
        result.estimated.direct_transmission,
        complex,
    )

    assert not np.allclose(
        result.true.alice_to_ris,
        result.estimated.alice_to_ris,
    )


def test_sampled_nmse_stays_inside_configured_range():
    config = load_config(
        CONFIG_PATH
    )

    rng = np.random.default_rng(
        103
    )

    sampled_values = []

    for _ in range(30):
        result = (
            sample_channels_with_imperfect_csi(
                config=config,
                rng=rng,
                num_elements=8,
            )
        )

        sampled_values.append(
            result.nmse_db
        )

    sampled_values = np.asarray(
        sampled_values,
        dtype=float,
    )

    assert np.all(
        sampled_values >= -25.0
    )

    assert np.all(
        sampled_values <= -10.0
    )


def test_vector_channel_empirical_nmse_matches_target():
    config = load_config(
        CONFIG_PATH
    )

    rng = np.random.default_rng(
        107
    )

    result = (
        sample_channels_with_imperfect_csi(
            config=config,
            rng=rng,
            num_elements=20_000,
            nmse_db=-10.0,
        )
    )

    empirical_nmse = (
        calculate_empirical_nmse(
            result.true.alice_to_ris,
            result.estimated.alice_to_ris,
        )
    )

    # -10 dB对应线性NMSE约为0.1。
    assert empirical_nmse == pytest.approx(
        0.1,
        rel=0.08,
    )