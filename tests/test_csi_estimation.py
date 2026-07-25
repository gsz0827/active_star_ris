import numpy as np
import pytest

from active_star_ris.csi_estimation import (
    CSIErrorConfig,
    calculate_empirical_nmse,
    generate_imperfect_csi,
    sample_nmse_db,
)


def test_generate_imperfect_csi_preserves_shape_and_relation():
    rng = np.random.default_rng(7)

    true_channel = np.ones(
        200_000,
        dtype=np.complex128,
    )

    result = generate_imperfect_csi(
        true_channel=true_channel,
        nmse_db=-10.0,
        rng=rng,
    )

    assert (
        result.true_channel.shape
        == true_channel.shape
    )

    assert (
        result.estimated_channel.shape
        == true_channel.shape
    )

    assert (
        result.estimation_error.shape
        == true_channel.shape
    )

    assert np.allclose(
        result.estimated_channel,
        result.true_channel
        + result.estimation_error,
    )

    # -10 dB对应线性NMSE约为0.1。
    empirical_nmse = calculate_empirical_nmse(
        result.true_channel,
        result.estimated_channel,
    )

    assert empirical_nmse == pytest.approx(
        0.1,
        rel=0.03,
    )


def test_smaller_nmse_produces_smaller_average_error():
    true_channel = np.ones(
        100_000,
        dtype=np.complex128,
    )

    small_error_result = generate_imperfect_csi(
        true_channel=true_channel,
        nmse_db=-25.0,
        rng=np.random.default_rng(17),
    )

    large_error_result = generate_imperfect_csi(
        true_channel=true_channel,
        nmse_db=-10.0,
        rng=np.random.default_rng(19),
    )

    small_error_nmse = calculate_empirical_nmse(
        small_error_result.true_channel,
        small_error_result.estimated_channel,
    )

    large_error_nmse = calculate_empirical_nmse(
        large_error_result.true_channel,
        large_error_result.estimated_channel,
    )

    assert small_error_nmse < large_error_nmse


def test_sample_nmse_db_stays_inside_configured_interval():
    rng = np.random.default_rng(11)

    config = CSIErrorConfig(
        nmse_db_min=-25.0,
        nmse_db_max=-10.0,
    )

    sampled_values = np.array(
        [
            sample_nmse_db(config, rng)
            for _ in range(100)
        ]
    )

    assert np.all(
        sampled_values >= -25.0
    )

    assert np.all(
        sampled_values <= -10.0
    )


def test_invalid_nmse_range_raises():
    rng = np.random.default_rng(13)

    config = CSIErrorConfig(
        nmse_db_min=-5.0,
        nmse_db_max=-20.0,
    )

    with pytest.raises(ValueError):
        sample_nmse_db(
            config,
            rng,
        )


def test_mismatched_shapes_raise():
    true_channel = np.ones(
        8,
        dtype=np.complex128,
    )

    estimated_channel = np.ones(
        9,
        dtype=np.complex128,
    )

    with pytest.raises(ValueError):
        calculate_empirical_nmse(
            true_channel,
            estimated_channel,
        )