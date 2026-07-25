from __future__ import annotations

import numpy as np
import pytest

from active_star_ris.key_generation import (
    complex_correlation,
    evaluate_key_generation,
    gaussian_mutual_information_bits,
    key_disagreement_rate,
    quantize_complex_sign,
)


def test_identical_observations_have_perfect_reciprocity() -> None:
    observations = np.array(
        [
            1 + 1j,
            2 - 1j,
            -1 + 2j,
            0.5 - 2j,
        ]
    )

    metrics = evaluate_key_generation(
        observations,
        observations,
    )

    assert (
        metrics.correlation_magnitude
        == pytest.approx(1.0)
    )
    assert (
        metrics.key_disagreement_rate
        == pytest.approx(0.0)
    )
    assert (
        metrics.num_raw_bits
        == 2 * observations.size
    )


def test_independent_observations_have_low_correlation() -> None:
    rng = np.random.default_rng(
        20260719
    )

    a = (
        rng.normal(size=10_000)
        + 1j * rng.normal(size=10_000)
    )
    b = (
        rng.normal(size=10_000)
        + 1j * rng.normal(size=10_000)
    )

    metrics = evaluate_key_generation(
        a,
        b,
    )

    assert (
        metrics.correlation_magnitude
        < 0.05
    )
    assert (
        0.45
        < metrics.key_disagreement_rate
        < 0.55
    )


def test_noisy_common_source_has_good_reciprocity() -> None:
    rng = np.random.default_rng(7)

    shared = (
        rng.normal(size=4_000)
        + 1j * rng.normal(size=4_000)
    )

    noise_a = 0.2 * (
        rng.normal(size=4_000)
        + 1j * rng.normal(size=4_000)
    )
    noise_b = 0.2 * (
        rng.normal(size=4_000)
        + 1j * rng.normal(size=4_000)
    )

    metrics = evaluate_key_generation(
        shared + noise_a,
        shared + noise_b,
    )

    assert (
        metrics.correlation_magnitude
        > 0.9
    )
    assert (
        metrics.key_disagreement_rate
        < 0.2
    )
    assert (
        metrics.mutual_information_bits_per_sample
        > 1.0
    )


def test_quantizer_and_kdr_shapes() -> None:
    samples = np.array(
        [
            1 + 2j,
            -1 - 2j,
            3 - 4j,
        ]
    )

    bits = quantize_complex_sign(
        samples
    )

    assert bits.shape == (6,)
    assert (
        key_disagreement_rate(
            bits,
            bits,
        )
        == pytest.approx(0.0)
    )


def test_invalid_inputs_raise_errors() -> None:
    with pytest.raises(ValueError):
        complex_correlation(
            [1 + 0j],
            [1 + 0j],
        )

    with pytest.raises(ValueError):
        key_disagreement_rate(
            [],
            [],
        )

    with pytest.raises(ValueError):
        gaussian_mutual_information_bits(
            1.1
        )