from __future__ import annotations

import numpy as np
import pytest

from active_star_ris.practical_key_generation import (
    cascade_reconcile,
    generate_end_to_end_key,
    quantize_with_guard_band,
    toeplitz_hash,
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


def test_identical_observations_quantize_identically() -> None:
    rng = np.random.default_rng(71)

    observations = _complex_gaussian(
        rng,
        (2000,),
    )

    result = quantize_with_guard_band(
        observations,
        observations,
        feature="real",
        guard_band_sigma=0.1,
    )

    assert result.retained_samples > 0

    assert (
        result.raw_key_disagreement_rate
        == pytest.approx(0.0)
    )

    assert np.array_equal(
        result.bits_at_a,
        result.bits_at_b,
    )


def test_cascade_corrects_sparse_errors() -> None:
    rng = np.random.default_rng(72)

    bits_at_a = rng.integers(
        0,
        2,
        size=5000,
        dtype=np.uint8,
    )

    bits_at_b = np.array(
        bits_at_a,
        copy=True,
    )

    error_mask = (
        rng.random(
            bits_at_a.size
        )
        < 0.05
    )

    bits_at_b[error_mask] ^= np.uint8(1)

    result = cascade_reconcile(
        bits_at_a,
        bits_at_b,
        initial_block_size=8,
        number_of_passes=8,
        verification_tag_bits=32,
        rng=np.random.default_rng(73),
    )

    assert (
        result.pre_reconciliation_kdr
        > 0.0
    )

    assert (
        result.post_reconciliation_kdr
        == pytest.approx(0.0)
    )

    assert result.corrections_applied > 0
    assert result.parity_leakage_bits > 0
    assert result.exact_match
    assert result.verification_passed


def test_verification_detects_residual_errors() -> None:
    bits_at_a = np.zeros(
        14,
        dtype=np.uint8,
    )

    bits_at_b = np.array(
        bits_at_a,
        copy=True,
    )

    public_seed = 12345

    permutation = (
        np.random.default_rng(
            public_seed
        )
        .permutation(
            bits_at_a.size
        )
    )

    # 在同一校验块内放置两个错误。
    # 偶数个错误不会触发该块的奇偶校验纠错。
    bits_at_b[
        permutation[0]
    ] = 1

    bits_at_b[
        permutation[1]
    ] = 1

    result = cascade_reconcile(
        bits_at_a,
        bits_at_b,
        initial_block_size=7,
        number_of_passes=1,
        verification_tag_bits=32,
        public_permutation_seeds=[
            public_seed
        ],
    )

    assert not result.exact_match
    assert not result.verification_passed

    assert (
        result.post_reconciliation_kdr
        > 0.0
    )


def test_toeplitz_hash_satisfies_linearity() -> None:
    rng = np.random.default_rng(74)

    input_length = 128
    output_length = 64

    x = rng.integers(
        0,
        2,
        size=input_length,
        dtype=np.uint8,
    )

    y = rng.integers(
        0,
        2,
        size=input_length,
        dtype=np.uint8,
    )

    required_seed_length = (
        input_length
        + output_length
        - 1
    )

    public_seed = rng.integers(
        0,
        2,
        size=required_seed_length,
        dtype=np.uint8,
    )

    hash_x, _ = toeplitz_hash(
        x,
        output_length,
        public_seed_bits=public_seed,
    )

    hash_y, _ = toeplitz_hash(
        y,
        output_length,
        public_seed_bits=public_seed,
    )

    hash_xor, _ = toeplitz_hash(
        np.bitwise_xor(
            x,
            y,
        ),
        output_length,
        public_seed_bits=public_seed,
    )

    assert np.array_equal(
        hash_xor,
        np.bitwise_xor(
            hash_x,
            hash_y,
        ),
    )


def test_end_to_end_pipeline_generates_matching_key() -> None:
    rng = np.random.default_rng(75)

    num_samples = 4000

    common_source = _complex_gaussian(
        rng,
        (num_samples,),
    )

    observation_at_a = (
        common_source
        + _complex_gaussian(
            rng,
            (num_samples,),
            scale=0.01,
        )
    )

    observation_at_b = (
        common_source
        + _complex_gaussian(
            rng,
            (num_samples,),
            scale=0.01,
        )
    )

    result = generate_end_to_end_key(
        observation_at_a,
        observation_at_b,
        pre_reconciliation_entropy_bound_bits=(
            2500
        ),
        feature="real",
        guard_band_sigma=0.1,
        initial_block_size=8,
        number_of_reconciliation_passes=8,
        verification_tag_bits=32,
        privacy_margin_bits=64,
        maximum_final_key_bits=128,
        rng=np.random.default_rng(76),
    )

    assert result.success

    assert (
        result.final_key_length_bits
        == 128
    )

    assert (
        result.reconciliation
        .post_reconciliation_kdr
        == pytest.approx(0.0)
    )

    assert (
        result.privacy_amplification
        is not None
    )

    assert (
        result
        .privacy_amplification
        .keys_match
    )

    assert (
        result
        .privacy_amplification
        .final_key_at_a_hex
        == result
        .privacy_amplification
        .final_key_at_b_hex
    )