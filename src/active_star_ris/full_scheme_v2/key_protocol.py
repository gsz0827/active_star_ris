from __future__ import annotations

from hashlib import blake2s
from math import ceil, floor, log2

import numpy as np

from .config import KeyGenerationConfig, ProbingConfig
from .models import BitArray, KeyRateResult


def _extract_feature(values: np.ndarray, feature: str) -> np.ndarray:
    data = np.asarray(values, dtype=np.complex128).reshape(-1)
    if feature == "real":
        result = data.real
    elif feature == "imag":
        result = data.imag
    elif feature == "magnitude":
        result = np.abs(data)
    elif feature == "phase":
        result = np.angle(data)
    else:
        raise ValueError("unsupported feature")
    return np.asarray(result, dtype=np.float64)


def _binary_entropy(probability: float) -> float:
    p = float(np.clip(probability, 0.0, 1.0))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * log2(p) - (1.0 - p) * log2(1.0 - p))


def quantize_with_guard_band(
    observation_a: np.ndarray,
    observation_b: np.ndarray,
    config: KeyGenerationConfig,
) -> tuple[BitArray, BitArray, int, int, float]:
    a = _extract_feature(observation_a, config.feature)
    b = _extract_feature(observation_b, config.feature)
    if a.size != b.size or a.size < 2:
        raise ValueError("observations must have equal length >= 2")

    threshold = float(np.median(a))
    scale = float(np.std(a, ddof=1))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        return (
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint8),
            0,
            0,
            1.0,
        )

    guard = config.guard_band_sigma * scale
    reliable_a = np.abs(a - threshold) > guard
    reliable_b = np.abs(b - threshold) > guard

    if config.selection_policy == "alice":
        retained = reliable_a
        disclosure_bits = int(a.size)
    else:
        retained = np.logical_and(reliable_a, reliable_b)
        disclosure_bits = int(2 * a.size)

    indices = np.flatnonzero(retained)
    if indices.size == 0:
        return (
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint8),
            disclosure_bits,
            0,
            1.0,
        )

    bits_a = (a[indices] > threshold).astype(np.uint8)
    bits_b = (b[indices] > threshold).astype(np.uint8)
    raw_kdr = float(np.mean(bits_a != bits_b))
    return bits_a, bits_b, disclosure_bits, int(indices.size), raw_kdr


def _parity(bits: BitArray, indices: np.ndarray) -> int:
    return int(np.sum(bits[indices], dtype=np.int64) % 2)


def _verification_tag(bits: BitArray, tag_length_bits: int) -> BitArray:
    packed = np.packbits(bits, bitorder="big").tobytes()
    prefix = int(bits.size).to_bytes(8, byteorder="big", signed=False)
    digest = blake2s(prefix + packed, digest_size=32).digest()
    digest_bits = np.unpackbits(
        np.frombuffer(digest, dtype=np.uint8),
        bitorder="big",
    )
    return np.asarray(digest_bits[:tag_length_bits], dtype=np.uint8)


def cascade_reconcile(
    bits_a: BitArray,
    bits_b: BitArray,
    config: KeyGenerationConfig,
    rng: np.random.Generator,
) -> tuple[BitArray, int, float, bool]:
    a = np.asarray(bits_a, dtype=np.uint8).reshape(-1)
    before = np.asarray(bits_b, dtype=np.uint8).reshape(-1)
    if a.size != before.size:
        raise ValueError("raw key sizes differ")
    if a.size < 2:
        return before.copy(), 0, 1.0, False

    corrected = before.copy()
    leakage = 0

    for pass_index in range(config.reconciliation_passes):
        block_size = min(
            a.size,
            config.initial_block_size
            * 2 ** min(pass_index, config.maximum_block_doublings),
        )
        permutation = rng.permutation(a.size).astype(np.int64)

        for start in range(0, a.size, block_size):
            block = permutation[start : start + block_size]
            leakage += 1
            if _parity(a, block) == _parity(corrected, block):
                continue

            search = block
            while search.size > 1:
                middle = search.size // 2
                left = search[:middle]
                right = search[middle:]
                leakage += 1
                if _parity(a, left) != _parity(corrected, left):
                    search = left
                else:
                    search = right
            corrected[int(search[0])] ^= np.uint8(1)

    post_kdr = float(np.mean(a != corrected))
    tag_a = _verification_tag(a, config.verification_tag_bits)
    tag_b = _verification_tag(corrected, config.verification_tag_bits)
    verified = bool(np.array_equal(tag_a, tag_b))
    return corrected, int(leakage), post_kdr, verified


def toeplitz_hash(
    bits: BitArray,
    output_length: int,
    rng: np.random.Generator,
) -> tuple[BitArray, BitArray]:
    source = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if not 1 <= output_length <= source.size:
        raise ValueError("invalid Toeplitz output length")

    seed_length = source.size + output_length - 1
    seed = rng.integers(0, 2, size=seed_length, dtype=np.uint8)
    result = np.zeros(output_length, dtype=np.uint8)
    positions = np.arange(source.size, dtype=np.int64)

    for row in range(output_length):
        seed_indices = row - positions + source.size - 1
        result[row] = np.uint8(
            np.sum(source * seed[seed_indices], dtype=np.int64) % 2
        )
    return result, seed


def _frame_duration(
    total_samples: int,
    public_bits: int,
    probing: ProbingConfig,
    key: KeyGenerationConfig,
    reverse_pilot_symbols: int,
) -> float:
    if reverse_pilot_symbols < 1:
        raise ValueError(
            "reverse_pilot_symbols must be positive"
        )

    # 一次双向信道观测：
    #
    # Controller -> user:
    #     L_controller 个 pilot symbols
    #
    # user -> Controller:
    #     L_user 个 pilot symbols
    #
    # 中间再加 forward/reverse guard。
    symbols_per_sample = (
        probing.pilot_symbols_controller
        + reverse_pilot_symbols
    )

    probing_time = total_samples * (
        symbols_per_sample
        * probing.pilot_symbol_duration_seconds
        + probing.forward_reverse_guard_seconds
    )

    public_time = (
        public_bits / key.public_channel_rate_bps
    )

    return max(
        probing_time
        + probing.branch_switch_guard_seconds
        + public_time
        + key.fixed_processing_delay_seconds,
        1.0e-12,
    )


def evaluate_key_rate(
    observation_a: np.ndarray,
    observation_b: np.ndarray,
    *,
    key_config: KeyGenerationConfig,
    probing_config: ProbingConfig,
    rng: np.random.Generator,
    full_protocol: bool,
    reverse_pilot_symbols: int | None = None,
) -> KeyRateResult:
    key_config.validate()
    if reverse_pilot_symbols is None:
        reverse_pilot_symbols = (
            probing_config.pilot_symbols_controller
        )

    if reverse_pilot_symbols < 1:
        raise ValueError(
            "reverse_pilot_symbols must be positive"
        )
    total_samples = int(np.asarray(observation_a).reshape(-1).size)

    bits_a, bits_b, selection_bits, retained, raw_kdr = quantize_with_guard_band(
        observation_a,
        observation_b,
        key_config,
    )

    if retained < 2:
        duration = _frame_duration(
            total_samples,
            selection_bits,
            probing_config,
            key_config,
            reverse_pilot_symbols,
        )
        return KeyRateResult(
            total_samples=total_samples,
            retained_samples=retained,
            retention_ratio=retained / max(total_samples, 1),
            raw_kdr=1.0,
            post_reconciliation_kdr=1.0,
            estimated_entropy_bits=0,
            reconciliation_leakage_bits=0,
            verification_leakage_bits=0,
            public_communication_bits=selection_bits,
            training_secret_bits=0,
            final_key_bits=0,
            frame_duration_seconds=duration,
            training_key_rate_bps=0.0,
            final_key_rate_bps=0.0,
            verification_passed=False,
            success=False,
        )

    entropy_bits = floor(
        retained * key_config.minimum_entropy_bits_per_retained_bit
    )

    if full_protocol:
        corrected_b, parity_leakage, post_kdr, verified = cascade_reconcile(
            bits_a,
            bits_b,
            key_config,
            rng,
        )
        verification_leakage = key_config.verification_tag_bits
        operational_bound = max(
            0,
            entropy_bits
            - parity_leakage
            - verification_leakage
            - key_config.privacy_margin_bits,
        )
        final_length = min(
            operational_bound,
            key_config.maximum_final_key_bits,
            retained,
        )

        seed_bits = 0
        success = bool(verified and final_length > 0)
        if success:
            final_a, seed = toeplitz_hash(bits_a, final_length, rng)
            final_b, _ = _toeplitz_hash_with_seed(corrected_b, final_length, seed)
            success = bool(np.array_equal(final_a, final_b))
            if key_config.include_toeplitz_seed_in_public_time:
                seed_bits = int(seed.size)
        final_bits = int(final_length if success else 0)
        training_bits = int(operational_bound)
        public_bits = int(
            selection_bits
            + parity_leakage
            + verification_leakage
            + seed_bits
        )
    else:
        estimated_leakage = ceil(
            key_config.reconciliation_efficiency
            * retained
            * _binary_entropy(raw_kdr)
        )
        parity_leakage = int(estimated_leakage)
        verification_leakage = key_config.verification_tag_bits
        trainable = raw_kdr <= key_config.maximum_trainable_raw_kdr
        training_bits = (
            max(
                0,
                entropy_bits
                - parity_leakage
                - verification_leakage
                - key_config.privacy_margin_bits,
            )
            if trainable
            else 0
        )
        post_kdr = 0.0 if trainable else raw_kdr
        verified = trainable
        final_bits = 0
        success = False
        public_bits = int(
            selection_bits + parity_leakage + verification_leakage
        )

    duration = _frame_duration(
        total_samples,
        public_bits,
        probing_config,
        key_config,
        reverse_pilot_symbols,
    )

    return KeyRateResult(
        total_samples=total_samples,
        retained_samples=retained,
        retention_ratio=retained / total_samples,
        raw_kdr=float(raw_kdr),
        post_reconciliation_kdr=float(post_kdr),
        estimated_entropy_bits=int(entropy_bits),
        reconciliation_leakage_bits=int(parity_leakage),
        verification_leakage_bits=int(verification_leakage),
        public_communication_bits=int(public_bits),
        training_secret_bits=int(training_bits),
        final_key_bits=int(final_bits),
        frame_duration_seconds=float(duration),
        training_key_rate_bps=float(training_bits / duration),
        final_key_rate_bps=float(final_bits / duration),
        verification_passed=bool(verified),
        success=bool(success),
    )


def _toeplitz_hash_with_seed(
    bits: BitArray,
    output_length: int,
    seed: BitArray,
) -> tuple[BitArray, BitArray]:
    source = np.asarray(bits, dtype=np.uint8).reshape(-1)
    public_seed = np.asarray(seed, dtype=np.uint8).reshape(-1)
    required = source.size + output_length - 1
    if public_seed.size != required:
        raise ValueError("invalid Toeplitz seed length")

    result = np.zeros(output_length, dtype=np.uint8)
    positions = np.arange(source.size, dtype=np.int64)
    for row in range(output_length):
        seed_indices = row - positions + source.size - 1
        result[row] = np.uint8(
            np.sum(source * public_seed[seed_indices], dtype=np.int64) % 2
        )
    return result, public_seed
