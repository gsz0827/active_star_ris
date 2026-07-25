from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2s
from math import floor
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]
BitArray = NDArray[np.uint8]
IndexArray = NDArray[np.int64]
SeedArray = NDArray[np.uint64]

FeatureName = Literal[
    "real",
    "imag",
    "magnitude",
    "phase",
]

SelectionPolicy = Literal[
    "alice",
    "intersection",
]


@dataclass(frozen=True)
class QuantizationResult:
    """保护间隔量化结果。"""

    total_samples: int
    retained_samples: int
    retention_ratio: float

    feature_name: str
    threshold: float
    feature_scale: float
    guard_band: float

    retained_indices: IndexArray

    bits_at_a: BitArray
    bits_at_b: BitArray

    raw_key_disagreement_rate: float

    selection_policy: str

    # 未压缩可靠性掩码的公开长度；它不直接等同于独立秘密信息泄漏。
    selection_disclosure_bits: int


@dataclass(frozen=True)
class ReconciliationResult:
    """多轮Cascade式信息协调结果。"""

    bits_at_a: BitArray

    bits_at_b_before_reconciliation: BitArray
    bits_at_b_after_reconciliation: BitArray

    pre_reconciliation_kdr: float
    post_reconciliation_kdr: float

    number_of_passes: int
    corrections_applied: int

    parity_leakage_bits: int
    verification_leakage_bits: int

    public_permutation_seeds: SeedArray
    block_sizes: tuple[int, ...]

    exact_match: bool
    verification_passed: bool


@dataclass(frozen=True)
class PrivacyAmplificationResult:
    """Toeplitz通用哈希隐私放大结果。"""

    input_length_bits: int
    output_length_bits: int

    public_seed_bits: BitArray

    final_key_at_a: BitArray
    final_key_at_b: BitArray

    final_key_at_a_hex: str
    final_key_at_b_hex: str

    keys_match: bool


@dataclass(frozen=True)
class EndToEndKeyGenerationResult:
    """量化、协调、验证和隐私放大的完整结果。"""

    quantization: QuantizationResult
    reconciliation: ReconciliationResult

    pre_reconciliation_entropy_bound_bits: int

    operational_secret_bit_bound: int
    requested_maximum_key_bits: int
    final_key_length_bits: int

    privacy_margin_bits: int

    privacy_amplification: PrivacyAmplificationResult | None

    success: bool


def _as_complex_vector(
    values: ArrayLike,
    name: str,
) -> ComplexArray:
    array = np.asarray(
        values,
        dtype=np.complex128,
    ).reshape(-1)

    if array.size < 2:
        raise ValueError(
            f"{name} must contain at least two samples"
        )

    if not np.all(np.isfinite(array.real)):
        raise ValueError(
            f"{name} contains non-finite real values"
        )

    if not np.all(np.isfinite(array.imag)):
        raise ValueError(
            f"{name} contains non-finite imaginary values"
        )

    return np.asarray(
        array,
        dtype=np.complex128,
    )


def _as_bit_vector(
    values: ArrayLike,
    name: str,
) -> BitArray:
    array = np.asarray(values).reshape(-1)

    if array.size == 0:
        raise ValueError(
            f"{name} cannot be empty"
        )

    if not np.all(
        np.logical_or(
            array == 0,
            array == 1,
        )
    ):
        raise ValueError(
            f"{name} must contain only 0 and 1"
        )

    return np.asarray(
        array,
        dtype=np.uint8,
    )


def _extract_feature(
    observations: ComplexArray,
    feature: FeatureName,
) -> NDArray[np.float64]:
    if feature == "real":
        values = observations.real
    elif feature == "imag":
        values = observations.imag
    elif feature == "magnitude":
        values = np.abs(observations)
    elif feature == "phase":
        values = np.angle(observations)
    else:
        raise ValueError(
            "feature must be one of "
            "'real', 'imag', 'magnitude', or 'phase'"
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def quantize_with_guard_band(
    observation_at_a: ArrayLike,
    observation_at_b: ArrayLike,
    *,
    feature: FeatureName = "real",
    guard_band_sigma: float = 0.1,
    threshold: float | None = None,
    selection_policy: SelectionPolicy = "intersection",
) -> QuantizationResult:
    """
    将相关复观测转换为原始二进制序列。

    默认阈值由A端特征中位数确定。保护间隔中的低可靠性样本会被丢弃。

    selection_policy="alice":
        仅由A端可靠性位置决定保留样本。

    selection_policy="intersection":
        使用A、B双方可靠性位置的交集。
    """
    if not np.isfinite(guard_band_sigma):
        raise ValueError(
            "guard_band_sigma must be finite"
        )

    if guard_band_sigma < 0.0:
        raise ValueError(
            "guard_band_sigma cannot be negative"
        )

    if selection_policy not in {
        "alice",
        "intersection",
    }:
        raise ValueError(
            "selection_policy must be "
            "'alice' or 'intersection'"
        )

    observation_a = _as_complex_vector(
        observation_at_a,
        "observation_at_a",
    )

    observation_b = _as_complex_vector(
        observation_at_b,
        "observation_at_b",
    )

    if observation_a.size != observation_b.size:
        raise ValueError(
            "observation_at_a and observation_at_b "
            "must contain the same number of samples"
        )

    feature_a = _extract_feature(
        observation_a,
        feature,
    )

    feature_b = _extract_feature(
        observation_b,
        feature,
    )

    if threshold is None:
        effective_threshold = float(
            np.median(feature_a)
        )
    else:
        if not np.isfinite(threshold):
            raise ValueError(
                "threshold must be finite"
            )

        effective_threshold = float(threshold)

    feature_scale = float(
        np.std(
            feature_a,
            ddof=1,
        )
    )

    if (
        not np.isfinite(feature_scale)
        or feature_scale <= np.finfo(np.float64).eps
    ):
        raise ValueError(
            "the selected feature has insufficient variation"
        )

    guard_band = guard_band_sigma * feature_scale

    reliable_at_a = (
        np.abs(
            feature_a - effective_threshold
        )
        > guard_band
    )

    reliable_at_b = (
        np.abs(
            feature_b - effective_threshold
        )
        > guard_band
    )

    if selection_policy == "alice":
        retained_mask = reliable_at_a
        selection_disclosure_bits = int(
            observation_a.size
        )
    else:
        retained_mask = np.logical_and(
            reliable_at_a,
            reliable_at_b,
        )
        selection_disclosure_bits = int(
            2 * observation_a.size
        )

    retained_indices = np.flatnonzero(
        retained_mask
    ).astype(np.int64)

    if retained_indices.size == 0:
        raise ValueError(
            "the guard band removed all samples"
        )

    bits_at_a = (
        feature_a[retained_indices]
        > effective_threshold
    ).astype(np.uint8)

    bits_at_b = (
        feature_b[retained_indices]
        > effective_threshold
    ).astype(np.uint8)

    raw_kdr = float(
        np.mean(bits_at_a != bits_at_b)
    )

    return QuantizationResult(
        total_samples=int(observation_a.size),
        retained_samples=int(retained_indices.size),
        retention_ratio=float(
            retained_indices.size
            / observation_a.size
        ),
        feature_name=str(feature),
        threshold=float(effective_threshold),
        feature_scale=float(feature_scale),
        guard_band=float(guard_band),
        retained_indices=np.asarray(
            retained_indices,
            dtype=np.int64,
        ),
        bits_at_a=np.asarray(
            bits_at_a,
            dtype=np.uint8,
        ),
        bits_at_b=np.asarray(
            bits_at_b,
            dtype=np.uint8,
        ),
        raw_key_disagreement_rate=float(raw_kdr),
        selection_policy=str(selection_policy),
        selection_disclosure_bits=selection_disclosure_bits,
    )


def _parity(
    bits: BitArray,
    indices: NDArray[np.int64],
) -> int:
    return int(
        np.sum(
            bits[indices],
            dtype=np.int64,
        )
        % 2
    )


def _verification_tag(
    bits: BitArray,
    tag_length_bits: int,
) -> BitArray:
    if not 1 <= tag_length_bits <= 256:
        raise ValueError(
            "tag_length_bits must lie within [1, 256]"
        )

    packed_bits = np.packbits(
        bits,
        bitorder="big",
    ).tobytes()

    length_prefix = int(bits.size).to_bytes(
        length=8,
        byteorder="big",
        signed=False,
    )

    digest = blake2s(
        length_prefix + packed_bits,
        digest_size=32,
    ).digest()

    digest_bits = np.unpackbits(
        np.frombuffer(
            digest,
            dtype=np.uint8,
        ),
        bitorder="big",
    )

    return np.asarray(
        digest_bits[:tag_length_bits],
        dtype=np.uint8,
    )


def cascade_reconcile(
    bits_at_a: ArrayLike,
    bits_at_b: ArrayLike,
    *,
    initial_block_size: int = 8,
    number_of_passes: int = 8,
    maximum_block_doublings: int = 3,
    verification_tag_bits: int = 32,
    public_permutation_seeds: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> ReconciliationResult:
    """
    使用多轮随机置换与分块奇偶校验执行Cascade式信息协调。

    每次公开的奇偶校验计为1 bit泄漏。发生奇偶不一致时，通过二分搜索
    定位并修正一个错误；多轮随机置换用于拆分同一块内的多个错误。
    """
    if isinstance(initial_block_size, bool) or not isinstance(
        initial_block_size,
        (int, np.integer),
    ):
        raise ValueError(
            "initial_block_size must be an integer"
        )

    if initial_block_size < 2:
        raise ValueError(
            "initial_block_size must be at least 2"
        )

    if isinstance(number_of_passes, bool) or not isinstance(
        number_of_passes,
        (int, np.integer),
    ):
        raise ValueError(
            "number_of_passes must be an integer"
        )

    if number_of_passes < 1:
        raise ValueError(
            "number_of_passes must be positive"
        )

    if isinstance(maximum_block_doublings, bool) or not isinstance(
        maximum_block_doublings,
        (int, np.integer),
    ):
        raise ValueError(
            "maximum_block_doublings must be an integer"
        )

    if maximum_block_doublings < 0:
        raise ValueError(
            "maximum_block_doublings cannot be negative"
        )

    bits_a = _as_bit_vector(
        bits_at_a,
        "bits_at_a",
    )

    bits_b_before = _as_bit_vector(
        bits_at_b,
        "bits_at_b",
    )

    if bits_a.size != bits_b_before.size:
        raise ValueError(
            "bits_at_a and bits_at_b must have the same length"
        )

    if bits_a.size < 2:
        raise ValueError(
            "at least two raw key bits are required"
        )

    corrected_bits_b = np.array(
        bits_b_before,
        dtype=np.uint8,
        copy=True,
    )

    generator = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    if public_permutation_seeds is None:
        seed_values = generator.integers(
            low=0,
            high=np.iinfo(np.uint64).max,
            size=number_of_passes,
            dtype=np.uint64,
        )
    else:
        supplied_seeds = np.asarray(
            public_permutation_seeds
        ).reshape(-1)

        if supplied_seeds.size != number_of_passes:
            raise ValueError(
                "public_permutation_seeds must contain "
                "one seed per reconciliation pass"
            )

        if not np.all(np.isfinite(supplied_seeds.astype(np.float64))):
            raise ValueError(
                "public permutation seeds must be finite"
            )

        if np.any(supplied_seeds < 0):
            raise ValueError(
                "public permutation seeds cannot be negative"
            )

        seed_values = np.asarray(
            supplied_seeds,
            dtype=np.uint64,
        )

    parity_leakage_bits = 0
    corrections_applied = 0
    block_sizes: list[int] = []

    num_bits = bits_a.size

    for pass_index in range(number_of_passes):
        block_size = min(
            num_bits,
            initial_block_size
            * (
                2
                ** min(
                    pass_index,
                    maximum_block_doublings,
                )
            ),
        )

        block_sizes.append(int(block_size))

        pass_rng = np.random.default_rng(
            int(seed_values[pass_index])
        )

        permutation = pass_rng.permutation(
            num_bits
        ).astype(np.int64)

        for block_start in range(
            0,
            num_bits,
            block_size,
        ):
            block_indices = permutation[
                block_start:
                block_start + block_size
            ]

            parity_leakage_bits += 1

            parity_a = _parity(
                bits_a,
                block_indices,
            )

            parity_b = _parity(
                corrected_bits_b,
                block_indices,
            )

            if parity_a == parity_b:
                continue

            search_indices = block_indices

            while search_indices.size > 1:
                midpoint = search_indices.size // 2

                left_indices = search_indices[:midpoint]
                right_indices = search_indices[midpoint:]

                parity_leakage_bits += 1

                left_parity_a = _parity(
                    bits_a,
                    left_indices,
                )

                left_parity_b = _parity(
                    corrected_bits_b,
                    left_indices,
                )

                if left_parity_a != left_parity_b:
                    search_indices = left_indices
                else:
                    search_indices = right_indices

            corrected_position = int(search_indices[0])
            corrected_bits_b[corrected_position] ^= np.uint8(1)
            corrections_applied += 1

    pre_kdr = float(
        np.mean(bits_a != bits_b_before)
    )

    post_kdr = float(
        np.mean(bits_a != corrected_bits_b)
    )

    verification_tag_a = _verification_tag(
        bits_a,
        verification_tag_bits,
    )

    verification_tag_b = _verification_tag(
        corrected_bits_b,
        verification_tag_bits,
    )

    verification_passed = bool(
        np.array_equal(
            verification_tag_a,
            verification_tag_b,
        )
    )

    exact_match = bool(
        np.array_equal(
            bits_a,
            corrected_bits_b,
        )
    )

    return ReconciliationResult(
        bits_at_a=np.asarray(
            bits_a,
            dtype=np.uint8,
        ),
        bits_at_b_before_reconciliation=np.asarray(
            bits_b_before,
            dtype=np.uint8,
        ),
        bits_at_b_after_reconciliation=np.asarray(
            corrected_bits_b,
            dtype=np.uint8,
        ),
        pre_reconciliation_kdr=float(pre_kdr),
        post_reconciliation_kdr=float(post_kdr),
        number_of_passes=int(number_of_passes),
        corrections_applied=int(corrections_applied),
        parity_leakage_bits=int(parity_leakage_bits),
        verification_leakage_bits=int(verification_tag_bits),
        public_permutation_seeds=np.asarray(
            seed_values,
            dtype=np.uint64,
        ),
        block_sizes=tuple(block_sizes),
        exact_match=exact_match,
        verification_passed=verification_passed,
    )


def toeplitz_hash(
    input_bits: ArrayLike,
    output_length_bits: int,
    *,
    public_seed_bits: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[BitArray, BitArray]:
    """
    使用二元Toeplitz矩阵执行通用哈希。

    对长度n的输入和长度m的输出，公开种子长度为n+m-1。
    """
    bits = _as_bit_vector(
        input_bits,
        "input_bits",
    )

    if isinstance(output_length_bits, bool) or not isinstance(
        output_length_bits,
        (int, np.integer),
    ):
        raise ValueError(
            "output_length_bits must be an integer"
        )

    if output_length_bits < 1:
        raise ValueError(
            "output_length_bits must be positive"
        )

    if output_length_bits > bits.size:
        raise ValueError(
            "output_length_bits cannot exceed the input bit length"
        )

    required_seed_length = (
        bits.size
        + output_length_bits
        - 1
    )

    generator = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    if public_seed_bits is None:
        seed_bits = generator.integers(
            low=0,
            high=2,
            size=required_seed_length,
            dtype=np.uint8,
        )
    else:
        seed_bits = _as_bit_vector(
            public_seed_bits,
            "public_seed_bits",
        )

        if seed_bits.size != required_seed_length:
            raise ValueError(
                "public_seed_bits has an invalid length"
            )

    output_bits = np.zeros(
        output_length_bits,
        dtype=np.uint8,
    )

    input_positions = np.arange(
        bits.size,
        dtype=np.int64,
    )

    for output_index in range(output_length_bits):
        seed_indices = (
            output_index
            - input_positions
            + bits.size
            - 1
        )

        output_bits[output_index] = np.uint8(
            np.sum(
                bits * seed_bits[seed_indices],
                dtype=np.int64,
            )
            % 2
        )

    return (
        np.asarray(
            output_bits,
            dtype=np.uint8,
        ),
        np.asarray(
            seed_bits,
            dtype=np.uint8,
        ),
    )


def _bits_to_hex(
    bits: BitArray,
) -> str:
    packed = np.packbits(
        bits,
        bitorder="big",
    )

    return packed.tobytes().hex()


def apply_privacy_amplification(
    reconciled_bits_at_a: ArrayLike,
    reconciled_bits_at_b: ArrayLike,
    output_length_bits: int,
    *,
    public_seed_bits: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> PrivacyAmplificationResult:
    """对协调后的两端比特执行相同的Toeplitz通用哈希。"""
    bits_a = _as_bit_vector(
        reconciled_bits_at_a,
        "reconciled_bits_at_a",
    )

    bits_b = _as_bit_vector(
        reconciled_bits_at_b,
        "reconciled_bits_at_b",
    )

    if bits_a.size != bits_b.size:
        raise ValueError(
            "the reconciled bit sequences must have the same length"
        )

    final_key_a, seed_bits = toeplitz_hash(
        bits_a,
        output_length_bits,
        public_seed_bits=public_seed_bits,
        rng=rng,
    )

    final_key_b, _ = toeplitz_hash(
        bits_b,
        output_length_bits,
        public_seed_bits=seed_bits,
    )

    return PrivacyAmplificationResult(
        input_length_bits=int(bits_a.size),
        output_length_bits=int(output_length_bits),
        public_seed_bits=np.asarray(
            seed_bits,
            dtype=np.uint8,
        ),
        final_key_at_a=np.asarray(
            final_key_a,
            dtype=np.uint8,
        ),
        final_key_at_b=np.asarray(
            final_key_b,
            dtype=np.uint8,
        ),
        final_key_at_a_hex=_bits_to_hex(final_key_a),
        final_key_at_b_hex=_bits_to_hex(final_key_b),
        keys_match=bool(
            np.array_equal(
                final_key_a,
                final_key_b,
            )
        ),
    )


def generate_end_to_end_key_from_quantization(
    quantization: QuantizationResult,
    *,
    pre_reconciliation_entropy_bound_bits: float,
    initial_block_size: int = 8,
    number_of_reconciliation_passes: int = 8,
    maximum_block_doublings: int = 3,
    verification_tag_bits: int = 32,
    privacy_margin_bits: int = 64,
    maximum_final_key_bits: int = 256,
    rng: np.random.Generator | None = None,
) -> EndToEndKeyGenerationResult:
    """
    从预先计算的量化结果开始执行信息协调、验证与隐私放大。

    该入口用于量化域Eve安全评估，确保安全评估和实际密钥生成使用完全相同的
    保留位置与原始比特序列。
    """
    if not np.isfinite(pre_reconciliation_entropy_bound_bits):
        raise ValueError(
            "pre_reconciliation_entropy_bound_bits must be finite"
        )

    if pre_reconciliation_entropy_bound_bits < 0.0:
        raise ValueError(
            "pre_reconciliation_entropy_bound_bits cannot be negative"
        )

    if isinstance(privacy_margin_bits, bool) or not isinstance(
        privacy_margin_bits,
        (int, np.integer),
    ):
        raise ValueError(
            "privacy_margin_bits must be an integer"
        )

    if privacy_margin_bits < 0:
        raise ValueError(
            "privacy_margin_bits cannot be negative"
        )

    if isinstance(maximum_final_key_bits, bool) or not isinstance(
        maximum_final_key_bits,
        (int, np.integer),
    ):
        raise ValueError(
            "maximum_final_key_bits must be an integer"
        )

    if maximum_final_key_bits < 1:
        raise ValueError(
            "maximum_final_key_bits must be positive"
        )

    generator = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    reconciliation = cascade_reconcile(
        quantization.bits_at_a,
        quantization.bits_at_b,
        initial_block_size=initial_block_size,
        number_of_passes=number_of_reconciliation_passes,
        maximum_block_doublings=maximum_block_doublings,
        verification_tag_bits=verification_tag_bits,
        rng=generator,
    )

    entropy_bound = max(
        0,
        floor(
            pre_reconciliation_entropy_bound_bits
        ),
    )

    input_bit_cap = int(
        reconciliation.bits_at_a.size
    )

    bounded_entropy = min(
        entropy_bound,
        input_bit_cap,
    )

    operational_secret_bit_bound = max(
        0,
        bounded_entropy
        - reconciliation.parity_leakage_bits
        - reconciliation.verification_leakage_bits
        - privacy_margin_bits,
    )

    final_key_length = min(
        int(maximum_final_key_bits),
        int(operational_secret_bit_bound),
    )

    if (
        not reconciliation.verification_passed
        or final_key_length <= 0
    ):
        return EndToEndKeyGenerationResult(
            quantization=quantization,
            reconciliation=reconciliation,
            pre_reconciliation_entropy_bound_bits=int(
                entropy_bound
            ),
            operational_secret_bit_bound=int(
                operational_secret_bit_bound
            ),
            requested_maximum_key_bits=int(
                maximum_final_key_bits
            ),
            final_key_length_bits=0,
            privacy_margin_bits=int(
                privacy_margin_bits
            ),
            privacy_amplification=None,
            success=False,
        )

    privacy_amplification = apply_privacy_amplification(
        reconciliation.bits_at_a,
        reconciliation.bits_at_b_after_reconciliation,
        final_key_length,
        rng=generator,
    )

    success = bool(
        reconciliation.verification_passed
        and privacy_amplification.keys_match
        and final_key_length > 0
    )

    return EndToEndKeyGenerationResult(
        quantization=quantization,
        reconciliation=reconciliation,
        pre_reconciliation_entropy_bound_bits=int(
            entropy_bound
        ),
        operational_secret_bit_bound=int(
            operational_secret_bit_bound
        ),
        requested_maximum_key_bits=int(
            maximum_final_key_bits
        ),
        final_key_length_bits=int(
            final_key_length
        ),
        privacy_margin_bits=int(
            privacy_margin_bits
        ),
        privacy_amplification=privacy_amplification,
        success=success,
    )


def generate_end_to_end_key(
    observation_at_a: ArrayLike,
    observation_at_b: ArrayLike,
    *,
    pre_reconciliation_entropy_bound_bits: float,
    feature: FeatureName = "real",
    guard_band_sigma: float = 0.1,
    selection_policy: SelectionPolicy = "intersection",
    initial_block_size: int = 8,
    number_of_reconciliation_passes: int = 8,
    maximum_block_doublings: int = 3,
    verification_tag_bits: int = 32,
    privacy_margin_bits: int = 64,
    maximum_final_key_bits: int = 256,
    rng: np.random.Generator | None = None,
) -> EndToEndKeyGenerationResult:
    """从复信道观测执行完整的实际密钥生成流程。"""
    quantization = quantize_with_guard_band(
        observation_at_a,
        observation_at_b,
        feature=feature,
        guard_band_sigma=guard_band_sigma,
        selection_policy=selection_policy,
    )

    return generate_end_to_end_key_from_quantization(
        quantization,
        pre_reconciliation_entropy_bound_bits=(
            pre_reconciliation_entropy_bound_bits
        ),
        initial_block_size=initial_block_size,
        number_of_reconciliation_passes=(
            number_of_reconciliation_passes
        ),
        maximum_block_doublings=maximum_block_doublings,
        verification_tag_bits=verification_tag_bits,
        privacy_margin_bits=privacy_margin_bits,
        maximum_final_key_bits=maximum_final_key_bits,
        rng=rng,
    )