from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from numpy.typing import ArrayLike, NDArray

from active_star_ris.practical_key_generation import (
    QuantizationResult,
)


FloatArray = NDArray[np.float64]
BitArray = NDArray[np.uint8]


@dataclass(frozen=True)
class QuantizedEveSecurityMetrics:
    """
    Eve针对量化后二进制原始密钥的安全评估结果。

    该模块使用交叉拟合的正则化LDA估计：

        P(Q_A = 1 | Z_E)

    Eve特征由双时隙复观测构成：

        [
            Re(Z_E_forward),
            Im(Z_E_forward),
            Re(Z_E_reverse),
            Im(Z_E_reverse)
        ]

    conditional_min_entropy_lower_bound_bits_per_retained_bit
    是加入有限样本猜测概率上置信修正后的最小熵代理。
    """

    total_observation_samples: int
    retained_samples: int

    class_zero_count: int
    class_one_count: int

    source_probability_zero: float
    source_probability_one: float

    source_shannon_entropy_bits_per_retained_bit: float
    source_min_entropy_bits_per_retained_bit: float

    conditional_shannon_entropy_proxy_bits_per_retained_bit: float
    eve_mutual_information_proxy_bits_per_retained_bit: float

    eve_guessing_probability_estimate: float
    guessing_probability_confidence_radius: float
    eve_guessing_probability_upper_bound: float

    conditional_min_entropy_lower_bound_bits_per_retained_bit: float

    cross_validation_folds: int
    estimation_failure_probability: float


@dataclass(frozen=True)
class QuantizedPreReconciliationBound:
    """
    实际Cascade运行之前的量化域熵界。

    此结果已经扣除认证泄漏，但尚未扣除：

        1. Cascade实际奇偶校验泄漏；
        2. 一致性验证标签；
        3. 隐私放大裕量。
    """

    retained_raw_key_bits: int

    conditional_min_entropy_lower_bound_bits_per_retained_bit: float

    gross_conditional_min_entropy_bits: int

    authentication_leakage_bits: int

    pre_reconciliation_entropy_bound_bits: int


def _binary_entropy(
    probability_one: float,
) -> float:
    probability = float(
        np.clip(
            probability_one,
            0.0,
            1.0,
        )
    )

    if probability <= 0.0 or probability >= 1.0:
        return 0.0

    return float(
        -probability
        * np.log2(
            probability
        )
        - (
            1.0
            - probability
        )
        * np.log2(
            1.0
            - probability
        )
    )


def _as_complex_vector(
    values: ArrayLike,
    name: str,
) -> NDArray[np.complex128]:
    array = np.asarray(
        values,
        dtype=np.complex128,
    ).reshape(-1)

    if array.size == 0:
        raise ValueError(
            f"{name} cannot be empty"
        )

    if not np.all(
        np.isfinite(
            array.real
        )
    ):
        raise ValueError(
            f"{name} contains non-finite real values"
        )

    if not np.all(
        np.isfinite(
            array.imag
        )
    ):
        raise ValueError(
            f"{name} contains non-finite imaginary values"
        )

    return np.asarray(
        array,
        dtype=np.complex128,
    )


def _build_eve_feature_matrix(
    quantization: QuantizationResult,
    eve_observation_forward: ArrayLike,
    eve_observation_reverse: ArrayLike,
) -> FloatArray:
    eve_forward = _as_complex_vector(
        eve_observation_forward,
        "eve_observation_forward",
    )

    eve_reverse = _as_complex_vector(
        eve_observation_reverse,
        "eve_observation_reverse",
    )

    if (
        eve_forward.size
        != quantization.total_samples
    ):
        raise ValueError(
            "eve_observation_forward must contain "
            "one value per quantization input sample"
        )

    if (
        eve_reverse.size
        != quantization.total_samples
    ):
        raise ValueError(
            "eve_observation_reverse must contain "
            "one value per quantization input sample"
        )

    all_features = np.column_stack(
        (
            eve_forward.real,
            eve_forward.imag,
            eve_reverse.real,
            eve_reverse.imag,
        )
    )

    retained_features = all_features[
        quantization.retained_indices
    ]

    return np.asarray(
        retained_features,
        dtype=np.float64,
    )


def _fit_predict_regularized_lda(
    training_features: FloatArray,
    training_bits: BitArray,
    testing_features: FloatArray,
    covariance_regularization: float,
) -> FloatArray:
    """
    拟合共享协方差的二分类LDA，并返回测试集P(Q=1|Z)。
    """
    num_training_samples = (
        training_features.shape[0]
    )

    class_zero_mask = (
        training_bits == 0
    )

    class_one_mask = (
        training_bits == 1
    )

    class_zero_count = int(
        np.sum(
            class_zero_mask
        )
    )

    class_one_count = int(
        np.sum(
            class_one_mask
        )
    )

    # 拉普拉斯平滑后的先验概率。
    prior_one = (
        class_one_count
        + 1.0
    ) / (
        num_training_samples
        + 2.0
    )

    prior_zero = (
        class_zero_count
        + 1.0
    ) / (
        num_training_samples
        + 2.0
    )

    if (
        class_zero_count < 2
        or class_one_count < 2
    ):
        return np.full(
            testing_features.shape[0],
            prior_one,
            dtype=np.float64,
        )

    feature_mean = np.mean(
        training_features,
        axis=0,
    )

    feature_scale = np.std(
        training_features,
        axis=0,
        ddof=1,
    )

    feature_scale = np.where(
        feature_scale
        > np.finfo(
            np.float64
        ).eps,
        feature_scale,
        1.0,
    )

    standardized_training = (
        training_features
        - feature_mean
    ) / feature_scale

    standardized_testing = (
        testing_features
        - feature_mean
    ) / feature_scale

    class_zero_mean = np.mean(
        standardized_training[
            class_zero_mask
        ],
        axis=0,
    )

    class_one_mean = np.mean(
        standardized_training[
            class_one_mask
        ],
        axis=0,
    )

    centered_training = np.empty_like(
        standardized_training
    )

    centered_training[
        class_zero_mask
    ] = (
        standardized_training[
            class_zero_mask
        ]
        - class_zero_mean
    )

    centered_training[
        class_one_mask
    ] = (
        standardized_training[
            class_one_mask
        ]
        - class_one_mean
    )

    covariance_denominator = max(
        num_training_samples
        - 2,
        1,
    )

    covariance = (
        centered_training.T
        @ centered_training
        / covariance_denominator
    )

    num_features = covariance.shape[0]

    covariance_scale = max(
        float(
            np.trace(
                covariance
            )
            / num_features
        ),
        np.finfo(
            np.float64
        ).eps,
    )

    regularized_covariance = (
        covariance
        + covariance_regularization
        * covariance_scale
        * np.eye(
            num_features,
            dtype=np.float64,
        )
    )

    mean_difference = (
        class_one_mean
        - class_zero_mean
    )

    try:
        discriminant_vector = np.linalg.solve(
            regularized_covariance,
            mean_difference,
        )
    except np.linalg.LinAlgError:
        discriminant_vector = (
            np.linalg.pinv(
                regularized_covariance
            )
            @ mean_difference
        )

    intercept = (
        -0.5
        * np.dot(
            class_one_mean
            + class_zero_mean,
            discriminant_vector,
        )
        + np.log(
            prior_one
            / prior_zero
        )
    )

    scores = (
        standardized_testing
        @ discriminant_vector
        + intercept
    )

    scores = np.clip(
        scores,
        -60.0,
        60.0,
    )

    probability_one = (
        1.0
        / (
            1.0
            + np.exp(
                -scores
            )
        )
    )

    return np.asarray(
        probability_one,
        dtype=np.float64,
    )


def _cross_fitted_lda_probabilities(
    features: FloatArray,
    bits: BitArray,
    *,
    num_folds: int,
    covariance_regularization: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, int]:
    """
    使用分层交叉拟合避免在同一数据上拟合和评估Eve。
    """
    class_zero_indices = np.flatnonzero(
        bits == 0
    )

    class_one_indices = np.flatnonzero(
        bits == 1
    )

    class_zero_count = int(
        class_zero_indices.size
    )

    class_one_count = int(
        class_one_indices.size
    )

    probability_one_prior = float(
        np.mean(
            bits
        )
    )

    if (
        class_zero_count < 2
        or class_one_count < 2
    ):
        return (
            np.full(
                bits.size,
                probability_one_prior,
                dtype=np.float64,
            ),
            1,
        )

    effective_folds = min(
        int(
            num_folds
        ),
        class_zero_count,
        class_one_count,
    )

    effective_folds = max(
        2,
        effective_folds,
    )

    shuffled_zero_indices = np.array(
        class_zero_indices,
        copy=True,
    )

    shuffled_one_indices = np.array(
        class_one_indices,
        copy=True,
    )

    rng.shuffle(
        shuffled_zero_indices
    )

    rng.shuffle(
        shuffled_one_indices
    )

    zero_folds = np.array_split(
        shuffled_zero_indices,
        effective_folds,
    )

    one_folds = np.array_split(
        shuffled_one_indices,
        effective_folds,
    )

    predicted_probability_one = np.empty(
        bits.size,
        dtype=np.float64,
    )

    all_indices = np.arange(
        bits.size,
        dtype=np.int64,
    )

    for fold_index in range(
        effective_folds
    ):
        testing_indices = np.concatenate(
            (
                zero_folds[
                    fold_index
                ],
                one_folds[
                    fold_index
                ],
            )
        ).astype(
            np.int64
        )

        training_mask = np.ones(
            bits.size,
            dtype=bool,
        )

        training_mask[
            testing_indices
        ] = False

        training_indices = all_indices[
            training_mask
        ]

        predicted_probability_one[
            testing_indices
        ] = _fit_predict_regularized_lda(
            features[
                training_indices
            ],
            bits[
                training_indices
            ],
            features[
                testing_indices
            ],
            covariance_regularization,
        )

    return (
        np.asarray(
            predicted_probability_one,
            dtype=np.float64,
        ),
        int(
            effective_folds
        ),
    )


def evaluate_quantized_eve_security(
    quantization: QuantizationResult,
    eve_observation_forward: ArrayLike,
    eve_observation_reverse: ArrayLike,
    *,
    num_cross_validation_folds: int = 5,
    covariance_regularization: float = 1.0e-3,
    estimation_failure_probability: float = 1.0e-6,
    rng: np.random.Generator | None = None,
) -> QuantizedEveSecurityMetrics:
    """
    评估Eve对Alice量化比特的猜测能力。

    该结果是交叉拟合、模型依赖的安全评估代理，
    不是完整的可组合安全证明。
    """
    if isinstance(
        num_cross_validation_folds,
        bool,
    ) or not isinstance(
        num_cross_validation_folds,
        (int, np.integer),
    ):
        raise ValueError(
            "num_cross_validation_folds "
            "must be an integer"
        )

    if num_cross_validation_folds < 2:
        raise ValueError(
            "num_cross_validation_folds "
            "must be at least 2"
        )

    if not np.isfinite(
        covariance_regularization
    ):
        raise ValueError(
            "covariance_regularization must be finite"
        )

    if covariance_regularization < 0.0:
        raise ValueError(
            "covariance_regularization "
            "cannot be negative"
        )

    if not np.isfinite(
        estimation_failure_probability
    ):
        raise ValueError(
            "estimation_failure_probability "
            "must be finite"
        )

    if not (
        0.0
        < estimation_failure_probability
        < 1.0
    ):
        raise ValueError(
            "estimation_failure_probability must "
            "lie strictly between 0 and 1"
        )

    bits = np.asarray(
        quantization.bits_at_a,
        dtype=np.uint8,
    ).reshape(-1)

    if bits.size < 4:
        raise ValueError(
            "at least four retained bits are required"
        )

    if not np.all(
        np.logical_or(
            bits == 0,
            bits == 1,
        )
    ):
        raise ValueError(
            "quantization bits must contain only 0 and 1"
        )

    features = _build_eve_feature_matrix(
        quantization,
        eve_observation_forward,
        eve_observation_reverse,
    )

    if features.shape[0] != bits.size:
        raise ValueError(
            "the retained Eve feature count must "
            "equal the raw key bit count"
        )

    generator = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    (
        probability_one_given_eve,
        effective_folds,
    ) = _cross_fitted_lda_probabilities(
        features,
        bits,
        num_folds=(
            num_cross_validation_folds
        ),
        covariance_regularization=(
            covariance_regularization
        ),
        rng=generator,
    )

    numerical_floor = 1.0e-12

    probability_one_given_eve = np.clip(
        probability_one_given_eve,
        numerical_floor,
        1.0
        - numerical_floor,
    )

    probability_one = float(
        np.mean(
            bits
        )
    )

    probability_zero = (
        1.0
        - probability_one
    )

    source_shannon_entropy = (
        _binary_entropy(
            probability_one
        )
    )

    majority_probability = max(
        probability_zero,
        probability_one,
    )

    source_min_entropy = float(
        -np.log2(
            max(
                majority_probability,
                numerical_floor,
            )
        )
    )

    conditional_cross_entropy = float(
        -np.mean(
            bits
            * np.log2(
                probability_one_given_eve
            )
            + (
                1
                - bits
            )
            * np.log2(
                1.0
                - probability_one_given_eve
            )
        )
    )

    conditional_shannon_entropy_proxy = float(
        np.clip(
            conditional_cross_entropy,
            0.0,
            source_shannon_entropy,
        )
    )

    eve_mutual_information_proxy = max(
        0.0,
        source_shannon_entropy
        - conditional_shannon_entropy_proxy,
    )

    posterior_guessing_probability = float(
        np.mean(
            np.maximum(
                probability_one_given_eve,
                1.0
                - probability_one_given_eve,
            )
        )
    )

    guessing_probability_estimate = max(
        posterior_guessing_probability,
        majority_probability,
    )

    confidence_radius = float(
        np.sqrt(
            np.log(
                1.0
                / estimation_failure_probability
            )
            / (
                2.0
                * bits.size
            )
        )
    )

    guessing_probability_upper_bound = min(
        1.0,
        guessing_probability_estimate
        + confidence_radius,
    )

    if guessing_probability_upper_bound >= 1.0:
        conditional_min_entropy_lower_bound = 0.0
    else:
        conditional_min_entropy_lower_bound = float(
            -np.log2(
                guessing_probability_upper_bound
            )
        )

    conditional_min_entropy_lower_bound = min(
        source_min_entropy,
        max(
            0.0,
            conditional_min_entropy_lower_bound,
        ),
    )

    class_zero_count = int(
        np.sum(
            bits == 0
        )
    )

    class_one_count = int(
        np.sum(
            bits == 1
        )
    )

    return QuantizedEveSecurityMetrics(
        total_observation_samples=int(
            quantization.total_samples
        ),
        retained_samples=int(
            bits.size
        ),
        class_zero_count=(
            class_zero_count
        ),
        class_one_count=(
            class_one_count
        ),
        source_probability_zero=float(
            probability_zero
        ),
        source_probability_one=float(
            probability_one
        ),
        source_shannon_entropy_bits_per_retained_bit=float(
            source_shannon_entropy
        ),
        source_min_entropy_bits_per_retained_bit=float(
            source_min_entropy
        ),
        conditional_shannon_entropy_proxy_bits_per_retained_bit=float(
            conditional_shannon_entropy_proxy
        ),
        eve_mutual_information_proxy_bits_per_retained_bit=float(
            eve_mutual_information_proxy
        ),
        eve_guessing_probability_estimate=float(
            guessing_probability_estimate
        ),
        guessing_probability_confidence_radius=float(
            confidence_radius
        ),
        eve_guessing_probability_upper_bound=float(
            guessing_probability_upper_bound
        ),
        conditional_min_entropy_lower_bound_bits_per_retained_bit=float(
            conditional_min_entropy_lower_bound
        ),
        cross_validation_folds=int(
            effective_folds
        ),
        estimation_failure_probability=float(
            estimation_failure_probability
        ),
    )


def evaluate_quantized_pre_reconciliation_bound(
    metrics: QuantizedEveSecurityMetrics,
    *,
    authentication_leakage_bits: float = 0.0,
) -> QuantizedPreReconciliationBound:
    """
    将每个保留比特的条件最小熵下界转换为总比特上界。
    """
    if not np.isfinite(
        authentication_leakage_bits
    ):
        raise ValueError(
            "authentication_leakage_bits must be finite"
        )

    if authentication_leakage_bits < 0.0:
        raise ValueError(
            "authentication_leakage_bits "
            "cannot be negative"
        )

    gross_conditional_min_entropy_bits = max(
        0,
        floor(
            metrics.retained_samples
            * metrics
            .conditional_min_entropy_lower_bound_bits_per_retained_bit
        ),
    )

    authentication_bits = int(
        ceil(
            authentication_leakage_bits
        )
    )

    pre_reconciliation_entropy_bound = max(
        0,
        gross_conditional_min_entropy_bits
        - authentication_bits,
    )

    return QuantizedPreReconciliationBound(
        retained_raw_key_bits=int(
            metrics.retained_samples
        ),
        conditional_min_entropy_lower_bound_bits_per_retained_bit=float(
            metrics
            .conditional_min_entropy_lower_bound_bits_per_retained_bit
        ),
        gross_conditional_min_entropy_bits=int(
            gross_conditional_min_entropy_bits
        ),
        authentication_leakage_bits=int(
            authentication_bits
        ),
        pre_reconciliation_entropy_bound_bits=int(
            pre_reconciliation_entropy_bound
        ),
    )