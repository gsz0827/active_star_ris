from __future__ import annotations

import csv
from dataclasses import (
    asdict,
    dataclass,
    fields,
    replace,
)
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from active_star_ris.finite_length_security import (
    FiniteLengthSecurityParameters,
    evaluate_dual_side_finite_length_security,
)
from active_star_ris.practical_key_generation import (
    generate_end_to_end_key_from_quantization,
    quantize_with_guard_band,
)
from active_star_ris.quantized_security import (
    evaluate_quantized_eve_security,
    evaluate_quantized_pre_reconciliation_bound,
)
from active_star_ris.secure_key_generation import (
    simulate_dual_side_secure_key_generation,
)
from active_star_ris.star_key_system import (
    build_star_coefficients,
)


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class SweepExperimentConfig:
    """参数扫描中保持固定的实验配置。"""

    num_samples: int = 2000
    num_elements: int = 16

    parameter_estimation_fraction: float = 0.1

    beta_transmission: float = 0.65

    active_noise_variance: float = 0.002

    # 仅用于连续域有限长度代理指标；实际协议路径使用Cascade真实泄漏。
    reconciliation_efficiency: float = 0.95

    epsilon_smoothing: float = 1.0e-10
    epsilon_parameter_estimation: float = 1.0e-10
    epsilon_privacy_amplification: float = 1.0e-10

    authentication_leakage_bits: float = 128.0
    implementation_margin_bits_per_sample: float = 0.01

    transmission_weight: float = 0.5
    reflection_weight: float = 0.5

    reconciliation_initial_block_size: int = 8
    reconciliation_passes: int = 8
    reconciliation_maximum_block_doublings: int = 3

    verification_tag_bits: int = 32
    privacy_margin_bits: int = 64
    maximum_final_key_bits: int = 128

    quantized_security_cross_validation_folds: int = 5
    quantized_security_covariance_regularization: float = 1.0e-3

    channel_seed: int = 20260719
    coefficient_seed: int = 20260720
    observation_seed: int = 20260721
    practical_seed: int = 20260722


@dataclass(frozen=True)
class SweepPoint:
    """
    单个参数扫描点。

    Eve信道采用：

        h_E = scale * (sqrt(rho) h_U + sqrt(1-rho) h_ind)

    因此eve_channel_correlation是功率相关参数。
    """

    sweep_name: str = "baseline"
    scenario: str = "baseline"

    active_fraction: float = 0.25
    active_gain: float = 1.5

    directional_gain_error_std_db: float = 0.0
    directional_phase_error_std_rad: float = 0.0

    eve_channel_scale: float = 0.2
    eve_channel_correlation: float = 0.0

    legitimate_receiver_noise_variance: float = 0.01
    eve_receiver_noise_variance: float = 0.5

    guard_band_sigma: float = 0.05
    selection_policy: str = "alice"


@dataclass(frozen=True)
class SweepPointResult:
    """单个参数点的一次Monte Carlo运行结果。"""

    repetition: int

    sweep_name: str
    scenario: str

    num_samples: int
    num_elements: int

    active_fraction: float
    active_count: int
    active_gain: float

    directional_gain_error_std_db: float
    directional_phase_error_std_rad: float

    eve_channel_scale: float
    eve_channel_correlation: float

    legitimate_receiver_noise_variance: float
    eve_receiver_noise_variance: float

    guard_band_sigma: float
    selection_policy: str

    weighted_legitimate_mi_bits_per_sample: float
    weighted_eve_leakage_bits_per_sample: float
    weighted_asymptotic_secret_rate_bits_per_sample: float

    # 连续高斯域、使用beta_rec的有限长度代理指标。
    weighted_finite_length_rate_bits_per_sample: float

    # 量化域Eve安全指标。
    weighted_quantized_eve_mutual_information_bits_per_retained_bit: float
    weighted_quantized_conditional_min_entropy_bits_per_retained_bit: float

    transmission_quantized_eve_guessing_probability: float
    reflection_quantized_eve_guessing_probability: float

    transmission_quantized_conditional_min_entropy_bits_per_retained_bit: float
    reflection_quantized_conditional_min_entropy_bits_per_retained_bit: float

    # 使用量化域熵界和Cascade实际泄漏得到的未截断运行指标。
    weighted_operational_bound_bits_per_sample: float

    transmission_pre_reconciliation_entropy_bound_bits: int
    reflection_pre_reconciliation_entropy_bound_bits: int

    transmission_operational_secret_bit_bound: int
    reflection_operational_secret_bit_bound: int
    aggregate_operational_secret_bit_bound: int

    transmission_retention_ratio: float
    reflection_retention_ratio: float

    transmission_raw_kdr: float
    reflection_raw_kdr: float

    transmission_post_reconciliation_kdr: float
    reflection_post_reconciliation_kdr: float

    transmission_corrections: int
    reflection_corrections: int

    transmission_parity_leakage_bits: int
    reflection_parity_leakage_bits: int

    transmission_final_key_bits: int
    reflection_final_key_bits: int
    aggregate_final_key_bits: int

    transmission_success: bool
    reflection_success: bool
    dual_side_success: bool


@dataclass(frozen=True)
class _BaseRealization:
    """
    同一次重复实验中所有参数点共享的随机实现。

    使用公共随机数可降低不同参数点比较时的Monte Carlo方差。
    """

    channel_controller_to_ris: ComplexArray

    channel_transmission_user: ComplexArray
    channel_reflection_user: ComplexArray

    independent_eve_transmission: ComplexArray
    independent_eve_reflection: ComplexArray

    base_phase_transmission: NDArray[np.float64]
    base_phase_reflection: NDArray[np.float64]

    gain_error_transmission_standard: NDArray[np.float64]
    gain_error_reflection_standard: NDArray[np.float64]

    phase_error_transmission_standard: NDArray[np.float64]
    phase_error_reflection_standard: NDArray[np.float64]

    active_element_order: NDArray[np.int64]


def _complex_gaussian(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> ComplexArray:
    values = (
        rng.normal(size=shape)
        + 1j * rng.normal(size=shape)
    ) / np.sqrt(2.0)

    return np.asarray(
        values,
        dtype=np.complex128,
    )


def _validate_config(
    config: SweepExperimentConfig,
) -> None:
    if isinstance(config.num_samples, bool) or not isinstance(
        config.num_samples,
        (int, np.integer),
    ):
        raise ValueError(
            "num_samples must be an integer"
        )

    if config.num_samples < 20:
        raise ValueError(
            "num_samples must be at least 20"
        )

    if isinstance(config.num_elements, bool) or not isinstance(
        config.num_elements,
        (int, np.integer),
    ):
        raise ValueError(
            "num_elements must be an integer"
        )

    if config.num_elements < 1:
        raise ValueError(
            "num_elements must be positive"
        )

    finite_values = {
        "parameter_estimation_fraction": (
            config.parameter_estimation_fraction
        ),
        "beta_transmission": config.beta_transmission,
        "active_noise_variance": config.active_noise_variance,
        "reconciliation_efficiency": (
            config.reconciliation_efficiency
        ),
        "epsilon_smoothing": config.epsilon_smoothing,
        "epsilon_parameter_estimation": (
            config.epsilon_parameter_estimation
        ),
        "epsilon_privacy_amplification": (
            config.epsilon_privacy_amplification
        ),
        "authentication_leakage_bits": (
            config.authentication_leakage_bits
        ),
        "implementation_margin_bits_per_sample": (
            config.implementation_margin_bits_per_sample
        ),
        "transmission_weight": config.transmission_weight,
        "reflection_weight": config.reflection_weight,
        "quantized_security_covariance_regularization": (
            config.quantized_security_covariance_regularization
        ),
    }

    for name, value in finite_values.items():
        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

    if not (
        0.0
        <= config.parameter_estimation_fraction
        < 1.0
    ):
        raise ValueError(
            "parameter_estimation_fraction must lie within [0, 1)"
        )

    if not 0.0 <= config.beta_transmission <= 1.0:
        raise ValueError(
            "beta_transmission must lie within [0, 1]"
        )

    if config.active_noise_variance < 0.0:
        raise ValueError(
            "active_noise_variance cannot be negative"
        )

    if not 0.0 < config.reconciliation_efficiency <= 1.0:
        raise ValueError(
            "reconciliation_efficiency must lie within (0, 1]"
        )

    probabilities = {
        "epsilon_smoothing": config.epsilon_smoothing,
        "epsilon_parameter_estimation": (
            config.epsilon_parameter_estimation
        ),
        "epsilon_privacy_amplification": (
            config.epsilon_privacy_amplification
        ),
    }

    for name, value in probabilities.items():
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"{name} must lie strictly between 0 and 1"
            )

    nonnegative_values = {
        "authentication_leakage_bits": (
            config.authentication_leakage_bits
        ),
        "implementation_margin_bits_per_sample": (
            config.implementation_margin_bits_per_sample
        ),
        "transmission_weight": config.transmission_weight,
        "reflection_weight": config.reflection_weight,
        "quantized_security_covariance_regularization": (
            config.quantized_security_covariance_regularization
        ),
    }

    for name, value in nonnegative_values.items():
        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative"
            )

    if (
        config.transmission_weight
        + config.reflection_weight
        <= 0.0
    ):
        raise ValueError(
            "at least one branch weight must be positive"
        )

    integer_values = {
        "reconciliation_initial_block_size": (
            config.reconciliation_initial_block_size
        ),
        "reconciliation_passes": config.reconciliation_passes,
        "reconciliation_maximum_block_doublings": (
            config.reconciliation_maximum_block_doublings
        ),
        "verification_tag_bits": config.verification_tag_bits,
        "privacy_margin_bits": config.privacy_margin_bits,
        "maximum_final_key_bits": config.maximum_final_key_bits,
        "quantized_security_cross_validation_folds": (
            config.quantized_security_cross_validation_folds
        ),
    }

    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError(
                f"{name} must be an integer"
            )

    if config.reconciliation_initial_block_size < 2:
        raise ValueError(
            "reconciliation_initial_block_size must be at least 2"
        )

    if config.reconciliation_passes < 1:
        raise ValueError(
            "reconciliation_passes must be positive"
        )

    if config.reconciliation_maximum_block_doublings < 0:
        raise ValueError(
            "reconciliation_maximum_block_doublings cannot be negative"
        )

    if not 1 <= config.verification_tag_bits <= 256:
        raise ValueError(
            "verification_tag_bits must lie within [1, 256]"
        )

    if config.privacy_margin_bits < 0:
        raise ValueError(
            "privacy_margin_bits cannot be negative"
        )

    if config.maximum_final_key_bits < 1:
        raise ValueError(
            "maximum_final_key_bits must be positive"
        )

    if config.quantized_security_cross_validation_folds < 2:
        raise ValueError(
            "quantized_security_cross_validation_folds must be at least 2"
        )


def _validate_point(
    point: SweepPoint,
) -> None:
    finite_values = {
        "active_fraction": point.active_fraction,
        "active_gain": point.active_gain,
        "directional_gain_error_std_db": (
            point.directional_gain_error_std_db
        ),
        "directional_phase_error_std_rad": (
            point.directional_phase_error_std_rad
        ),
        "eve_channel_scale": point.eve_channel_scale,
        "eve_channel_correlation": point.eve_channel_correlation,
        "legitimate_receiver_noise_variance": (
            point.legitimate_receiver_noise_variance
        ),
        "eve_receiver_noise_variance": (
            point.eve_receiver_noise_variance
        ),
        "guard_band_sigma": point.guard_band_sigma,
    }

    for name, value in finite_values.items():
        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

    if not 0.0 <= point.active_fraction <= 1.0:
        raise ValueError(
            "active_fraction must lie within [0, 1]"
        )

    if point.active_gain <= 0.0:
        raise ValueError(
            "active_gain must be positive"
        )

    nonnegative_values = {
        "directional_gain_error_std_db": (
            point.directional_gain_error_std_db
        ),
        "directional_phase_error_std_rad": (
            point.directional_phase_error_std_rad
        ),
        "eve_channel_scale": point.eve_channel_scale,
        "legitimate_receiver_noise_variance": (
            point.legitimate_receiver_noise_variance
        ),
        "eve_receiver_noise_variance": (
            point.eve_receiver_noise_variance
        ),
        "guard_band_sigma": point.guard_band_sigma,
    }

    for name, value in nonnegative_values.items():
        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative"
            )

    if not 0.0 <= point.eve_channel_correlation <= 1.0:
        raise ValueError(
            "eve_channel_correlation must lie within [0, 1]"
        )

    if point.selection_policy not in {
        "alice",
        "intersection",
    }:
        raise ValueError(
            "selection_policy must be 'alice' or 'intersection'"
        )


def _repetition_seed(
    base_seed: int,
    repetition: int,
) -> int:
    if isinstance(repetition, bool) or not isinstance(
        repetition,
        (int, np.integer),
    ):
        raise ValueError(
            "repetition must be an integer"
        )

    if repetition < 0:
        raise ValueError(
            "repetition cannot be negative"
        )

    return int(
        base_seed
        + 1_000_003 * repetition
    )


def _build_base_realization(
    config: SweepExperimentConfig,
    repetition: int,
) -> _BaseRealization:
    channel_rng = np.random.default_rng(
        _repetition_seed(
            config.channel_seed,
            repetition,
        )
    )

    coefficient_rng = np.random.default_rng(
        _repetition_seed(
            config.coefficient_seed,
            repetition,
        )
    )

    matrix_shape = (
        config.num_samples,
        config.num_elements,
    )

    return _BaseRealization(
        channel_controller_to_ris=_complex_gaussian(
            channel_rng,
            matrix_shape,
        ),
        channel_transmission_user=_complex_gaussian(
            channel_rng,
            matrix_shape,
        ),
        channel_reflection_user=_complex_gaussian(
            channel_rng,
            matrix_shape,
        ),
        independent_eve_transmission=_complex_gaussian(
            channel_rng,
            matrix_shape,
        ),
        independent_eve_reflection=_complex_gaussian(
            channel_rng,
            matrix_shape,
        ),
        base_phase_transmission=np.asarray(
            coefficient_rng.uniform(
                -np.pi,
                np.pi,
                size=config.num_elements,
            ),
            dtype=np.float64,
        ),
        base_phase_reflection=np.asarray(
            coefficient_rng.uniform(
                -np.pi,
                np.pi,
                size=config.num_elements,
            ),
            dtype=np.float64,
        ),
        gain_error_transmission_standard=np.asarray(
            coefficient_rng.normal(
                size=config.num_elements
            ),
            dtype=np.float64,
        ),
        gain_error_reflection_standard=np.asarray(
            coefficient_rng.normal(
                size=config.num_elements
            ),
            dtype=np.float64,
        ),
        phase_error_transmission_standard=np.asarray(
            coefficient_rng.normal(
                size=config.num_elements
            ),
            dtype=np.float64,
        ),
        phase_error_reflection_standard=np.asarray(
            coefficient_rng.normal(
                size=config.num_elements
            ),
            dtype=np.float64,
        ),
        active_element_order=np.asarray(
            coefficient_rng.permutation(
                config.num_elements
            ),
            dtype=np.int64,
        ),
    )


def _active_mask(
    config: SweepExperimentConfig,
    point: SweepPoint,
    base: _BaseRealization,
) -> NDArray[np.bool_]:
    active_count = int(
        np.clip(
            round(
                point.active_fraction
                * config.num_elements
            ),
            0,
            config.num_elements,
        )
    )

    mask = np.zeros(
        config.num_elements,
        dtype=bool,
    )

    if active_count > 0:
        mask[
            base.active_element_order[:active_count]
        ] = True

    return np.asarray(
        mask,
        dtype=bool,
    )


def _build_coefficients(
    config: SweepExperimentConfig,
    point: SweepPoint,
    base: _BaseRealization,
    active_mask: NDArray[np.bool_],
):
    amplitudes_forward = np.ones(
        config.num_elements,
        dtype=np.float64,
    )

    amplitudes_forward[active_mask] = point.active_gain

    beta_transmission = np.full(
        config.num_elements,
        config.beta_transmission,
        dtype=np.float64,
    )

    beta_reflection = 1.0 - beta_transmission

    forward_coefficients = build_star_coefficients(
        amplitudes=amplitudes_forward,
        beta_transmission=beta_transmission,
        beta_reflection=beta_reflection,
        phase_transmission=base.base_phase_transmission,
        phase_reflection=base.base_phase_reflection,
    )

    gain_error_transmission_db = (
        point.directional_gain_error_std_db
        * base.gain_error_transmission_standard
    )

    gain_error_reflection_db = (
        point.directional_gain_error_std_db
        * base.gain_error_reflection_standard
    )

    reverse_amplitudes_transmission = (
        amplitudes_forward
        * np.power(
            10.0,
            gain_error_transmission_db / 20.0,
        )
    )

    reverse_amplitudes_reflection = (
        amplitudes_forward
        * np.power(
            10.0,
            gain_error_reflection_db / 20.0,
        )
    )

    reverse_phase_transmission = (
        base.base_phase_transmission
        + point.directional_phase_error_std_rad
        * base.phase_error_transmission_standard
    )

    reverse_phase_reflection = (
        base.base_phase_reflection
        + point.directional_phase_error_std_rad
        * base.phase_error_reflection_standard
    )

    reverse_from_transmission = build_star_coefficients(
        amplitudes=reverse_amplitudes_transmission,
        beta_transmission=beta_transmission,
        beta_reflection=beta_reflection,
        phase_transmission=reverse_phase_transmission,
        phase_reflection=reverse_phase_reflection,
    )

    reverse_from_reflection = build_star_coefficients(
        amplitudes=reverse_amplitudes_reflection,
        beta_transmission=beta_transmission,
        beta_reflection=beta_reflection,
        phase_transmission=reverse_phase_transmission,
        phase_reflection=reverse_phase_reflection,
    )

    # 保留StarCoefficientPair中的amplitudes及beta字段，仅替换实际反向系数。
    reverse_coefficients = replace(
        forward_coefficients,
        transmission=(
            reverse_from_transmission.transmission
        ),
        reflection=(
            reverse_from_reflection.reflection
        ),
    )

    return (
        forward_coefficients,
        reverse_coefficients,
    )


def _build_eve_channels(
    point: SweepPoint,
    base: _BaseRealization,
) -> tuple[ComplexArray, ComplexArray]:
    rho = point.eve_channel_correlation

    correlated_weight = np.sqrt(rho)
    independent_weight = np.sqrt(1.0 - rho)

    eve_transmission = point.eve_channel_scale * (
        correlated_weight
        * base.channel_transmission_user
        + independent_weight
        * base.independent_eve_transmission
    )

    eve_reflection = point.eve_channel_scale * (
        correlated_weight
        * base.channel_reflection_user
        + independent_weight
        * base.independent_eve_reflection
    )

    return (
        np.asarray(
            eve_transmission,
            dtype=np.complex128,
        ),
        np.asarray(
            eve_reflection,
            dtype=np.complex128,
        ),
    )


def _parameter_estimation_samples(
    config: SweepExperimentConfig,
) -> int:
    count = int(
        round(
            config.parameter_estimation_fraction
            * config.num_samples
        )
    )

    return int(
        np.clip(
            count,
            0,
            config.num_samples - 1,
        )
    )


def _run_point_with_base(
    config: SweepExperimentConfig,
    point: SweepPoint,
    base: _BaseRealization,
    repetition: int,
) -> SweepPointResult:
    active_mask = _active_mask(
        config,
        point,
        base,
    )

    active_count = int(
        np.sum(active_mask)
    )

    (
        forward_coefficients,
        reverse_coefficients,
    ) = _build_coefficients(
        config,
        point,
        base,
        active_mask,
    )

    (
        eve_transmission,
        eve_reflection,
    ) = _build_eve_channels(
        point,
        base,
    )

    secure_result = simulate_dual_side_secure_key_generation(
        channel_controller_to_ris=(
            base.channel_controller_to_ris
        ),
        channel_ris_to_transmission_user=(
            base.channel_transmission_user
        ),
        channel_ris_to_reflection_user=(
            base.channel_reflection_user
        ),
        channel_ris_to_eve_transmission=eve_transmission,
        channel_ris_to_eve_reflection=eve_reflection,
        coefficients=forward_coefficients,
        reverse_coefficients=reverse_coefficients,
        active_mask=active_mask,
        active_noise_variance=config.active_noise_variance,
        receiver_noise_variance_controller=(
            point.legitimate_receiver_noise_variance
        ),
        receiver_noise_variance_transmission_user=(
            point.legitimate_receiver_noise_variance
        ),
        receiver_noise_variance_reflection_user=(
            point.legitimate_receiver_noise_variance
        ),
        receiver_noise_variance_eve_transmission=(
            point.eve_receiver_noise_variance
        ),
        receiver_noise_variance_eve_reflection=(
            point.eve_receiver_noise_variance
        ),
        # 实际协议路径使用Cascade真实泄漏，不使用固定公开泄漏占位值。
        public_leakage_transmission_bits_per_sample=0.0,
        public_leakage_reflection_bits_per_sample=0.0,
        transmission_weight=config.transmission_weight,
        reflection_weight=config.reflection_weight,
        rng=np.random.default_rng(
            _repetition_seed(
                config.observation_seed,
                repetition,
            )
        ),
    )

    parameter_estimation_samples = _parameter_estimation_samples(
        config
    )

    finite_parameters = FiniteLengthSecurityParameters(
        block_length=config.num_samples,
        parameter_estimation_samples=(
            parameter_estimation_samples
        ),
        reconciliation_efficiency=(
            config.reconciliation_efficiency
        ),
        epsilon_smoothing=config.epsilon_smoothing,
        epsilon_parameter_estimation=(
            config.epsilon_parameter_estimation
        ),
        epsilon_privacy_amplification=(
            config.epsilon_privacy_amplification
        ),
        authentication_leakage_bits=(
            config.authentication_leakage_bits
        ),
        implementation_margin_bits_per_sample=(
            config.implementation_margin_bits_per_sample
        ),
    )

    # 保留连续高斯域有限长度代理，便于与第6步结果比较。
    finite_result = evaluate_dual_side_finite_length_security(
        transmission_secrecy=(
            secure_result.transmission_secrecy
        ),
        reflection_secrecy=(
            secure_result.reflection_secrecy
        ),
        transmission_parameters=finite_parameters,
        reflection_parameters=finite_parameters,
        transmission_weight=config.transmission_weight,
        reflection_weight=config.reflection_weight,
    )

    transmission_probing = (
        secure_result
        .legitimate
        .transmission
        .probing
    )

    reflection_probing = (
        secure_result
        .legitimate
        .reflection
        .probing
    )

    transmission_observation_a = (
        transmission_probing.observation_at_a[
            parameter_estimation_samples:
        ]
    )

    transmission_observation_b = (
        transmission_probing.observation_at_b[
            parameter_estimation_samples:
        ]
    )

    reflection_observation_a = (
        reflection_probing.observation_at_a[
            parameter_estimation_samples:
        ]
    )

    reflection_observation_b = (
        reflection_probing.observation_at_b[
            parameter_estimation_samples:
        ]
    )

    transmission_eve_forward = (
        secure_result.transmission_eve.observation_forward[
            parameter_estimation_samples:
        ]
    )

    transmission_eve_reverse = (
        secure_result.transmission_eve.observation_reverse[
            parameter_estimation_samples:
        ]
    )

    reflection_eve_forward = (
        secure_result.reflection_eve.observation_forward[
            parameter_estimation_samples:
        ]
    )

    reflection_eve_reverse = (
        secure_result.reflection_eve.observation_reverse[
            parameter_estimation_samples:
        ]
    )

    transmission_quantization = quantize_with_guard_band(
        transmission_observation_a,
        transmission_observation_b,
        feature="real",
        guard_band_sigma=point.guard_band_sigma,
        selection_policy=point.selection_policy,
    )

    reflection_quantization = quantize_with_guard_band(
        reflection_observation_a,
        reflection_observation_b,
        feature="real",
        guard_band_sigma=point.guard_band_sigma,
        selection_policy=point.selection_policy,
    )

    practical_seed = _repetition_seed(
        config.practical_seed,
        repetition,
    )

    transmission_quantized_security = evaluate_quantized_eve_security(
        transmission_quantization,
        transmission_eve_forward,
        transmission_eve_reverse,
        num_cross_validation_folds=(
            config.quantized_security_cross_validation_folds
        ),
        covariance_regularization=(
            config.quantized_security_covariance_regularization
        ),
        estimation_failure_probability=(
            config.epsilon_parameter_estimation
        ),
        rng=np.random.default_rng(
            practical_seed + 10
        ),
    )

    reflection_quantized_security = evaluate_quantized_eve_security(
        reflection_quantization,
        reflection_eve_forward,
        reflection_eve_reverse,
        num_cross_validation_folds=(
            config.quantized_security_cross_validation_folds
        ),
        covariance_regularization=(
            config.quantized_security_covariance_regularization
        ),
        estimation_failure_probability=(
            config.epsilon_parameter_estimation
        ),
        rng=np.random.default_rng(
            practical_seed + 11
        ),
    )

    transmission_quantized_bound = (
        evaluate_quantized_pre_reconciliation_bound(
            transmission_quantized_security,
            authentication_leakage_bits=(
                config.authentication_leakage_bits
            ),
        )
    )

    reflection_quantized_bound = (
        evaluate_quantized_pre_reconciliation_bound(
            reflection_quantized_security,
            authentication_leakage_bits=(
                config.authentication_leakage_bits
            ),
        )
    )

    transmission_key = generate_end_to_end_key_from_quantization(
        transmission_quantization,
        pre_reconciliation_entropy_bound_bits=(
            transmission_quantized_bound
            .pre_reconciliation_entropy_bound_bits
        ),
        initial_block_size=(
            config.reconciliation_initial_block_size
        ),
        number_of_reconciliation_passes=(
            config.reconciliation_passes
        ),
        maximum_block_doublings=(
            config.reconciliation_maximum_block_doublings
        ),
        verification_tag_bits=config.verification_tag_bits,
        privacy_margin_bits=config.privacy_margin_bits,
        maximum_final_key_bits=config.maximum_final_key_bits,
        rng=np.random.default_rng(practical_seed),
    )

    reflection_key = generate_end_to_end_key_from_quantization(
        reflection_quantization,
        pre_reconciliation_entropy_bound_bits=(
            reflection_quantized_bound
            .pre_reconciliation_entropy_bound_bits
        ),
        initial_block_size=(
            config.reconciliation_initial_block_size
        ),
        number_of_reconciliation_passes=(
            config.reconciliation_passes
        ),
        maximum_block_doublings=(
            config.reconciliation_maximum_block_doublings
        ),
        verification_tag_bits=config.verification_tag_bits,
        privacy_margin_bits=config.privacy_margin_bits,
        maximum_final_key_bits=config.maximum_final_key_bits,
        rng=np.random.default_rng(practical_seed + 1),
    )

    transmission_operational_bits = int(
        transmission_key.operational_secret_bit_bound
    )

    reflection_operational_bits = int(
        reflection_key.operational_secret_bit_bound
    )

    aggregate_operational_bits = int(
        transmission_operational_bits
        + reflection_operational_bits
    )

    weight_sum = (
        config.transmission_weight
        + config.reflection_weight
    )

    normalized_transmission_weight = (
        config.transmission_weight
        / weight_sum
    )

    normalized_reflection_weight = (
        config.reflection_weight
        / weight_sum
    )

    weighted_quantized_eve_mi = (
        normalized_transmission_weight
        * transmission_quantized_security
        .eve_mutual_information_proxy_bits_per_retained_bit
        + normalized_reflection_weight
        * reflection_quantized_security
        .eve_mutual_information_proxy_bits_per_retained_bit
    )

    weighted_quantized_min_entropy = (
        normalized_transmission_weight
        * transmission_quantized_security
        .conditional_min_entropy_lower_bound_bits_per_retained_bit
        + normalized_reflection_weight
        * reflection_quantized_security
        .conditional_min_entropy_lower_bound_bits_per_retained_bit
    )

    key_generation_sample_count = (
        config.num_samples
        - parameter_estimation_samples
    )

    weighted_operational_rate = (
        normalized_transmission_weight
        * transmission_operational_bits
        / key_generation_sample_count
        + normalized_reflection_weight
        * reflection_operational_bits
        / key_generation_sample_count
    )

    transmission_final_bits = int(
        transmission_key.final_key_length_bits
    )

    reflection_final_bits = int(
        reflection_key.final_key_length_bits
    )

    aggregate_final_bits = int(
        transmission_final_bits
        + reflection_final_bits
    )

    return SweepPointResult(
        repetition=int(repetition),
        sweep_name=point.sweep_name,
        scenario=point.scenario,
        num_samples=int(config.num_samples),
        num_elements=int(config.num_elements),
        active_fraction=float(point.active_fraction),
        active_count=active_count,
        active_gain=float(point.active_gain),
        directional_gain_error_std_db=float(
            point.directional_gain_error_std_db
        ),
        directional_phase_error_std_rad=float(
            point.directional_phase_error_std_rad
        ),
        eve_channel_scale=float(point.eve_channel_scale),
        eve_channel_correlation=float(
            point.eve_channel_correlation
        ),
        legitimate_receiver_noise_variance=float(
            point.legitimate_receiver_noise_variance
        ),
        eve_receiver_noise_variance=float(
            point.eve_receiver_noise_variance
        ),
        guard_band_sigma=float(point.guard_band_sigma),
        selection_policy=str(point.selection_policy),
        weighted_legitimate_mi_bits_per_sample=float(
            secure_result
            .legitimate
            .weighted_mutual_information
        ),
        weighted_eve_leakage_bits_per_sample=float(
            secure_result
            .weighted_eve_leakage_bits_per_sample
        ),
        weighted_asymptotic_secret_rate_bits_per_sample=float(
            secure_result
            .weighted_secret_key_rate_bits_per_sample
        ),
        weighted_finite_length_rate_bits_per_sample=float(
            finite_result
            .weighted_finite_length_rate_bits_per_sample
        ),
        weighted_quantized_eve_mutual_information_bits_per_retained_bit=float(
            weighted_quantized_eve_mi
        ),
        weighted_quantized_conditional_min_entropy_bits_per_retained_bit=float(
            weighted_quantized_min_entropy
        ),
        transmission_quantized_eve_guessing_probability=float(
            transmission_quantized_security
            .eve_guessing_probability_upper_bound
        ),
        reflection_quantized_eve_guessing_probability=float(
            reflection_quantized_security
            .eve_guessing_probability_upper_bound
        ),
        transmission_quantized_conditional_min_entropy_bits_per_retained_bit=float(
            transmission_quantized_security
            .conditional_min_entropy_lower_bound_bits_per_retained_bit
        ),
        reflection_quantized_conditional_min_entropy_bits_per_retained_bit=float(
            reflection_quantized_security
            .conditional_min_entropy_lower_bound_bits_per_retained_bit
        ),
        weighted_operational_bound_bits_per_sample=float(
            weighted_operational_rate
        ),
        transmission_pre_reconciliation_entropy_bound_bits=int(
            transmission_quantized_bound
            .pre_reconciliation_entropy_bound_bits
        ),
        reflection_pre_reconciliation_entropy_bound_bits=int(
            reflection_quantized_bound
            .pre_reconciliation_entropy_bound_bits
        ),
        transmission_operational_secret_bit_bound=(
            transmission_operational_bits
        ),
        reflection_operational_secret_bit_bound=(
            reflection_operational_bits
        ),
        aggregate_operational_secret_bit_bound=(
            aggregate_operational_bits
        ),
        transmission_retention_ratio=float(
            transmission_key.quantization.retention_ratio
        ),
        reflection_retention_ratio=float(
            reflection_key.quantization.retention_ratio
        ),
        transmission_raw_kdr=float(
            transmission_key
            .quantization
            .raw_key_disagreement_rate
        ),
        reflection_raw_kdr=float(
            reflection_key
            .quantization
            .raw_key_disagreement_rate
        ),
        transmission_post_reconciliation_kdr=float(
            transmission_key
            .reconciliation
            .post_reconciliation_kdr
        ),
        reflection_post_reconciliation_kdr=float(
            reflection_key
            .reconciliation
            .post_reconciliation_kdr
        ),
        transmission_corrections=int(
            transmission_key
            .reconciliation
            .corrections_applied
        ),
        reflection_corrections=int(
            reflection_key
            .reconciliation
            .corrections_applied
        ),
        transmission_parity_leakage_bits=int(
            transmission_key
            .reconciliation
            .parity_leakage_bits
        ),
        reflection_parity_leakage_bits=int(
            reflection_key
            .reconciliation
            .parity_leakage_bits
        ),
        transmission_final_key_bits=(
            transmission_final_bits
        ),
        reflection_final_key_bits=(
            reflection_final_bits
        ),
        aggregate_final_key_bits=aggregate_final_bits,
        transmission_success=bool(
            transmission_key.success
        ),
        reflection_success=bool(
            reflection_key.success
        ),
        dual_side_success=bool(
            transmission_key.success
            and reflection_key.success
        ),
    )


def run_parameter_point(
    config: SweepExperimentConfig,
    point: SweepPoint,
    *,
    repetition: int = 0,
) -> SweepPointResult:
    """运行一个参数点。"""
    _validate_config(config)
    _validate_point(point)

    base = _build_base_realization(
        config,
        repetition,
    )

    return _run_point_with_base(
        config,
        point,
        base,
        repetition,
    )


def run_parameter_sweep(
    config: SweepExperimentConfig,
    points: Sequence[SweepPoint],
    *,
    num_repetitions: int = 1,
) -> tuple[SweepPointResult, ...]:
    """
    运行一组参数点。

    同一repetition内的所有参数点共享基础随机信道；不同repetition相互独立。
    """
    _validate_config(config)

    if isinstance(num_repetitions, bool) or not isinstance(
        num_repetitions,
        (int, np.integer),
    ):
        raise ValueError(
            "num_repetitions must be an integer"
        )

    if num_repetitions < 1:
        raise ValueError(
            "num_repetitions must be positive"
        )

    if len(points) == 0:
        raise ValueError(
            "points cannot be empty"
        )

    for point in points:
        _validate_point(point)

    results: list[SweepPointResult] = []

    for repetition in range(num_repetitions):
        base = _build_base_realization(
            config,
            repetition,
        )

        for point in points:
            results.append(
                _run_point_with_base(
                    config,
                    point,
                    base,
                    repetition,
                )
            )

    return tuple(results)


def build_one_factor_sweep(
    base_point: SweepPoint,
    parameter_name: str,
    values: Iterable[object],
    *,
    sweep_name: str | None = None,
) -> tuple[SweepPoint, ...]:
    """构建单因素扫描点，除目标参数外其余参数保持不变。"""
    valid_names = {
        field.name
        for field in fields(SweepPoint)
    }

    protected_names = {
        "sweep_name",
        "scenario",
    }

    if (
        parameter_name not in valid_names
        or parameter_name in protected_names
    ):
        raise ValueError(
            f"invalid sweep parameter: {parameter_name}"
        )

    effective_sweep_name = (
        parameter_name
        if sweep_name is None
        else sweep_name
    )

    points: list[SweepPoint] = []

    for value in values:
        point = replace(
            base_point,
            sweep_name=effective_sweep_name,
            scenario=f"{parameter_name}={value}",
            **{
                parameter_name: value,
            },
        )

        _validate_point(point)
        points.append(point)

    if len(points) == 0:
        raise ValueError(
            "values cannot be empty"
        )

    return tuple(points)


def write_sweep_results_csv(
    results: Sequence[SweepPointResult],
    output_path: str | Path,
) -> Path:
    """将原始参数扫描结果写入CSV。"""
    if len(results) == 0:
        raise ValueError(
            "results cannot be empty"
        )

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        asdict(result)
        for result in results
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    return path