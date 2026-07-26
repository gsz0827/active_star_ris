from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from active_star_ris.star_key_system import (
    StarCoefficientPair,
    build_star_coefficients,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class HardwareMismatchParameters:
    """STAR-RIS硬件失配参数。

    静态误差和方向相关误差在一个episode内保持固定；
    fast_phase_jitter_std_rad描述每次探测重新产生的快速抖动。
    """

    # 前向与反向共同存在的静态增益误差，单位dB。
    static_gain_error_std_db: float = 0.0

    # 前向与反向分别存在的固定增益非互易偏差，单位dB。
    directional_gain_error_std_db: float = 0.0

    # 透射和反射支路各自的静态相位误差，单位rad。
    static_phase_error_std_rad: float = 0.0

    # 前向与反向分别存在的固定相位非互易偏差，单位rad。
    directional_phase_error_std_rad: float = 0.0

    # 每次探测重新采样的快速相位抖动，单位rad。
    fast_phase_jitter_std_rad: float = 0.0

    # 有源单元透射支路的一阶幅相耦合系数，单位rad/dB。
    transmission_amplitude_phase_coupling_rad_per_db: float = 0.0

    # 有源单元反射支路的一阶幅相耦合系数，单位rad/dB。
    reflection_amplitude_phase_coupling_rad_per_db: float = 0.0

    # 有源单元实际增益比例的上下界。
    gain_scale_min: float = 0.25
    gain_scale_max: float = 4.0


@dataclass(frozen=True)
class HardwareMismatchRealization:
    forward_coefficients: StarCoefficientPair
    reverse_coefficients: StarCoefficientPair

    static_gain_scale: FloatArray
    forward_directional_gain_scale: FloatArray
    reverse_directional_gain_scale: FloatArray

    static_phase_error_transmission: FloatArray
    static_phase_error_reflection: FloatArray

    forward_phase_jitter_transmission: FloatArray
    reverse_phase_jitter_transmission: FloatArray

    forward_phase_jitter_reflection: FloatArray
    reverse_phase_jitter_reflection: FloatArray

    # 每次调用重新产生的快速相位抖动。
    forward_fast_phase_jitter_transmission: FloatArray
    reverse_fast_phase_jitter_transmission: FloatArray

    forward_fast_phase_jitter_reflection: FloatArray
    reverse_fast_phase_jitter_reflection: FloatArray

    # 由有源增益偏差引起的幅相耦合误差。
    forward_amplitude_phase_coupling_transmission: FloatArray
    reverse_amplitude_phase_coupling_transmission: FloatArray

    forward_amplitude_phase_coupling_reflection: FloatArray
    reverse_amplitude_phase_coupling_reflection: FloatArray


def _validate_parameters(
    parameters: HardwareMismatchParameters,
) -> None:
    nonnegative_parameters = {
        "static_gain_error_std_db": (
            parameters.static_gain_error_std_db
        ),
        "directional_gain_error_std_db": (
            parameters.directional_gain_error_std_db
        ),
        "static_phase_error_std_rad": (
            parameters.static_phase_error_std_rad
        ),
        "directional_phase_error_std_rad": (
            parameters.directional_phase_error_std_rad
        ),
        "fast_phase_jitter_std_rad": (
            parameters.fast_phase_jitter_std_rad
        ),
        "transmission_amplitude_phase_coupling_rad_per_db": (
            parameters
            .transmission_amplitude_phase_coupling_rad_per_db
        ),
        "reflection_amplitude_phase_coupling_rad_per_db": (
            parameters
            .reflection_amplitude_phase_coupling_rad_per_db
        ),
    }

    for name, value in nonnegative_parameters.items():
        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative"
            )

    if parameters.gain_scale_min <= 0.0:
        raise ValueError(
            "gain_scale_min must be positive"
        )

    if (
        parameters.gain_scale_max
        < parameters.gain_scale_min
    ):
        raise ValueError(
            "gain_scale_max must be greater than "
            "or equal to gain_scale_min"
        )


def _validate_active_mask(
    active_mask: ArrayLike,
    num_elements: int,
) -> BoolArray:
    mask = np.asarray(
        active_mask,
        dtype=bool,
    ).reshape(-1)

    if mask.size != num_elements:
        raise ValueError(
            "active_mask length must equal the "
            "number of STAR-RIS elements"
        )

    return mask


def _gain_scale_from_db(
    gain_error_db: FloatArray,
    parameters: HardwareMismatchParameters,
) -> FloatArray:
    scale = np.power(
        10.0,
        gain_error_db / 20.0,
    )

    return np.asarray(
        np.clip(
            scale,
            parameters.gain_scale_min,
            parameters.gain_scale_max,
        ),
        dtype=np.float64,
    )


def apply_hardware_mismatch(
    ideal_coefficients: StarCoefficientPair,
    active_mask: ArrayLike,
    parameters: HardwareMismatchParameters,
    *,
    rng: np.random.Generator | None = None,
    dynamic_rng: np.random.Generator | None = None,
) -> HardwareMismatchRealization:
    """把硬件失配施加到理想STAR-RIS系数上。

    参数
    ----------
    rng:
        用于生成episode内固定的制造偏差和方向非互易偏差。

    dynamic_rng:
        用于生成每次探测变化的快速相位抖动。若未提供，
        则与rng共用同一个生成器，以保持向后兼容。
    """
    _validate_parameters(parameters)

    generator = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    dynamic_generator = (
        generator
        if dynamic_rng is None
        else dynamic_rng
    )

    num_elements = ideal_coefficients.amplitudes.size

    mask = _validate_active_mask(
        active_mask,
        num_elements,
    )

    # ============================================================
    # 1. episode内固定的增益误差
    # ============================================================
    static_gain_error_db = generator.normal(
        loc=0.0,
        scale=parameters.static_gain_error_std_db,
        size=num_elements,
    )

    forward_gain_error_db = generator.normal(
        loc=0.0,
        scale=parameters.directional_gain_error_std_db,
        size=num_elements,
    )

    reverse_gain_error_db = generator.normal(
        loc=0.0,
        scale=parameters.directional_gain_error_std_db,
        size=num_elements,
    )

    static_gain_scale = _gain_scale_from_db(
        static_gain_error_db,
        parameters,
    )

    forward_directional_gain_scale = _gain_scale_from_db(
        forward_gain_error_db,
        parameters,
    )

    reverse_directional_gain_scale = _gain_scale_from_db(
        reverse_gain_error_db,
        parameters,
    )

    forward_total_gain_scale = (
        static_gain_scale
        * forward_directional_gain_scale
    )

    reverse_total_gain_scale = (
        static_gain_scale
        * reverse_directional_gain_scale
    )

    forward_amplitudes = (
        ideal_coefficients.amplitudes
        * forward_total_gain_scale
    )

    reverse_amplitudes = (
        ideal_coefficients.amplitudes
        * reverse_total_gain_scale
    )

    # 无源单元可以出现插入损耗，但不能因为误差产生净放大。
    passive_mask = ~mask

    forward_amplitudes[passive_mask] = np.minimum(
        forward_amplitudes[passive_mask],
        ideal_coefficients.amplitudes[passive_mask],
    )

    reverse_amplitudes[passive_mask] = np.minimum(
        reverse_amplitudes[passive_mask],
        ideal_coefficients.amplitudes[passive_mask],
    )

    # ============================================================
    # 2. episode内固定的相位误差
    # ============================================================
    static_phase_error_t = generator.normal(
        loc=0.0,
        scale=parameters.static_phase_error_std_rad,
        size=num_elements,
    )

    static_phase_error_r = generator.normal(
        loc=0.0,
        scale=parameters.static_phase_error_std_rad,
        size=num_elements,
    )

    forward_directional_phase_t = generator.normal(
        loc=0.0,
        scale=parameters.directional_phase_error_std_rad,
        size=num_elements,
    )

    reverse_directional_phase_t = generator.normal(
        loc=0.0,
        scale=parameters.directional_phase_error_std_rad,
        size=num_elements,
    )

    forward_directional_phase_r = generator.normal(
        loc=0.0,
        scale=parameters.directional_phase_error_std_rad,
        size=num_elements,
    )

    reverse_directional_phase_r = generator.normal(
        loc=0.0,
        scale=parameters.directional_phase_error_std_rad,
        size=num_elements,
    )

    # ============================================================
    # 3. 每次探测变化的快速相位抖动
    # ============================================================
    fast_std = parameters.fast_phase_jitter_std_rad

    forward_fast_phase_t = dynamic_generator.normal(
        loc=0.0,
        scale=fast_std,
        size=num_elements,
    )

    reverse_fast_phase_t = dynamic_generator.normal(
        loc=0.0,
        scale=fast_std,
        size=num_elements,
    )

    forward_fast_phase_r = dynamic_generator.normal(
        loc=0.0,
        scale=fast_std,
        size=num_elements,
    )

    reverse_fast_phase_r = dynamic_generator.normal(
        loc=0.0,
        scale=fast_std,
        size=num_elements,
    )

    # ============================================================
    # 4. 一阶幅相耦合
    # ============================================================
    minimum_scale = 1.0e-12

    forward_actual_gain_error_db = (
        20.0
        * np.log10(
            np.maximum(
                forward_total_gain_scale,
                minimum_scale,
            )
        )
    )

    reverse_actual_gain_error_db = (
        20.0
        * np.log10(
            np.maximum(
                reverse_total_gain_scale,
                minimum_scale,
            )
        )
    )

    # 幅相耦合只作用于有源单元。
    forward_coupling_t = np.where(
        mask,
        parameters
        .transmission_amplitude_phase_coupling_rad_per_db
        * forward_actual_gain_error_db,
        0.0,
    )

    reverse_coupling_t = np.where(
        mask,
        parameters
        .transmission_amplitude_phase_coupling_rad_per_db
        * reverse_actual_gain_error_db,
        0.0,
    )

    forward_coupling_r = np.where(
        mask,
        parameters
        .reflection_amplitude_phase_coupling_rad_per_db
        * forward_actual_gain_error_db,
        0.0,
    )

    reverse_coupling_r = np.where(
        mask,
        parameters
        .reflection_amplitude_phase_coupling_rad_per_db
        * reverse_actual_gain_error_db,
        0.0,
    )

    # ============================================================
    # 5. 构造实际透射和反射系数
    # ============================================================
    ideal_phase_t = np.angle(
        ideal_coefficients.transmission
    )

    ideal_phase_r = np.angle(
        ideal_coefficients.reflection
    )

    forward_coefficients = build_star_coefficients(
        amplitudes=forward_amplitudes,
        beta_transmission=(
            ideal_coefficients.beta_transmission
        ),
        beta_reflection=(
            ideal_coefficients.beta_reflection
        ),
        phase_transmission=(
            ideal_phase_t
            + static_phase_error_t
            + forward_directional_phase_t
            + forward_coupling_t
            + forward_fast_phase_t
        ),
        phase_reflection=(
            ideal_phase_r
            + static_phase_error_r
            + forward_directional_phase_r
            + forward_coupling_r
            + forward_fast_phase_r
        ),
    )

    reverse_coefficients = build_star_coefficients(
        amplitudes=reverse_amplitudes,
        beta_transmission=(
            ideal_coefficients.beta_transmission
        ),
        beta_reflection=(
            ideal_coefficients.beta_reflection
        ),
        phase_transmission=(
            ideal_phase_t
            + static_phase_error_t
            + reverse_directional_phase_t
            + reverse_coupling_t
            + reverse_fast_phase_t
        ),
        phase_reflection=(
            ideal_phase_r
            + static_phase_error_r
            + reverse_directional_phase_r
            + reverse_coupling_r
            + reverse_fast_phase_r
        ),
    )

    return HardwareMismatchRealization(
        forward_coefficients=forward_coefficients,
        reverse_coefficients=reverse_coefficients,
        static_gain_scale=np.asarray(
            static_gain_scale,
            dtype=np.float64,
        ),
        forward_directional_gain_scale=np.asarray(
            forward_directional_gain_scale,
            dtype=np.float64,
        ),
        reverse_directional_gain_scale=np.asarray(
            reverse_directional_gain_scale,
            dtype=np.float64,
        ),
        static_phase_error_transmission=np.asarray(
            static_phase_error_t,
            dtype=np.float64,
        ),
        static_phase_error_reflection=np.asarray(
            static_phase_error_r,
            dtype=np.float64,
        ),
        forward_phase_jitter_transmission=np.asarray(
            forward_directional_phase_t,
            dtype=np.float64,
        ),
        reverse_phase_jitter_transmission=np.asarray(
            reverse_directional_phase_t,
            dtype=np.float64,
        ),
        forward_phase_jitter_reflection=np.asarray(
            forward_directional_phase_r,
            dtype=np.float64,
        ),
        reverse_phase_jitter_reflection=np.asarray(
            reverse_directional_phase_r,
            dtype=np.float64,
        ),
        forward_fast_phase_jitter_transmission=np.asarray(
            forward_fast_phase_t,
            dtype=np.float64,
        ),
        reverse_fast_phase_jitter_transmission=np.asarray(
            reverse_fast_phase_t,
            dtype=np.float64,
        ),
        forward_fast_phase_jitter_reflection=np.asarray(
            forward_fast_phase_r,
            dtype=np.float64,
        ),
        reverse_fast_phase_jitter_reflection=np.asarray(
            reverse_fast_phase_r,
            dtype=np.float64,
        ),
        forward_amplitude_phase_coupling_transmission=np.asarray(
            forward_coupling_t,
            dtype=np.float64,
        ),
        reverse_amplitude_phase_coupling_transmission=np.asarray(
            reverse_coupling_t,
            dtype=np.float64,
        ),
        forward_amplitude_phase_coupling_reflection=np.asarray(
            forward_coupling_r,
            dtype=np.float64,
        ),
        reverse_amplitude_phase_coupling_reflection=np.asarray(
            reverse_coupling_r,
            dtype=np.float64,
        ),
    )