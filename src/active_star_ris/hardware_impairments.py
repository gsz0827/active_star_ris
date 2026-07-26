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
    # 前向和反向共同存在的静态增益误差
    static_gain_error_std_db: float = 0.0

    # 前向和反向各自独立、在一个episode内固定的方向相关增益偏差
    directional_gain_error_std_db: float = 0.0

    # 前向和反向共同存在、在一个episode内固定的静态相位偏差
    static_phase_error_std_rad: float = 0.0

    # 前向和反向各自独立、在一个episode内固定的方向相关相位偏差
    directional_phase_error_std_rad: float = 0.0

    # 有源单元实际增益比例的上下界
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
) -> HardwareMismatchRealization:
    _validate_parameters(parameters)

    generator = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    num_elements = (
        ideal_coefficients.amplitudes.size
    )

    mask = _validate_active_mask(
        active_mask,
        num_elements,
    )

    static_gain_error_db = generator.normal(
        loc=0.0,
        scale=parameters.static_gain_error_std_db,
        size=num_elements,
    )

    forward_gain_error_db = generator.normal(
        loc=0.0,
        scale=(
            parameters
            .directional_gain_error_std_db
        ),
        size=num_elements,
    )

    reverse_gain_error_db = generator.normal(
        loc=0.0,
        scale=(
            parameters
            .directional_gain_error_std_db
        ),
        size=num_elements,
    )

    static_gain_scale = _gain_scale_from_db(
        static_gain_error_db,
        parameters,
    )

    forward_directional_gain_scale = (
        _gain_scale_from_db(
            forward_gain_error_db,
            parameters,
        )
    )

    reverse_directional_gain_scale = (
        _gain_scale_from_db(
            reverse_gain_error_db,
            parameters,
        )
    )

    forward_amplitudes = (
        ideal_coefficients.amplitudes
        * static_gain_scale
        * forward_directional_gain_scale
    )

    reverse_amplitudes = (
        ideal_coefficients.amplitudes
        * static_gain_scale
        * reverse_directional_gain_scale
    )

    # 无源单元不能因误差产生净放大。
    passive_mask = ~mask

    forward_amplitudes[passive_mask] = np.minimum(
        forward_amplitudes[passive_mask],
        ideal_coefficients.amplitudes[
            passive_mask
        ],
    )

    reverse_amplitudes[passive_mask] = np.minimum(
        reverse_amplitudes[passive_mask],
        ideal_coefficients.amplitudes[
            passive_mask
        ],
    )

    static_phase_error_t = generator.normal(
        loc=0.0,
        scale=(
            parameters.static_phase_error_std_rad
        ),
        size=num_elements,
    )

    static_phase_error_r = generator.normal(
        loc=0.0,
        scale=(
            parameters.static_phase_error_std_rad
        ),
        size=num_elements,
    )

    forward_phase_jitter_t = generator.normal(
        loc=0.0,
        scale=(
            parameters
            .directional_phase_error_std_rad
        ),
        size=num_elements,
    )

    reverse_phase_jitter_t = generator.normal(
        loc=0.0,
        scale=(
            parameters
            .directional_phase_error_std_rad
        ),
        size=num_elements,
    )

    forward_phase_jitter_r = generator.normal(
        loc=0.0,
        scale=(
            parameters
            .directional_phase_error_std_rad
        ),
        size=num_elements,
    )

    reverse_phase_jitter_r = generator.normal(
        loc=0.0,
        scale=(
            parameters
            .directional_phase_error_std_rad
        ),
        size=num_elements,
    )

    ideal_phase_t = np.angle(
        ideal_coefficients.transmission
    )

    ideal_phase_r = np.angle(
        ideal_coefficients.reflection
    )

    forward_coefficients = build_star_coefficients(
        amplitudes=forward_amplitudes,
        beta_transmission=(
            ideal_coefficients
            .beta_transmission
        ),
        beta_reflection=(
            ideal_coefficients
            .beta_reflection
        ),
        phase_transmission=(
            ideal_phase_t
            + static_phase_error_t
            + forward_phase_jitter_t
        ),
        phase_reflection=(
            ideal_phase_r
            + static_phase_error_r
            + forward_phase_jitter_r
        ),
    )

    reverse_coefficients = build_star_coefficients(
        amplitudes=reverse_amplitudes,
        beta_transmission=(
            ideal_coefficients
            .beta_transmission
        ),
        beta_reflection=(
            ideal_coefficients
            .beta_reflection
        ),
        phase_transmission=(
            ideal_phase_t
            + static_phase_error_t
            + reverse_phase_jitter_t
        ),
        phase_reflection=(
            ideal_phase_r
            + static_phase_error_r
            + reverse_phase_jitter_r
        ),
    )

    return HardwareMismatchRealization(
        forward_coefficients=(
            forward_coefficients
        ),
        reverse_coefficients=(
            reverse_coefficients
        ),
        static_gain_scale=(
            static_gain_scale
        ),
        forward_directional_gain_scale=(
            forward_directional_gain_scale
        ),
        reverse_directional_gain_scale=(
            reverse_directional_gain_scale
        ),
        static_phase_error_transmission=(
            static_phase_error_t
        ),
        static_phase_error_reflection=(
            static_phase_error_r
        ),
        forward_phase_jitter_transmission=(
            forward_phase_jitter_t
        ),
        reverse_phase_jitter_transmission=(
            reverse_phase_jitter_t
        ),
        forward_phase_jitter_reflection=(
            forward_phase_jitter_r
        ),
        reverse_phase_jitter_reflection=(
            reverse_phase_jitter_r
        ),
    )