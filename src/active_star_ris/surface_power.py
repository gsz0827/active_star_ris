from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .surface import SurfaceCoefficients


@dataclass(frozen=True)
class BidirectionalSurfacePower:
    """STAR-RIS在三个探测方向下的功率结果。"""

    output_power_controller: float
    output_power_transmission_user: float
    output_power_reflection_user: float

    maximum_output_power: float
    output_power_budget: float
    power_violation: float

    amplifier_dc_power: float
    total_surface_power: float


@dataclass(frozen=True)
class RobustAmplitudeProjection:
    """公共有源增益的鲁棒投影结果。"""

    common_active_amplitude: float

    robust_input_upper_controller: float
    robust_input_upper_transmission_user: float
    robust_input_upper_reflection_user: float

    robust_output_upper_controller: float
    robust_output_upper_transmission_user: float
    robust_output_upper_reflection_user: float

    maximum_robust_output_upper: float
    is_feasible_at_unit_gain: bool


@dataclass(frozen=True)
class RobustAmplitudeVectorProjection:
    """逐有源单元增益向量的鲁棒功率投影结果。"""

    requested_amplitudes: np.ndarray
    projected_amplitudes: np.ndarray

    # 从候选增益向量投影到可行域时使用的缩放比例。
    projection_scale: float

    robust_output_upper_controller: float
    robust_output_upper_transmission_user: float
    robust_output_upper_reflection_user: float

    maximum_robust_output_upper: float

    # 单位增益下是否可行。
    is_feasible_at_unit_gain: bool


def _prepare_channel(
    channel: ArrayLike,
    expected_size: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(
        channel,
        dtype=np.complex128,
    ).reshape(-1)

    if values.size != expected_size:
        raise ValueError(
            f"{name} must contain "
            f"{expected_size} entries"
        )

    return values


def _direction_output_power(
    channel: np.ndarray,
    amplitude_gain: np.ndarray,
    active_mask: np.ndarray,
    pilot_power: float,
    ris_internal_noise_variance: float,
) -> float:
    """计算一个探测方向对应的有源单元输出功率。"""

    input_power = (
        pilot_power
        * np.abs(channel[active_mask]) ** 2
        + ris_internal_noise_variance
    )

    return float(
        np.sum(
            amplitude_gain[active_mask] ** 2
            * input_power
        )
    )


def evaluate_bidirectional_surface_power(
    controller_to_ris: ArrayLike,
    transmission_user_to_ris: ArrayLike,
    reflection_user_to_ris: ArrayLike,
    surface: SurfaceCoefficients,
    controller_pilot_power: float,
    transmission_user_pilot_power: float,
    reflection_user_pilot_power: float,
    ris_internal_noise_variance: float,
    output_power_budget: float,
    amplifier_efficiency: float = 0.35,
    controller_static_power: float = 0.10,
    active_element_bias_power: float = 0.01,
) -> BidirectionalSurfacePower:
    """计算双向探测中的STAR-RIS功率。

    三个探测方向分别为：

    1. 控制端发送导频；
    2. 透射侧用户发送导频；
    3. 反射侧用户发送导频。

    功率约束使用三个方向输出功率的最大值。
    """

    if controller_pilot_power < 0.0:
        raise ValueError(
            "controller_pilot_power "
            "must be non-negative"
        )

    if transmission_user_pilot_power < 0.0:
        raise ValueError(
            "transmission_user_pilot_power "
            "must be non-negative"
        )

    if reflection_user_pilot_power < 0.0:
        raise ValueError(
            "reflection_user_pilot_power "
            "must be non-negative"
        )

    if ris_internal_noise_variance < 0.0:
        raise ValueError(
            "ris_internal_noise_variance "
            "must be non-negative"
        )

    if output_power_budget < 0.0:
        raise ValueError(
            "output_power_budget "
            "must be non-negative"
        )

    if not 0.0 < amplifier_efficiency <= 1.0:
        raise ValueError(
            "amplifier_efficiency "
            "must lie in (0, 1]"
        )

    if controller_static_power < 0.0:
        raise ValueError(
            "controller_static_power "
            "must be non-negative"
        )

    if active_element_bias_power < 0.0:
        raise ValueError(
            "active_element_bias_power "
            "must be non-negative"
        )

    n = surface.num_elements

    g = _prepare_channel(
        controller_to_ris,
        n,
        "controller_to_ris",
    )

    h_t = _prepare_channel(
        transmission_user_to_ris,
        n,
        "transmission_user_to_ris",
    )

    h_r = _prepare_channel(
        reflection_user_to_ris,
        n,
        "reflection_user_to_ris",
    )

    active_mask = np.asarray(
        surface.active_mask,
        dtype=bool,
    ).reshape(-1)

    amplitude_gain = np.asarray(
        surface.amplitude_gain,
        dtype=np.float64,
    ).reshape(-1)

    output_controller = (
        _direction_output_power(
            channel=g,
            amplitude_gain=amplitude_gain,
            active_mask=active_mask,
            pilot_power=controller_pilot_power,
            ris_internal_noise_variance=(
                ris_internal_noise_variance
            ),
        )
    )

    output_transmission = (
        _direction_output_power(
            channel=h_t,
            amplitude_gain=amplitude_gain,
            active_mask=active_mask,
            pilot_power=(
                transmission_user_pilot_power
            ),
            ris_internal_noise_variance=(
                ris_internal_noise_variance
            ),
        )
    )

    output_reflection = (
        _direction_output_power(
            channel=h_r,
            amplitude_gain=amplitude_gain,
            active_mask=active_mask,
            pilot_power=(
                reflection_user_pilot_power
            ),
            ris_internal_noise_variance=(
                ris_internal_noise_variance
            ),
        )
    )

    maximum_output = max(
        output_controller,
        output_transmission,
        output_reflection,
    )

    violation = max(
        0.0,
        maximum_output
        - output_power_budget,
    )

    # 计算三个方向的平均输入功率，
    # 用于估算放大器的额外射频功率。
    input_controller = (
        controller_pilot_power
        * np.abs(g[active_mask]) ** 2
        + ris_internal_noise_variance
    )

    input_transmission = (
        transmission_user_pilot_power
        * np.abs(h_t[active_mask]) ** 2
        + ris_internal_noise_variance
    )

    input_reflection = (
        reflection_user_pilot_power
        * np.abs(h_r[active_mask]) ** 2
        + ris_internal_noise_variance
    )

    if np.any(active_mask):
        mean_input_power = (
            input_controller
            + input_transmission
            + input_reflection
        ) / 3.0

        additional_rf_power = float(
            np.sum(
                np.maximum(
                    amplitude_gain[
                        active_mask
                    ] ** 2
                    - 1.0,
                    0.0,
                )
                * mean_input_power
            )
        )
    else:
        additional_rf_power = 0.0

    amplifier_dc_power = (
        additional_rf_power
        / amplifier_efficiency
    )

    number_of_active_elements = int(
        np.sum(active_mask)
    )

    total_surface_power = (
        controller_static_power
        + number_of_active_elements
        * active_element_bias_power
        + amplifier_dc_power
    )

    return BidirectionalSurfacePower(
        output_power_controller=(
            output_controller
        ),
        output_power_transmission_user=(
            output_transmission
        ),
        output_power_reflection_user=(
            output_reflection
        ),
        maximum_output_power=float(
            maximum_output
        ),
        output_power_budget=float(
            output_power_budget
        ),
        power_violation=float(
            violation
        ),
        amplifier_dc_power=float(
            amplifier_dc_power
        ),
        total_surface_power=float(
            total_surface_power
        ),
    )


def _robust_direction_input_upper_per_element(
    estimated_channel: np.ndarray,
    active_mask: np.ndarray,
    pilot_power: float,
    ris_internal_noise_variance: float,
    nmse_linear: float,
    robust_margin_multiplier: float,
) -> np.ndarray:
    """计算每个单元对应的鲁棒输入功率上界。

    对每个有源单元使用：

        |h_true,n|
        <= |h_hat,n| + gamma * sigma_e

    返回长度为N的功率向量。
    """

    estimate = np.asarray(
        estimated_channel,
        dtype=np.complex128,
    ).reshape(-1)

    mask = np.asarray(
        active_mask,
        dtype=bool,
    ).reshape(-1)

    if estimate.size != mask.size:
        raise ValueError(
            "estimated_channel and active_mask "
            "must have equal length"
        )

    result = np.zeros(
        estimate.size,
        dtype=np.float64,
    )

    if not np.any(mask):
        return result

    active_estimate = estimate[mask]

    estimated_channel_power = float(
        np.mean(
            np.abs(active_estimate) ** 2
        )
    )

    # 由：
    # E|h_hat|²
    # = E|h_true|² + E|e|²
    #
    # 且：
    # E|e|² = NMSE × E|h_true|²
    estimated_true_power = (
        estimated_channel_power
        / (1.0 + nmse_linear)
    )

    error_variance = (
        nmse_linear
        * estimated_true_power
    )

    error_standard_deviation = float(
        np.sqrt(
            max(
                error_variance,
                0.0,
            )
        )
    )

    upper_magnitude = (
        np.abs(active_estimate)
        + robust_margin_multiplier
        * error_standard_deviation
    )

    result[mask] = (
        pilot_power
        * upper_magnitude**2
        + ris_internal_noise_variance
    )

    return result


def _robust_direction_input_upper(
    estimated_channel: np.ndarray,
    active_mask: np.ndarray,
    pilot_power: float,
    ris_internal_noise_variance: float,
    nmse_linear: float,
    robust_margin_multiplier: float,
) -> float:
    """构造一个方向的输入功率鲁棒上界。

    估计模型为：

        h_hat = h_true + e

    这里用目标NMSE估计误差标准差，并构造：

        |h_true,n|
        <= |h_hat,n| + gamma * sigma_e

    其中gamma为robust_margin_multiplier。
    """

    if not np.any(active_mask):
        return 0.0

    active_estimate = estimated_channel[
        active_mask
    ]

    estimated_channel_power = float(
        np.mean(
            np.abs(active_estimate) ** 2
        )
    )

    # 因为：
    # E|h_hat|² = E|h_true|² + E|e|²
    # 且 E|e|² = NMSE × E|h_true|²
    estimated_true_power = (
        estimated_channel_power
        / (1.0 + nmse_linear)
    )

    error_variance = (
        nmse_linear
        * estimated_true_power
    )

    error_standard_deviation = float(
        np.sqrt(
            max(
                error_variance,
                0.0,
            )
        )
    )

    upper_magnitude = (
        np.abs(active_estimate)
        + robust_margin_multiplier
        * error_standard_deviation
    )

    return float(
        np.sum(
            pilot_power
            * upper_magnitude**2
            + ris_internal_noise_variance
        )
    )


def project_common_active_amplitude_robust(
    controller_to_ris_estimate: ArrayLike,
    transmission_user_to_ris_estimate: ArrayLike,
    reflection_user_to_ris_estimate: ArrayLike,
    active_mask: ArrayLike,
    controller_pilot_power: float,
    transmission_user_pilot_power: float,
    reflection_user_pilot_power: float,
    ris_internal_noise_variance: float,
    output_power_budget: float,
    maximum_active_amplitude: float,
    nmse_db: float,
    robust_margin_multiplier: float,
) -> RobustAmplitudeProjection:
    """对公共有源增益执行鲁棒功率投影。

    在配置的不确定集合内，使三个探测方向的
    输出功率上界均不超过功率预算。
    """

    if maximum_active_amplitude < 1.0:
        raise ValueError(
            "maximum_active_amplitude "
            "must be at least 1"
        )

    if output_power_budget < 0.0:
        raise ValueError(
            "output_power_budget "
            "must be non-negative"
        )

    if robust_margin_multiplier < 0.0:
        raise ValueError(
            "robust_margin_multiplier "
            "must be non-negative"
        )

    if not np.isfinite(nmse_db):
        raise ValueError(
            "nmse_db must be finite"
        )

    mask = np.asarray(
        active_mask,
        dtype=bool,
    ).reshape(-1)

    n = mask.size

    g_hat = _prepare_channel(
        controller_to_ris_estimate,
        n,
        "controller_to_ris_estimate",
    )

    h_t_hat = _prepare_channel(
        transmission_user_to_ris_estimate,
        n,
        "transmission_user_to_ris_estimate",
    )

    h_r_hat = _prepare_channel(
        reflection_user_to_ris_estimate,
        n,
        "reflection_user_to_ris_estimate",
    )

    if not np.any(mask):
        return RobustAmplitudeProjection(
            common_active_amplitude=1.0,
            robust_input_upper_controller=0.0,
            robust_input_upper_transmission_user=0.0,
            robust_input_upper_reflection_user=0.0,
            robust_output_upper_controller=0.0,
            robust_output_upper_transmission_user=0.0,
            robust_output_upper_reflection_user=0.0,
            maximum_robust_output_upper=0.0,
            is_feasible_at_unit_gain=True,
        )

    nmse_linear = float(
        10.0 ** (nmse_db / 10.0)
    )

    input_upper_controller = (
        _robust_direction_input_upper(
            estimated_channel=g_hat,
            active_mask=mask,
            pilot_power=controller_pilot_power,
            ris_internal_noise_variance=(
                ris_internal_noise_variance
            ),
            nmse_linear=nmse_linear,
            robust_margin_multiplier=(
                robust_margin_multiplier
            ),
        )
    )

    input_upper_transmission = (
        _robust_direction_input_upper(
            estimated_channel=h_t_hat,
            active_mask=mask,
            pilot_power=(
                transmission_user_pilot_power
            ),
            ris_internal_noise_variance=(
                ris_internal_noise_variance
            ),
            nmse_linear=nmse_linear,
            robust_margin_multiplier=(
                robust_margin_multiplier
            ),
        )
    )

    input_upper_reflection = (
        _robust_direction_input_upper(
            estimated_channel=h_r_hat,
            active_mask=mask,
            pilot_power=(
                reflection_user_pilot_power
            ),
            ris_internal_noise_variance=(
                ris_internal_noise_variance
            ),
            nmse_linear=nmse_linear,
            robust_margin_multiplier=(
                robust_margin_multiplier
            ),
        )
    )

    worst_input_upper = max(
        input_upper_controller,
        input_upper_transmission,
        input_upper_reflection,
    )

    feasible_at_unit_gain = (
        worst_input_upper
        <= output_power_budget
        + 1.0e-12
    )

    if worst_input_upper <= 0.0:
        amplitude = (
            maximum_active_amplitude
        )
    else:
        budget_limited_amplitude = float(
            np.sqrt(
                output_power_budget
                / worst_input_upper
            )
        )

        amplitude = float(
            min(
                maximum_active_amplitude,
                budget_limited_amplitude,
            )
        )

        amplitude = max(
            1.0,
            amplitude,
        )

    output_upper_controller = float(
        amplitude**2
        * input_upper_controller
    )

    output_upper_transmission = float(
        amplitude**2
        * input_upper_transmission
    )

    output_upper_reflection = float(
        amplitude**2
        * input_upper_reflection
    )

    maximum_output_upper = max(
        output_upper_controller,
        output_upper_transmission,
        output_upper_reflection,
    )

    return RobustAmplitudeProjection(
        common_active_amplitude=amplitude,
        robust_input_upper_controller=(
            input_upper_controller
        ),
        robust_input_upper_transmission_user=(
            input_upper_transmission
        ),
        robust_input_upper_reflection_user=(
            input_upper_reflection
        ),
        robust_output_upper_controller=(
            output_upper_controller
        ),
        robust_output_upper_transmission_user=(
            output_upper_transmission
        ),
        robust_output_upper_reflection_user=(
            output_upper_reflection
        ),
        maximum_robust_output_upper=(
            maximum_output_upper
        ),
        is_feasible_at_unit_gain=(
            feasible_at_unit_gain
        ),
    )


def project_active_amplitude_vector_robust(
    controller_to_ris_estimate: ArrayLike,
    transmission_user_to_ris_estimate: ArrayLike,
    reflection_user_to_ris_estimate: ArrayLike,
    requested_amplitudes: ArrayLike,
    active_mask: ArrayLike,
    controller_pilot_power: float,
    transmission_user_pilot_power: float,
    reflection_user_pilot_power: float,
    ris_internal_noise_variance: float,
    output_power_budget: float,
    maximum_active_amplitude: float,
    nmse_db: float,
    robust_margin_multiplier: float,
) -> RobustAmplitudeVectorProjection:
    """将逐单元候选增益投影到三方向鲁棒功率可行域。

    候选增益为b_n，投影后增益为a_n。

    投影采用：

        a_n² = 1 + s(b_n² - 1)

    其中：

        0 <= s <= 1

    该映射具有三个优点：

    1. 单位增益下保持a_n=1；
    2. 保留不同单元之间的相对增益关系；
    3. 三方向输出功率关于s为线性函数，
       因而可以直接计算最大可行s。
    """

    if maximum_active_amplitude < 1.0:
        raise ValueError(
            "maximum_active_amplitude "
            "must be at least 1"
        )

    if output_power_budget < 0.0:
        raise ValueError(
            "output_power_budget "
            "must be non-negative"
        )

    if robust_margin_multiplier < 0.0:
        raise ValueError(
            "robust_margin_multiplier "
            "must be non-negative"
        )

    if not np.isfinite(nmse_db):
        raise ValueError(
            "nmse_db must be finite"
        )

    mask = np.asarray(
        active_mask,
        dtype=bool,
    ).reshape(-1)

    n = mask.size

    g_hat = _prepare_channel(
        controller_to_ris_estimate,
        n,
        "controller_to_ris_estimate",
    )

    h_t_hat = _prepare_channel(
        transmission_user_to_ris_estimate,
        n,
        "transmission_user_to_ris_estimate",
    )

    h_r_hat = _prepare_channel(
        reflection_user_to_ris_estimate,
        n,
        "reflection_user_to_ris_estimate",
    )

    requested = np.asarray(
        requested_amplitudes,
        dtype=np.float64,
    ).reshape(-1)

    if requested.size != n:
        raise ValueError(
            "requested_amplitudes and active_mask "
            "must have equal length"
        )

    if not np.all(
        np.isfinite(requested)
    ):
        raise ValueError(
            "requested_amplitudes must be finite"
        )

    # 无源单元的幅度固定为1。
    requested_clipped = np.ones(
        n,
        dtype=np.float64,
    )

    requested_clipped[mask] = np.clip(
        requested[mask],
        1.0,
        maximum_active_amplitude,
    )

    if not np.any(mask):
        return RobustAmplitudeVectorProjection(
            requested_amplitudes=(
                requested_clipped
            ),
            projected_amplitudes=np.ones(
                n,
                dtype=np.float64,
            ),
            projection_scale=1.0,
            robust_output_upper_controller=0.0,
            robust_output_upper_transmission_user=0.0,
            robust_output_upper_reflection_user=0.0,
            maximum_robust_output_upper=0.0,
            is_feasible_at_unit_gain=True,
        )

    nmse_linear = float(
        10.0 ** (nmse_db / 10.0)
    )

    input_upper_controller = (
        _robust_direction_input_upper_per_element(
            estimated_channel=g_hat,
            active_mask=mask,
            pilot_power=(
                controller_pilot_power
            ),
            ris_internal_noise_variance=(
                ris_internal_noise_variance
            ),
            nmse_linear=nmse_linear,
            robust_margin_multiplier=(
                robust_margin_multiplier
            ),
        )
    )

    input_upper_transmission = (
        _robust_direction_input_upper_per_element(
            estimated_channel=h_t_hat,
            active_mask=mask,
            pilot_power=(
                transmission_user_pilot_power
            ),
            ris_internal_noise_variance=(
                ris_internal_noise_variance
            ),
            nmse_linear=nmse_linear,
            robust_margin_multiplier=(
                robust_margin_multiplier
            ),
        )
    )

    input_upper_reflection = (
        _robust_direction_input_upper_per_element(
            estimated_channel=h_r_hat,
            active_mask=mask,
            pilot_power=(
                reflection_user_pilot_power
            ),
            ris_internal_noise_variance=(
                ris_internal_noise_variance
            ),
            nmse_linear=nmse_linear,
            robust_margin_multiplier=(
                robust_margin_multiplier
            ),
        )
    )

    # 单位增益情况下的三个方向输出功率。
    base_controller = float(
        np.sum(
            input_upper_controller[mask]
        )
    )

    base_transmission = float(
        np.sum(
            input_upper_transmission[mask]
        )
    )

    base_reflection = float(
        np.sum(
            input_upper_reflection[mask]
        )
    )

    maximum_base_output = max(
        base_controller,
        base_transmission,
        base_reflection,
    )

    feasible_at_unit_gain = (
        maximum_base_output
        <= output_power_budget
        + 1.0e-12
    )

    # 如果单位增益都不可行，调用方需要减少有源单元数量。
    if not feasible_at_unit_gain:
        unit_amplitudes = np.ones(
            n,
            dtype=np.float64,
        )

        return RobustAmplitudeVectorProjection(
            requested_amplitudes=(
                requested_clipped
            ),
            projected_amplitudes=(
                unit_amplitudes
            ),
            projection_scale=0.0,
            robust_output_upper_controller=(
                base_controller
            ),
            robust_output_upper_transmission_user=(
                base_transmission
            ),
            robust_output_upper_reflection_user=(
                base_reflection
            ),
            maximum_robust_output_upper=(
                maximum_base_output
            ),
            is_feasible_at_unit_gain=False,
        )

    requested_excess_squared = (
        requested_clipped**2
        - 1.0
    )

    # 三个方向中，由候选增益额外引入的功率。
    extra_controller = float(
        np.sum(
            requested_excess_squared[mask]
            * input_upper_controller[mask]
        )
    )

    extra_transmission = float(
        np.sum(
            requested_excess_squared[mask]
            * input_upper_transmission[mask]
        )
    )

    extra_reflection = float(
        np.sum(
            requested_excess_squared[mask]
            * input_upper_reflection[mask]
        )
    )

    scale_candidates = [1.0]

    for base_power, extra_power in (
        (
            base_controller,
            extra_controller,
        ),
        (
            base_transmission,
            extra_transmission,
        ),
        (
            base_reflection,
            extra_reflection,
        ),
    ):
        if extra_power > 1.0e-15:
            scale_candidates.append(
                (
                    output_power_budget
                    - base_power
                )
                / extra_power
            )

    projection_scale = float(
        np.clip(
            min(scale_candidates),
            0.0,
            1.0,
        )
    )

    projected_squared = (
        1.0
        + projection_scale
        * requested_excess_squared
    )

    projected = np.sqrt(
        np.maximum(
            projected_squared,
            1.0,
        )
    )

    projected[~mask] = 1.0

    output_controller = float(
        np.sum(
            projected[mask] ** 2
            * input_upper_controller[mask]
        )
    )

    output_transmission = float(
        np.sum(
            projected[mask] ** 2
            * input_upper_transmission[mask]
        )
    )

    output_reflection = float(
        np.sum(
            projected[mask] ** 2
            * input_upper_reflection[mask]
        )
    )

    maximum_output = max(
        output_controller,
        output_transmission,
        output_reflection,
    )

    return RobustAmplitudeVectorProjection(
        requested_amplitudes=(
            requested_clipped
        ),
        projected_amplitudes=(
            projected
        ),
        projection_scale=(
            projection_scale
        ),
        robust_output_upper_controller=(
            output_controller
        ),
        robust_output_upper_transmission_user=(
            output_transmission
        ),
        robust_output_upper_reflection_user=(
            output_reflection
        ),
        maximum_robust_output_upper=(
            maximum_output
        ),
        is_feasible_at_unit_gain=True,
    )