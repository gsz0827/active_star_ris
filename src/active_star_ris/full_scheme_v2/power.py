from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import HardwareConfig, PowerConfig, ProbingConfig
from .hardware import quantize_gain_downward
from .models import IdealSurfaceCommand, PowerResult


def _normalized_time_fractions(config: PowerConfig) -> np.ndarray:
    values = np.asarray(
        [
            config.controller_time_fraction,
            config.transmission_time_fraction,
            config.reflection_time_fraction,
        ],
        dtype=np.float64,
    )
    if np.any(values < 0.0) or float(np.sum(values)) <= 0.0:
        raise ValueError("invalid time fractions")
    return values / np.sum(values)


def conservative_input_powers(
    controller_to_ris_estimate: np.ndarray,
    transmission_to_ris_estimate: np.ndarray,
    reflection_to_ris_estimate: np.ndarray,
    *,
    nmse_db: float,
    probing: ProbingConfig,
    power: PowerConfig,
    amplifier_noise_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if amplifier_noise_scale < 0.0:
        raise ValueError("amplifier_noise_scale cannot be negative")
    nmse_linear = 10.0 ** (nmse_db / 10.0)

    def robust_power(
        values: np.ndarray,
        pilot_power: float,
    ) -> np.ndarray:
        magnitude = np.abs(
            np.asarray(values, dtype=np.complex128)
        )

        # 与 channels.estimate_channel() 中的 NMSE 定义保持一致：
        # sigma_e^2 = P_h * NMSE
        link_power = max(
            float(np.mean(magnitude**2)),
            1.0e-12,
        )

        error_std = np.sqrt(
            max(nmse_linear, 0.0) * link_power
        )

        upper_magnitude = (
            magnitude
            + power.csi_power_margin_std * error_std
        )

    return (
        pilot_power * upper_magnitude**2
        + probing.input_referred_amplifier_noise_variance
        * amplifier_noise_scale
    )

    return (
        robust_power(
            controller_to_ris_estimate,
            probing.pilot_power_controller,
        ),
        robust_power(
            transmission_to_ris_estimate,
            probing.pilot_power_transmission_user,
        ),
        robust_power(
            reflection_to_ris_estimate,
            probing.pilot_power_reflection_user,
        ),
    )


def evaluate_power(
    command: IdealSurfaceCommand,
    input_power_controller: np.ndarray,
    input_power_transmission: np.ndarray,
    input_power_reflection: np.ndarray,
    *,
    power_config: PowerConfig,
    hardware_config: HardwareConfig,
    rf_budget: float | None = None,
    dc_budget: float | None = None,
) -> PowerResult:
    active = np.asarray(command.active_mask, dtype=bool)
    gain = np.asarray(command.gain, dtype=np.float64)
    if gain.size != active.size:
        raise ValueError("gain and active_mask size mismatch")

    for values in (
        input_power_controller,
        input_power_transmission,
        input_power_reflection,
    ):
        if np.asarray(values).reshape(-1).size != gain.size:
            raise ValueError("input power size mismatch")

    input_c = np.asarray(input_power_controller, dtype=np.float64).reshape(-1)
    input_t = np.asarray(input_power_transmission, dtype=np.float64).reshape(-1)
    input_r = np.asarray(input_power_reflection, dtype=np.float64).reshape(-1)

    active_gain_sq = np.where(active, gain**2, 0.0)
    rf_c = float(np.sum(active_gain_sq * input_c))
    rf_t = float(np.sum(active_gain_sq * input_t))
    rf_r = float(np.sum(active_gain_sq * input_r))

    maximum_rf = max(rf_c, rf_t, rf_r)
    fractions = _normalized_time_fractions(power_config)
    average_rf = float(fractions @ np.asarray([rf_c, rf_t, rf_r]))

    extra_gain_sq = np.where(active, np.maximum(gain**2 - 1.0, 0.0), 0.0)
    additional = np.asarray(
        [
            np.sum(extra_gain_sq * input_c),
            np.sum(extra_gain_sq * input_t),
            np.sum(extra_gain_sq * input_r),
        ],
        dtype=np.float64,
    )
    additional_average = float(fractions @ additional)
    amplifier_dc = additional_average / max(
        power_config.amplifier_efficiency,
        1.0e-12,
    )

    num_active = int(np.count_nonzero(active))
    num_passive = int(active.size - num_active)
    total_dc = float(
        amplifier_dc
        + power_config.controller_static_power
        + power_config.switching_network_static_power
        + num_passive * power_config.passive_element_control_power
        + num_active * power_config.active_element_control_power
        + num_active * power_config.active_element_bias_power
    )

    effective_rf_budget = (
        power_config.maximum_rf_output_power
        if rf_budget is None
        else float(rf_budget)
    )
    effective_dc_budget = (
        power_config.maximum_total_dc_power
        if dc_budget is None
        else float(dc_budget)
    )

    worst_input = np.maximum.reduce((input_c, input_t, input_r))
    element_output = gain**2 * worst_input
    saturation_excess = np.where(
        active,
        np.maximum(
            element_output - hardware_config.per_active_element_saturation_power,
            0.0,
        ),
        0.0,
    )
    saturation_violation = float(np.max(saturation_excess, initial=0.0))

    rf_violation = max(0.0, maximum_rf - effective_rf_budget)
    dc_violation = max(0.0, total_dc - effective_dc_budget)
    tolerance = power_config.projection_tolerance

    return PowerResult(
        rf_output_controller=rf_c,
        rf_output_transmission=rf_t,
        rf_output_reflection=rf_r,
        maximum_rf_output=maximum_rf,
        average_rf_output=average_rf,
        additional_rf_power_average=additional_average,
        amplifier_dc_power=float(amplifier_dc),
        total_surface_dc_power=total_dc,
        rf_violation=float(rf_violation),
        dc_violation=float(dc_violation),
        per_element_saturation_violation=saturation_violation,
        rf_feasible=bool(rf_violation <= tolerance),
        dc_feasible=bool(dc_violation <= tolerance),
        saturation_feasible=bool(saturation_violation <= tolerance),
        fully_feasible=bool(
            rf_violation <= tolerance
            and dc_violation <= tolerance
            and saturation_violation <= tolerance
        ),
    )


def _scale_active_gain(
    gain: np.ndarray,
    active_mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    result = np.ones_like(gain, dtype=np.float64)
    result[active_mask] = 1.0 + scale * (gain[active_mask] - 1.0)
    return result


def project_command_to_power_constraints(
    command: IdealSurfaceCommand,
    input_power_controller: np.ndarray,
    input_power_transmission: np.ndarray,
    input_power_reflection: np.ndarray,
    *,
    power_config: PowerConfig,
    hardware_config: HardwareConfig,
    rf_budget: float | None = None,
    dc_budget: float | None = None,
) -> tuple[IdealSurfaceCommand, PowerResult]:
    """在最坏硬件增益失配下，对实际下发的增益命令进行鲁棒功率投影。

    注意：
    1. requested_gain 是控制器真正想下发的增益；
    2. worst_case_gain 只用于检查最坏情况下是否满足功率约束；
    3. 最终不能把 worst_case_gain 直接下发给 STAR-RIS。
    """

    active = np.asarray(command.active_mask, dtype=bool).reshape(-1)
    requested_gain = np.asarray(command.gain, dtype=np.float64).reshape(-1)

    if requested_gain.size != active.size:
        raise ValueError("gain and active_mask size mismatch")

    # 真正准备下发给硬件的增益命令。
    requested_gain = np.where(
        active,
        np.clip(
            requested_gain,
            1.0,
            hardware_config.maximum_active_gain,
        ),
        1.0,
    )

    # 正向硬件增益失配的保守裕量。
    mismatch_margin = 10.0 ** (
        power_config.hardware_gain_margin_db / 20.0
    )

    def make_candidate(gain: np.ndarray) -> IdealSurfaceCommand:
        return replace(
            command,
            gain=np.asarray(gain, dtype=np.float64),
        )

    def worst_case_gain(command_gain: np.ndarray) -> np.ndarray:
        """把控制增益映射为功率检查使用的最坏实际增益。"""
        gain = np.ones_like(command_gain, dtype=np.float64)

        gain[active] = np.minimum(
            command_gain[active] * mismatch_margin,
            hardware_config.maximum_active_gain,
        )

        return gain

    def robust_result_for(command_gain: np.ndarray) -> PowerResult:
        """按最坏实际增益检查 RF/DC/单元饱和约束。"""
        conservative_gain = worst_case_gain(command_gain)

        return evaluate_power(
            make_candidate(conservative_gain),
            input_power_controller,
            input_power_transmission,
            input_power_reflection,
            power_config=power_config,
            hardware_config=hardware_config,
            rf_budget=rf_budget,
            dc_budget=dc_budget,
        )

    passive_gain = np.ones_like(requested_gain, dtype=np.float64)

    # 请求的控制增益在最坏硬件失配下仍然可行。
    if robust_result_for(requested_gain).fully_feasible:
        projected_gain = requested_gain.copy()

    # 连单位增益在当前预算下都不可行，则只能退化到单位增益。
    elif not robust_result_for(passive_gain).fully_feasible:
        projected_gain = passive_gain

    else:
        # 在 gain=1 与 requested_gain 之间做二分投影。
        lower = 0.0
        upper = 1.0

        for _ in range(power_config.projection_iterations):
            middle = 0.5 * (lower + upper)

            candidate_gain = _scale_active_gain(
                requested_gain,
                active,
                middle,
            )

            if robust_result_for(candidate_gain).fully_feasible:
                lower = middle
            else:
                upper = middle

        projected_gain = _scale_active_gain(
            requested_gain,
            active,
            lower,
        )

    # 向下量化，避免量化后重新违反功率约束。
    projected_gain = quantize_gain_downward(
        projected_gain,
        hardware_config.maximum_active_gain,
        hardware_config.gain_quantization_bits,
    )

    projected_gain[~active] = 1.0

    projected = make_candidate(projected_gain)

    # 返回的是“这个控制命令在最坏硬件失配下”的功率检查结果。
    conservative_result = robust_result_for(projected_gain)

    return projected, conservative_result
