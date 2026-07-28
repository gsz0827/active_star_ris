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
        raise ValueError(
            "amplifier_noise_scale cannot be negative"
        )

    nmse_linear = 10.0 ** (nmse_db / 10.0)

    def robust_power(
        values: np.ndarray,
        pilot_power: float,
    ) -> np.ndarray:
        magnitude = np.abs(
            np.asarray(values, dtype=np.complex128)
        )

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
    element_utility: np.ndarray | None = None,
    rf_budget: float | None = None,
    dc_budget: float | None = None,
) -> tuple[IdealSurfaceCommand, PowerResult]:
    """将增益命令投影到鲁棒功率约束可行域。

    若有源单元在单位增益下仍不可行，则将本次命令中的
    有源单元切换到无源 bypass。
    """

    original_active = np.asarray(
        command.active_mask,
        dtype=bool,
    ).reshape(-1)

    requested_gain = np.asarray(
        command.gain,
        dtype=np.float64,
    ).reshape(-1)

    if requested_gain.size != original_active.size:
        raise ValueError(
            "gain and active_mask size mismatch"
        )

    requested_gain = np.where(
        original_active,
        np.clip(
            requested_gain,
            1.0,
            hardware_config.maximum_active_gain,
        ),
        1.0,
    )

    mismatch_margin = 10.0 ** (
        power_config.hardware_gain_margin_db
        / 20.0
    )

    def make_candidate(
        gain: np.ndarray,
        active_mask: np.ndarray,
    ) -> IdealSurfaceCommand:
        candidate_mask = np.asarray(
            active_mask,
            dtype=bool,
        ).reshape(-1)

        candidate_gain = np.asarray(
            gain,
            dtype=np.float64,
        ).reshape(-1)

        candidate_gain = np.where(
            candidate_mask,
            candidate_gain,
            1.0,
        )

        return replace(
            command,
            gain=candidate_gain,
            active_mask=candidate_mask,
        )

    def worst_case_gain(
        command_gain: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        candidate_mask = np.asarray(
            active_mask,
            dtype=bool,
        ).reshape(-1)

        gain = np.ones_like(
            command_gain,
            dtype=np.float64,
        )

        gain[candidate_mask] = np.minimum(
            command_gain[candidate_mask]
            * mismatch_margin,
            hardware_config.maximum_active_gain,
        )

        return gain

    def robust_result_for(
        command_gain: np.ndarray,
        active_mask: np.ndarray,
    ) -> PowerResult:
        conservative_gain = worst_case_gain(
            command_gain,
            active_mask,
        )

        conservative_command = make_candidate(
            conservative_gain,
            active_mask,
        )

        return evaluate_power(
            conservative_command,
            input_power_controller,
            input_power_transmission,
            input_power_reflection,
            power_config=power_config,
            hardware_config=hardware_config,
            rf_budget=rf_budget,
            dc_budget=dc_budget,
        )

    utility = np.ones_like(
        requested_gain,
        dtype=np.float64,
    )

    if element_utility is not None:
        utility_input = np.asarray(
            element_utility,
            dtype=np.float64,
        ).reshape(-1)

        if utility_input.size != requested_gain.size:
            raise ValueError(
                "element_utility size mismatch"
            )

        utility = np.maximum(
            utility_input,
            0.0,
        )

    worst_input_power = np.maximum.reduce(
        (
            np.asarray(
                input_power_controller,
                dtype=np.float64,
            ).reshape(-1),
            np.asarray(
                input_power_transmission,
                dtype=np.float64,
            ).reshape(-1),
            np.asarray(
                input_power_reflection,
                dtype=np.float64,
            ).reshape(-1),
        )
    )

    burden = (
        requested_gain**2
        * worst_input_power
        + power_config.active_element_bias_power
    )

    utility_to_burden = utility / np.maximum(
        burden,
        1.0e-12,
    )

    projected_mask = original_active.copy()
    unit_gain = np.ones_like(
        requested_gain,
        dtype=np.float64,
    )

    unit_result = robust_result_for(
        unit_gain,
        projected_mask,
    )

    # 单位增益仍不可行：逐个关闭低效用单元
    if not unit_result.fully_feasible:
        active_indices = np.flatnonzero(
            projected_mask
        )

        removal_order = active_indices[
            np.argsort(
                utility_to_burden[
                    active_indices
                ]
            )
        ]

        for element_index in removal_order:
            projected_mask[
                element_index
            ] = False

            unit_result = robust_result_for(
                unit_gain,
                projected_mask,
            )

            if unit_result.fully_feasible:
                break

    # 全部旁路仍不可行
    if not unit_result.fully_feasible:
        projected_mask = np.zeros_like(
            original_active,
            dtype=bool,
        )

        projected_gain = np.ones_like(
            requested_gain,
            dtype=np.float64,
        )

        projected_command = make_candidate(
            projected_gain,
            projected_mask,
        )

        return (
            projected_command,
            robust_result_for(
                projected_gain,
                projected_mask,
            ),
        )

    masked_requested_gain = np.where(
        projected_mask,
        requested_gain,
        1.0,
    )

    requested_result = robust_result_for(
        masked_requested_gain,
        projected_mask,
    )

    if requested_result.fully_feasible:
        projected_gain = (
            masked_requested_gain.copy()
        )
    else:
        lower = 0.0
        upper = 1.0

        for _ in range(
            power_config.projection_iterations
        ):
            middle = 0.5 * (
                lower + upper
            )

            candidate_gain = (
                _scale_active_gain(
                    masked_requested_gain,
                    projected_mask,
                    middle,
                )
            )

            candidate_result = (
                robust_result_for(
                    candidate_gain,
                    projected_mask,
                )
            )

            if candidate_result.fully_feasible:
                lower = middle
            else:
                upper = middle

        projected_gain = _scale_active_gain(
            masked_requested_gain,
            projected_mask,
            lower,
        )

    # 向下量化，防止量化后超出约束
    projected_gain = quantize_gain_downward(
        projected_gain,
        hardware_config.maximum_active_gain,
        hardware_config.gain_quantization_bits,
    )

    projected_gain[~projected_mask] = 1.0

    projected_command = make_candidate(
        projected_gain,
        projected_mask,
    )

    conservative_result = robust_result_for(
        projected_gain,
        projected_mask,
    )

    return (
        projected_command,
        conservative_result,
    )
