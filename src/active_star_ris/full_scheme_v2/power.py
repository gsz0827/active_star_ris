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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nmse_linear = 10.0 ** (nmse_db / 10.0)

    def robust_power(values: np.ndarray, pilot_power: float) -> np.ndarray:
        magnitude = np.abs(np.asarray(values, dtype=np.complex128))
        uncertainty = np.sqrt(np.maximum(nmse_linear, 0.0)) * np.maximum(
            magnitude,
            1.0e-12,
        )
        upper_magnitude = magnitude + power.csi_power_margin_std * uncertainty
        return (
            pilot_power * upper_magnitude**2
            + probing.input_referred_amplifier_noise_variance
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
    active = command.active_mask
    requested_gain = np.asarray(command.gain, dtype=np.float64)

    # Conservative hardware margin protects against positive gain mismatch.
    mismatch_margin = 10.0 ** (power_config.hardware_gain_margin_db / 20.0)
    conservative_requested = np.where(
        active,
        np.minimum(
            requested_gain * mismatch_margin,
            hardware_config.maximum_active_gain,
        ),
        1.0,
    )

    worst_input = np.maximum.reduce(
        (
            np.asarray(input_power_controller, dtype=np.float64),
            np.asarray(input_power_transmission, dtype=np.float64),
            np.asarray(input_power_reflection, dtype=np.float64),
        )
    )
    saturation_gain = np.sqrt(
        hardware_config.per_active_element_saturation_power
        / np.maximum(worst_input, 1.0e-12)
    )
    conservative_requested = np.where(
        active,
        np.minimum(conservative_requested, saturation_gain),
        1.0,
    )
    conservative_requested = np.clip(
        conservative_requested,
        1.0,
        hardware_config.maximum_active_gain,
    )

    def make_candidate(gain: np.ndarray) -> IdealSurfaceCommand:
        return replace(command, gain=np.asarray(gain, dtype=np.float64))

    def result_for(gain: np.ndarray) -> PowerResult:
        return evaluate_power(
            make_candidate(gain),
            input_power_controller,
            input_power_transmission,
            input_power_reflection,
            power_config=power_config,
            hardware_config=hardware_config,
            rf_budget=rf_budget,
            dc_budget=dc_budget,
        )

    if result_for(conservative_requested).fully_feasible:
        projected_gain = conservative_requested
    else:
        passive_gain = np.ones_like(requested_gain)
        if not result_for(passive_gain).fully_feasible:
            projected_gain = passive_gain
        else:
            lower = 0.0
            upper = 1.0
            for _ in range(power_config.projection_iterations):
                middle = 0.5 * (lower + upper)
                candidate_gain = _scale_active_gain(
                    conservative_requested,
                    active,
                    middle,
                )
                if result_for(candidate_gain).fully_feasible:
                    lower = middle
                else:
                    upper = middle
            projected_gain = _scale_active_gain(
                conservative_requested,
                active,
                lower,
            )

    # Downward quantization preserves feasibility better than nearest rounding.
    projected_gain = quantize_gain_downward(
        projected_gain,
        hardware_config.maximum_active_gain,
        hardware_config.gain_quantization_bits,
    )
    projected_gain[~active] = 1.0

    projected = make_candidate(projected_gain)
    final_result = evaluate_power(
        projected,
        input_power_controller,
        input_power_transmission,
        input_power_reflection,
        power_config=power_config,
        hardware_config=hardware_config,
        rf_budget=rf_budget,
        dc_budget=dc_budget,
    )
    return projected, final_result
