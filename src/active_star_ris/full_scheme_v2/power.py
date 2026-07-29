from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import HardwareConfig, PowerConfig, ProbingConfig
from .models import CSIState, GainProjectionResult, IdealSurfaceAction, PowerMetrics, StaticChannels


def _upper_magnitude(estimate: np.ndarray, std: np.ndarray, margin: float) -> np.ndarray:
    return np.abs(estimate) + margin * std


def _incident_vectors(
    csi: CSIState,
    probing: ProbingConfig,
    power: PowerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = _upper_magnitude(
        csi.controller_ris.estimate,
        csi.controller_ris.error_standard_deviation,
        power.csi_power_margin_std,
    )
    ht = _upper_magnitude(
        csi.ris_transmission.estimate,
        csi.ris_transmission.error_standard_deviation,
        power.csi_power_margin_std,
    )
    hr = _upper_magnitude(
        csi.ris_reflection.estimate,
        csi.ris_reflection.error_standard_deviation,
        power.csi_power_margin_std,
    )
    amplifier_noise = probing.input_referred_amplifier_noise_variance
    return (
        probing.pilot_power_controller * g**2 + amplifier_noise,
        probing.pilot_power_transmission_user * ht**2 + amplifier_noise,
        probing.pilot_power_reflection_user * hr**2 + amplifier_noise,
    )


def _rf_outputs(gain: np.ndarray, active: np.ndarray, incidents: tuple[np.ndarray, ...]) -> tuple[float, float, float]:
    gain_squared = np.where(active, gain**2, 0.0)
    return tuple(float(np.sum(gain_squared * incident)) for incident in incidents)  # type: ignore[return-value]


def _total_dc_power(
    gain: np.ndarray,
    active: np.ndarray,
    incidents: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: PowerConfig,
) -> float:
    n = gain.size
    active_count = int(np.count_nonzero(active))
    passive_count = n - active_count
    time_fractions = np.asarray(
        [
            config.controller_time_fraction,
            config.transmission_time_fraction,
            config.reflection_time_fraction,
        ],
        dtype=np.float64,
    )
    if np.sum(time_fractions) > 0.0:
        time_fractions /= np.sum(time_fractions)
    additional_rf = 0.0
    for fraction, incident in zip(time_fractions, incidents, strict=True):
        additional_rf += float(
            fraction * np.sum(np.where(active, np.maximum(gain**2 - 1.0, 0.0) * incident, 0.0))
        )
    return float(
        config.controller_static_power
        + config.switching_network_static_power
        + passive_count * config.passive_element_control_power
        + active_count * (config.active_element_control_power + config.active_element_bias_power)
        + additional_rf / config.amplifier_efficiency
    )


def _is_feasible(
    gain: np.ndarray,
    active: np.ndarray,
    incidents: tuple[np.ndarray, np.ndarray, np.ndarray],
    hardware: HardwareConfig,
    power: PowerConfig,
) -> tuple[bool, tuple[float, float, float], float]:
    outputs = _rf_outputs(gain, active, incidents)
    dc = _total_dc_power(gain, active, incidents, power)
    per_element_ok = True
    for incident in incidents:
        per_element_ok = per_element_ok and bool(
            np.all(np.where(active, gain**2 * incident, 0.0) <= hardware.per_active_element_saturation_power + power.projection_tolerance)
        )
    feasible = (
        max(outputs) <= power.maximum_rf_output_power + power.projection_tolerance
        and dc <= power.maximum_total_dc_power + power.projection_tolerance
        and per_element_ok
    )
    return feasible, outputs, dc


def project_gains(
    ideal: IdealSurfaceAction,
    csi: CSIState,
    probing: ProbingConfig,
    hardware: HardwareConfig,
    power: PowerConfig,
) -> GainProjectionResult:
    active = ideal.active_mask
    requested = ideal.gain
    incidents = _incident_vectors(csi, probing, power)
    hardware_margin = 10.0 ** (power.hardware_gain_margin_db / 20.0)
    requested_robust = np.where(active, np.clip(requested * hardware_margin, 1.0, hardware.maximum_active_gain), 1.0)
    unit = np.ones_like(requested_robust)
    unit_feasible, unit_outputs, unit_dc = _is_feasible(unit, active, incidents, hardware, power)
    requested_feasible, requested_outputs, requested_dc = _is_feasible(
        requested_robust,
        active,
        incidents,
        hardware,
        power,
    )
    if requested_feasible:
        projected = requested
        outputs = requested_outputs
        dc = requested_dc
        scale = 1.0
    elif not unit_feasible:
        projected = unit
        outputs = unit_outputs
        dc = unit_dc
        scale = 0.0
    else:
        low, high = 0.0, 1.0
        best = unit
        best_outputs = unit_outputs
        best_dc = unit_dc
        for _ in range(power.projection_iterations):
            middle = 0.5 * (low + high)
            trial_nominal = 1.0 + middle * (requested - 1.0)
            trial_robust = np.where(active, np.clip(trial_nominal * hardware_margin, 1.0, hardware.maximum_active_gain), 1.0)
            feasible, trial_outputs, trial_dc = _is_feasible(
                trial_robust,
                active,
                incidents,
                hardware,
                power,
            )
            if feasible:
                low = middle
                best = trial_nominal
                best_outputs = trial_outputs
                best_dc = trial_dc
            else:
                high = middle
        projected = best
        outputs = best_outputs
        dc = best_dc
        scale = low
    return GainProjectionResult(
        projected_gain=np.asarray(projected, dtype=np.float64),
        robust_rf_output_controller=outputs[0],
        robust_rf_output_transmission=outputs[1],
        robust_rf_output_reflection=outputs[2],
        robust_total_dc_power=dc,
        unit_gain_feasible=unit_feasible,
        projection_scale=float(scale),
    )


def replace_gain(action: IdealSurfaceAction, projected: GainProjectionResult) -> IdealSurfaceAction:
    return replace(action, gain=projected.projected_gain)


def actual_power_metrics(
    channels: StaticChannels,
    gain_forward: np.ndarray,
    gain_reverse: np.ndarray,
    active: np.ndarray,
    probing: ProbingConfig,
    hardware: HardwareConfig,
    power: PowerConfig,
) -> PowerMetrics:
    controller_incident = probing.pilot_power_controller * np.abs(channels.controller_ris) ** 2 + probing.input_referred_amplifier_noise_variance
    transmission_incident = probing.pilot_power_transmission_user * np.abs(channels.ris_transmission) ** 2 + probing.input_referred_amplifier_noise_variance
    reflection_incident = probing.pilot_power_reflection_user * np.abs(channels.ris_reflection) ** 2 + probing.input_referred_amplifier_noise_variance

    controller_per_element = np.where(
        active,
        gain_forward**2 * controller_incident,
        0.0,
    )
    transmission_per_element = np.where(
        active,
        gain_reverse**2 * transmission_incident,
        0.0,
    )
    reflection_per_element = np.where(
        active,
        gain_reverse**2 * reflection_incident,
        0.0,
    )

    rf_controller = float(np.sum(controller_per_element))
    rf_transmission = float(np.sum(transmission_per_element))
    rf_reflection = float(np.sum(reflection_per_element))

    average_gain = np.maximum(
        0.5 * (gain_forward + gain_reverse),
        1.0,
    )

    dc = _total_dc_power(
        average_gain,
        active,
        (
            controller_incident,
            transmission_incident,
            reflection_incident,
        ),
        power,
    )

    maximum_element_output = float(
        max(
            np.max(controller_per_element),
            np.max(transmission_per_element),
            np.max(reflection_per_element),
        )
    )

    rf_violation = max(
        0.0,
        max(
            rf_controller,
            rf_transmission,
            rf_reflection,
        )
        - power.maximum_rf_output_power,
    )

    dc_violation = max(
        0.0,
        dc - power.maximum_total_dc_power,
    )

    saturation_violation = max(
        0.0,
        maximum_element_output
        - hardware.per_active_element_saturation_power,
    )

    return PowerMetrics(
        rf_output_controller=rf_controller,
        rf_output_transmission=rf_transmission,
        rf_output_reflection=rf_reflection,
        total_dc_power=dc,
        rf_violation=rf_violation,
        dc_violation=dc_violation,
        saturation_violation=saturation_violation,
        any_violation=(
            rf_violation > 0.0
            or dc_violation > 0.0
            or saturation_violation > 0.0
        ),
    )
