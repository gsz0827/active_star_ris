from __future__ import annotations

import numpy as np

from .config import EnvironmentConfig
from .hardware import apply_hardware
from .key_protocol import evaluate_key_rate
from .models import (
    BidirectionalChannelBlock,
    HardwareStaticRealization,
    IdealSurfaceCommand,
    ObjectiveResult,
    PowerResult,
)
from .power import evaluate_power
from .probing import simulate_dual_side_probing


def complex_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.complex128).reshape(-1)
    y = np.asarray(b, dtype=np.complex128).reshape(-1)
    if x.size != y.size or x.size < 2:
        return 0.0
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.sqrt(
        np.sum(np.abs(x_centered) ** 2)
        * np.sum(np.abs(y_centered) ** 2)
    )
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(
        np.clip(
            np.abs(np.vdot(x_centered, y_centered)) / denominator,
            0.0,
            1.0,
        )
    )


def gaussian_mutual_information(correlation: float) -> float:
    rho_sq = min(float(correlation) ** 2, 1.0 - 1.0e-12)
    return float(-np.log2(max(1.0 - rho_sq, 1.0e-12)))


def evaluate_objective(
    block: BidirectionalChannelBlock,
    command: IdealSurfaceCommand,
    static_hardware: HardwareStaticRealization,
    config: EnvironmentConfig,
    rng: np.random.Generator,
    *,
    full_protocol: bool,
    amplifier_noise_scale: float = 1.0,
    receiver_noise_scale: float = 1.0,
    rf_budget: float | None = None,
    dc_budget: float | None = None,
) -> ObjectiveResult:
    actual_surface = apply_hardware(
        command,
        static_hardware,
        config.hardware,
        rng,
    )

    probing = simulate_dual_side_probing(
        block,
        actual_surface,
        command.active_mask,
        static_hardware.endpoint_rf,
        config.probing,
        rng,
        amplifier_noise_scale=amplifier_noise_scale,
        receiver_noise_scale=receiver_noise_scale,
    )

    transmission_key = evaluate_key_rate(
        probing.transmission.observation_forward,
        probing.transmission.observation_reverse,
        key_config=config.key_generation,
        probing_config=config.probing,
        rng=rng,
        full_protocol=full_protocol,
        reverse_pilot_symbols=(
            config.probing.pilot_symbols_transmission_user
        ),
    )
    reflection_key = evaluate_key_rate(
        probing.reflection.observation_forward,
        probing.reflection.observation_reverse,
        key_config=config.key_generation,
        probing_config=config.probing,
        rng=rng,
        full_protocol=full_protocol,
        reverse_pilot_symbols=(
            config.probing.pilot_symbols_reflection_user
        ),
    )
    weight_sum = (
        config.objective.transmission_weight
        + config.objective.reflection_weight
    )
    weight_t = config.objective.transmission_weight / weight_sum
    weight_r = config.objective.reflection_weight / weight_sum

    reciprocity_t = complex_correlation(
        probing.transmission.observation_forward,
        probing.transmission.observation_reverse,
    )
    reciprocity_r = complex_correlation(
        probing.reflection.observation_forward,
        probing.reflection.observation_reverse,
    )
    reciprocity = weight_t * reciprocity_t + weight_r * reciprocity_r

    theoretical_mi = (
        weight_t * gaussian_mutual_information(reciprocity_t)
        + weight_r * gaussian_mutual_information(reciprocity_r)
    )
    training_rate = (
        weight_t * transmission_key.training_key_rate_bps
        + weight_r * reflection_key.training_key_rate_bps
    )
    final_rate = (
        weight_t * transmission_key.final_key_rate_bps
        + weight_r * reflection_key.final_key_rate_bps
    )
    raw_kdr = (
        weight_t * transmission_key.raw_kdr
        + weight_r * reflection_key.raw_kdr
    )
    post_kdr = (
        weight_t * transmission_key.post_reconciliation_kdr
        + weight_r * reflection_key.post_reconciliation_kdr
    )
    retention = (
        weight_t * transmission_key.retention_ratio
        + weight_r * reflection_key.retention_ratio
    )
    success_rate = (
        weight_t * float(transmission_key.success)
        + weight_r * float(reflection_key.success)
    )

    # 功率投影使用估计CSI的保守上界；这里的性能报告改用真实双向信道块。
    actual_input_power_controller = (
        config.probing.pilot_power_controller
        * np.mean(
            np.abs(block.controller_to_ris_forward) ** 2,
            axis=0,
        )
        + config.probing.input_referred_amplifier_noise_variance
        * amplifier_noise_scale
    )
    actual_input_power_transmission = (
        config.probing.pilot_power_transmission_user
        * np.mean(
            np.abs(block.transmission_to_ris_reverse) ** 2,
            axis=0,
        )
        + config.probing.input_referred_amplifier_noise_variance
        * amplifier_noise_scale
    )
    actual_input_power_reflection = (
        config.probing.pilot_power_reflection_user
        * np.mean(
            np.abs(block.reflection_to_ris_reverse) ** 2,
            axis=0,
        )
        + config.probing.input_referred_amplifier_noise_variance
        * amplifier_noise_scale
    )

    # 前后向最不利实际增益用于真实功率和饱和检查。
    power_gain = np.maximum(
        actual_surface.gain_forward,
        actual_surface.gain_reverse,
    )
    power_command = IdealSurfaceCommand(
        gain=power_gain,
        beta_transmission=actual_surface.beta_transmission,
        phase_transmission=np.angle(actual_surface.transmission_forward),
        phase_reflection=np.angle(actual_surface.reflection_forward),
        active_mask=command.active_mask,
    )
    power_result: PowerResult = evaluate_power(
        power_command,
        actual_input_power_controller,
        actual_input_power_transmission,
        actual_input_power_reflection,
        power_config=config.power,
        hardware_config=config.hardware,
        rf_budget=rf_budget,
        dc_budget=dc_budget,
    )

    rewarded_rate = (
        training_rate
        if config.objective.key_rate_mode == "training_bound"
        else final_rate
    )
    normalized_key_rate = float(
        np.log1p(max(rewarded_rate, 0.0))
        / np.log1p(config.objective.key_rate_reference_bps)
    )
    normalized_kdr = float(
        config.objective.raw_kdr_weight
        * raw_kdr
        / config.objective.raw_kdr_reference
        + config.objective.post_reconciliation_kdr_weight
        * post_kdr
        / config.objective.post_reconciliation_kdr_reference
    )
    normalized_power = float(
        power_result.total_surface_dc_power
        / config.objective.surface_power_reference_watt
    )

    effective_rf_budget = (
        config.power.maximum_rf_output_power
        if rf_budget is None
        else rf_budget
    )
    effective_dc_budget = (
        config.power.maximum_total_dc_power
        if dc_budget is None
        else dc_budget
    )
    normalized_violation = float(
        (power_result.rf_violation / max(effective_rf_budget, 1.0e-12)) ** 2
        + (power_result.dc_violation / max(effective_dc_budget, 1.0e-12)) ** 2
        + (
            power_result.per_element_saturation_violation
            / config.hardware.per_active_element_saturation_power
        )
        ** 2
    )

    reward = float(
        config.objective.key_rate_weight * normalized_key_rate
        - normalized_kdr
        + config.objective.reciprocity_weight * reciprocity
        - config.objective.surface_power_weight * normalized_power
        - config.objective.constraint_violation_weight * normalized_violation
    )

    return ObjectiveResult(
        reward=reward,
        theoretical_mutual_information_bits_per_sample=theoretical_mi,
        training_key_rate_bps=float(training_rate),
        final_key_rate_bps=float(final_rate),
        raw_kdr=float(raw_kdr),
        post_reconciliation_kdr=float(post_kdr),
        reciprocity=float(reciprocity),
        retention_ratio=float(retention),
        success_rate=float(success_rate),
        normalized_key_rate=normalized_key_rate,
        normalized_kdr=normalized_kdr,
        normalized_power=normalized_power,
        normalized_constraint_violation=normalized_violation,
        transmission_key=transmission_key,
        reflection_key=reflection_key,
        power=power_result,
        probing=probing,
    )
