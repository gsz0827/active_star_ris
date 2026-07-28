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

from .channels import correlated_eve_channel
from .security import (
    estimate_eve_leakage_bits_per_sample,
    simulate_eve_branch,
)

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


def feature_correlation(
    a: np.ndarray,
    b: np.ndarray,
    feature: str,
) -> float:
    """计算与实际量化特征一致的互易性。"""
    x_complex = np.asarray(a, dtype=np.complex128).reshape(-1)
    y_complex = np.asarray(b, dtype=np.complex128).reshape(-1)

    if x_complex.size != y_complex.size or x_complex.size < 2:
        return 0.0

    if feature == "real":
        x = x_complex.real
        y = y_complex.real
    elif feature == "imag":
        x = x_complex.imag
        y = y_complex.imag
    elif feature == "magnitude":
        x = np.abs(x_complex)
        y = np.abs(y_complex)
    elif feature == "phase":
        # 相位特征使用圆统计一致性
        phase_difference = np.angle(
            np.exp(1j * (np.angle(x_complex) - np.angle(y_complex)))
        )
        return float(
            np.clip(
                np.abs(np.mean(np.exp(1j * phase_difference))),
                0.0,
                1.0,
            )
        )
    else:
        raise ValueError(f"unsupported feature: {feature}")

    x = x - np.mean(x)
    y = y - np.mean(y)

    denominator = np.sqrt(
        np.sum(x**2) * np.sum(y**2)
    )
    if denominator <= np.finfo(np.float64).eps:
        return 0.0

    return float(
        np.clip(
            np.abs(np.dot(x, y)) / denominator,
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

    if config.channel.eve_enabled:
        rho_eve = (
            config.channel
            .eve_spatial_correlation
        )

        # ====================================================
        # 透射侧Eve信道
        # ====================================================

        ris_eve_t_forward = (
            correlated_eve_channel(
                block.ris_to_transmission_forward,
                config.channel
                .ris_eve_transmission_power,
                rho_eve,
                rng,
            )
        )

        ris_eve_t_reverse = (
            correlated_eve_channel(
                block.transmission_to_ris_reverse,
                config.channel
                .ris_eve_transmission_power,
                rho_eve,
                rng,
            )
        )

        direct_controller_eve_t = (
            correlated_eve_channel(
                block.direct_transmission_forward,
                config.channel
                .direct_controller_eve_transmission_power,
                rho_eve,
                rng,
            )
        )

        direct_user_eve_t = (
            correlated_eve_channel(
                block.direct_transmission_reverse,
                config.channel
                .direct_transmission_user_eve_power,
                rho_eve,
                rng,
            )
        )

        transmission_eve = simulate_eve_branch(
            controller_to_ris_forward=(
                block.controller_to_ris_forward
            ),
            user_to_ris_reverse=(
                block.transmission_to_ris_reverse
            ),
            ris_to_eve_forward=(
                ris_eve_t_forward
            ),
            ris_to_eve_reverse=(
                ris_eve_t_reverse
            ),
            direct_controller_to_eve=(
                direct_controller_eve_t
            ),
            direct_user_to_eve=(
                direct_user_eve_t
            ),
            phi_forward=(
                actual_surface
                .transmission_forward
            ),
            phi_reverse=(
                actual_surface
                .transmission_reverse
            ),
            active_noise_forward=(
                probing.transmission
                .active_noise_forward
            ),
            active_noise_reverse=(
                probing.transmission
                .active_noise_reverse
            ),
            pilot_power_forward=(
                config.probing
                .pilot_power_controller
            ),
            pilot_power_reverse=(
                config.probing
                .pilot_power_transmission_user
            ),
            pilot_symbols_forward=(
                config.probing
                .pilot_symbols_controller
            ),
            pilot_symbols_reverse=(
                config.probing
                .pilot_symbols_transmission_user
            ),
            receiver_noise_variance=(
                config.probing
                .receiver_noise_variance_eve_transmission
                * receiver_noise_scale
            ),
            tx_coefficient_forward=(
                static_hardware
                .endpoint_rf
                .controller_tx
            ),
            tx_coefficient_reverse=(
                static_hardware
                .endpoint_rf
                .transmission_tx
            ),
            rng=rng,
        )

        transmission_eve_leakage = (
            estimate_eve_leakage_bits_per_sample(
                probing.transmission
                .observation_forward,
                probing.transmission
                .observation_reverse,
                transmission_eve,
            )
        )

        # ====================================================
        # 反射侧Eve信道
        # ====================================================

        ris_eve_r_forward = (
            correlated_eve_channel(
                block.ris_to_reflection_forward,
                config.channel
                .ris_eve_reflection_power,
                rho_eve,
                rng,
            )
        )

        ris_eve_r_reverse = (
            correlated_eve_channel(
                block.reflection_to_ris_reverse,
                config.channel
                .ris_eve_reflection_power,
                rho_eve,
                rng,
            )
        )

        direct_controller_eve_r = (
            correlated_eve_channel(
                block.direct_reflection_forward,
                config.channel
                .direct_controller_eve_reflection_power,
                rho_eve,
                rng,
            )
        )

        direct_user_eve_r = (
            correlated_eve_channel(
                block.direct_reflection_reverse,
                config.channel
                .direct_reflection_user_eve_power,
                rho_eve,
                rng,
            )
        )

        reflection_eve = simulate_eve_branch(
            controller_to_ris_forward=(
                block.controller_to_ris_forward
            ),
            user_to_ris_reverse=(
                block.reflection_to_ris_reverse
            ),
            ris_to_eve_forward=(
                ris_eve_r_forward
            ),
            ris_to_eve_reverse=(
                ris_eve_r_reverse
            ),
            direct_controller_to_eve=(
                direct_controller_eve_r
            ),
            direct_user_to_eve=(
                direct_user_eve_r
            ),
            phi_forward=(
                actual_surface
                .reflection_forward
            ),
            phi_reverse=(
                actual_surface
                .reflection_reverse
            ),
            active_noise_forward=(
                probing.reflection
                .active_noise_forward
            ),
            active_noise_reverse=(
                probing.reflection
                .active_noise_reverse
            ),
            pilot_power_forward=(
                config.probing
                .pilot_power_controller
            ),
            pilot_power_reverse=(
                config.probing
                .pilot_power_reflection_user
            ),
            pilot_symbols_forward=(
                config.probing
                .pilot_symbols_controller
            ),
            pilot_symbols_reverse=(
                config.probing
                .pilot_symbols_reflection_user
            ),
            receiver_noise_variance=(
                config.probing
                .receiver_noise_variance_eve_reflection
                * receiver_noise_scale
            ),
            tx_coefficient_forward=(
                static_hardware
                .endpoint_rf
                .controller_tx
            ),
            tx_coefficient_reverse=(
                static_hardware
                .endpoint_rf
                .reflection_tx
            ),
            rng=rng,
        )

        reflection_eve_leakage = (
            estimate_eve_leakage_bits_per_sample(
                probing.reflection
                .observation_forward,
                probing.reflection
                .observation_reverse,
                reflection_eve,
            )
        )

    else:
        transmission_eve_leakage = 0.0
        reflection_eve_leakage = 0.0

    transmission_key = evaluate_key_rate(
        probing.transmission.observation_forward,
        probing.transmission.observation_reverse,
        key_config=config.key_generation,
        probing_config=config.probing,
        rng=rng,
        full_protocol=full_protocol,
        reverse_pilot_symbols=(
            config.probing
            .pilot_symbols_transmission_user
        ),
        eve_leakage_bits_per_retained_bit=(
            transmission_eve_leakage
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
            config.probing
            .pilot_symbols_reflection_user
        ),
        eve_leakage_bits_per_retained_bit=(
            reflection_eve_leakage
        ),
    )
    weight_sum = (
        config.objective.transmission_weight
        + config.objective.reflection_weight
    )
    weight_t = config.objective.transmission_weight / weight_sum
    weight_r = config.objective.reflection_weight / weight_sum
    weighted_eve_leakage = (
        weight_t
        * transmission_eve_leakage
        + weight_r
        * reflection_eve_leakage
    )
    # 复数相关性仅用于高斯互信息代理
    complex_reciprocity_t = complex_correlation(
        probing.transmission.observation_forward,
        probing.transmission.observation_reverse,
    )
    complex_reciprocity_r = complex_correlation(
        probing.reflection.observation_forward,
        probing.reflection.observation_reverse,
    )

    # 奖励中的互易性必须与实际量化特征一致
    reciprocity_t = feature_correlation(
        probing.transmission.observation_forward,
        probing.transmission.observation_reverse,
        config.key_generation.feature,
    )
    reciprocity_r = feature_correlation(
        probing.reflection.observation_forward,
        probing.reflection.observation_reverse,
        config.key_generation.feature,
    )

    reciprocity = (
        weight_t * reciprocity_t
        + weight_r * reciprocity_r
    )

    theoretical_mi = (
        weight_t
        * gaussian_mutual_information(complex_reciprocity_t)
        + weight_r
        * gaussian_mutual_information(complex_reciprocity_r)
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

    joint_frame_duration = max(
        transmission_key.frame_duration_seconds
        + reflection_key.frame_duration_seconds,
        1.0e-12,
    )

    system_training_rate = (
        transmission_key.training_secret_bits
        + reflection_key.training_secret_bits
    ) / joint_frame_duration

    system_final_rate = (
        transmission_key.final_key_bits
        + reflection_key.final_key_bits
    ) / joint_frame_duration

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
        system_training_rate
        if config.objective.key_rate_mode
        == "training_bound"
        else system_final_rate
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
        system_training_key_rate_bps=float(
            system_training_rate
        ),
        system_final_key_rate_bps=float(
            system_final_rate
        ),
        transmission_eve_leakage_bits_per_sample=float(
            transmission_eve_leakage
        ),
        reflection_eve_leakage_bits_per_sample=float(
            reflection_eve_leakage
        ),
        eve_leakage_bits_per_sample=float(
            weighted_eve_leakage
        ),
    )
