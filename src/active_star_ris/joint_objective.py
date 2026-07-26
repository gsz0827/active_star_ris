from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from active_star_ris.hardware_impairments import (
    HardwareMismatchParameters,
    HardwareMismatchRealization,
    apply_hardware_mismatch,
)
from active_star_ris.star_key_system import (
    DualSideKeyGenerationResult,
    StarCoefficientPair,
    build_star_coefficients,
    simulate_dual_side_key_generation,
)
from active_star_ris.surface import EnergySplit, SurfaceCoefficients
from active_star_ris.surface_power import (
    BidirectionalSurfacePower,
    evaluate_bidirectional_surface_power,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class JointObjectiveConfig:
    """联合密钥生成目标的权重与归一化参数。"""

    key_rate_weight: float = 1.0
    key_disagreement_weight: float = 1.0
    reciprocity_weight: float = 0.5
    surface_power_weight: float = 0.1
    constraint_violation_weight: float = 10.0

    key_rate_reference: float = 10.0
    key_disagreement_reference: float = 0.5
    surface_power_reference: float = 1.0

    def validate(self) -> None:
        weights = {
            "key_rate_weight": self.key_rate_weight,
            "key_disagreement_weight": self.key_disagreement_weight,
            "reciprocity_weight": self.reciprocity_weight,
            "surface_power_weight": self.surface_power_weight,
            "constraint_violation_weight": self.constraint_violation_weight,
        }
        for name, value in weights.items():
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")

        references = {
            "key_rate_reference": self.key_rate_reference,
            "key_disagreement_reference": self.key_disagreement_reference,
            "surface_power_reference": self.surface_power_reference,
        }
        for name, value in references.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class JointObjectiveResult:
    """一次联合性能评价的完整结果。"""

    reward: float

    normalized_key_rate: float
    normalized_key_disagreement: float
    normalized_reciprocity: float
    normalized_surface_power: float
    normalized_power_violation: float

    key_generation: DualSideKeyGenerationResult
    surface_power: BidirectionalSurfacePower
    hardware_mismatch: HardwareMismatchRealization


def _surface_to_star_pair(
    surface: SurfaceCoefficients,
) -> StarCoefficientPair:
    """把通信基线使用的SurfaceCoefficients转换为密钥模块格式。"""
    return build_star_coefficients(
        amplitudes=surface.amplitude_gain,
        beta_transmission=(
            surface.energy_split.beta_transmission
        ),
        beta_reflection=(
            surface.energy_split.beta_reflection
        ),
        phase_transmission=np.angle(
            surface.phi_transmission
        ),
        phase_reflection=np.angle(
            surface.phi_reflection
        ),
    )


def _rms_equivalent_channel(
    channel: ArrayLike,
    name: str,
) -> ComplexArray:
    """把多次探测信道转换为保持平均逐单元功率的等效向量。"""
    values = np.asarray(
        channel,
        dtype=np.complex128,
    )

    if values.ndim == 1:
        if values.size == 0:
            raise ValueError(f"{name} cannot be empty")
        return np.asarray(values, dtype=np.complex128)

    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(
            f"{name} must be a non-empty vector or matrix"
        )

    rms_magnitude = np.sqrt(
        np.mean(np.abs(values) ** 2, axis=0)
    )
    return np.asarray(
        rms_magnitude.astype(np.complex128),
        dtype=np.complex128,
    )


def _power_surface_from_actual_coefficients(
    ideal_surface: SurfaceCoefficients,
    mismatch: HardwareMismatchRealization,
) -> SurfaceCoefficients:
    """用前后向最不利实际增益构造功率评价表面。"""
    active_mask = np.asarray(
        ideal_surface.active_mask,
        dtype=bool,
    )

    actual_amplitude = np.maximum(
        mismatch.forward_coefficients.amplitudes,
        mismatch.reverse_coefficients.amplitudes,
    )

    # 无源单元不计入有源输出功率，保留单位幅度便于表达。
    power_amplitude = np.where(
        active_mask,
        actual_amplitude,
        1.0,
    )

    beta_t = np.asarray(
        ideal_surface.energy_split.beta_transmission,
        dtype=np.float64,
    )
    beta_r = np.asarray(
        ideal_surface.energy_split.beta_reflection,
        dtype=np.float64,
    )

    phase_t = np.angle(
        mismatch.forward_coefficients.transmission
    )
    phase_r = np.angle(
        mismatch.forward_coefficients.reflection
    )

    phi_t = (
        power_amplitude
        * np.sqrt(beta_t)
        * np.exp(1j * phase_t)
    )
    phi_r = (
        power_amplitude
        * np.sqrt(beta_r)
        * np.exp(1j * phase_r)
    )

    return SurfaceCoefficients(
        phi_transmission=np.asarray(
            phi_t,
            dtype=np.complex128,
        ),
        phi_reflection=np.asarray(
            phi_r,
            dtype=np.complex128,
        ),
        amplitude_gain=np.asarray(
            power_amplitude,
            dtype=np.float64,
        ),
        active_mask=active_mask,
        energy_split=EnergySplit(
            beta_transmission=beta_t,
            beta_reflection=beta_r,
        ),
    )


def evaluate_joint_objective(
    channel_controller_to_ris: ArrayLike,
    channel_ris_to_transmission_user: ArrayLike,
    channel_ris_to_reflection_user: ArrayLike,
    ideal_surface: SurfaceCoefficients,
    *,
    direct_channel_transmission: ArrayLike = 0.0j,
    direct_channel_reflection: ArrayLike = 0.0j,
    pilot_power_controller: float = 1.0,
    pilot_power_transmission_user: float = 1.0,
    pilot_power_reflection_user: float = 1.0,
    ris_internal_noise_variance: float = 0.0,
    receiver_noise_variance_controller: float = 0.0,
    receiver_noise_variance_transmission_user: float = 0.0,
    receiver_noise_variance_reflection_user: float = 0.0,
    output_power_budget: float = 1.0,
    amplifier_efficiency: float = 0.35,
    controller_static_power: float = 0.10,
    active_element_bias_power: float = 0.01,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    hardware_parameters: HardwareMismatchParameters | None = None,
    hardware_rng: np.random.Generator | None = None,
    objective_config: JointObjectiveConfig | None = None,
    rng: np.random.Generator | None = None,
) -> JointObjectiveResult:
    """统一计算KGR、KDR、互易性、表面功耗及联合奖励。

    智能体或其他优化器应只输出理想控制量。该函数随后施加真实
    硬件失配、前后向独立内部噪声和接收机噪声，并基于实际观测
    计算奖励。因此它将作为后续TD3/SAC环境的核心评价函数。
    """
    config = (
        JointObjectiveConfig()
        if objective_config is None
        else objective_config
    )
    config.validate()

    parameters = (
        HardwareMismatchParameters()
        if hardware_parameters is None
        else hardware_parameters
    )

    generator = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    hardware_generator = (
        generator
        if hardware_rng is None
        else hardware_rng
    )

    ideal_pair = _surface_to_star_pair(
        ideal_surface
    )
    mismatch = apply_hardware_mismatch(
        ideal_coefficients=ideal_pair,
        active_mask=ideal_surface.active_mask,
        parameters=parameters,

        # episode内固定的制造误差和方向非互易误差。
        rng=hardware_generator,

        # 每次step变化的快速相位抖动。
        dynamic_rng=generator,
    )

    key_result = simulate_dual_side_key_generation(
        channel_controller_to_ris=(
            channel_controller_to_ris
        ),
        channel_ris_to_transmission_user=(
            channel_ris_to_transmission_user
        ),
        channel_ris_to_reflection_user=(
            channel_ris_to_reflection_user
        ),
        coefficients=mismatch.forward_coefficients,
        reverse_coefficients=(
            mismatch.reverse_coefficients
        ),
        active_mask=ideal_surface.active_mask,
        direct_channel_transmission=(
            direct_channel_transmission
        ),
        direct_channel_reflection=(
            direct_channel_reflection
        ),
        pilot_power_controller=(
            pilot_power_controller
        ),
        pilot_power_transmission_user=(
            pilot_power_transmission_user
        ),
        pilot_power_reflection_user=(
            pilot_power_reflection_user
        ),
        active_noise_variance=(
            ris_internal_noise_variance
        ),
        receiver_noise_variance_controller=(
            receiver_noise_variance_controller
        ),
        receiver_noise_variance_transmission_user=(
            receiver_noise_variance_transmission_user
        ),
        receiver_noise_variance_reflection_user=(
            receiver_noise_variance_reflection_user
        ),
        transmission_weight=transmission_weight,
        reflection_weight=reflection_weight,
        rng=generator,
    )

    power_surface = (
        _power_surface_from_actual_coefficients(
            ideal_surface,
            mismatch,
        )
    )

    power_result = evaluate_bidirectional_surface_power(
        controller_to_ris=_rms_equivalent_channel(
            channel_controller_to_ris,
            "channel_controller_to_ris",
        ),
        transmission_user_to_ris=(
            _rms_equivalent_channel(
                channel_ris_to_transmission_user,
                "channel_ris_to_transmission_user",
            )
        ),
        reflection_user_to_ris=(
            _rms_equivalent_channel(
                channel_ris_to_reflection_user,
                "channel_ris_to_reflection_user",
            )
        ),
        surface=power_surface,
        controller_pilot_power=(
            pilot_power_controller
        ),
        transmission_user_pilot_power=(
            pilot_power_transmission_user
        ),
        reflection_user_pilot_power=(
            pilot_power_reflection_user
        ),
        ris_internal_noise_variance=(
            ris_internal_noise_variance
        ),
        output_power_budget=output_power_budget,
        amplifier_efficiency=amplifier_efficiency,
        controller_static_power=controller_static_power,
        active_element_bias_power=(
            active_element_bias_power
        ),
    )

    normalized_key_rate = (
        key_result.weighted_mutual_information
        / config.key_rate_reference
    )
    normalized_kdr = (
        key_result.weighted_key_disagreement_rate
        / config.key_disagreement_reference
    )
    normalized_reciprocity = (
        key_result.weighted_correlation
    )
    normalized_surface_power = (
        power_result.total_surface_power
        / config.surface_power_reference
    )
    normalized_violation = (
        power_result.power_violation
        / max(output_power_budget, 1.0e-12)
    )

    reward = (
        config.key_rate_weight
        * normalized_key_rate
        - config.key_disagreement_weight
        * normalized_kdr
        + config.reciprocity_weight
        * normalized_reciprocity
        - config.surface_power_weight
        * normalized_surface_power
        - config.constraint_violation_weight
        * normalized_violation**2
    )

    return JointObjectiveResult(
        reward=float(reward),
        normalized_key_rate=float(
            normalized_key_rate
        ),
        normalized_key_disagreement=float(
            normalized_kdr
        ),
        normalized_reciprocity=float(
            normalized_reciprocity
        ),
        normalized_surface_power=float(
            normalized_surface_power
        ),
        normalized_power_violation=float(
            normalized_violation
        ),
        key_generation=key_result,
        surface_power=power_result,
        hardware_mismatch=mismatch,
    )
