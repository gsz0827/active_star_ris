from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, get_type_hints

import yaml


PhaseCouplingMode = Literal[
    "independent",
    "quadrature",
    "hybrid",
]
FeatureName = Literal["real", "imag", "magnitude", "phase"]
SelectionPolicy = Literal["alice", "intersection"]
KeyRateMode = Literal["training_bound", "final_key"]


@dataclass(frozen=True)
class ChannelConfig:
    num_elements: int = 32
    active_ratio: float = 0.25
    rician_k_factor: float = 3.0

    controller_ris_power: float = 1.0
    ris_transmission_power: float = 1.0
    ris_reflection_power: float = 1.0
    direct_transmission_power: float = 0.10
    direct_reflection_power: float = 0.10

    # ========================================================
    # Eve信道参数
    # ========================================================

    eve_enabled: bool = True

    # 这里表示Eve信道与对应合法用户信道的复相关系数。
    eve_spatial_correlation: float = 0.20

    # RIS -> Eve平均信道功率
    ris_eve_transmission_power: float = 0.50
    ris_eve_reflection_power: float = 0.50

    # Controller/User -> Eve直达链路平均功率
    direct_controller_eve_transmission_power: float = 0.05
    direct_transmission_user_eve_power: float = 0.05

    direct_controller_eve_reflection_power: float = 0.05
    direct_reflection_user_eve_power: float = 0.05
    
    within_block_correlation: float = 0.995
    between_step_correlation: float = 0.98
    forward_reverse_delay_seconds: float = 1.0e-3
    channel_coherence_time_seconds: float = 1.0e-2

    control_csi_nmse_db: float = -15.0

    ris_eve_transmission_power: float = 0.50
    ris_eve_reflection_power: float = 0.50

    direct_controller_eve_transmission_power: float = 0.05
    direct_transmission_user_eve_power: float = 0.05

    direct_controller_eve_reflection_power: float = 0.05
    direct_reflection_user_eve_power: float = 0.05

    eve_legitimate_spatial_correlation: float = 0.20

    def validate(self) -> None:
        if self.num_elements < 2:
            raise ValueError("num_elements must be at least 2")
        if not 0.0 <= self.active_ratio <= 1.0:
            raise ValueError("active_ratio must lie in [0, 1]")
        if self.rician_k_factor < 0.0:
            raise ValueError("rician_k_factor cannot be negative")
        for name in (
            "controller_ris_power",
            "ris_transmission_power",
            "ris_reflection_power",
            "direct_transmission_power",
            "direct_reflection_power",
            "ris_eve_transmission_power",
            "ris_eve_reflection_power",
            "direct_controller_eve_transmission_power",
            "direct_transmission_user_eve_power",
            "direct_controller_eve_reflection_power",
            "direct_reflection_user_eve_power",
        ):
        if not 0.0 <= self.eve_spatial_correlation <= 1.0:
            raise ValueError(
                "eve_spatial_correlation must lie in [0, 1]"
            )
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("within_block_correlation", "between_step_correlation"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.forward_reverse_delay_seconds < 0.0:
            raise ValueError("forward_reverse_delay_seconds cannot be negative")
        if self.channel_coherence_time_seconds <= 0.0:
            raise ValueError("channel_coherence_time_seconds must be positive")

    @property
    def forward_reverse_correlation(self) -> float:
        import numpy as np

        return float(
            np.exp(
                -self.forward_reverse_delay_seconds
                / self.channel_coherence_time_seconds
            )
        )


@dataclass(frozen=True)
class ProbingConfig:
    samples_per_step: int = 256
    pilot_symbols_controller: int = 16
    pilot_symbols_transmission_user: int = 16
    pilot_symbols_reflection_user: int = 16

    pilot_power_controller: float = 1.0
    pilot_power_transmission_user: float = 1.0
    pilot_power_reflection_user: float = 1.0

    input_referred_amplifier_noise_variance: float = 1.0e-3
    receiver_noise_variance_controller: float = 1.0e-3
    receiver_noise_variance_transmission_user: float = 1.0e-3
    receiver_noise_variance_reflection_user: float = 1.0e-3

    receiver_noise_variance_eve_transmission: float = 1.0e-3
    receiver_noise_variance_eve_reflection: float = 1.0e-3

    pilot_symbol_duration_seconds: float = 1.0e-4
    forward_reverse_guard_seconds: float = 5.0e-5
    branch_switch_guard_seconds: float = 1.0e-4

    def validate(self) -> None:
        if self.samples_per_step < 8:
            raise ValueError("samples_per_step must be at least 8")
        for name in (
            "pilot_symbols_controller",
            "pilot_symbols_transmission_user",
            "pilot_symbols_reflection_user",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "pilot_power_controller",
            "pilot_power_transmission_user",
            "pilot_power_reflection_user",
            "pilot_symbol_duration_seconds",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "input_referred_amplifier_noise_variance",
            "receiver_noise_variance_controller",
            "receiver_noise_variance_transmission_user",
            "receiver_noise_variance_reflection_user",
            "receiver_noise_variance_eve_transmission",
            "receiver_noise_variance_eve_reflection",
            "forward_reverse_guard_seconds",
            "branch_switch_guard_seconds",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class HardwareConfig:
    maximum_active_gain: float = 4.0

    # 被动单元插入损耗
    passive_transmission_insertion_loss_db: float = 1.0
    passive_reflection_insertion_loss_db: float = 1.0

    static_gain_error_std_db: float = 0.20
    directional_gain_error_std_db: float = 0.10
    static_phase_error_std_rad: float = 0.05
    directional_phase_error_std_rad: float = 0.03
    fast_phase_jitter_std_rad: float = 0.01
    transmission_split_error_std: float = 0.02

    transmission_amplitude_phase_coupling_rad_per_db: float = 0.01
    reflection_amplitude_phase_coupling_rad_per_db: float = 0.01

    phase_quantization_bits: int | None = 4
    gain_quantization_bits: int | None = 4

    # 主论文建议使用 quadrature
    phase_coupling_mode: PhaseCouplingMode = "quadrature"

    per_active_element_saturation_power: float = 5.0

    endpoint_gain_error_std_db: float = 0.10
    endpoint_phase_error_std_rad: float = 0.03

    def validate(self) -> None:
        if self.maximum_active_gain < 1.0:
            raise ValueError("maximum_active_gain must be at least 1")
        for name in (
            "static_gain_error_std_db",
            "directional_gain_error_std_db",
            "static_phase_error_std_rad",
            "directional_phase_error_std_rad",
            "fast_phase_jitter_std_rad",
            "transmission_split_error_std",
            "endpoint_gain_error_std_db",
            "endpoint_phase_error_std_rad",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.phase_quantization_bits is not None and self.phase_quantization_bits < 1:
            raise ValueError("phase_quantization_bits must be positive or None")
        if self.gain_quantization_bits is not None and self.gain_quantization_bits < 1:
            raise ValueError("gain_quantization_bits must be positive or None")
        if self.phase_coupling_mode not in {"independent", "quadrature"}:
            raise ValueError("invalid phase_coupling_mode")
        if self.per_active_element_saturation_power <= 0.0:
            raise ValueError("per_active_element_saturation_power must be positive")
        for name in (
            "passive_transmission_insertion_loss_db",
            "passive_reflection_insertion_loss_db",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")

        if self.phase_coupling_mode not in {
            "independent",
            "quadrature",
            "hybrid",
        }:
            raise ValueError("invalid phase_coupling_mode")

@dataclass(frozen=True)
class PowerConfig:
    maximum_rf_output_power: float = 35.0
    maximum_total_dc_power: float = 5.0
    amplifier_efficiency: float = 0.35

    controller_static_power: float = 0.10
    passive_element_control_power: float = 0.001
    active_element_control_power: float = 0.005
    active_element_bias_power: float = 0.01
    switching_network_static_power: float = 0.05

    controller_time_fraction: float = 1.0 / 3.0
    transmission_time_fraction: float = 1.0 / 3.0
    reflection_time_fraction: float = 1.0 / 3.0

    projection_iterations: int = 40
    projection_tolerance: float = 1.0e-9
    csi_power_margin_std: float = 2.0
    hardware_gain_margin_db: float = 0.60

    def validate(self) -> None:
        if self.maximum_rf_output_power <= 0.0:
            raise ValueError("maximum_rf_output_power must be positive")
        if self.maximum_total_dc_power <= 0.0:
            raise ValueError("maximum_total_dc_power must be positive")
        if not 0.0 < self.amplifier_efficiency <= 1.0:
            raise ValueError("amplifier_efficiency must lie in (0, 1]")
        for name in (
            "controller_static_power",
            "passive_element_control_power",
            "active_element_control_power",
            "active_element_bias_power",
            "switching_network_static_power",
            "controller_time_fraction",
            "transmission_time_fraction",
            "reflection_time_fraction",
            "csi_power_margin_std",
            "hardware_gain_margin_db",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.projection_iterations < 1:
            raise ValueError("projection_iterations must be positive")
        if self.projection_tolerance < 0.0:
            raise ValueError("projection_tolerance cannot be negative")
        if (
            self.controller_time_fraction
            + self.transmission_time_fraction
            + self.reflection_time_fraction
            <= 0.0
        ):
            raise ValueError("at least one time fraction must be positive")


@dataclass(frozen=True)
class KeyGenerationConfig:
    feature: FeatureName = "real"
    guard_band_sigma: float = 0.10
    selection_policy: SelectionPolicy = "alice"

    initial_block_size: int = 8
    reconciliation_passes: int = 8
    maximum_block_doublings: int = 3
    verification_tag_bits: int = 32
    privacy_margin_bits: int = 64
    maximum_final_key_bits: int = 256

    # 仅用于TD3训练阶段的边际熵代理。
    # full_protocol=True时不得直接使用该固定值。
    minimum_entropy_bits_per_retained_bit: float = 0.80
    
    reconciliation_efficiency: float = 1.15
    maximum_trainable_raw_kdr: float = 0.30

    public_channel_rate_bps: float = 1.0e6
    fixed_processing_delay_seconds: float = 0.0
    include_toeplitz_seed_in_public_time: bool = False

    def validate(self) -> None:
        if self.feature not in {"real", "imag", "magnitude", "phase"}:
            raise ValueError("invalid feature")
        if self.selection_policy not in {"alice", "intersection"}:
            raise ValueError("invalid selection_policy")
        if self.guard_band_sigma < 0.0:
            raise ValueError("guard_band_sigma cannot be negative")
        if self.initial_block_size < 2:
            raise ValueError("initial_block_size must be at least 2")
        if self.reconciliation_passes < 1:
            raise ValueError("reconciliation_passes must be positive")
        if self.maximum_block_doublings < 0:
            raise ValueError("maximum_block_doublings cannot be negative")
        if not 1 <= self.verification_tag_bits <= 256:
            raise ValueError("verification_tag_bits must lie in [1, 256]")
        if self.privacy_margin_bits < 0:
            raise ValueError("privacy_margin_bits cannot be negative")
        if self.maximum_final_key_bits < 1:
            raise ValueError("maximum_final_key_bits must be positive")
        if not 0.0 <= self.minimum_entropy_bits_per_retained_bit <= 1.0:
            raise ValueError("minimum entropy rate must lie in [0, 1]")
        if self.reconciliation_efficiency < 1.0:
            raise ValueError("reconciliation_efficiency must be at least 1")
        if not 0.0 < self.maximum_trainable_raw_kdr < 0.5:
            raise ValueError("maximum_trainable_raw_kdr must lie in (0, 0.5)")
        if self.public_channel_rate_bps <= 0.0:
            raise ValueError("public_channel_rate_bps must be positive")
        if self.fixed_processing_delay_seconds < 0.0:
            raise ValueError("fixed_processing_delay_seconds cannot be negative")


@dataclass(frozen=True)
class ObjectiveConfig:
    key_rate_mode: KeyRateMode = "training_bound"

    key_rate_weight: float = 1.0
    raw_kdr_weight: float = 0.50
    post_reconciliation_kdr_weight: float = 1.0
    reciprocity_weight: float = 0.20
    surface_power_weight: float = 0.10
    constraint_violation_weight: float = 10.0

    key_rate_reference_bps: float = 10_000.0
    raw_kdr_reference: float = 0.50
    post_reconciliation_kdr_reference: float = 0.01
    surface_power_reference_watt: float = 5.0

    transmission_weight: float = 0.5
    reflection_weight: float = 0.5

    def validate(self) -> None:
        if self.key_rate_mode not in {"training_bound", "final_key"}:
            raise ValueError("invalid key_rate_mode")
        for name in (
            "key_rate_weight",
            "raw_kdr_weight",
            "post_reconciliation_kdr_weight",
            "reciprocity_weight",
            "surface_power_weight",
            "constraint_violation_weight",
            "transmission_weight",
            "reflection_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "key_rate_reference_bps",
            "raw_kdr_reference",
            "post_reconciliation_kdr_reference",
            "surface_power_reference_watt",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.transmission_weight + self.reflection_weight <= 0.0:
            raise ValueError("branch weights must have a positive sum")


@dataclass(frozen=True)
class RobustConfig:
    objective_samples: int = 16
    cvar_alpha: float = 0.25
    mean_weight: float = 0.50
    cvar_weight: float = 0.50
    minimum_tail_samples: int = 4

    nmse_db_min: float = -20.0
    nmse_db_max: float = -10.0
    amplifier_noise_scale_min: float = 0.5
    amplifier_noise_scale_max: float = 2.0
    receiver_noise_scale_min: float = 0.5
    receiver_noise_scale_max: float = 2.0
    rf_budget_scale_min: float = 0.80
    rf_budget_scale_max: float = 1.20
    dc_budget_scale_min: float = 0.80
    dc_budget_scale_max: float = 1.20

    include_known_system_context: bool = True
    include_oracle_impairment_context: bool = False

    hardware_error_scale_min: float = 0.5
    hardware_error_scale_max: float = 2.0

    def validate(self) -> None:
        if self.objective_samples < 1:
            raise ValueError("objective_samples must be positive")
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError("cvar_alpha must lie in (0, 1]")
        if self.mean_weight < 0.0 or self.cvar_weight < 0.0:
            raise ValueError("robust weights cannot be negative")
        if self.mean_weight + self.cvar_weight <= 0.0:
            raise ValueError("robust weights must have a positive sum")
        if self.minimum_tail_samples < 1:
            raise ValueError("minimum_tail_samples must be positive")
        import math

        required = math.floor(
            (self.minimum_tail_samples - 1) / self.cvar_alpha
        ) + 1
        if self.objective_samples < required:
            raise ValueError(
                "objective_samples is too small for the requested CVaR tail; "
                f"need at least {required}"
            )
        for low_name, high_name in (
            ("nmse_db_min", "nmse_db_max"),
            ("amplifier_noise_scale_min", "amplifier_noise_scale_max"),
            ("receiver_noise_scale_min", "receiver_noise_scale_max"),
            ("rf_budget_scale_min", "rf_budget_scale_max"),
            ("dc_budget_scale_min", "dc_budget_scale_max"),
            ("hardware_error_scale_min", "hardware_error_scale_max"),
        ):
            if getattr(self, low_name) > getattr(self, high_name):
                raise ValueError(f"{low_name} cannot exceed {high_name}")


@dataclass(frozen=True)
class EnvironmentConfig:
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    probing: ProbingConfig = field(default_factory=ProbingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    key_generation: KeyGenerationConfig = field(default_factory=KeyGenerationConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    robust: RobustConfig = field(default_factory=RobustConfig)

    max_episode_steps: int = 50

    def validate(self) -> None:
        self.channel.validate()
        self.probing.validate()
        self.hardware.validate()
        self.power.validate()
        self.key_generation.validate()
        self.objective.validate()
        self.robust.validate()
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")


T = TypeVar("T")


def _dataclass_from_dict(cls: type[T], values: dict[str, Any]) -> T:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in values:
            continue
        value = values[item.name]
        hinted_type = hints.get(item.name)
        if isinstance(value, dict) and isinstance(hinted_type, type) and is_dataclass(hinted_type):
            kwargs[item.name] = _dataclass_from_dict(hinted_type, value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)


def load_environment_config(path: str | Path) -> EnvironmentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    config = _dataclass_from_dict(EnvironmentConfig, data)
    config.validate()
    return config
