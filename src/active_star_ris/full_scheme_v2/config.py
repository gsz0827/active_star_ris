from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, get_type_hints

import yaml

PhaseCouplingMode = Literal["independent", "quadrature", "hybrid"]
Architecture = Literal[
    "passive",
    "partially_active_fixed",
    "partially_active_dynamic",
    "fully_active_fixed",
]
FeatureName = Literal["real", "imag", "magnitude", "phase"]
SelectionPolicy = Literal["alice", "intersection"]
CSIModel = Literal[
    "ls",
    "lmmse",
    "nmse",
    "nmse_oracle",
]


@dataclass(frozen=True)
class GeometryConfig:
    carrier_frequency_hz: float = 3.5e9
    ris_rows: int = 4
    ris_columns: int = 8
    element_spacing_wavelengths: float = 0.5
    controller_position_m: list[float] = field(default_factory=lambda: [-15.0, 0.0, 2.0])
    ris_position_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 5.0])
    transmission_user_position_m: list[float] = field(default_factory=lambda: [12.0, 8.0, 1.5])
    reflection_user_position_m: list[float] = field(default_factory=lambda: [-10.0, 9.0, 1.5])
    eve_transmission_position_m: list[float] = field(default_factory=lambda: [10.0, 12.0, 1.5])
    eve_reflection_position_m: list[float] = field(default_factory=lambda: [-8.0, 13.0, 1.5])
    reference_distance_m: float = 1.0
    path_loss_exponent_ris: float = 2.2
    path_loss_exponent_direct: float = 3.0
    additional_ris_loss_db: float = 0.0
    additional_direct_loss_db: float = 20.0

    def validate(self) -> None:
        if self.carrier_frequency_hz <= 0.0:
            raise ValueError("carrier_frequency_hz must be positive")
        if self.ris_rows < 1 or self.ris_columns < 1:
            raise ValueError("ris_rows and ris_columns must be positive")
        if self.element_spacing_wavelengths <= 0.0:
            raise ValueError("element_spacing_wavelengths must be positive")
        if self.reference_distance_m <= 0.0:
            raise ValueError("reference_distance_m must be positive")
        if self.path_loss_exponent_ris <= 0.0 or self.path_loss_exponent_direct <= 0.0:
            raise ValueError("path-loss exponents must be positive")
        for name in (
            "controller_position_m",
            "ris_position_m",
            "transmission_user_position_m",
            "reflection_user_position_m",
            "eve_transmission_position_m",
            "eve_reflection_position_m",
        ):
            value = getattr(self, name)
            if len(value) != 3:
                raise ValueError(f"{name} must contain three coordinates")

    @property
    def num_elements(self) -> int:
        return self.ris_rows * self.ris_columns


@dataclass(frozen=True)
class ChannelConfig:
    rician_k_factor: float = 3.0
    within_block_correlation: float = 0.995
    between_step_correlation: float = 0.98
    forward_reverse_delay_seconds: float = 1.0e-3
    channel_coherence_time_seconds: float = 1.0e-2
    eve_enabled: bool = True
    eve_spatial_correlation: float = 0.20
    control_csi_model: CSIModel = "lmmse"
    control_csi_nmse_db: float = -15.0
    csi_pilot_symbols: int = 32
    csi_pilot_power: float = 0.1
    csi_receiver_noise_variance: float = 2.0e-14

    def validate(self) -> None:
        if self.rician_k_factor < 0.0:
            raise ValueError("rician_k_factor cannot be negative")
        for name in ("within_block_correlation", "between_step_correlation", "eve_spatial_correlation"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.forward_reverse_delay_seconds < 0.0:
            raise ValueError("forward_reverse_delay_seconds cannot be negative")
        if self.channel_coherence_time_seconds <= 0.0:
            raise ValueError("channel_coherence_time_seconds must be positive")
        if self.control_csi_model not in {
            "ls",
            "lmmse",
            "nmse",
            "nmse_oracle",
        }:
            raise ValueError("invalid control_csi_model")
        if self.csi_pilot_symbols < 1 or self.csi_pilot_power <= 0.0:
            raise ValueError("CSI pilot length and power must be positive")
        if self.csi_receiver_noise_variance < 0.0:
            raise ValueError("csi_receiver_noise_variance cannot be negative")

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
    pilot_power_controller: float = 0.1
    pilot_power_transmission_user: float = 0.1
    pilot_power_reflection_user: float = 0.1
    input_referred_amplifier_noise_variance: float = 1.0e-12
    receiver_noise_variance_controller: float = 2.0e-14
    receiver_noise_variance_transmission_user: float = 2.0e-14
    receiver_noise_variance_reflection_user: float = 2.0e-14
    receiver_noise_variance_eve: float = 2.0e-13
    pilot_symbol_duration_seconds: float = 1.0e-6
    forward_reverse_guard_seconds: float = 5.0e-7
    branch_switch_guard_seconds: float = 1.0e-6

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
            "receiver_noise_variance_eve",
            "forward_reverse_guard_seconds",
            "branch_switch_guard_seconds",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class HardwareConfig:
    active_ratio: float = 0.25
    maximum_active_gain: float = 4.0
    phase_coupling_mode: PhaseCouplingMode = "quadrature"
    passive_transmission_insertion_loss_db: float = 1.0
    passive_reflection_insertion_loss_db: float = 1.0
    static_gain_error_std_db: float = 0.20
    directional_gain_error_std_db: float = 0.10
    static_phase_error_std_rad: float = 0.05
    directional_phase_error_std_rad: float = 0.03
    fast_phase_jitter_std_rad: float = 0.01
    fast_jitter_forward_reverse_correlation: float = 0.0
    transmission_split_error_std: float = 0.02
    transmission_amplitude_phase_coupling_rad_per_db: float = 0.01
    reflection_amplitude_phase_coupling_rad_per_db: float = 0.01
    phase_quantization_bits: int | None = 4
    gain_quantization_bits: int | None = 4
    per_active_element_saturation_power: float = 5.0e-7

    def validate(self) -> None:
        if not 0.0 <= self.active_ratio <= 1.0:
            raise ValueError("active_ratio must lie in [0, 1]")
        if self.maximum_active_gain < 1.0:
            raise ValueError("maximum_active_gain must be at least one")
        if self.phase_coupling_mode not in {"independent", "quadrature", "hybrid"}:
            raise ValueError("invalid phase_coupling_mode")
        for name in (
            "passive_transmission_insertion_loss_db",
            "passive_reflection_insertion_loss_db",
            "static_gain_error_std_db",
            "directional_gain_error_std_db",
            "static_phase_error_std_rad",
            "directional_phase_error_std_rad",
            "fast_phase_jitter_std_rad",
            "transmission_split_error_std",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if not -1.0 <= self.fast_jitter_forward_reverse_correlation <= 1.0:
            raise ValueError("fast jitter correlation must lie in [-1, 1]")
        if self.phase_quantization_bits is not None and self.phase_quantization_bits < 1:
            raise ValueError("phase_quantization_bits must be positive or None")
        if self.gain_quantization_bits is not None and self.gain_quantization_bits < 1:
            raise ValueError("gain_quantization_bits must be positive or None")
        if self.per_active_element_saturation_power <= 0.0:
            raise ValueError("per_active_element_saturation_power must be positive")


@dataclass(frozen=True)
class PowerConfig:
    maximum_rf_output_power: float = 1.0e-6
    maximum_total_dc_power: float = 1.0
    amplifier_efficiency: float = 0.35
    controller_static_power: float = 0.10
    passive_element_control_power: float = 0.001
    active_element_control_power: float = 0.005
    active_element_bias_power: float = 0.01
    switching_network_static_power: float = 0.05
    controller_time_fraction: float = 1.0 / 3.0
    transmission_time_fraction: float = 1.0 / 3.0
    reflection_time_fraction: float = 1.0 / 3.0
    projection_iterations: int = 50
    projection_tolerance: float = 1.0e-9
    csi_power_margin_std: float = 2.0
    hardware_gain_margin_db: float = 0.60

    def validate(self) -> None:
        if self.maximum_rf_output_power <= 0.0 or self.maximum_total_dc_power <= 0.0:
            raise ValueError("power budgets must be positive")
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
        if self.projection_iterations < 1 or self.projection_tolerance < 0.0:
            raise ValueError("invalid projection settings")


@dataclass(frozen=True)
class KeyGenerationConfig:
    feature: FeatureName = "real"
    guard_band_sigma: float = 0.10
    selection_policy: SelectionPolicy = "alice"
    reconciliation_efficiency: float = 1.15
    initial_block_size: int = 8
    reconciliation_passes: int = 8
    maximum_block_doublings: int = 3
    verification_tag_bits: int = 32
    privacy_margin_bits: int = 32
    maximum_final_key_bits: int = 256
    epsilon_security: float = 1.0e-9
    public_channel_rate_bps: float = 1.0e6
    fixed_processing_delay_seconds: float = 0.0

    def validate(self) -> None:
        if self.feature not in {"real", "imag", "magnitude", "phase"}:
            raise ValueError("invalid feature")
        if self.selection_policy not in {"alice", "intersection"}:
            raise ValueError("invalid selection_policy")
        if self.guard_band_sigma < 0.0:
            raise ValueError("guard_band_sigma cannot be negative")
        if self.reconciliation_efficiency < 1.0:
            raise ValueError("reconciliation_efficiency must be at least one")
        if self.initial_block_size < 2 or self.reconciliation_passes < 1:
            raise ValueError("invalid reconciliation settings")
        if self.maximum_block_doublings < 0:
            raise ValueError("maximum_block_doublings cannot be negative")
        if not 1 <= self.verification_tag_bits <= 256:
            raise ValueError("verification_tag_bits must lie in [1, 256]")
        if self.privacy_margin_bits < 0 or self.maximum_final_key_bits < 1:
            raise ValueError("invalid privacy-amplification settings")
        if not 0.0 < self.epsilon_security < 1.0:
            raise ValueError("epsilon_security must lie in (0, 1)")
        if self.public_channel_rate_bps <= 0.0:
            raise ValueError("public_channel_rate_bps must be positive")


@dataclass(frozen=True)
class ObjectiveConfig:
    key_rate_weight: float = 1.0
    key_margin_weight: float = 0.0
    worst_branch_key_margin_weight: float = 0.0
    branch_imbalance_penalty_weight: float = 0.0
    raw_kdr_weight: float = 0.25
    post_reconciliation_kdr_weight: float = 0.50
    reciprocity_weight: float = 0.20
    surface_power_weight: float = 0.10
    projection_penalty_weight: float = 0.0
    constraint_violation_weight: float = 10.0
    key_rate_reference_bps: float = 10_000.0
    key_margin_reference_bits: float = 256.0
    raw_kdr_reference: float = 0.50
    post_reconciliation_kdr_reference: float = 0.01
    surface_power_reference_watt: float = 1.0
    transmission_weight: float = 0.5
    reflection_weight: float = 0.5

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value < 0.0:
                raise ValueError(f"{item.name} cannot be negative")
        if self.transmission_weight + self.reflection_weight <= 0.0:
            raise ValueError("branch weights cannot both be zero")
        for name in (
            "key_rate_reference_bps",
            "key_margin_reference_bits",
            "raw_kdr_reference",
            "post_reconciliation_kdr_reference",
            "surface_power_reference_watt",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

@dataclass(frozen=True)
class RobustConfig:
    objective_samples: int = 16
    cvar_alpha: float = 0.25
    mean_weight: float = 0.5
    cvar_weight: float = 0.5

    def validate(self) -> None:
        if self.objective_samples < 2:
            raise ValueError("objective_samples must be at least two")
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError("cvar_alpha must lie in (0, 1]")
        if self.mean_weight < 0.0 or self.cvar_weight < 0.0:
            raise ValueError("robust weights cannot be negative")
        if self.mean_weight + self.cvar_weight <= 0.0:
            raise ValueError("robust weights cannot both be zero")


@dataclass(frozen=True)
class EnvironmentConfig:
    architecture: Architecture = "partially_active_fixed"
    episode_length: int = 100
    seed: int = 0

    def validate(self) -> None:
        if self.architecture not in {
            "passive",
            "partially_active_fixed",
            "partially_active_dynamic",
            "fully_active_fixed",
        }:
            raise ValueError("invalid architecture")
        if self.episode_length < 1:
            raise ValueError("episode_length must be positive")


@dataclass(frozen=True)
class TD3Config:
    hidden_dimensions: list[int] = field(default_factory=lambda: [256, 256])
    actor_learning_rate: float = 1.0e-4
    critic_learning_rate: float = 1.0e-3
    discount_factor: float = 0.99
    target_update_rate: float = 0.005
    policy_noise: float = 0.20
    noise_clip: float = 0.50
    policy_delay: int = 2
    replay_capacity: int = 200_000
    batch_size: int = 256
    warmup_steps: int = 2_000
    exploration_noise: float = 0.10

    def validate(self) -> None:
        if not self.hidden_dimensions or any(x < 1 for x in self.hidden_dimensions):
            raise ValueError("hidden_dimensions must be positive")
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must lie in [0, 1]")
        if not 0.0 < self.target_update_rate <= 1.0:
            raise ValueError("target_update_rate must lie in (0, 1]")
        if self.policy_delay < 1 or self.replay_capacity < 1 or self.batch_size < 1:
            raise ValueError("invalid TD3 integer settings")


@dataclass(frozen=True)
class ExperimentConfig:
    training_steps: int = 100_000
    evaluation_episodes: int = 100
    seeds: list[int] = field(default_factory=lambda: list(range(8)))
    architectures: list[Architecture] = field(
        default_factory=lambda: [
            "passive",
            "partially_active_fixed",
            "partially_active_dynamic",
            "fully_active_fixed",
        ]
    )

    def validate(self) -> None:
        if self.training_steps < 1 or self.evaluation_episodes < 1:
            raise ValueError("experiment lengths must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        for architecture in self.architectures:
            EnvironmentConfig(architecture=architecture).validate()


@dataclass(frozen=True)
class FullSchemeConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    probing: ProbingConfig = field(default_factory=ProbingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    key_generation: KeyGenerationConfig = field(default_factory=KeyGenerationConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    robust: RobustConfig = field(default_factory=RobustConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    td3: TD3Config = field(default_factory=TD3Config)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if hasattr(value, "validate"):
                value.validate()
        if self.geometry.num_elements < 2:
            raise ValueError("the RIS must have at least two elements")


T = TypeVar("T")


def _convert_dataclass(cls: type[T], data: dict[str, Any]) -> T:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in data:
            continue
        value = data[item.name]
        expected = hints.get(item.name)
        if isinstance(expected, type) and is_dataclass(expected) and isinstance(value, dict):
            value = _convert_dataclass(expected, value)
        kwargs[item.name] = value
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> FullSchemeConfig:
    config = _convert_dataclass(FullSchemeConfig, data)
    config.validate()
    return config


def load_config(path: str | Path) -> FullSchemeConfig:
    content = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError("configuration root must be a mapping")
    return config_from_dict(content)


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _to_plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def save_config(config: FullSchemeConfig, path: str | Path) -> None:
    config.validate()
    Path(path).write_text(
        yaml.safe_dump(_to_plain(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
