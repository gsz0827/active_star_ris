from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class StaticChannels:
    controller_ris: ComplexArray
    ris_transmission: ComplexArray
    ris_reflection: ComplexArray
    ris_eve_transmission: ComplexArray
    ris_eve_reflection: ComplexArray
    direct_transmission: complex
    direct_reflection: complex
    direct_controller_eve_transmission: complex
    direct_user_eve_transmission: complex
    direct_controller_eve_reflection: complex
    direct_user_eve_reflection: complex


@dataclass(frozen=True)
class CSIResult:
    estimate: ComplexArray
    error_standard_deviation: FloatArray
    pilot_symbols: int


@dataclass(frozen=True)
class CSIState:
    controller_ris: CSIResult
    ris_transmission: CSIResult
    ris_reflection: CSIResult
    direct_transmission: CSIResult
    direct_reflection: CSIResult


@dataclass(frozen=True)
class IdealSurfaceAction:
    gain: FloatArray
    phase_transmission: FloatArray
    phase_reflection: FloatArray
    transmission_split: FloatArray
    active_mask: BoolArray


@dataclass(frozen=True)
class StaticHardwareState:
    common_gain_error_db: FloatArray
    forward_gain_error_db: FloatArray
    reverse_gain_error_db: FloatArray
    transmission_static_phase_error: FloatArray
    reflection_static_phase_error: FloatArray
    forward_directional_phase_error: FloatArray
    reverse_directional_phase_error: FloatArray


@dataclass(frozen=True)
class DirectionalSurfaceCoefficients:
    transmission_forward: ComplexArray
    transmission_reverse: ComplexArray
    reflection_forward: ComplexArray
    reflection_reverse: ComplexArray
    actual_gain_forward: FloatArray
    actual_gain_reverse: FloatArray
    actual_transmission_split: FloatArray


@dataclass(frozen=True)
class GainProjectionResult:
    projected_gain: FloatArray
    robust_rf_output_controller: float
    robust_rf_output_transmission: float
    robust_rf_output_reflection: float
    robust_total_dc_power: float
    unit_gain_feasible: bool
    projection_scale: float


@dataclass(frozen=True)
class BranchObservations:
    observation_alice: ComplexArray
    observation_bob: ComplexArray
    observation_eve_forward: ComplexArray
    observation_eve_reverse: ComplexArray
    effective_forward: ComplexArray
    effective_reverse: ComplexArray


@dataclass(frozen=True)
class BranchKeyMetrics:
    mutual_information_ab: float
    eve_information: float
    reciprocity: float
    raw_kdr: float
    retained_bits: int
    post_reconciliation_kdr: float
    public_leakage_bits: int
    finite_length_secret_bits: int
    final_key_bits: int
    final_keys_match: bool
    secure_key_rate_bps: float
    finite_penalty_bits: int = 0
    conditional_min_entropy_bits: int = 0
    key_margin_bits: float = 0.0


@dataclass(frozen=True)
class JointKeyMetrics:
    transmission: BranchKeyMetrics
    reflection: BranchKeyMetrics
    weighted_secure_key_rate_bps: float
    weighted_raw_kdr: float
    weighted_post_reconciliation_kdr: float
    weighted_reciprocity: float
    weighted_key_margin_bits: float = 0.0

@dataclass(frozen=True)
class PowerMetrics:
    rf_output_controller: float
    rf_output_transmission: float
    rf_output_reflection: float
    total_dc_power: float
    rf_violation: float
    dc_violation: float
    saturation_violation: float
    any_violation: bool


@dataclass(frozen=True)
class ObjectiveSample:
    reward: float
    key_metrics: JointKeyMetrics
    power_metrics: PowerMetrics
    active_elements: int
    projection_scale: float
    architecture_feasible: bool


@dataclass(frozen=True)
class RobustSummary:
    robust_reward: float
    mean_reward: float
    cvar_reward: float
    worst_reward: float
    mean_secure_key_rate_bps: float
    cvar_secure_key_rate_bps: float
    worst_secure_key_rate_bps: float
    mean_raw_kdr: float
    cvar_raw_kdr: float
    worst_raw_kdr: float
    mean_reciprocity: float
    cvar_reciprocity: float
    worst_reciprocity: float
    mean_surface_power_watt: float
    cvar_surface_power_watt: float
    worst_surface_power_watt: float
    power_violation_probability: float
    mean_active_elements: float
    mean_projection_scale: float
    mean_post_reconciliation_kdr: float = 0.0
    cvar_post_reconciliation_kdr: float = 0.0
    worst_post_reconciliation_kdr: float = 0.0
