from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]
BitArray = NDArray[np.uint8]


@dataclass(frozen=True)
class ChannelSnapshot:
    controller_to_ris: ComplexArray
    ris_to_transmission: ComplexArray
    ris_to_reflection: ComplexArray
    direct_transmission: complex
    direct_reflection: complex


@dataclass(frozen=True)
class BidirectionalChannelBlock:
    controller_to_ris_forward: ComplexArray
    ris_to_transmission_forward: ComplexArray
    ris_to_reflection_forward: ComplexArray
    direct_transmission_forward: ComplexArray
    direct_reflection_forward: ComplexArray

    transmission_to_ris_reverse: ComplexArray
    reflection_to_ris_reverse: ComplexArray
    ris_to_controller_reverse: ComplexArray
    direct_transmission_reverse: ComplexArray
    direct_reflection_reverse: ComplexArray


@dataclass(frozen=True)
class IdealSurfaceCommand:
    gain: FloatArray
    beta_transmission: FloatArray
    phase_transmission: FloatArray
    phase_reflection: FloatArray
    active_mask: BoolArray


@dataclass(frozen=True)
class ActualSurfaceCoefficients:
    gain_forward: FloatArray
    gain_reverse: FloatArray
    beta_transmission: FloatArray
    beta_reflection: FloatArray

    transmission_forward: ComplexArray
    reflection_forward: ComplexArray
    transmission_reverse: ComplexArray
    reflection_reverse: ComplexArray


@dataclass(frozen=True)
class EndpointRFRealization:
    controller_tx: complex
    controller_rx: complex
    transmission_tx: complex
    transmission_rx: complex
    reflection_tx: complex
    reflection_rx: complex


@dataclass(frozen=True)
class HardwareStaticRealization:
    gain_error_common_db: FloatArray
    gain_error_forward_db: FloatArray
    gain_error_reverse_db: FloatArray

    phase_error_transmission_common: FloatArray
    phase_error_reflection_common: FloatArray
    phase_error_forward: FloatArray
    phase_error_reverse: FloatArray

    beta_error: FloatArray
    endpoint_rf: EndpointRFRealization


@dataclass(frozen=True)
class BranchProbingResult:
    observation_forward: ComplexArray
    observation_reverse: ComplexArray
    effective_channel_forward: ComplexArray
    effective_channel_reverse: ComplexArray
    forwarded_active_noise_forward: ComplexArray
    forwarded_active_noise_reverse: ComplexArray


@dataclass(frozen=True)
class DualProbingResult:
    transmission: BranchProbingResult
    reflection: BranchProbingResult


@dataclass(frozen=True)
class PowerResult:
    rf_output_controller: float
    rf_output_transmission: float
    rf_output_reflection: float
    maximum_rf_output: float
    average_rf_output: float

    additional_rf_power_average: float
    amplifier_dc_power: float
    total_surface_dc_power: float

    rf_violation: float
    dc_violation: float
    per_element_saturation_violation: float

    rf_feasible: bool
    dc_feasible: bool
    saturation_feasible: bool
    fully_feasible: bool


@dataclass(frozen=True)
class KeyRateResult:
    total_samples: int
    retained_samples: int
    retention_ratio: float

    raw_kdr: float
    post_reconciliation_kdr: float

    estimated_entropy_bits: int
    reconciliation_leakage_bits: int
    verification_leakage_bits: int
    public_communication_bits: int

    training_secret_bits: int
    final_key_bits: int
    frame_duration_seconds: float
    training_key_rate_bps: float
    final_key_rate_bps: float

    verification_passed: bool
    success: bool


@dataclass(frozen=True)
class ObjectiveResult:
    reward: float

    theoretical_mutual_information_bits_per_sample: float
    training_key_rate_bps: float
    final_key_rate_bps: float

    raw_kdr: float
    post_reconciliation_kdr: float
    reciprocity: float
    retention_ratio: float
    success_rate: float

    normalized_key_rate: float
    normalized_kdr: float
    normalized_power: float
    normalized_constraint_violation: float

    transmission_key: KeyRateResult
    reflection_key: KeyRateResult
    power: PowerResult
    probing: DualProbingResult

    system_training_key_rate_bps: float
    system_final_key_rate_bps: float
