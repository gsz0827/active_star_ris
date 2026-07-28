from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .config import Architecture, HardwareConfig
from .models import (
    BoolArray,
    DirectionalSurfaceCoefficients,
    FloatArray,
    IdealSurfaceAction,
    StaticHardwareState,
)


def action_dimension(num_elements: int) -> int:
    return 5 * num_elements


def fixed_active_mask(num_elements: int, active_ratio: float) -> BoolArray:
    count = int(round(num_elements * active_ratio))
    count = min(max(count, 0), num_elements)
    mask = np.zeros(num_elements, dtype=bool)
    if count == 0:
        return mask
    indices = np.linspace(0, num_elements - 1, count, dtype=int)
    mask[np.unique(indices)] = True
    if np.count_nonzero(mask) < count:
        missing = count - np.count_nonzero(mask)
        available = np.flatnonzero(~mask)
        mask[available[:missing]] = True
    return mask


def architecture_active_mask(
    architecture: Architecture,
    active_ratio: float,
    gate_action: ArrayLike,
) -> BoolArray:
    gates = np.asarray(gate_action, dtype=np.float64)
    n = gates.size
    if architecture == "passive":
        return np.zeros(n, dtype=bool)
    if architecture == "fully_active_fixed":
        return np.ones(n, dtype=bool)
    if architecture == "partially_active_fixed":
        return fixed_active_mask(n, active_ratio)
    count = int(round(n * active_ratio))
    count = min(max(count, 0), n)
    mask = np.zeros(n, dtype=bool)
    if count:
        selected = np.argpartition(gates, -count)[-count:]
        mask[selected] = True
    return mask


def _quantize(values: FloatArray, minimum: float, maximum: float, bits: int | None) -> FloatArray:
    clipped = np.clip(values, minimum, maximum)
    if bits is None:
        return np.asarray(clipped, dtype=np.float64)
    levels = 2**bits
    if levels <= 1 or maximum <= minimum:
        return np.full_like(clipped, minimum)
    step = (maximum - minimum) / (levels - 1)
    return np.asarray(minimum + np.round((clipped - minimum) / step) * step, dtype=np.float64)


def decode_action(
    action: ArrayLike,
    *,
    num_elements: int,
    architecture: Architecture,
    config: HardwareConfig,
) -> IdealSurfaceAction:
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size != action_dimension(num_elements):
        raise ValueError(f"action must contain {action_dimension(num_elements)} values")
    values = np.clip(values, -1.0, 1.0)
    gain_raw, phase_t_raw, phase_r_raw, split_raw, gate_raw = np.split(values, 5)
    active_mask = architecture_active_mask(architecture, config.active_ratio, gate_raw)
    requested_gain = 1.0 + 0.5 * (gain_raw + 1.0) * (config.maximum_active_gain - 1.0)
    requested_gain = _quantize(
        requested_gain,
        1.0,
        config.maximum_active_gain,
        config.gain_quantization_bits,
    )
    gain = np.where(active_mask, requested_gain, 1.0)
    phase_t = np.mod(np.pi * (phase_t_raw + 1.0), 2.0 * np.pi)
    phase_r_independent = np.mod(np.pi * (phase_r_raw + 1.0), 2.0 * np.pi)
    phase_t = _quantize(phase_t, 0.0, 2.0 * np.pi, config.phase_quantization_bits)
    phase_r_independent = _quantize(
        phase_r_independent,
        0.0,
        2.0 * np.pi,
        config.phase_quantization_bits,
    )
    coupled = np.mod(phase_t + np.pi / 2.0, 2.0 * np.pi)
    if config.phase_coupling_mode == "quadrature":
        phase_r = coupled
    elif config.phase_coupling_mode == "hybrid":
        phase_r = np.where(active_mask, phase_r_independent, coupled)
    else:
        phase_r = phase_r_independent
    transmission_split = np.clip(0.5 * (split_raw + 1.0), 0.0, 1.0)
    return IdealSurfaceAction(
        gain=np.asarray(gain, dtype=np.float64),
        phase_transmission=np.asarray(phase_t, dtype=np.float64),
        phase_reflection=np.asarray(phase_r, dtype=np.float64),
        transmission_split=np.asarray(transmission_split, dtype=np.float64),
        active_mask=np.asarray(active_mask, dtype=bool),
    )


def sample_static_hardware(
    num_elements: int,
    config: HardwareConfig,
    rng: np.random.Generator,
) -> StaticHardwareState:
    return StaticHardwareState(
        common_gain_error_db=np.asarray(
            rng.normal(0.0, config.static_gain_error_std_db, num_elements),
            dtype=np.float64,
        ),
        forward_gain_error_db=np.asarray(
            rng.normal(0.0, config.directional_gain_error_std_db, num_elements),
            dtype=np.float64,
        ),
        reverse_gain_error_db=np.asarray(
            rng.normal(0.0, config.directional_gain_error_std_db, num_elements),
            dtype=np.float64,
        ),
        transmission_static_phase_error=np.asarray(
            rng.normal(0.0, config.static_phase_error_std_rad, num_elements),
            dtype=np.float64,
        ),
        reflection_static_phase_error=np.asarray(
            rng.normal(0.0, config.static_phase_error_std_rad, num_elements),
            dtype=np.float64,
        ),
        forward_directional_phase_error=np.asarray(
            rng.normal(0.0, config.directional_phase_error_std_rad, num_elements),
            dtype=np.float64,
        ),
        reverse_directional_phase_error=np.asarray(
            rng.normal(0.0, config.directional_phase_error_std_rad, num_elements),
            dtype=np.float64,
        ),
    )


def _sample_fast_jitter(
    samples: int,
    elements: int,
    config: HardwareConfig,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    std = config.fast_phase_jitter_std_rad
    if std == 0.0:
        zeros = np.zeros((samples, elements), dtype=np.float64)
        return zeros, zeros.copy()
    forward = rng.normal(0.0, std, (samples, elements))
    independent = rng.normal(0.0, std, (samples, elements))
    rho = config.fast_jitter_forward_reverse_correlation
    reverse = rho * forward + np.sqrt(max(0.0, 1.0 - rho**2)) * independent
    return np.asarray(forward), np.asarray(reverse)


def realize_coefficients(
    ideal: IdealSurfaceAction,
    static: StaticHardwareState,
    *,
    samples: int,
    config: HardwareConfig,
    rng: np.random.Generator,
) -> DirectionalSurfaceCoefficients:
    n = ideal.gain.size
    forward_jitter, reverse_jitter = _sample_fast_jitter(samples, n, config, rng)
    active = ideal.active_mask
    passive_t = 10.0 ** (-config.passive_transmission_insertion_loss_db / 20.0)
    passive_r = 10.0 ** (-config.passive_reflection_insertion_loss_db / 20.0)
    gain_forward_db = 20.0 * np.log10(np.maximum(ideal.gain, 1.0e-12))
    gain_forward_db += static.common_gain_error_db + static.forward_gain_error_db
    gain_reverse_db = 20.0 * np.log10(np.maximum(ideal.gain, 1.0e-12))
    gain_reverse_db += static.common_gain_error_db + static.reverse_gain_error_db
    gain_forward = 10.0 ** (gain_forward_db / 20.0)
    gain_reverse = 10.0 ** (gain_reverse_db / 20.0)
    gain_forward = np.where(active, np.clip(gain_forward, 1.0, config.maximum_active_gain), 1.0)
    gain_reverse = np.where(active, np.clip(gain_reverse, 1.0, config.maximum_active_gain), 1.0)
    split = np.clip(
        ideal.transmission_split
        + rng.normal(0.0, config.transmission_split_error_std, n),
        0.0,
        1.0,
    )
    gain_error_forward_db = 20.0 * np.log10(np.maximum(gain_forward / ideal.gain, 1.0e-12))
    gain_error_reverse_db = 20.0 * np.log10(np.maximum(gain_reverse / ideal.gain, 1.0e-12))

    def coefficients(branch: str, direction: str) -> np.ndarray:
        if branch == "t":
            base_phase = ideal.phase_transmission
            static_phase = static.transmission_static_phase_error
            amplitude = np.sqrt(split)
            passive_loss = passive_t
            coupling = config.transmission_amplitude_phase_coupling_rad_per_db
        else:
            base_phase = ideal.phase_reflection
            static_phase = static.reflection_static_phase_error
            amplitude = np.sqrt(1.0 - split)
            passive_loss = passive_r
            coupling = config.reflection_amplitude_phase_coupling_rad_per_db
        if direction == "f":
            gain = gain_forward
            directional = static.forward_directional_phase_error
            jitter = forward_jitter
            gain_error_db = gain_error_forward_db
        else:
            gain = gain_reverse
            directional = static.reverse_directional_phase_error
            jitter = reverse_jitter
            gain_error_db = gain_error_reverse_db
        phase = (
            base_phase[None, :]
            + static_phase[None, :]
            + directional[None, :]
            + coupling * gain_error_db[None, :]
            + jitter
        )
        branch_gain = gain * amplitude
        branch_gain = np.where(active, branch_gain, passive_loss * amplitude)
        return np.asarray(branch_gain[None, :] * np.exp(1j * phase), dtype=np.complex128)

    return DirectionalSurfaceCoefficients(
        transmission_forward=coefficients("t", "f"),
        transmission_reverse=coefficients("t", "r"),
        reflection_forward=coefficients("r", "f"),
        reflection_reverse=coefficients("r", "r"),
        actual_gain_forward=np.asarray(gain_forward, dtype=np.float64),
        actual_gain_reverse=np.asarray(gain_reverse, dtype=np.float64),
        actual_transmission_split=np.asarray(split, dtype=np.float64),
    )
