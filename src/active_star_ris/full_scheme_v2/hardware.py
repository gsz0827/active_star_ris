from __future__ import annotations

import numpy as np

from .config import HardwareConfig
from .models import (
    ActualSurfaceCoefficients,
    EndpointRFRealization,
    HardwareStaticRealization,
    IdealSurfaceCommand,
)


def build_active_mask(num_elements: int, active_ratio: float) -> np.ndarray:
    if num_elements < 1:
        raise ValueError("num_elements must be positive")
    if not 0.0 <= active_ratio <= 1.0:
        raise ValueError("active_ratio must lie in [0, 1]")

    num_active = int(round(num_elements * active_ratio))
    mask = np.zeros(num_elements, dtype=bool)
    if num_active == 0:
        return mask
    indices = np.linspace(0, num_elements - 1, num_active, dtype=int)
    mask[np.unique(indices)] = True
    return mask


def action_dimension(num_elements: int, active_mask: np.ndarray) -> int:
    return int(np.count_nonzero(active_mask) + 3 * num_elements)


def decode_action(
    action: np.ndarray,
    active_mask: np.ndarray,
    config: HardwareConfig,
) -> IdealSurfaceCommand:
    config.validate()
    action_array = np.asarray(action, dtype=np.float64).reshape(-1)
    n = active_mask.size
    num_active = int(np.count_nonzero(active_mask))
    expected = num_active + 3 * n
    if action_array.size != expected:
        raise ValueError(f"action has length {action_array.size}; expected {expected}")

    action_array = np.clip(action_array, -1.0, 1.0)
    cursor = 0

    active_gain_action = action_array[cursor : cursor + num_active]
    cursor += num_active
    phase_t_action = action_array[cursor : cursor + n]
    cursor += n
    phase_r_action = action_array[cursor : cursor + n]
    cursor += n
    beta_action = action_array[cursor : cursor + n]

    gain = np.ones(n, dtype=np.float64)
    if num_active:
        gain[active_mask] = 1.0 + 0.5 * (active_gain_action + 1.0) * (
            config.maximum_active_gain - 1.0
        )

    phase_t = np.mod(
        np.pi * (phase_t_action + 1.0),
        2.0 * np.pi,
    )

    phase_r_independent = np.mod(
        np.pi * (phase_r_action + 1.0),
        2.0 * np.pi,
    )

    branch_sign = np.where(
        phase_r_action >= 0.0,
        1.0,
        -1.0,
    )

    phase_r_quadrature = np.mod(
        phase_t + branch_sign * np.pi / 2.0,
        2.0 * np.pi,
    )

    if config.phase_coupling_mode == "independent":
        phase_r = phase_r_independent
    elif config.phase_coupling_mode == "quadrature":
        phase_r = phase_r_quadrature
    elif config.phase_coupling_mode == "hybrid":
        phase_r = np.where(
            active_mask,
            phase_r_independent,
            phase_r_quadrature,
        )
    else:
        raise ValueError(
            f"unsupported phase coupling mode: "
            f"{config.phase_coupling_mode}"
        )

    beta_t = np.clip(0.5 * (beta_action + 1.0), 0.0, 1.0)

    return IdealSurfaceCommand(
        gain=gain,
        beta_transmission=beta_t,
        phase_transmission=phase_t,
        phase_reflection=phase_r,
        active_mask=np.asarray(active_mask, dtype=bool),
    )


def _sample_complex_coefficient(
    rng: np.random.Generator,
    gain_std_db: float,
    phase_std_rad: float,
) -> complex:
    gain_db = float(rng.normal(0.0, gain_std_db))
    phase = float(rng.normal(0.0, phase_std_rad))
    return complex(10.0 ** (gain_db / 20.0) * np.exp(1j * phase))


def sample_static_hardware(
    num_elements: int,
    config: HardwareConfig,
    rng: np.random.Generator,
) -> HardwareStaticRealization:
    config.validate()
    if num_elements < 1:
        raise ValueError("num_elements must be positive")

    endpoint = EndpointRFRealization(
        controller_tx=_sample_complex_coefficient(
            rng,
            config.endpoint_gain_error_std_db,
            config.endpoint_phase_error_std_rad,
        ),
        controller_rx=_sample_complex_coefficient(
            rng,
            config.endpoint_gain_error_std_db,
            config.endpoint_phase_error_std_rad,
        ),
        transmission_tx=_sample_complex_coefficient(
            rng,
            config.endpoint_gain_error_std_db,
            config.endpoint_phase_error_std_rad,
        ),
        transmission_rx=_sample_complex_coefficient(
            rng,
            config.endpoint_gain_error_std_db,
            config.endpoint_phase_error_std_rad,
        ),
        reflection_tx=_sample_complex_coefficient(
            rng,
            config.endpoint_gain_error_std_db,
            config.endpoint_phase_error_std_rad,
        ),
        reflection_rx=_sample_complex_coefficient(
            rng,
            config.endpoint_gain_error_std_db,
            config.endpoint_phase_error_std_rad,
        ),
    )

    return HardwareStaticRealization(
        gain_error_common_db=np.asarray(
            rng.normal(0.0, config.static_gain_error_std_db, size=num_elements),
            dtype=np.float64,
        ),
        gain_error_forward_db=np.asarray(
            rng.normal(0.0, config.directional_gain_error_std_db, size=num_elements),
            dtype=np.float64,
        ),
        gain_error_reverse_db=np.asarray(
            rng.normal(0.0, config.directional_gain_error_std_db, size=num_elements),
            dtype=np.float64,
        ),
        phase_error_transmission_common=np.asarray(
            rng.normal(0.0, config.static_phase_error_std_rad, size=num_elements),
            dtype=np.float64,
        ),
        phase_error_reflection_common=np.asarray(
            rng.normal(0.0, config.static_phase_error_std_rad, size=num_elements),
            dtype=np.float64,
        ),
        phase_error_forward=np.asarray(
            rng.normal(0.0, config.directional_phase_error_std_rad, size=num_elements),
            dtype=np.float64,
        ),
        phase_error_reverse=np.asarray(
            rng.normal(0.0, config.directional_phase_error_std_rad, size=num_elements),
            dtype=np.float64,
        ),
        beta_error=np.asarray(
            rng.normal(0.0, config.transmission_split_error_std, size=num_elements),
            dtype=np.float64,
        ),
        endpoint_rf=endpoint,
    )


def quantize_phase(phase: np.ndarray, bits: int | None) -> np.ndarray:
    wrapped = np.mod(np.asarray(phase, dtype=np.float64), 2.0 * np.pi)
    if bits is None:
        return wrapped
    levels = 2**bits
    step = 2.0 * np.pi / levels
    return np.mod(np.round(wrapped / step) * step, 2.0 * np.pi)


def quantize_gain_downward(
    gain: np.ndarray,
    maximum_gain: float,
    bits: int | None,
) -> np.ndarray:
    values = np.clip(np.asarray(gain, dtype=np.float64), 1.0, maximum_gain)
    if bits is None:
        return values
    levels = 2**bits
    if levels <= 1 or maximum_gain <= 1.0:
        return np.ones_like(values)
    normalized = (values - 1.0) / (maximum_gain - 1.0)
    index = np.floor(normalized * (levels - 1) + 1.0e-12)
    return 1.0 + index / (levels - 1) * (maximum_gain - 1.0)


def apply_hardware(
    command: IdealSurfaceCommand,
    static: HardwareStaticRealization,
    config: HardwareConfig,
    rng: np.random.Generator,
) -> ActualSurfaceCoefficients:
    n = command.gain.size
    for array in (
        static.gain_error_common_db,
        static.gain_error_forward_db,
        static.gain_error_reverse_db,
        static.phase_error_transmission_common,
        static.phase_error_reflection_common,
        static.phase_error_forward,
        static.phase_error_reverse,
        static.beta_error,
    ):
        if array.size != n:
            raise ValueError("hardware realization size mismatch")

    active = np.asarray(
        command.active_mask,
        dtype=bool,
    )

    # 1. 先量化数字控制命令
    command_gain = quantize_gain_downward(
        command.gain,
        config.maximum_active_gain,
        config.gain_quantization_bits,
    )
    command_gain = np.where(
        active,
        command_gain,
        1.0,
    )

    command_phase_t = quantize_phase(
        command.phase_transmission,
        config.phase_quantization_bits,
    )
    command_phase_r = quantize_phase(
        command.phase_reflection,
        config.phase_quantization_bits,
    )

    # 2. 模拟硬件在量化命令的基础上产生增益误差
    gain_forward = command_gain * 10.0 ** (
        (
            static.gain_error_common_db
            + static.gain_error_forward_db
        )
        / 20.0
    )

    gain_reverse = command_gain * 10.0 ** (
        (
            static.gain_error_common_db
            + static.gain_error_reverse_db
        )
        / 20.0
    )

    gain_forward = np.where(
        active,
        np.clip(
            gain_forward,
            1.0,
            config.maximum_active_gain,
        ),
        1.0,
    )

    gain_reverse = np.where(
        active,
        np.clip(
            gain_reverse,
            1.0,
            config.maximum_active_gain,
        ),
        1.0,
    )

    beta_t = np.clip(
        command.beta_transmission + static.beta_error,
        0.0,
        1.0,
    )
    beta_r = 1.0 - beta_t

    jitter_t_forward = rng.normal(
        0.0,
        config.fast_phase_jitter_std_rad,
        size=n,
    )
    jitter_r_forward = rng.normal(
        0.0,
        config.fast_phase_jitter_std_rad,
        size=n,
    )
    jitter_t_reverse = rng.normal(
        0.0,
        config.fast_phase_jitter_std_rad,
        size=n,
    )
    jitter_r_reverse = rng.normal(
        0.0,
        config.fast_phase_jitter_std_rad,
        size=n,
    )

    # 使用实际方向增益计算幅相耦合
    gain_forward_db = 20.0 * np.log10(
        np.maximum(gain_forward, 1.0)
    )
    gain_reverse_db = 20.0 * np.log10(
        np.maximum(gain_reverse, 1.0)
    )

    coupling_t_forward = (
        config.transmission_amplitude_phase_coupling_rad_per_db
        * gain_forward_db
    )
    coupling_t_reverse = (
        config.transmission_amplitude_phase_coupling_rad_per_db
        * gain_reverse_db
    )
    coupling_r_forward = (
        config.reflection_amplitude_phase_coupling_rad_per_db
        * gain_forward_db
    )
    coupling_r_reverse = (
        config.reflection_amplitude_phase_coupling_rad_per_db
        * gain_reverse_db
    )

    # 3. 量化后不再二次量化，直接加入模拟硬件误差
    phase_t_forward = np.mod(
        command_phase_t
        + static.phase_error_transmission_common
        + static.phase_error_forward
        + coupling_t_forward
        + jitter_t_forward,
        2.0 * np.pi,
    )

    phase_r_forward = np.mod(
        command_phase_r
        + static.phase_error_reflection_common
        + static.phase_error_forward
        + coupling_r_forward
        + jitter_r_forward,
        2.0 * np.pi,
    )

    phase_t_reverse = np.mod(
        command_phase_t
        + static.phase_error_transmission_common
        + static.phase_error_reverse
        + coupling_t_reverse
        + jitter_t_reverse,
        2.0 * np.pi,
    )

    phase_r_reverse = np.mod(
        command_phase_r
        + static.phase_error_reflection_common
        + static.phase_error_reverse
        + coupling_r_reverse
        + jitter_r_reverse,
        2.0 * np.pi,
    )

    passive_amplitude_t = 10.0 ** (
        -config.passive_transmission_insertion_loss_db / 20.0
    )
    passive_amplitude_r = 10.0 ** (
        -config.passive_reflection_insertion_loss_db / 20.0
    )

    amplitude_t_forward = np.where(
        active,
        gain_forward,
        passive_amplitude_t,
    )
    amplitude_r_forward = np.where(
        active,
        gain_forward,
        passive_amplitude_r,
    )
    amplitude_t_reverse = np.where(
        active,
        gain_reverse,
        passive_amplitude_t,
    )
    amplitude_r_reverse = np.where(
        active,
        gain_reverse,
        passive_amplitude_r,
    )

    transmission_forward = (
        amplitude_t_forward
        * np.sqrt(beta_t)
        * np.exp(1j * phase_t_forward)
    )

    reflection_forward = (
        amplitude_r_forward
        * np.sqrt(beta_r)
        * np.exp(1j * phase_r_forward)
    )

    transmission_reverse = (
        amplitude_t_reverse
        * np.sqrt(beta_t)
        * np.exp(1j * phase_t_reverse)
    )

    reflection_reverse = (
        amplitude_r_reverse
        * np.sqrt(beta_r)
        * np.exp(1j * phase_r_reverse)
    )

    return ActualSurfaceCoefficients(
        gain_forward=np.asarray(gain_forward, dtype=np.float64),
        gain_reverse=np.asarray(gain_reverse, dtype=np.float64),
        beta_transmission=np.asarray(beta_t, dtype=np.float64),
        beta_reflection=np.asarray(beta_r, dtype=np.float64),
        transmission_forward=np.asarray(transmission_forward, dtype=np.complex128),
        reflection_forward=np.asarray(reflection_forward, dtype=np.complex128),
        transmission_reverse=np.asarray(transmission_reverse, dtype=np.complex128),
        reflection_reverse=np.asarray(reflection_reverse, dtype=np.complex128),
    )
