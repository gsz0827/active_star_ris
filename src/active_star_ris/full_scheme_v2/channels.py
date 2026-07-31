from __future__ import annotations

from dataclasses import replace

import numpy as np

from .fast_ar import evolve_gauss_markov
from numpy.typing import ArrayLike

from .config import ChannelConfig, GeometryConfig
from .models import CSIResult, CSIState, ComplexArray, StaticChannels

SPEED_OF_LIGHT = 299_792_458.0


def complex_normal(rng: np.random.Generator, shape: tuple[int, ...] | int | None = None) -> np.ndarray:
    return (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ) / np.sqrt(2.0)


def _position(value: ArrayLike) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError("positions must be three-dimensional")
    return result


def distance(a: ArrayLike, b: ArrayLike) -> float:
    return float(np.linalg.norm(_position(a) - _position(b)))


def reference_free_space_loss_power(config: GeometryConfig) -> float:
    wavelength = SPEED_OF_LIGHT / config.carrier_frequency_hz
    return float((wavelength / (4.0 * np.pi * config.reference_distance_m)) ** 2)


def path_loss_power(
    distance_m: float,
    *,
    exponent: float,
    config: GeometryConfig,
    additional_loss_db: float,
) -> float:
    effective_distance = max(distance_m, config.reference_distance_m)
    reference = reference_free_space_loss_power(config)
    distance_factor = (effective_distance / config.reference_distance_m) ** (-exponent)
    extra = 10.0 ** (-additional_loss_db / 10.0)
    return float(reference * distance_factor * extra)


def ris_element_positions(config: GeometryConfig) -> np.ndarray:
    wavelength = SPEED_OF_LIGHT / config.carrier_frequency_hz
    spacing = config.element_spacing_wavelengths * wavelength
    rows = np.arange(config.ris_rows, dtype=np.float64)
    columns = np.arange(config.ris_columns, dtype=np.float64)
    row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")
    row_grid -= np.mean(rows)
    column_grid -= np.mean(columns)
    # RIS位于y-z平面，法向沿x轴。
    offsets = np.stack(
        [
            np.zeros(row_grid.size),
            column_grid.reshape(-1) * spacing,
            row_grid.reshape(-1) * spacing,
        ],
        axis=1,
    )
    return offsets + _position(config.ris_position_m)[None, :]


def steering_vector(node_position: ArrayLike, config: GeometryConfig) -> ComplexArray:
    node = _position(node_position)
    ris = _position(config.ris_position_m)
    direction = node - ris
    norm = np.linalg.norm(direction)
    if norm <= 0.0:
        raise ValueError("node cannot coincide with RIS center")
    unit = direction / norm
    relative = ris_element_positions(config) - ris[None, :]
    wavelength = SPEED_OF_LIGHT / config.carrier_frequency_hz
    phase = 2.0 * np.pi / wavelength * (relative @ unit)
    return np.asarray(np.exp(1j * phase), dtype=np.complex128)


def rician_vector(
    node_position: ArrayLike,
    *,
    geometry: GeometryConfig,
    channel: ChannelConfig,
    rng: np.random.Generator,
) -> ComplexArray:
    loss = path_loss_power(
        distance(node_position, geometry.ris_position_m),
        exponent=geometry.path_loss_exponent_ris,
        config=geometry,
        additional_loss_db=geometry.additional_ris_loss_db,
    )
    k = channel.rician_k_factor
    los = steering_vector(node_position, geometry)
    nlos = complex_normal(rng, geometry.num_elements)
    result = np.sqrt(loss) * (
        np.sqrt(k / (k + 1.0)) * los
        + np.sqrt(1.0 / (k + 1.0)) * nlos
    )
    return np.asarray(result, dtype=np.complex128)


def direct_scalar(
    a: ArrayLike,
    b: ArrayLike,
    *,
    geometry: GeometryConfig,
    rng: np.random.Generator,
) -> complex:
    loss = path_loss_power(
        distance(a, b),
        exponent=geometry.path_loss_exponent_direct,
        config=geometry,
        additional_loss_db=geometry.additional_direct_loss_db,
    )
    return complex(np.sqrt(loss) * complex_normal(rng))


def correlated_eve_channel(
    legitimate: ComplexArray,
    independent: ComplexArray,
    correlation: float,
) -> ComplexArray:
    rho = float(np.clip(correlation, 0.0, 1.0))
    legitimate_power = float(np.mean(np.abs(legitimate) ** 2))
    independent_power = float(np.mean(np.abs(independent) ** 2))
    normalized_legitimate = legitimate / np.sqrt(max(legitimate_power, 1.0e-30))
    normalized_independent = independent / np.sqrt(max(independent_power, 1.0e-30))
    target_power = independent_power
    result = np.sqrt(target_power) * (
        rho * normalized_legitimate
        + np.sqrt(max(0.0, 1.0 - rho**2)) * normalized_independent
    )
    return np.asarray(result, dtype=np.complex128)


def sample_static_channels(
    geometry: GeometryConfig,
    channel: ChannelConfig,
    rng: np.random.Generator,
) -> StaticChannels:
    controller = rician_vector(
        geometry.controller_position_m,
        geometry=geometry,
        channel=channel,
        rng=rng,
    )
    transmission = rician_vector(
        geometry.transmission_user_position_m,
        geometry=geometry,
        channel=channel,
        rng=rng,
    )
    reflection = rician_vector(
        geometry.reflection_user_position_m,
        geometry=geometry,
        channel=channel,
        rng=rng,
    )
    eve_t_independent = rician_vector(
        geometry.eve_transmission_position_m,
        geometry=geometry,
        channel=channel,
        rng=rng,
    )
    eve_r_independent = rician_vector(
        geometry.eve_reflection_position_m,
        geometry=geometry,
        channel=channel,
        rng=rng,
    )
    if channel.eve_enabled:
        eve_t = correlated_eve_channel(
            transmission,
            eve_t_independent,
            channel.eve_spatial_correlation,
        )
        eve_r = correlated_eve_channel(
            reflection,
            eve_r_independent,
            channel.eve_spatial_correlation,
        )
    else:
        eve_t = np.zeros_like(transmission)
        eve_r = np.zeros_like(reflection)

    if channel.eve_enabled:
        direct_controller_eve_transmission = direct_scalar(
            geometry.controller_position_m,
            geometry.eve_transmission_position_m,
            geometry=geometry,
            rng=rng,
        )
        direct_user_eve_transmission = direct_scalar(
            geometry.transmission_user_position_m,
            geometry.eve_transmission_position_m,
            geometry=geometry,
            rng=rng,
        )
        direct_controller_eve_reflection = direct_scalar(
            geometry.controller_position_m,
            geometry.eve_reflection_position_m,
            geometry=geometry,
            rng=rng,
        )
        direct_user_eve_reflection = direct_scalar(
            geometry.reflection_user_position_m,
            geometry.eve_reflection_position_m,
            geometry=geometry,
            rng=rng,
        )
    else:
        direct_controller_eve_transmission = 0.0j
        direct_user_eve_transmission = 0.0j
        direct_controller_eve_reflection = 0.0j
        direct_user_eve_reflection = 0.0j

    return StaticChannels(
        controller_ris=controller,
        ris_transmission=transmission,
        ris_reflection=reflection,
        ris_eve_transmission=eve_t,
        ris_eve_reflection=eve_r,
        direct_transmission=direct_scalar(
            geometry.controller_position_m,
            geometry.transmission_user_position_m,
            geometry=geometry,
            rng=rng,
        ),
        direct_reflection=direct_scalar(
            geometry.controller_position_m,
            geometry.reflection_user_position_m,
            geometry=geometry,
            rng=rng,
        ),
        direct_controller_eve_transmission=(
            direct_controller_eve_transmission
        ),
        direct_user_eve_transmission=(
            direct_user_eve_transmission
        ),
        direct_controller_eve_reflection=(
            direct_controller_eve_reflection
        ),
        direct_user_eve_reflection=(
            direct_user_eve_reflection
        ),
    )


def gauss_markov_update(
    value: ArrayLike,
    correlation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    current = np.asarray(value, dtype=np.complex128)

    if not np.any(current):
        return np.zeros_like(current)

    rho = float(np.clip(correlation, 0.0, 1.0))
    power = max(float(np.mean(np.abs(current) ** 2)), 1.0e-30)
    innovation = complex_normal(rng, current.shape) * np.sqrt(power)
    return np.asarray(
        rho * current + np.sqrt(max(0.0, 1.0 - rho**2)) * innovation,
        dtype=np.complex128,
    )


def advance_static_channels(
    channels: StaticChannels,
    correlation: float,
    rng: np.random.Generator,
) -> StaticChannels:
    def vector(value: ComplexArray) -> ComplexArray:
        return np.asarray(gauss_markov_update(value, correlation, rng), dtype=np.complex128)

    def scalar(value: complex) -> complex:
        return complex(gauss_markov_update(np.asarray(value), correlation, rng))

    return replace(
        channels,
        controller_ris=vector(channels.controller_ris),
        ris_transmission=vector(channels.ris_transmission),
        ris_reflection=vector(channels.ris_reflection),
        ris_eve_transmission=vector(channels.ris_eve_transmission),
        ris_eve_reflection=vector(channels.ris_eve_reflection),
        direct_transmission=scalar(channels.direct_transmission),
        direct_reflection=scalar(channels.direct_reflection),
        direct_controller_eve_transmission=scalar(channels.direct_controller_eve_transmission),
        direct_user_eve_transmission=scalar(channels.direct_user_eve_transmission),
        direct_controller_eve_reflection=scalar(channels.direct_controller_eve_reflection),
        direct_user_eve_reflection=scalar(channels.direct_user_eve_reflection),
    )


def estimate_channel(
    true_channel: ArrayLike,
    config: ChannelConfig,
    rng: np.random.Generator,
) -> CSIResult:
    h = np.asarray(true_channel, dtype=np.complex128)
    if config.control_csi_model in {
        "nmse",
        "nmse_oracle",
    }:
        signal_power = float(np.mean(np.abs(h) ** 2))
        error_variance = signal_power * 10.0 ** (config.control_csi_nmse_db / 10.0)
        error = complex_normal(rng, h.shape) * np.sqrt(error_variance)
        estimate = h + error
        std = np.full(h.shape, np.sqrt(error_variance), dtype=np.float64)
        return CSIResult(np.asarray(estimate), std, 0)

    effective_pilot_energy = config.csi_pilot_symbols * config.csi_pilot_power
    noise_variance = config.csi_receiver_noise_variance
    noise = complex_normal(rng, h.shape) * np.sqrt(noise_variance)
    observation = np.sqrt(effective_pilot_energy) * h + noise
    if config.control_csi_model == "ls":
        estimate = observation / np.sqrt(effective_pilot_energy)
        error_variance = noise_variance / effective_pilot_energy
    else:
        prior_variance = max(float(np.mean(np.abs(h) ** 2)), 1.0e-30)
        gain = (
            np.sqrt(effective_pilot_energy) * prior_variance
            / (effective_pilot_energy * prior_variance + noise_variance)
        )
        estimate = gain * observation
        error_variance = (
            prior_variance * noise_variance
            / (effective_pilot_energy * prior_variance + noise_variance)
        )
    std = np.full(h.shape, np.sqrt(max(error_variance, 0.0)), dtype=np.float64)
    return CSIResult(np.asarray(estimate, dtype=np.complex128), std, config.csi_pilot_symbols)


def estimate_control_csi(
    channels: StaticChannels,
    config: ChannelConfig,
    rng: np.random.Generator,
) -> CSIState:
    return CSIState(
        controller_ris=estimate_channel(channels.controller_ris, config, rng),
        ris_transmission=estimate_channel(channels.ris_transmission, config, rng),
        ris_reflection=estimate_channel(channels.ris_reflection, config, rng),
        direct_transmission=estimate_channel(np.asarray([channels.direct_transmission]), config, rng),
        direct_reflection=estimate_channel(np.asarray([channels.direct_reflection]), config, rng),
    )


def evolve_block(
    initial: ArrayLike,
    samples: int,
    correlation: float,
    rng: np.random.Generator,
) -> ComplexArray:
    """Generate one time-correlated block with a compiled sequential recurrence.

    The number of samples and the channel law are unchanged. Random values are
    generated in the same real/imaginary order as the former Python loop.
    """
    if samples < 1:
        raise ValueError("samples must be positive")

    initial_array = np.asarray(initial, dtype=np.complex128)
    original_shape = initial_array.shape
    initial_flat = initial_array.reshape(-1)

    if samples == 1:
        return np.asarray(initial_array[None, ...], dtype=np.complex128)

    # The original update does not consume random numbers for an all-zero block.
    if not np.any(initial_flat):
        return np.zeros((samples,) + original_shape, dtype=np.complex128)

    # One batched call preserves the original RNG ordering:
    # real(t, all elements), imag(t, all elements), then t+1.
    standard = rng.normal(size=(samples - 1, 2, initial_flat.size))
    result_flat = evolve_gauss_markov(
        initial_flat,
        np.asarray(standard, dtype=np.float64),
        float(correlation),
    )
    return np.asarray(
        result_flat.reshape((samples,) + original_shape),
        dtype=np.complex128,
    )

def delayed_reverse_block(
    forward: ComplexArray,
    delay_correlation: float,
    rng: np.random.Generator,
) -> ComplexArray:
    power = np.maximum(
        np.mean(np.abs(forward) ** 2, axis=0, keepdims=True),
        1.0e-30,
    )
    innovation = complex_normal(rng, forward.shape) * np.sqrt(power)
    rho = float(np.clip(delay_correlation, 0.0, 1.0))
    return np.asarray(
        rho * forward + np.sqrt(max(0.0, 1.0 - rho**2)) * innovation,
        dtype=np.complex128,
    )
