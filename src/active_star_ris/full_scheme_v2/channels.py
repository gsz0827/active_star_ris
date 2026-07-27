from __future__ import annotations

import numpy as np

from .config import ChannelConfig
from .models import BidirectionalChannelBlock, ChannelSnapshot, ComplexArray


def complex_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...] | int,
    variance: float = 1.0,
) -> ComplexArray:
    if variance < 0.0:
        raise ValueError("variance cannot be negative")
    scale = np.sqrt(variance / 2.0)
    return np.asarray(
        scale
        * (
            rng.normal(size=shape)
            + 1j * rng.normal(size=shape)
        ),
        dtype=np.complex128,
    )


def sample_rician(
    rng: np.random.Generator,
    shape: tuple[int, ...] | int,
    *,
    average_power: float,
    k_factor: float,
) -> ComplexArray:
    if average_power < 0.0:
        raise ValueError("average_power cannot be negative")
    if k_factor < 0.0:
        raise ValueError("k_factor cannot be negative")

    if average_power == 0.0:
        return np.zeros(shape, dtype=np.complex128)

    los_phase = rng.uniform(0.0, 2.0 * np.pi, size=shape)
    los = np.exp(1j * los_phase)
    nlos = complex_normal(rng, shape, variance=1.0)

    los_weight = np.sqrt(k_factor / (k_factor + 1.0))
    nlos_weight = np.sqrt(1.0 / (k_factor + 1.0))

    return np.asarray(
        np.sqrt(average_power)
        * (los_weight * los + nlos_weight * nlos),
        dtype=np.complex128,
    )


def sample_channel_snapshot(
    config: ChannelConfig,
    rng: np.random.Generator,
) -> ChannelSnapshot:
    config.validate()
    n = config.num_elements

    g = sample_rician(
        rng,
        n,
        average_power=config.controller_ris_power,
        k_factor=config.rician_k_factor,
    )
    h_t = sample_rician(
        rng,
        n,
        average_power=config.ris_transmission_power,
        k_factor=config.rician_k_factor,
    )
    h_r = sample_rician(
        rng,
        n,
        average_power=config.ris_reflection_power,
        k_factor=config.rician_k_factor,
    )
    d_t = complex(
        sample_rician(
            rng,
            1,
            average_power=config.direct_transmission_power,
            k_factor=0.0,
        )[0]
    )
    d_r = complex(
        sample_rician(
            rng,
            1,
            average_power=config.direct_reflection_power,
            k_factor=0.0,
        )[0]
    )

    return ChannelSnapshot(
        controller_to_ris=g,
        ris_to_transmission=h_t,
        ris_to_reflection=h_r,
        direct_transmission=d_t,
        direct_reflection=d_r,
    )


def evolve_snapshot(
    snapshot: ChannelSnapshot,
    config: ChannelConfig,
    rng: np.random.Generator,
) -> ChannelSnapshot:
    rho = config.between_step_correlation

    def evolve_vector(values: ComplexArray, power: float) -> ComplexArray:
        innovation = complex_normal(
            rng,
            values.shape,
            variance=max(power, 1.0e-12),
        )
        return np.asarray(
            rho * values + np.sqrt(max(0.0, 1.0 - rho**2)) * innovation,
            dtype=np.complex128,
        )

    def evolve_scalar(value: complex, power: float) -> complex:
        innovation = complex(
            complex_normal(rng, 1, variance=max(power, 1.0e-12))[0]
        )
        return complex(
            rho * value + np.sqrt(max(0.0, 1.0 - rho**2)) * innovation
        )

    return ChannelSnapshot(
        controller_to_ris=evolve_vector(
            snapshot.controller_to_ris,
            config.controller_ris_power,
        ),
        ris_to_transmission=evolve_vector(
            snapshot.ris_to_transmission,
            config.ris_transmission_power,
        ),
        ris_to_reflection=evolve_vector(
            snapshot.ris_to_reflection,
            config.ris_reflection_power,
        ),
        direct_transmission=evolve_scalar(
            snapshot.direct_transmission,
            config.direct_transmission_power,
        ),
        direct_reflection=evolve_scalar(
            snapshot.direct_reflection,
            config.direct_reflection_power,
        ),
    )


def gauss_markov_block(
    initial: ComplexArray | complex,
    num_samples: int,
    correlation: float,
    rng: np.random.Generator,
) -> ComplexArray:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must lie in [0, 1]")

    initial_array = np.asarray(initial, dtype=np.complex128)
    scalar = initial_array.ndim == 0
    if scalar:
        initial_array = initial_array.reshape(1)

    result = np.empty((num_samples, initial_array.size), dtype=np.complex128)
    result[0] = initial_array.reshape(-1)

    element_power = np.maximum(
        np.abs(initial_array.reshape(-1)) ** 2,
        1.0e-12,
    )
    innovation_scale = np.sqrt(max(0.0, 1.0 - correlation**2))

    for index in range(1, num_samples):
        innovation = complex_normal(
            rng,
            initial_array.size,
            variance=1.0,
        ) * np.sqrt(element_power)
        result[index] = correlation * result[index - 1] + innovation_scale * innovation

    if scalar:
        return np.asarray(result[:, 0], dtype=np.complex128)
    return result


def delayed_reciprocal(
    forward: ComplexArray,
    correlation: float,
    rng: np.random.Generator,
) -> ComplexArray:
    values = np.asarray(forward, dtype=np.complex128)
    if values.ndim not in {1, 2}:
        raise ValueError("forward channel must be one- or two-dimensional")

    if values.ndim == 1:
        power = np.maximum(np.abs(values) ** 2, 1.0e-12)
    else:
        power = np.maximum(
            np.mean(np.abs(values) ** 2, axis=0, keepdims=True),
            1.0e-12,
        )

    innovation = complex_normal(rng, values.shape, variance=1.0) * np.sqrt(power)
    return np.asarray(
        correlation * values
        + np.sqrt(max(0.0, 1.0 - correlation**2)) * innovation,
        dtype=np.complex128,
    )


def build_bidirectional_block(
    snapshot: ChannelSnapshot,
    config: ChannelConfig,
    num_samples: int,
    rng: np.random.Generator,
) -> BidirectionalChannelBlock:
    g_forward = gauss_markov_block(
        snapshot.controller_to_ris,
        num_samples,
        config.within_block_correlation,
        rng,
    )
    h_t_forward = gauss_markov_block(
        snapshot.ris_to_transmission,
        num_samples,
        config.within_block_correlation,
        rng,
    )
    h_r_forward = gauss_markov_block(
        snapshot.ris_to_reflection,
        num_samples,
        config.within_block_correlation,
        rng,
    )
    d_t_forward = gauss_markov_block(
        snapshot.direct_transmission,
        num_samples,
        config.within_block_correlation,
        rng,
    )
    d_r_forward = gauss_markov_block(
        snapshot.direct_reflection,
        num_samples,
        config.within_block_correlation,
        rng,
    )

    delay_rho = config.forward_reverse_correlation
    transmission_to_ris_reverse = delayed_reciprocal(
        h_t_forward,
        delay_rho,
        rng,
    )
    reflection_to_ris_reverse = delayed_reciprocal(
        h_r_forward,
        delay_rho,
        rng,
    )
    ris_to_controller_reverse = delayed_reciprocal(
        g_forward,
        delay_rho,
        rng,
    )
    d_t_reverse = delayed_reciprocal(
        d_t_forward,
        delay_rho,
        rng,
    )
    d_r_reverse = delayed_reciprocal(
        d_r_forward,
        delay_rho,
        rng,
    )

    return BidirectionalChannelBlock(
        controller_to_ris_forward=g_forward,
        ris_to_transmission_forward=h_t_forward,
        ris_to_reflection_forward=h_r_forward,
        direct_transmission_forward=np.asarray(d_t_forward, dtype=np.complex128),
        direct_reflection_forward=np.asarray(d_r_forward, dtype=np.complex128),
        transmission_to_ris_reverse=transmission_to_ris_reverse,
        reflection_to_ris_reverse=reflection_to_ris_reverse,
        ris_to_controller_reverse=ris_to_controller_reverse,
        direct_transmission_reverse=np.asarray(d_t_reverse, dtype=np.complex128),
        direct_reflection_reverse=np.asarray(d_r_reverse, dtype=np.complex128),
    )


def estimate_channel(
    true_channel: ComplexArray | complex,
    nmse_db: float,
    rng: np.random.Generator,
) -> ComplexArray:
    values = np.asarray(true_channel, dtype=np.complex128)
    signal_power = float(np.mean(np.abs(values) ** 2))
    error_variance = max(signal_power, 1.0e-12) * 10.0 ** (nmse_db / 10.0)
    error = complex_normal(rng, values.shape, variance=error_variance)
    return np.asarray(values + error, dtype=np.complex128)
