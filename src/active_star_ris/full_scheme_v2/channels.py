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


def correlated_eve_channel(
    legitimate_channel: np.ndarray,
    average_power: float,
    correlation: float,
    rng: np.random.Generator,
) -> ComplexArray:
    """生成与合法信道具有可控复相关性的Eve信道。

    模型：
        h_e = sqrt(P_e) * (
            rho * h_normalized
            + sqrt(1-rho^2) * w
        )

    其中：
        rho为复相关系数；
        P_e为Eve链路平均功率；
        w为单位功率复高斯随机变量。
    """
    if average_power < 0.0:
        raise ValueError(
            "average_power cannot be negative"
        )

    reference = np.asarray(
        legitimate_channel,
        dtype=np.complex128,
    )

    if average_power == 0.0:
        return np.zeros_like(
            reference,
            dtype=np.complex128,
        )

    rho = float(
        np.clip(correlation, 0.0, 1.0)
    )

    reference_power = float(
        np.mean(np.abs(reference) ** 2)
    )

    if (
        not np.isfinite(reference_power)
        or reference_power <= 1.0e-12
    ):
        normalized_reference = complex_normal(
            rng,
            reference.shape,
            variance=1.0,
        )
    else:
        normalized_reference = (
            reference / np.sqrt(reference_power)
        )

    innovation = complex_normal(
        rng,
        reference.shape,
        variance=1.0,
    )

    eve_channel = np.sqrt(average_power) * (
        rho * normalized_reference
        + np.sqrt(max(0.0, 1.0 - rho**2))
        * innovation
    )

    return np.asarray(
        eve_channel,
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
    *,
    average_power: float,
) -> ComplexArray:
    if num_samples < 1:
        raise ValueError(
            "num_samples must be positive"
        )
    if not 0.0 <= correlation <= 1.0:
        raise ValueError(
            "correlation must lie in [0, 1]"
        )
    if average_power < 0.0:
        raise ValueError(
            "average_power cannot be negative"
        )

    initial_array = np.asarray(
        initial,
        dtype=np.complex128,
    )
    scalar = initial_array.ndim == 0

    if scalar:
        initial_array = initial_array.reshape(1)

    result = np.empty(
        (num_samples, initial_array.size),
        dtype=np.complex128,
    )
    result[0] = initial_array.reshape(-1)

    innovation_scale = np.sqrt(
        max(0.0, 1.0 - correlation**2)
    )

    for index in range(1, num_samples):
        innovation = complex_normal(
            rng,
            initial_array.size,
            variance=max(
                average_power,
                1.0e-12,
            ),
        )

        result[index] = (
            correlation * result[index - 1]
            + innovation_scale * innovation
        )

    if scalar:
        return np.asarray(
            result[:, 0],
            dtype=np.complex128,
        )

    return result


def delayed_reciprocal(
    forward: ComplexArray,
    correlation: float,
    rng: np.random.Generator,
    *,
    average_power: float,
) -> ComplexArray:
    values = np.asarray(
        forward,
        dtype=np.complex128,
    )

    if values.ndim not in {1, 2}:
        raise ValueError(
            "forward channel must be "
            "one- or two-dimensional"
        )

    innovation = complex_normal(
        rng,
        values.shape,
        variance=max(
            average_power,
            1.0e-12,
        ),
    )

    return np.asarray(
        correlation * values
        + np.sqrt(
            max(
                0.0,
                1.0 - correlation**2,
            )
        )
        * innovation,
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
        average_power=config.controller_ris_power,
    )

    h_t_forward = gauss_markov_block(
        snapshot.ris_to_transmission,
        num_samples,
        config.within_block_correlation,
        rng,
        average_power=config.ris_transmission_power,
    )

    h_r_forward = gauss_markov_block(
        snapshot.ris_to_reflection,
        num_samples,
        config.within_block_correlation,
        rng,
        average_power=config.ris_reflection_power,
    )

    d_t_forward = gauss_markov_block(
        snapshot.direct_transmission,
        num_samples,
        config.within_block_correlation,
        rng,
        average_power=config.direct_transmission_power,
    )

    d_r_forward = gauss_markov_block(
        snapshot.direct_reflection,
        num_samples,
        config.within_block_correlation,
        rng,
        average_power=config.direct_reflection_power,
    )

    delay_rho = config.forward_reverse_correlation
    transmission_to_ris_reverse = delayed_reciprocal(
        h_t_forward,
        delay_rho,
        rng,
        average_power=config.ris_transmission_power,
    )

    reflection_to_ris_reverse = delayed_reciprocal(
        h_r_forward,
        delay_rho,
        rng,
        average_power=config.ris_reflection_power,
    )

    ris_to_controller_reverse = delayed_reciprocal(
        g_forward,
        delay_rho,
        rng,
        average_power=config.controller_ris_power,
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
