from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
ChannelModel = Literal["rayleigh", "rician"]


@dataclass(frozen=True)
class ChannelConfig:
    model: ChannelModel
    large_scale_power: float = 1.0
    k_factor_db: float = 0.0

    def validate(self) -> None:
        if self.model not in ("rayleigh", "rician"):
            raise ValueError(f"unsupported channel model: {self.model}")
        if self.large_scale_power < 0:
            raise ValueError("large_scale_power must be non-negative")


def db_to_linear(value_db: float | ArrayLike) -> NDArray[np.float64]:
    return np.asarray(
        10.0 ** (np.asarray(value_db, dtype=float) / 10.0),
        dtype=float,
    )


def complex_normal(
    shape: int | Sequence[int],
    rng: np.random.Generator,
    variance: float = 1.0,
) -> ComplexArray:
    if variance < 0:
        raise ValueError("variance must be non-negative")
    scale = np.sqrt(variance / 2.0)
    real = rng.normal(0.0, scale, size=shape)
    imag = rng.normal(0.0, scale, size=shape)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def rayleigh_channel(
    shape: int | Sequence[int],
    rng: np.random.Generator,
    large_scale_power: float = 1.0,
) -> ComplexArray:
    if large_scale_power < 0:
        raise ValueError("large_scale_power must be non-negative")
    return complex_normal(shape, rng, variance=large_scale_power)


def rician_channel(
    shape: int | Sequence[int],
    rng: np.random.Generator,
    k_factor_db: float,
    large_scale_power: float = 1.0,
    los_component: ArrayLike | None = None,
) -> ComplexArray:
    if large_scale_power < 0:
        raise ValueError("large_scale_power must be non-negative")

    target_shape = np.empty(shape).shape
    k_linear = float(db_to_linear(k_factor_db))

    if los_component is None:
        los = np.ones(target_shape, dtype=np.complex128)
    else:
        los = np.asarray(los_component, dtype=np.complex128)
        if los.shape != target_shape:
            raise ValueError(
                f"los_component shape {los.shape} does not match {target_shape}"
            )
        magnitude = np.abs(los)
        los = np.divide(
            los,
            magnitude,
            out=np.ones_like(los),
            where=magnitude > 0,
        )

    nlos = complex_normal(target_shape, rng, variance=1.0)
    normalized = (
        np.sqrt(k_linear / (k_linear + 1.0)) * los
        + np.sqrt(1.0 / (k_linear + 1.0)) * nlos
    )
    return np.asarray(
        np.sqrt(large_scale_power) * normalized,
        dtype=np.complex128,
    )


def generate_channel(
    shape: int | Sequence[int],
    rng: np.random.Generator,
    config: ChannelConfig,
) -> ComplexArray:
    config.validate()
    if config.model == "rayleigh":
        return rayleigh_channel(shape, rng, config.large_scale_power)
    return rician_channel(
        shape,
        rng,
        k_factor_db=config.k_factor_db,
        large_scale_power=config.large_scale_power,
    )


def gauss_markov_sequence(
    initial_channel: ArrayLike,
    num_steps: int,
    correlation: float,
    rng: np.random.Generator,
    innovation_variance: float = 1.0,
) -> ComplexArray:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must lie within [0, 1]")
    if innovation_variance < 0:
        raise ValueError("innovation_variance must be non-negative")

    initial = np.asarray(initial_channel, dtype=np.complex128)
    sequence = np.empty((num_steps,) + initial.shape, dtype=np.complex128)
    sequence[0] = initial

    innovation_scale = np.sqrt(1.0 - correlation**2)
    for index in range(1, num_steps):
        innovation = complex_normal(
            initial.shape,
            rng,
            variance=innovation_variance,
        )
        sequence[index] = (
            correlation * sequence[index - 1]
            + innovation_scale * innovation
        )

    return sequence
