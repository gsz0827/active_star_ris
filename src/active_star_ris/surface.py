from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class EnergySplit:
    beta_transmission: FloatArray
    beta_reflection: FloatArray

    @classmethod
    def from_transmission(
        cls,
        beta_transmission: float | ArrayLike,
        num_elements: int,
    ) -> "EnergySplit":
        if num_elements <= 0:
            raise ValueError("num_elements must be positive")

        beta_t = np.asarray(beta_transmission, dtype=float)
        if beta_t.ndim == 0:
            beta_t = np.full(num_elements, float(beta_t), dtype=float)
        else:
            beta_t = beta_t.reshape(-1)

        if beta_t.size != num_elements:
            raise ValueError(
                "beta_transmission must be scalar or contain num_elements entries"
            )
        if np.any((beta_t < 0.0) | (beta_t > 1.0)):
            raise ValueError("beta_transmission must lie within [0, 1]")

        return cls(
            beta_transmission=np.asarray(beta_t, dtype=float),
            beta_reflection=np.asarray(1.0 - beta_t, dtype=float),
        )

    def maximum_constraint_error(self) -> float:
        return float(
            np.max(
                np.abs(
                    self.beta_transmission
                    + self.beta_reflection
                    - 1.0
                )
            )
        )


@dataclass(frozen=True)
class SurfaceCoefficients:
    phi_transmission: ComplexArray
    phi_reflection: ComplexArray
    amplitude_gain: FloatArray
    active_mask: BoolArray
    energy_split: EnergySplit

    def __post_init__(self) -> None:
        n = self.energy_split.beta_transmission.size
        arrays = (
            self.phi_transmission,
            self.phi_reflection,
            self.amplitude_gain,
            self.active_mask,
        )
        if any(np.asarray(item).reshape(-1).size != n for item in arrays):
            raise ValueError("all surface arrays must have the same length")

    @property
    def num_elements(self) -> int:
        return int(self.amplitude_gain.size)

    def maximum_energy_error(self) -> float:
        normalized_t = np.divide(
            np.abs(self.phi_transmission) ** 2,
            self.amplitude_gain**2,
            out=np.zeros_like(self.amplitude_gain, dtype=float),
            where=self.amplitude_gain > 0,
        )
        normalized_r = np.divide(
            np.abs(self.phi_reflection) ** 2,
            self.amplitude_gain**2,
            out=np.zeros_like(self.amplitude_gain, dtype=float),
            where=self.amplitude_gain > 0,
        )
        return float(np.max(np.abs(normalized_t + normalized_r - 1.0)))


def wrap_phase(phase_rad: ArrayLike) -> FloatArray:
    return np.asarray(
        np.mod(np.asarray(phase_rad, dtype=float), 2.0 * np.pi),
        dtype=float,
    )


def build_surface_coefficients(
    energy_split: EnergySplit,
    phase_transmission_rad: ArrayLike,
    phase_reflection_rad: ArrayLike,
    amplitude_gain: float | ArrayLike = 1.0,
    active_mask: ArrayLike | None = None,
) -> SurfaceCoefficients:
    n = energy_split.beta_transmission.size
    theta_t = wrap_phase(phase_transmission_rad).reshape(-1)
    theta_r = wrap_phase(phase_reflection_rad).reshape(-1)

    if theta_t.size != n or theta_r.size != n:
        raise ValueError("phase arrays must contain num_elements entries")

    gain = np.asarray(amplitude_gain, dtype=float)
    if gain.ndim == 0:
        gain = np.full(n, float(gain), dtype=float)
    else:
        gain = gain.reshape(-1)
    if gain.size != n:
        raise ValueError("amplitude_gain must be scalar or contain num_elements entries")
    if np.any(gain < 1.0):
        raise ValueError("amplitude_gain must be at least 1")

    if active_mask is None:
        mask = np.zeros(n, dtype=bool)
    else:
        mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if mask.size != n:
            raise ValueError("active_mask must contain num_elements entries")

    # Passive elements must retain unit amplitude.
    gain = np.where(mask, gain, 1.0)

    phi_t = (
        gain
        * np.sqrt(energy_split.beta_transmission)
        * np.exp(1j * theta_t)
    )
    phi_r = (
        gain
        * np.sqrt(energy_split.beta_reflection)
        * np.exp(1j * theta_r)
    )

    return SurfaceCoefficients(
        phi_transmission=np.asarray(phi_t, dtype=np.complex128),
        phi_reflection=np.asarray(phi_r, dtype=np.complex128),
        amplitude_gain=np.asarray(gain, dtype=float),
        active_mask=np.asarray(mask, dtype=bool),
        energy_split=energy_split,
    )


def random_phases(
    num_elements: int,
    rng: np.random.Generator,
) -> FloatArray:
    if num_elements <= 0:
        raise ValueError("num_elements must be positive")
    return np.asarray(
        rng.uniform(0.0, 2.0 * np.pi, size=num_elements),
        dtype=float,
    )
