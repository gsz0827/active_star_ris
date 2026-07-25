from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class KeyGenerationMetrics:
    correlation_magnitude: float
    correlation_squared: float
    mutual_information_bits_per_sample: float
    key_disagreement_rate: float
    num_raw_bits: int


def _complex_vector(
    samples: ArrayLike,
    name: str,
) -> NDArray[np.complex128]:
    values = np.asarray(
        samples,
        dtype=np.complex128,
    ).reshape(-1)

    if values.size < 2:
        raise ValueError(
            f"{name} must contain at least two samples"
        )

    real_is_finite = np.all(np.isfinite(values.real))
    imag_is_finite = np.all(np.isfinite(values.imag))
    if not real_is_finite or not imag_is_finite:
        raise ValueError(
            f"{name} contains non-finite values"
        )

    return values


def complex_correlation(
    observations_a: ArrayLike,
    observations_b: ArrayLike,
) -> complex:
    a = _complex_vector(
        observations_a,
        "observations_a",
    )
    b = _complex_vector(
        observations_b,
        "observations_b",
    )

    if a.size != b.size:
        raise ValueError(
            "observation vectors must have equal length"
        )

    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)

    energy_a = float(
        np.vdot(a_centered, a_centered).real
    )
    energy_b = float(
        np.vdot(b_centered, b_centered).real
    )

    if energy_a <= 0.0 or energy_b <= 0.0:
        return 0.0j

    rho = np.vdot(
        a_centered,
        b_centered,
    ) / np.sqrt(energy_a * energy_b)

    rho_abs = float(abs(rho))
    magnitude = min(1.0, rho_abs)

    if rho_abs == 0.0:
        return 0.0j

    return complex(
        rho / rho_abs * magnitude
    )


def gaussian_mutual_information_bits(
    correlation_magnitude: float,
    numerical_floor: float = 1.0e-12,
) -> float:
    if not 0.0 <= correlation_magnitude <= 1.0:
        raise ValueError(
            "correlation_magnitude must lie within [0, 1]"
        )

    if not 0.0 < numerical_floor < 1.0:
        raise ValueError(
            "numerical_floor must lie within (0, 1)"
        )

    one_minus_rho_squared = max(
        numerical_floor,
        1.0 - correlation_magnitude**2,
    )

    return float(
        -np.log2(one_minus_rho_squared)
    )


def quantize_complex_sign(
    samples: ArrayLike,
) -> BoolArray:
    values = _complex_vector(
        samples,
        "samples",
    )

    real_threshold = float(
        np.median(values.real)
    )
    imag_threshold = float(
        np.median(values.imag)
    )

    bits = np.empty(
        2 * values.size,
        dtype=bool,
    )

    bits[0::2] = (
        values.real >= real_threshold
    )
    bits[1::2] = (
        values.imag >= imag_threshold
    )

    return bits


def key_disagreement_rate(
    bits_a: ArrayLike,
    bits_b: ArrayLike,
) -> float:
    a = np.asarray(
        bits_a,
        dtype=bool,
    ).reshape(-1)

    b = np.asarray(
        bits_b,
        dtype=bool,
    ).reshape(-1)

    if a.size == 0:
        raise ValueError(
            "bit vectors cannot be empty"
        )

    if a.size != b.size:
        raise ValueError(
            "bit vectors must have equal length"
        )

    return float(
        np.mean(a != b)
    )


def evaluate_key_generation(
    observations_a: ArrayLike,
    observations_b: ArrayLike,
) -> KeyGenerationMetrics:
    a = _complex_vector(
        observations_a,
        "observations_a",
    )
    b = _complex_vector(
        observations_b,
        "observations_b",
    )

    if a.size != b.size:
        raise ValueError(
            "observation vectors must have equal length"
        )

    rho = complex_correlation(a, b)
    rho_magnitude = min(
        1.0,
        float(abs(rho)),
    )

    bits_a = quantize_complex_sign(a)
    bits_b = quantize_complex_sign(b)

    return KeyGenerationMetrics(
        correlation_magnitude=rho_magnitude,
        correlation_squared=float(
            rho_magnitude**2
        ),
        mutual_information_bits_per_sample=(
            gaussian_mutual_information_bits(
                rho_magnitude
            )
        ),
        key_disagreement_rate=(
            key_disagreement_rate(
                bits_a,
                bits_b,
            )
        ),
        num_raw_bits=int(bits_a.size),
    )