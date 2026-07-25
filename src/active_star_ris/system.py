from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .surface import SurfaceCoefficients


@dataclass(frozen=True)
class LinkMetrics:
    effective_channel: complex
    signal_power: float
    forwarded_noise_variance: float
    total_noise_variance: float
    snr_linear: float
    rate_bps_hz: float


@dataclass(frozen=True)
class TwoUserMetrics:
    transmission: LinkMetrics
    reflection: LinkMetrics
    weighted_sum_rate: float
    ris_output_power: float
    ris_power_violation: float


def effective_channel(
    alice_to_ris: ArrayLike,
    ris_to_user: ArrayLike,
    coefficients: ArrayLike,
    direct_channel: complex = 0.0j,
) -> complex:
    g = np.asarray(alice_to_ris, dtype=np.complex128).reshape(-1)
    h = np.asarray(ris_to_user, dtype=np.complex128).reshape(-1)
    phi = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    if not (g.size == h.size == phi.size):
        raise ValueError("all channel and coefficient arrays must have equal length")
    return complex(direct_channel + np.sum(h * phi * g))


def forwarded_active_noise_variance(
    ris_to_user: ArrayLike,
    coefficients: ArrayLike,
    active_mask: ArrayLike,
    ris_internal_noise_variance: float,
) -> float:
    if ris_internal_noise_variance < 0:
        raise ValueError("ris_internal_noise_variance must be non-negative")

    h = np.asarray(ris_to_user, dtype=np.complex128).reshape(-1)
    phi = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    if not (h.size == phi.size == mask.size):
        raise ValueError("all arrays must have equal length")

    return float(
        ris_internal_noise_variance
        * np.sum(np.abs(h[mask] * phi[mask]) ** 2)
    )


def ris_output_power(
    alice_to_ris: ArrayLike,
    surface: SurfaceCoefficients,
    transmit_power: float,
    ris_internal_noise_variance: float,
) -> float:
    if transmit_power < 0:
        raise ValueError("transmit_power must be non-negative")
    if ris_internal_noise_variance < 0:
        raise ValueError("ris_internal_noise_variance must be non-negative")

    g = np.asarray(alice_to_ris, dtype=np.complex128).reshape(-1)
    if g.size != surface.num_elements:
        raise ValueError("alice_to_ris length does not match surface")

    mask = surface.active_mask
    input_power_per_active_element = (
        transmit_power * np.abs(g[mask]) ** 2
        + ris_internal_noise_variance
    )
    return float(
        np.sum(
            surface.amplitude_gain[mask] ** 2
            * input_power_per_active_element
        )
    )


def evaluate_link(
    alice_to_ris: ArrayLike,
    ris_to_user: ArrayLike,
    coefficients: ArrayLike,
    active_mask: ArrayLike,
    transmit_power: float,
    user_noise_variance: float,
    ris_internal_noise_variance: float,
    direct_channel: complex = 0.0j,
) -> LinkMetrics:
    if transmit_power < 0:
        raise ValueError("transmit_power must be non-negative")
    if user_noise_variance <= 0:
        raise ValueError("user_noise_variance must be positive")

    h_eff = effective_channel(
        alice_to_ris,
        ris_to_user,
        coefficients,
        direct_channel,
    )
    signal_power = float(transmit_power * abs(h_eff) ** 2)
    forwarded_noise = forwarded_active_noise_variance(
        ris_to_user,
        coefficients,
        active_mask,
        ris_internal_noise_variance,
    )
    total_noise = float(user_noise_variance + forwarded_noise)
    snr = float(signal_power / total_noise)
    rate = float(np.log2(1.0 + snr))

    return LinkMetrics(
        effective_channel=h_eff,
        signal_power=signal_power,
        forwarded_noise_variance=forwarded_noise,
        total_noise_variance=total_noise,
        snr_linear=snr,
        rate_bps_hz=rate,
    )


def evaluate_two_user_system(
    alice_to_ris: ArrayLike,
    ris_to_transmission_user: ArrayLike,
    ris_to_reflection_user: ArrayLike,
    surface: SurfaceCoefficients,
    transmit_power: float,
    user_noise_variance: float,
    ris_internal_noise_variance: float,
    ris_output_power_budget: float,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    direct_transmission: complex = 0.0j,
    direct_reflection: complex = 0.0j,
) -> TwoUserMetrics:
    if ris_output_power_budget < 0:
        raise ValueError("ris_output_power_budget must be non-negative")
    if transmission_weight < 0 or reflection_weight < 0:
        raise ValueError("weights must be non-negative")
    if transmission_weight + reflection_weight <= 0:
        raise ValueError("at least one weight must be positive")

    total_weight = transmission_weight + reflection_weight
    wt = transmission_weight / total_weight
    wr = reflection_weight / total_weight

    transmission = evaluate_link(
        alice_to_ris,
        ris_to_transmission_user,
        surface.phi_transmission,
        surface.active_mask,
        transmit_power,
        user_noise_variance,
        ris_internal_noise_variance,
        direct_transmission,
    )
    reflection = evaluate_link(
        alice_to_ris,
        ris_to_reflection_user,
        surface.phi_reflection,
        surface.active_mask,
        transmit_power,
        user_noise_variance,
        ris_internal_noise_variance,
        direct_reflection,
    )

    output_power = ris_output_power(
        alice_to_ris,
        surface,
        transmit_power,
        ris_internal_noise_variance,
    )
    violation = max(0.0, output_power - ris_output_power_budget)

    return TwoUserMetrics(
        transmission=transmission,
        reflection=reflection,
        weighted_sum_rate=float(
            wt * transmission.rate_bps_hz
            + wr * reflection.rate_bps_hz
        ),
        ris_output_power=output_power,
        ris_power_violation=float(violation),
    )
