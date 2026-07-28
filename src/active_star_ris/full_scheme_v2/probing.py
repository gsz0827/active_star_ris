from __future__ import annotations

import numpy as np

from .channels import complex_normal
from .config import ProbingConfig
from .models import (
    ActualSurfaceCoefficients,
    BidirectionalChannelBlock,
    BranchProbingResult,
    DualProbingResult,
    EndpointRFRealization,
)


def _simulate_branch(
    input_forward: np.ndarray,
    output_forward: np.ndarray,
    input_reverse: np.ndarray,
    output_reverse: np.ndarray,
    direct_forward: np.ndarray,
    direct_reverse: np.ndarray,
    phi_forward: np.ndarray,
    phi_reverse: np.ndarray,
    active_mask: np.ndarray,
    *,
    pilot_power_forward: float,
    pilot_power_reverse: float,
    pilot_symbols_forward: int,
    pilot_symbols_reverse: int,
    amplifier_noise_variance: float,
    receiver_noise_variance_forward: float,
    receiver_noise_variance_reverse: float,
    tx_coefficient_forward: complex,
    rx_coefficient_forward: complex,
    tx_coefficient_reverse: complex,
    rx_coefficient_reverse: complex,
    rng: np.random.Generator,
) -> BranchProbingResult:
    x_f = np.asarray(input_forward, dtype=np.complex128)
    y_f = np.asarray(output_forward, dtype=np.complex128)
    x_r = np.asarray(input_reverse, dtype=np.complex128)
    y_r = np.asarray(output_reverse, dtype=np.complex128)

    if x_f.ndim != 2 or y_f.shape != x_f.shape:
        raise ValueError("forward channels must be equal-sized matrices")
    if x_r.shape != x_f.shape or y_r.shape != x_f.shape:
        raise ValueError("reverse channels must match forward channel shape")

    samples, num_elements = x_f.shape
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    if active.size != num_elements:
        raise ValueError("active_mask size mismatch")

    phi_f = np.asarray(phi_forward, dtype=np.complex128).reshape(-1)
    phi_r = np.asarray(phi_reverse, dtype=np.complex128).reshape(-1)
    if phi_f.size != num_elements or phi_r.size != num_elements:
        raise ValueError("surface coefficient size mismatch")

    d_f = np.asarray(direct_forward, dtype=np.complex128).reshape(-1)
    d_r = np.asarray(direct_reverse, dtype=np.complex128).reshape(-1)
    if d_f.size != samples or d_r.size != samples:
        raise ValueError("direct channel sample count mismatch")

    effective_forward = (
        d_f + np.sum(x_f * phi_f[None, :] * y_f, axis=1)
    ) * tx_coefficient_forward * rx_coefficient_forward

    effective_reverse = (
        d_r + np.sum(x_r * phi_r[None, :] * y_r, axis=1)
    ) * tx_coefficient_reverse * rx_coefficient_reverse

    active_noise_forward = np.zeros((samples, num_elements), dtype=np.complex128)
    active_noise_reverse = np.zeros((samples, num_elements), dtype=np.complex128)
    if np.any(active) and amplifier_noise_variance > 0.0:
        active_noise_forward[:, active] = complex_normal(
            rng,
            (samples, int(np.count_nonzero(active))),
            variance=amplifier_noise_variance,
        )
        active_noise_reverse[:, active] = complex_normal(
            rng,
            (samples, int(np.count_nonzero(active))),
            variance=amplifier_noise_variance,
        )

    forwarded_forward = np.sum(
        y_f * phi_f[None, :] * active_noise_forward,
        axis=1,
    ) * rx_coefficient_forward
    forwarded_reverse = np.sum(
        y_r * phi_r[None, :] * active_noise_reverse,
        axis=1,
    ) * rx_coefficient_reverse

    receiver_forward = complex_normal(
        rng,
        samples,
        variance=receiver_noise_variance_forward,
    )
    receiver_reverse = complex_normal(
        rng,
        samples,
        variance=receiver_noise_variance_reverse,
    )

    estimation_scale_forward = np.sqrt(
        max(pilot_power_forward * pilot_symbols_forward, 1.0e-12)
    )
    estimation_scale_reverse = np.sqrt(
        max(pilot_power_reverse * pilot_symbols_reverse, 1.0e-12)
    )

    observation_forward = (
        effective_forward
        + forwarded_forward / estimation_scale_forward
        + receiver_forward / estimation_scale_forward
    )
    observation_reverse = (
        effective_reverse
        + forwarded_reverse / estimation_scale_reverse
        + receiver_reverse / estimation_scale_reverse
    )

    return BranchProbingResult(
        observation_forward=np.asarray(
            observation_forward,
            dtype=np.complex128,
        ),
        observation_reverse=np.asarray(
            observation_reverse,
            dtype=np.complex128,
        ),
        effective_channel_forward=np.asarray(
            effective_forward,
            dtype=np.complex128,
        ),
        effective_channel_reverse=np.asarray(
            effective_reverse,
            dtype=np.complex128,
        ),
        forwarded_active_noise_forward=np.asarray(
            forwarded_forward,
            dtype=np.complex128,
        ),
        forwarded_active_noise_reverse=np.asarray(
            forwarded_reverse,
            dtype=np.complex128,
        ),
        active_noise_forward=np.asarray(
            active_noise_forward,
            dtype=np.complex128,
        ),
        active_noise_reverse=np.asarray(
            active_noise_reverse,
            dtype=np.complex128,
        ),
    )


def simulate_dual_side_probing(
    block: BidirectionalChannelBlock,
    surface: ActualSurfaceCoefficients,
    active_mask: np.ndarray,
    endpoint: EndpointRFRealization,
    config: ProbingConfig,
    rng: np.random.Generator,
    *,
    amplifier_noise_scale: float = 1.0,
    receiver_noise_scale: float = 1.0,
) -> DualProbingResult:
    config.validate()

    transmission = _simulate_branch(
        block.controller_to_ris_forward,
        block.ris_to_transmission_forward,
        block.transmission_to_ris_reverse,
        block.ris_to_controller_reverse,
        block.direct_transmission_forward,
        block.direct_transmission_reverse,
        surface.transmission_forward,
        surface.transmission_reverse,
        active_mask,
        pilot_power_forward=config.pilot_power_controller,
        pilot_power_reverse=config.pilot_power_transmission_user,
        pilot_symbols_forward=config.pilot_symbols_controller,
        pilot_symbols_reverse=config.pilot_symbols_transmission_user,
        amplifier_noise_variance=(
            config.input_referred_amplifier_noise_variance
            * amplifier_noise_scale
        ),
        receiver_noise_variance_forward=(
            config.receiver_noise_variance_transmission_user
            * receiver_noise_scale
        ),
        receiver_noise_variance_reverse=(
            config.receiver_noise_variance_controller
            * receiver_noise_scale
        ),
        tx_coefficient_forward=endpoint.controller_tx,
        rx_coefficient_forward=endpoint.transmission_rx,
        tx_coefficient_reverse=endpoint.transmission_tx,
        rx_coefficient_reverse=endpoint.controller_rx,
        rng=rng,
    )

    reflection = _simulate_branch(
        block.controller_to_ris_forward,
        block.ris_to_reflection_forward,
        block.reflection_to_ris_reverse,
        block.ris_to_controller_reverse,
        block.direct_reflection_forward,
        block.direct_reflection_reverse,
        surface.reflection_forward,
        surface.reflection_reverse,
        active_mask,
        pilot_power_forward=config.pilot_power_controller,
        pilot_power_reverse=config.pilot_power_reflection_user,
        pilot_symbols_forward=config.pilot_symbols_controller,
        pilot_symbols_reverse=config.pilot_symbols_reflection_user,
        amplifier_noise_variance=(
            config.input_referred_amplifier_noise_variance
            * amplifier_noise_scale
        ),
        receiver_noise_variance_forward=(
            config.receiver_noise_variance_reflection_user
            * receiver_noise_scale
        ),
        receiver_noise_variance_reverse=(
            config.receiver_noise_variance_controller
            * receiver_noise_scale
        ),
        tx_coefficient_forward=endpoint.controller_tx,
        rx_coefficient_forward=endpoint.reflection_rx,
        tx_coefficient_reverse=endpoint.reflection_tx,
        rx_coefficient_reverse=endpoint.controller_rx,
        rng=rng,
    )

    return DualProbingResult(
        transmission=transmission,
        reflection=reflection,
    )
