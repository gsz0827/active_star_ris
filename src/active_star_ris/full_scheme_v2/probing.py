from __future__ import annotations

import numpy as np

from .channels import delayed_reverse_block, evolve_block, complex_normal
from .config import ChannelConfig, ProbingConfig
from .models import BranchObservations, DirectionalSurfaceCoefficients, StaticChannels


def _active_noise(
    downstream: np.ndarray,
    coefficients: np.ndarray,
    active_mask: np.ndarray,
    variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if variance <= 0.0 or not np.any(active_mask):
        return np.zeros(downstream.shape[0], dtype=np.complex128)
    source = complex_normal(rng, downstream.shape) * np.sqrt(variance)
    source[:, ~active_mask] = 0.0
    return np.sum(downstream * coefficients * source, axis=1)


def _receiver_noise(samples: int, variance: float, rng: np.random.Generator) -> np.ndarray:
    if variance <= 0.0:
        return np.zeros(samples, dtype=np.complex128)
    return complex_normal(rng, samples) * np.sqrt(variance)


def simulate_branch(
    *,
    controller_ris: np.ndarray,
    ris_user: np.ndarray,
    ris_eve: np.ndarray,
    direct_legitimate: complex,
    direct_controller_eve: complex,
    direct_user_eve: complex,
    coefficients_forward: np.ndarray,
    coefficients_reverse: np.ndarray,
    active_mask: np.ndarray,
    pilot_power_controller: float,
    pilot_power_user: float,
    receiver_noise_variance_alice: float,
    receiver_noise_variance_bob: float,
    receiver_noise_variance_eve: float,
    amplifier_noise_variance: float,
    channel_config: ChannelConfig,
    probing_config: ProbingConfig,
    rng: np.random.Generator,
) -> BranchObservations:
    samples = probing_config.samples_per_step
    within = channel_config.within_block_correlation
    delay = channel_config.forward_reverse_correlation
    g_forward = evolve_block(controller_ris, samples, within, rng)
    h_forward = evolve_block(ris_user, samples, within, rng)
    e_forward = evolve_block(ris_eve, samples, within, rng)
    g_reverse = delayed_reverse_block(g_forward, delay, rng)
    h_reverse = delayed_reverse_block(h_forward, delay, rng)
    e_reverse = delayed_reverse_block(e_forward, delay, rng)
    direct_forward = evolve_block(np.asarray(direct_legitimate), samples, within, rng).reshape(-1)
    direct_reverse = delayed_reverse_block(direct_forward, delay, rng).reshape(-1)

    effective_forward = direct_forward + np.sum(h_forward * coefficients_forward * g_forward, axis=1)
    effective_reverse = direct_reverse + np.sum(g_reverse * coefficients_reverse * h_reverse, axis=1)
    active_noise_bob = _active_noise(
        h_forward,
        coefficients_forward,
        active_mask,
        amplifier_noise_variance,
        rng,
    )
    active_noise_alice = _active_noise(
        g_reverse,
        coefficients_reverse,
        active_mask,
        amplifier_noise_variance,
        rng,
    )
    observation_bob = (
        np.sqrt(pilot_power_controller) * effective_forward
        + active_noise_bob
        + _receiver_noise(samples, receiver_noise_variance_bob, rng)
    )
    observation_alice = (
        np.sqrt(pilot_power_user) * effective_reverse
        + active_noise_alice
        + _receiver_noise(samples, receiver_noise_variance_alice, rng)
    )

    direct_eve_f = evolve_block(np.asarray(direct_controller_eve), samples, within, rng).reshape(-1)
    direct_eve_r = evolve_block(np.asarray(direct_user_eve), samples, within, rng).reshape(-1)
    eve_effective_forward = direct_eve_f + np.sum(e_forward * coefficients_forward * g_forward, axis=1)
    eve_effective_reverse = direct_eve_r + np.sum(e_reverse * coefficients_reverse * h_reverse, axis=1)
    active_noise_eve_f = _active_noise(
        e_forward,
        coefficients_forward,
        active_mask,
        amplifier_noise_variance,
        rng,
    )
    active_noise_eve_r = _active_noise(
        e_reverse,
        coefficients_reverse,
        active_mask,
        amplifier_noise_variance,
        rng,
    )
    observation_eve_forward = (
        np.sqrt(pilot_power_controller) * eve_effective_forward
        + active_noise_eve_f
        + _receiver_noise(samples, receiver_noise_variance_eve, rng)
    )
    observation_eve_reverse = (
        np.sqrt(pilot_power_user) * eve_effective_reverse
        + active_noise_eve_r
        + _receiver_noise(samples, receiver_noise_variance_eve, rng)
    )
    return BranchObservations(
        observation_alice=np.asarray(observation_alice, dtype=np.complex128),
        observation_bob=np.asarray(observation_bob, dtype=np.complex128),
        observation_eve_forward=np.asarray(observation_eve_forward, dtype=np.complex128),
        observation_eve_reverse=np.asarray(observation_eve_reverse, dtype=np.complex128),
        effective_forward=np.asarray(effective_forward, dtype=np.complex128),
        effective_reverse=np.asarray(effective_reverse, dtype=np.complex128),
    )


def simulate_dual_side_probing(
    channels: StaticChannels,
    coefficients: DirectionalSurfaceCoefficients,
    active_mask: np.ndarray,
    channel_config: ChannelConfig,
    probing: ProbingConfig,
    rng: np.random.Generator,
) -> tuple[BranchObservations, BranchObservations]:
    transmission = simulate_branch(
        controller_ris=channels.controller_ris,
        ris_user=channels.ris_transmission,
        ris_eve=channels.ris_eve_transmission,
        direct_legitimate=channels.direct_transmission,
        direct_controller_eve=channels.direct_controller_eve_transmission,
        direct_user_eve=channels.direct_user_eve_transmission,
        coefficients_forward=coefficients.transmission_forward,
        coefficients_reverse=coefficients.transmission_reverse,
        active_mask=active_mask,
        pilot_power_controller=probing.pilot_power_controller,
        pilot_power_user=probing.pilot_power_transmission_user,
        receiver_noise_variance_alice=probing.receiver_noise_variance_controller,
        receiver_noise_variance_bob=probing.receiver_noise_variance_transmission_user,
        receiver_noise_variance_eve=probing.receiver_noise_variance_eve,
        amplifier_noise_variance=probing.input_referred_amplifier_noise_variance,
        channel_config=channel_config,
        probing_config=probing,
        rng=rng,
    )
    reflection = simulate_branch(
        controller_ris=channels.controller_ris,
        ris_user=channels.ris_reflection,
        ris_eve=channels.ris_eve_reflection,
        direct_legitimate=channels.direct_reflection,
        direct_controller_eve=channels.direct_controller_eve_reflection,
        direct_user_eve=channels.direct_user_eve_reflection,
        coefficients_forward=coefficients.reflection_forward,
        coefficients_reverse=coefficients.reflection_reverse,
        active_mask=active_mask,
        pilot_power_controller=probing.pilot_power_controller,
        pilot_power_user=probing.pilot_power_reflection_user,
        receiver_noise_variance_alice=probing.receiver_noise_variance_controller,
        receiver_noise_variance_bob=probing.receiver_noise_variance_reflection_user,
        receiver_noise_variance_eve=probing.receiver_noise_variance_eve,
        amplifier_noise_variance=probing.input_referred_amplifier_noise_variance,
        channel_config=channel_config,
        probing_config=probing,
        rng=rng,
    )
    return transmission, reflection
