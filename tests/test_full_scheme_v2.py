from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from active_star_ris.full_scheme_v2.channels import (
    build_bidirectional_block,
    sample_channel_snapshot,
)
from active_star_ris.full_scheme_v2.config import (
    EnvironmentConfig,
    HardwareConfig,
    KeyGenerationConfig,
    PowerConfig,
    ProbingConfig,
    RobustConfig,
    load_environment_config,
)
from active_star_ris.full_scheme_v2.environment import RobustFullSchemeEnvironment
from active_star_ris.full_scheme_v2.hardware import build_active_mask, decode_action
from active_star_ris.full_scheme_v2.key_protocol import evaluate_key_rate
from active_star_ris.full_scheme_v2.power import project_command_to_power_constraints
from active_star_ris.full_scheme_v2.td3 import ReplayBuffer


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "full_scheme_v2.yaml"
)


def test_configuration_loads() -> None:
    config = load_environment_config(CONFIG_PATH)
    assert config.channel.num_elements == 32
    assert config.robust.objective_samples == 16


def test_delay_reduces_average_reciprocity() -> None:
    base = EnvironmentConfig().channel
    correlations: dict[str, list[float]] = {"zero": [], "delayed": []}

    for seed in range(20):
        rng_zero = np.random.default_rng(seed)
        rng_delay = np.random.default_rng(seed)
        snapshot_zero = sample_channel_snapshot(base, rng_zero)
        snapshot_delay = sample_channel_snapshot(base, rng_delay)

        zero_config = replace(
            base,
            forward_reverse_delay_seconds=0.0,
        )
        delayed_config = replace(
            base,
            forward_reverse_delay_seconds=0.02,
            channel_coherence_time_seconds=0.01,
        )
        block_zero = build_bidirectional_block(
            snapshot_zero,
            zero_config,
            128,
            rng_zero,
        )
        block_delay = build_bidirectional_block(
            snapshot_delay,
            delayed_config,
            128,
            rng_delay,
        )

        def corr(a: np.ndarray, b: np.ndarray) -> float:
            return float(
                np.abs(np.vdot(a.reshape(-1), b.reshape(-1)))
                / np.sqrt(
                    np.vdot(a.reshape(-1), a.reshape(-1)).real
                    * np.vdot(b.reshape(-1), b.reshape(-1)).real
                )
            )

        correlations["zero"].append(
            corr(
                block_zero.ris_to_transmission_forward,
                block_zero.transmission_to_ris_reverse,
            )
        )
        correlations["delayed"].append(
            corr(
                block_delay.ris_to_transmission_forward,
                block_delay.transmission_to_ris_reverse,
            )
        )

    assert np.mean(correlations["zero"]) > np.mean(correlations["delayed"])


def test_practical_key_rate_positive_for_matching_observations() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    observation_a = source + 0.01 * (
        rng.normal(size=4096) + 1j * rng.normal(size=4096)
    )
    observation_b = source + 0.01 * (
        rng.normal(size=4096) + 1j * rng.normal(size=4096)
    )

    result = evaluate_key_rate(
        observation_a,
        observation_b,
        key_config=KeyGenerationConfig(),
        probing_config=ProbingConfig(samples_per_step=4096),
        rng=rng,
        full_protocol=True,
    )

    assert result.raw_kdr < 0.05
    assert result.post_reconciliation_kdr == 0.0
    assert result.final_key_bits > 0
    assert result.final_key_rate_bps > 0.0
    assert result.success


def test_projection_respects_rf_dc_and_saturation_constraints() -> None:
    n = 16
    active = build_active_mask(n, 0.5)
    hardware = HardwareConfig(
        maximum_active_gain=5.0,
        per_active_element_saturation_power=2.0,
    )
    power = PowerConfig(
        maximum_rf_output_power=8.0,
        maximum_total_dc_power=2.0,
    )
    action_dim = int(np.count_nonzero(active) + 3 * n)
    command = decode_action(
        np.ones(action_dim),
        active,
        hardware,
    )
    input_power = np.full(n, 0.5)
    projected, result = project_command_to_power_constraints(
        command,
        input_power,
        input_power,
        input_power,
        power_config=power,
        hardware_config=hardware,
    )

    assert result.fully_feasible
    assert np.all(projected.gain[~active] == 1.0)
    assert np.all(
        projected.gain[active] ** 2 * input_power[active]
        <= hardware.per_active_element_saturation_power + 1.0e-8
    )


def test_time_limit_truncation_can_be_stored_as_nonterminal() -> None:
    replay = ReplayBuffer(3, 2, capacity=4, seed=0)
    replay.add(
        np.zeros(3),
        np.zeros(2),
        0.0,
        np.ones(3),
        terminal=False,
    )
    _, _, _, _, terminals = replay.sample(1, torch.device("cpu"))
    assert float(terminals.item()) == 0.0


def test_environment_smoke_step() -> None:
    config = load_environment_config(CONFIG_PATH)
    config = replace(
        config,
        probing=replace(config.probing, samples_per_step=64),
        robust=RobustConfig(
            objective_samples=4,
            cvar_alpha=1.0,
            minimum_tail_samples=4,
        ),
        max_episode_steps=2,
    )
    environment = RobustFullSchemeEnvironment(config, seed=0)
    state, _ = environment.reset()
    assert state.size == environment.state_dim
    action = environment.heuristic_action()
    next_state, reward, terminated, truncated, info = environment.step(action)
    assert next_state.size == environment.state_dim
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert "training_key_rate_bps" in info
