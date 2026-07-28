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
from active_star_ris.full_scheme_v2.power import (
    conservative_input_powers,
    project_command_to_power_constraints,
)
from active_star_ris.full_scheme_v2.power import (
    conservative_input_powers,
    project_command_to_power_constraints,
)
from active_star_ris.full_scheme_v2.models import (
    IdealSurfaceCommand,
)
from active_star_ris.full_scheme_v2.td3 import ReplayBuffer

from active_star_ris.full_scheme_v2.channels import (
    complex_normal,
    correlated_eve_channel,
)

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
    assert "final_key_rate_bps" in info

    assert (
        "system_training_key_rate_bps"
        in info
    )
    assert (
        "system_final_key_rate_bps"
        in info
    )

    assert np.isfinite(
        info[
            "system_training_key_rate_bps"
        ]
    )
    assert np.isfinite(
        info[
            "system_final_key_rate_bps"
        ]
    )

    assert (
        info[
            "system_training_key_rate_bps"
        ]
        >= 0.0
    )
    assert (
        info[
            "system_final_key_rate_bps"
        ]
        >= 0.0
    )

    assert "bypassed_active_elements" in info
    assert "remaining_active_elements" in info


def test_hardware_margin_does_not_increase_commanded_gain() -> None:
    n = 8
    active = build_active_mask(n, 0.5)

    hardware = HardwareConfig(
        maximum_active_gain=4.0,
        per_active_element_saturation_power=100.0,
    )

    power = PowerConfig(
        maximum_rf_output_power=1000.0,
        maximum_total_dc_power=1000.0,
        hardware_gain_margin_db=1.0,
    )

    action_dim = int(
        np.count_nonzero(active) + 3 * n
    )

    action = np.zeros(action_dim, dtype=np.float64)

    command = decode_action(
        action,
        active,
        hardware,
    )

    requested_gain = command.gain.copy()

    input_power = np.full(
        n,
        0.01,
        dtype=np.float64,
    )

    projected, result = (
        project_command_to_power_constraints(
            command,
            input_power,
            input_power,
            input_power,
            power_config=power,
            hardware_config=hardware,
        )
    )

    assert result.fully_feasible

    # 鲁棒裕量只能使允许的控制增益更保守，
    # 绝不能主动增加控制增益。
    assert np.all(
        projected.gain[active]
        <= requested_gain[active] + 1.0e-12
    )


def test_key_rate_duration_accounts_for_pilot_symbols() -> None:
    source = np.linspace(
        -2.0,
        2.0,
        1024,
        dtype=np.float64,
    ).astype(np.complex128)

    key_config = KeyGenerationConfig(
        privacy_margin_bits=0,
        verification_tag_bits=1,
    )

    probing = ProbingConfig(
        samples_per_step=1024,
        pilot_symbols_controller=8,
    )

    short_result = evaluate_key_rate(
        source,
        source,
        key_config=key_config,
        probing_config=probing,
        rng=np.random.default_rng(1),
        full_protocol=False,
        reverse_pilot_symbols=8,
    )

    long_result = evaluate_key_rate(
        source,
        source,
        key_config=key_config,
        probing_config=probing,
        rng=np.random.default_rng(1),
        full_protocol=False,
        reverse_pilot_symbols=32,
    )

    assert (
        long_result.frame_duration_seconds
        > short_result.frame_duration_seconds
    )


def test_hardware_scale_is_in_oracle_context() -> None:
    config = load_environment_config(
        CONFIG_PATH
    )

    robust = replace(
        config.robust,
        include_oracle_impairment_context=True,
        hardware_error_scale_min=1.5,
        hardware_error_scale_max=1.5,
    )

    config = replace(
        config,
        robust=robust,
        max_episode_steps=1,
    )

    environment = RobustFullSchemeEnvironment(
        config,
        seed=10,
    )

    state, info = environment.reset(
        seed=10
    )

    assert state.size == environment.state_dim

    domain = info["domain"]

    assert np.isclose(
        domain.hardware_error_scale,
        1.5,
    )

    # 上下界相同时，归一化硬件误差特征为 0
    assert np.isclose(
        state[-1],
        0.0,
        atol=1.0e-6,
    )


def test_amplifier_noise_scale_increases_input_power() -> None:
    n = 8

    channel = np.ones(
        n,
        dtype=np.complex128,
    )

    probing = ProbingConfig(
        input_referred_amplifier_noise_variance=0.1,
    )

    power = PowerConfig()

    low = conservative_input_powers(
        channel,
        channel,
        channel,
        nmse_db=-20.0,
        probing=probing,
        power=power,
        amplifier_noise_scale=0.5,
    )

    high = conservative_input_powers(
        channel,
        channel,
        channel,
        nmse_db=-20.0,
        probing=probing,
        power=power,
        amplifier_noise_scale=2.0,
    )

    for low_values, high_values in zip(
        low,
        high,
        strict=True,
    ):
        assert np.all(
            high_values > low_values
        )


def test_power_projection_uses_passive_bypass() -> None:
    n = 4

    active_mask = np.ones(
        n,
        dtype=bool,
    )

    command = IdealSurfaceCommand(
        gain=np.full(
            n,
            2.0,
            dtype=np.float64,
        ),
        beta_transmission=np.full(
            n,
            0.5,
            dtype=np.float64,
        ),
        phase_transmission=np.zeros(
            n,
            dtype=np.float64,
        ),
        phase_reflection=np.zeros(
            n,
            dtype=np.float64,
        ),
        active_mask=active_mask,
    )

    hardware = HardwareConfig(
        maximum_active_gain=4.0,
        per_active_element_saturation_power=100.0,
        gain_quantization_bits=None,
    )

    # active 单元的固定控制与偏置功耗很高，
    # passive 单元功耗很低。
    power = PowerConfig(
        maximum_rf_output_power=100.0,
        maximum_total_dc_power=0.1,
        controller_static_power=0.0,
        switching_network_static_power=0.0,
        passive_element_control_power=0.001,
        active_element_control_power=0.2,
        active_element_bias_power=0.2,
        hardware_gain_margin_db=0.0,
    )

    input_power = np.zeros(
        n,
        dtype=np.float64,
    )

    projected, result = (
        project_command_to_power_constraints(
            command,
            input_power,
            input_power,
            input_power,
            power_config=power,
            hardware_config=hardware,
        )
    )

    # 所有原有源单元应切换到 passive bypass
    assert not np.any(
        projected.active_mask
    )

    assert np.allclose(
        projected.gain,
        1.0,
    )

    assert result.fully_feasible

    expected_passive_power = (
        n
        * power.passive_element_control_power
    )

    assert np.isclose(
        result.total_surface_dc_power,
        expected_passive_power,
    )


def test_experiment_summary_contains_system_rate() -> None:
    config = load_environment_config(
        CONFIG_PATH
    )

    config = replace(
        config,
        probing=replace(
            config.probing,
            samples_per_step=64,
        ),
        robust=replace(
            config.robust,
            objective_samples=4,
            cvar_alpha=1.0,
            minimum_tail_samples=4,
        ),
        max_episode_steps=1,
    )

    environment = RobustFullSchemeEnvironment(
        config,
        seed=20,
    )

    summary = evaluate_policy(
        environment,
        heuristic_policy,
        method="test",
        episodes=2,
        seed=20,
    )

    assert np.isfinite(
        summary.mean_system_training_key_rate_bps
    )
    assert np.isfinite(
        summary.mean_system_final_key_rate_bps
    )

    assert (
        summary.std_system_final_key_rate_bps
        >= 0.0
    )
    assert (
        summary.ci95_system_final_key_rate_bps
        >= 0.0
    )


def test_correlated_eve_channel_power_and_correlation():
    rng = np.random.default_rng(100)

    legitimate = complex_normal(
        rng,
        (10000, 4),
        variance=1.0,
    )

    eve = correlated_eve_channel(
        legitimate,
        average_power=0.5,
        correlation=0.7,
        rng=rng,
    )

    measured_power = float(
        np.mean(np.abs(eve) ** 2)
    )

    measured_correlation = float(
        np.abs(
            np.vdot(
                legitimate.reshape(-1),
                eve.reshape(-1),
            )
        )
        / (
            np.linalg.norm(
                legitimate.reshape(-1)
            )
            * np.linalg.norm(
                eve.reshape(-1)
            )
        )
    )

    assert np.isclose(
        measured_power,
        0.5,
        rtol=0.05,
    )

    assert np.isclose(
        measured_correlation,
        0.7,
        atol=0.05,
    )


def test_eve_leakage_reduces_training_key_bits():
    rng = np.random.default_rng(101)

    source = (
        rng.normal(size=4096)
        + 1j * rng.normal(size=4096)
    )

    observation_a = source
    observation_b = (
        source
        + 0.01
        * (
            rng.normal(size=4096)
            + 1j * rng.normal(size=4096)
        )
    )

    no_eve = evaluate_key_rate(
        observation_a,
        observation_b,
        key_config=KeyGenerationConfig(),
        probing_config=ProbingConfig(
            samples_per_step=4096
        ),
        rng=rng,
        full_protocol=False,
        eve_leakage_bits_per_retained_bit=0.0,
    )

    strong_eve = evaluate_key_rate(
        observation_a,
        observation_b,
        key_config=KeyGenerationConfig(),
        probing_config=ProbingConfig(
            samples_per_step=4096
        ),
        rng=rng,
        full_protocol=False,
        eve_leakage_bits_per_retained_bit=0.5,
    )

    assert (
        strong_eve.training_secret_bits
        < no_eve.training_secret_bits
    )


def test_full_protocol_uses_external_min_entropy():
    rng = np.random.default_rng(102)

    source = (
        rng.normal(size=4096)
        + 1j * rng.normal(size=4096)
    )

    result = evaluate_key_rate(
        source,
        source,
        key_config=KeyGenerationConfig(
            privacy_margin_bits=0,
            verification_tag_bits=1,
        ),
        probing_config=ProbingConfig(
            samples_per_step=4096
        ),
        rng=rng,
        full_protocol=True,
        conditional_min_entropy_bits=100,
    )

    assert result.estimated_entropy_bits == 100

