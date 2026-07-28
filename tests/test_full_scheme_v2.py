from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from active_star_ris.full_scheme_v2.channels import estimate_channel
from active_star_ris.full_scheme_v2.config import FullSchemeConfig, load_config
from active_star_ris.full_scheme_v2.environment import ActiveStarRisKeyEnvironment
from active_star_ris.full_scheme_v2.hardware import decode_action, realize_coefficients, sample_static_hardware
from active_star_ris.full_scheme_v2.key_protocol import evaluate_branch_key_metrics
from active_star_ris.full_scheme_v2.models import BranchObservations, ObjectiveSample, PowerMetrics
from active_star_ris.full_scheme_v2.objective import aggregate_robust_samples


def small_config(**changes) -> FullSchemeConfig:
    config = load_config(ROOT / "configs/full_scheme_v2_paper.yaml")
    config = replace(
        config,
        geometry=replace(config.geometry, ris_rows=2, ris_columns=4),
        probing=replace(config.probing, samples_per_step=32),
        robust=replace(config.robust, objective_samples=4),
        environment=replace(config.environment, episode_length=3),
        hardware=replace(config.hardware, active_ratio=0.5),
    )
    for key, value in changes.items():
        config = replace(config, **{key: value})
    config.validate()
    return config


def test_quadrature_phase_constraint() -> None:
    config = small_config()
    n = config.geometry.num_elements
    action = np.random.default_rng(0).uniform(-1.0, 1.0, 5 * n)
    result = decode_action(
        action,
        num_elements=n,
        architecture="partially_active_fixed",
        config=config.hardware,
    )
    error = np.angle(np.exp(1j * (result.phase_reflection - result.phase_transmission - np.pi / 2.0)))
    assert np.max(np.abs(error)) < 1.0e-12


def test_fast_jitter_changes_inside_block() -> None:
    config = small_config()
    n = config.geometry.num_elements
    ideal = decode_action(
        np.zeros(5 * n),
        num_elements=n,
        architecture="partially_active_fixed",
        config=replace(config.hardware, phase_quantization_bits=None, gain_quantization_bits=None),
    )
    rng = np.random.default_rng(1)
    static = sample_static_hardware(n, config.hardware, rng)
    realized = realize_coefficients(
        ideal,
        static,
        samples=32,
        config=config.hardware,
        rng=rng,
    )
    assert realized.transmission_forward.shape == (32, n)
    assert np.var(np.angle(realized.transmission_forward[:, 0])) > 0.0


def test_zero_fast_jitter_is_constant_inside_block() -> None:
    config = small_config()
    hardware = replace(
        config.hardware,
        fast_phase_jitter_std_rad=0.0,
        static_gain_error_std_db=0.0,
        directional_gain_error_std_db=0.0,
        static_phase_error_std_rad=0.0,
        directional_phase_error_std_rad=0.0,
        transmission_split_error_std=0.0,
    )
    n = config.geometry.num_elements
    ideal = decode_action(np.zeros(5 * n), num_elements=n, architecture="passive", config=hardware)
    rng = np.random.default_rng(2)
    static = sample_static_hardware(n, hardware, rng)
    realized = realize_coefficients(ideal, static, samples=16, config=hardware, rng=rng)
    assert np.allclose(realized.transmission_forward, realized.transmission_forward[0])


def test_architecture_masks_have_exact_semantics() -> None:
    config = small_config()
    n = config.geometry.num_elements
    action = np.linspace(-1.0, 1.0, 5 * n)
    passive = decode_action(action, num_elements=n, architecture="passive", config=config.hardware)
    partial = decode_action(action, num_elements=n, architecture="partially_active_fixed", config=config.hardware)
    dynamic = decode_action(action, num_elements=n, architecture="partially_active_dynamic", config=config.hardware)
    full = decode_action(action, num_elements=n, architecture="fully_active_fixed", config=config.hardware)
    assert np.count_nonzero(passive.active_mask) == 0
    expected = round(n * config.hardware.active_ratio)
    assert np.count_nonzero(partial.active_mask) == expected
    assert np.count_nonzero(dynamic.active_mask) == expected
    assert np.count_nonzero(full.active_mask) == n


def test_lmmse_pilot_length_reduces_mean_error() -> None:
    base = small_config().channel
    h = np.ones(128, dtype=np.complex128)
    errors_short = []
    errors_long = []
    for seed in range(100):
        short = estimate_channel(h, replace(base, csi_pilot_symbols=4), np.random.default_rng(seed))
        long = estimate_channel(h, replace(base, csi_pilot_symbols=64), np.random.default_rng(seed))
        errors_short.append(np.mean(np.abs(short.estimate - h) ** 2))
        errors_long.append(np.mean(np.abs(long.estimate - h) ** 2))
    assert np.mean(errors_long) < np.mean(errors_short)


def _fake_observations(eve_quality: float, seed: int = 0) -> BranchObservations:
    rng = np.random.default_rng(seed)
    source = rng.normal(size=256) + 1j * rng.normal(size=256)
    alice = source + 0.05 * (rng.normal(size=256) + 1j * rng.normal(size=256))
    bob = source + 0.05 * (rng.normal(size=256) + 1j * rng.normal(size=256))
    independent = rng.normal(size=256) + 1j * rng.normal(size=256)
    eve = eve_quality * source + np.sqrt(max(0.0, 1.0 - eve_quality**2)) * independent
    return BranchObservations(alice, bob, eve, eve.copy(), source, source.copy())


def test_secure_rate_decreases_when_eve_improves() -> None:
    config = small_config()
    weak = evaluate_branch_key_metrics(
        _fake_observations(0.0), config.key_generation, config.probing, np.random.default_rng(0), full_protocol=False
    )
    strong = evaluate_branch_key_metrics(
        _fake_observations(0.99), config.key_generation, config.probing, np.random.default_rng(0), full_protocol=False
    )
    assert strong.secure_key_rate_bps <= weak.secure_key_rate_bps


def test_robust_summary_separates_mean_cvar_and_worst() -> None:
    config = small_config()
    env = ActiveStarRisKeyEnvironment(config)
    env.reset(seed=3)
    action = np.zeros(env.action_dimension)
    summary = env.evaluate_action(action, objective_samples=4)
    assert summary.worst_reward <= summary.cvar_reward <= summary.mean_reward + 1.0e-12
    assert 0.0 <= summary.power_violation_probability <= 1.0


def test_environment_state_and_info_use_mean_metrics() -> None:
    config = small_config()
    env = ActiveStarRisKeyEnvironment(config)
    state, _ = env.reset(seed=4)
    next_state, reward, terminated, truncated, info = env.step(np.zeros(env.action_dimension))
    assert state.shape == next_state.shape == (env.state_dimension,)
    assert np.isfinite(reward)
    assert not truncated
    assert "mean_secure_key_rate_bps" in info
    assert "cvar_secure_key_rate_bps" in info
    assert "worst_secure_key_rate_bps" in info


def test_full_active_never_silently_bypasses() -> None:
    config = small_config(environment=replace(small_config().environment, architecture="fully_active_fixed"))
    env = ActiveStarRisKeyEnvironment(config)
    env.reset(seed=5)
    summary = env.evaluate_action(np.zeros(env.action_dimension), objective_samples=2)
    assert summary.mean_active_elements == env.num_elements


def test_full_protocol_reports_operational_rate() -> None:
    config = small_config()
    metrics = evaluate_branch_key_metrics(
        _fake_observations(0.0, seed=10),
        replace(config.key_generation, privacy_margin_bits=0, verification_tag_bits=8),
        config.probing,
        np.random.default_rng(10),
        full_protocol=True,
    )
    assert metrics.final_key_bits >= 0
    assert metrics.secure_key_rate_bps >= 0.0
