from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from active_star_ris.full_scheme_v2.channels import (
    complex_normal,
    evolve_block,
    gauss_markov_update,
)
from active_star_ris.full_scheme_v2.config import load_config
from active_star_ris.full_scheme_v2.environment import ActiveStarRisKeyEnvironment
from active_star_ris.full_scheme_v2.power import project_gains as real_project_gains


def _reference_block(
    initial: np.ndarray | complex,
    samples: int,
    correlation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    initial_array = np.asarray(initial, dtype=np.complex128)
    result = np.empty((samples,) + initial_array.shape, dtype=np.complex128)
    result[0] = initial_array
    for index in range(1, samples):
        result[index] = gauss_markov_update(
            result[index - 1],
            correlation,
            rng,
        )
    return result


def test_fast_evolve_block_preserves_rng_and_numerics() -> None:
    initial = np.asarray([1.0 + 2.0j, 0.2 - 0.1j, -0.4 + 0.8j])
    reference_rng = np.random.default_rng(12345)
    fast_rng = np.random.default_rng(12345)
    reference = _reference_block(initial, 64, 0.995, reference_rng)
    fast = evolve_block(initial, 64, 0.995, fast_rng)
    np.testing.assert_allclose(fast, reference, rtol=5.0e-13, atol=5.0e-13)
    assert fast_rng.random() == reference_rng.random()


def test_fast_evolve_block_zero_input_does_not_advance_rng() -> None:
    reference_rng = np.random.default_rng(88)
    fast_rng = np.random.default_rng(88)
    reference = _reference_block(np.zeros(4, dtype=np.complex128), 16, 0.9, reference_rng)
    fast = evolve_block(np.zeros(4, dtype=np.complex128), 16, 0.9, fast_rng)
    np.testing.assert_array_equal(fast, reference)
    assert fast_rng.random() == reference_rng.random()


def test_gain_projection_runs_once_per_robust_action() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/full_scheme_v2.yaml")
    config = replace(
        config,
        probing=replace(config.probing, samples_per_step=8),
        robust=replace(config.robust, objective_samples=3),
        environment=replace(config.environment, episode_length=2),
    )
    env = ActiveStarRisKeyEnvironment(config)
    env.reset(seed=7)
    action = np.zeros(env.action_dimension, dtype=np.float32)
    with patch(
        "active_star_ris.full_scheme_v2.environment.project_gains",
        wraps=real_project_gains,
    ) as mocked:
        summary = env.evaluate_action(action, objective_samples=3)
    assert mocked.call_count == 1
    assert np.isfinite(summary.robust_reward)
    assert hasattr(summary, "mean_post_reconciliation_kdr")
