from __future__ import annotations

from dataclasses import replace

import numpy as np

from active_star_ris.rl_environment import (
    RobustActiveStarRISEnv,
    RobustEnvironmentConfig,
)
from active_star_ris.td3_agent import TD3Agent, TD3Config
from active_star_ris.td3_training import (
    TD3TrainingConfig,
    evaluate_td3_policy,
    train_td3,
)


def tiny_environment(seed: int) -> RobustActiveStarRISEnv:
    config = RobustEnvironmentConfig(
        num_elements=4,
        num_active_elements=1,
        max_episode_steps=4,
        probing_samples_per_step=8,
    )
    return RobustActiveStarRISEnv(config, seed=seed)


def test_short_td3_training_loop_runs_end_to_end() -> None:
    environment = tiny_environment(10)
    evaluation_environment = tiny_environment(20)
    agent = TD3Agent(
        environment.state_dim,
        environment.action_dim,
        TD3Config(
            hidden_dimensions=(32, 32),
            policy_delay=2,
            exploration_noise_std=0.05,
        ),
        device="cpu",
        seed=10,
    )
    result = train_td3(
        environment,
        agent,
        TD3TrainingConfig(
            total_environment_steps=12,
            replay_capacity=64,
            random_action_steps=4,
            learning_starts=4,
            batch_size=4,
            gradient_steps_per_environment_step=1,
            evaluation_interval=6,
            evaluation_episodes=1,
            seed=10,
        ),
        evaluation_environment=evaluation_environment,
    )
    assert len(result.replay_buffer) == 12
    assert agent.update_count > 0
    assert len(result.history.episode_returns) == 3
    assert len(result.history.critic_losses) > 0
    assert len(result.history.actor_losses) > 0
    assert result.final_evaluation is not None
    assert np.isfinite(result.final_evaluation.mean_return)
    assert 0.0 <= result.final_evaluation.robust_feasibility_rate <= 1.0


def test_evaluation_does_not_add_exploration_noise() -> None:
    environment = tiny_environment(30)
    agent = TD3Agent(
        environment.state_dim,
        environment.action_dim,
        TD3Config(hidden_dimensions=(16, 16)),
        device="cpu",
        seed=30,
    )
    summary = evaluate_td3_policy(environment, agent, episodes=2, seed=40)
    assert np.isfinite(summary.mean_return)
    assert np.isfinite(summary.mean_key_rate)
    assert np.isfinite(summary.mean_surface_power)
