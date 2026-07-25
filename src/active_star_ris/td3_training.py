from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .rl_environment import RobustActiveStarRISEnv
from .td3_agent import ReplayBuffer, TD3Agent, TD3UpdateMetrics


@dataclass(frozen=True)
class TD3TrainingConfig:
    """Training-loop settings separated from TD3 network hyperparameters."""

    total_environment_steps: int = 100_000
    replay_capacity: int = 200_000
    random_action_steps: int = 2_000
    learning_starts: int = 2_000
    batch_size: int = 256
    gradient_steps_per_environment_step: int = 1
    evaluation_interval: int = 5_000
    evaluation_episodes: int = 10
    seed: int = 0

    def validate(self) -> None:
        positive = {
            "total_environment_steps": self.total_environment_steps,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "gradient_steps_per_environment_step": (
                self.gradient_steps_per_environment_step
            ),
            "evaluation_episodes": self.evaluation_episodes,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "random_action_steps": self.random_action_steps,
            "learning_starts": self.learning_starts,
            "evaluation_interval": self.evaluation_interval,
        }.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")


@dataclass(frozen=True)
class EvaluationSummary:
    mean_return: float
    mean_key_rate: float
    mean_key_disagreement_rate: float
    mean_reciprocity: float
    mean_surface_power: float
    robust_feasibility_rate: float


@dataclass
class TD3TrainingHistory:
    environment_steps: list[int] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    critic_losses: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    evaluation_steps: list[int] = field(default_factory=list)
    evaluation_returns: list[float] = field(default_factory=list)
    evaluation_key_rates: list[float] = field(default_factory=list)
    evaluation_key_disagreement_rates: list[float] = field(default_factory=list)
    evaluation_reciprocities: list[float] = field(default_factory=list)
    evaluation_surface_powers: list[float] = field(default_factory=list)
    evaluation_feasibility_rates: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class TD3TrainingResult:
    history: TD3TrainingHistory
    replay_buffer: ReplayBuffer
    final_evaluation: EvaluationSummary | None


def evaluate_td3_policy(
    environment: RobustActiveStarRISEnv,
    agent: TD3Agent,
    *,
    episodes: int,
    seed: int = 10_000,
) -> EvaluationSummary:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    returns: list[float] = []
    key_rates: list[float] = []
    kdrs: list[float] = []
    reciprocities: list[float] = []
    powers: list[float] = []
    feasibility: list[float] = []

    for episode in range(episodes):
        state, _ = environment.reset(seed=seed + episode)
        episode_return = 0.0
        while True:
            action = agent.select_action(state, explore=False)
            state, reward, terminated, truncated, info = environment.step(
                action
            )
            episode_return += reward
            key_rates.append(float(info["weighted_key_rate"]))
            kdrs.append(float(info["weighted_key_disagreement_rate"]))
            reciprocities.append(float(info["weighted_reciprocity"]))
            powers.append(float(info["total_surface_power"]))
            feasibility.append(float(bool(info["robustly_feasible"])))
            if terminated or truncated:
                break
        returns.append(episode_return)

    return EvaluationSummary(
        mean_return=float(np.mean(returns)),
        mean_key_rate=float(np.mean(key_rates)),
        mean_key_disagreement_rate=float(np.mean(kdrs)),
        mean_reciprocity=float(np.mean(reciprocities)),
        mean_surface_power=float(np.mean(powers)),
        robust_feasibility_rate=float(np.mean(feasibility)),
    )


def train_td3(
    environment: RobustActiveStarRISEnv,
    agent: TD3Agent,
    config: TD3TrainingConfig | None = None,
    *,
    evaluation_environment: RobustActiveStarRISEnv | None = None,
    progress_callback: Callable[[int, TD3TrainingHistory], None] | None = None,
) -> TD3TrainingResult:
    """Train TD3 against the robust STAR-RIS environment.

    The replay terminal flag is true for both environment termination and time
    truncation.  This prevents bootstrapping beyond a deliberately finite
    episode and keeps the training definition consistent with the configured
    maximum episode length.
    """

    training_config = TD3TrainingConfig() if config is None else config
    training_config.validate()
    replay_buffer = ReplayBuffer(
        environment.state_dim,
        environment.action_dim,
        training_config.replay_capacity,
        seed=training_config.seed,
    )
    history = TD3TrainingHistory()
    state, _ = environment.reset(seed=training_config.seed)
    episode_return = 0.0
    episode_length = 0

    for step in range(1, training_config.total_environment_steps + 1):
        if step <= training_config.random_action_steps:
            action = environment.sample_action()
        else:
            action = agent.select_action(state, explore=True)

        next_state, reward, terminated, truncated, _ = environment.step(
            action
        )
        done = bool(terminated or truncated)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward
        episode_length += 1

        if (
            step >= training_config.learning_starts
            and len(replay_buffer) >= training_config.batch_size
        ):
            for _ in range(
                training_config.gradient_steps_per_environment_step
            ):
                metrics: TD3UpdateMetrics | None = agent.train_step(
                    replay_buffer, training_config.batch_size
                )
                if metrics is not None:
                    history.critic_losses.append(metrics.critic_loss)
                    if metrics.actor_loss is not None:
                        history.actor_losses.append(metrics.actor_loss)

        if done:
            history.environment_steps.append(step)
            history.episode_returns.append(float(episode_return))
            history.episode_lengths.append(int(episode_length))
            state, _ = environment.reset()
            episode_return = 0.0
            episode_length = 0

        if (
            evaluation_environment is not None
            and training_config.evaluation_interval > 0
            and step % training_config.evaluation_interval == 0
        ):
            summary = evaluate_td3_policy(
                evaluation_environment,
                agent,
                episodes=training_config.evaluation_episodes,
                seed=training_config.seed + 100_000 + step,
            )
            history.evaluation_steps.append(step)
            history.evaluation_returns.append(summary.mean_return)
            history.evaluation_key_rates.append(summary.mean_key_rate)
            history.evaluation_key_disagreement_rates.append(
                summary.mean_key_disagreement_rate
            )
            history.evaluation_reciprocities.append(
                summary.mean_reciprocity
            )
            history.evaluation_surface_powers.append(
                summary.mean_surface_power
            )
            history.evaluation_feasibility_rates.append(
                summary.robust_feasibility_rate
            )

        if progress_callback is not None:
            progress_callback(step, history)

    final_evaluation = None
    if evaluation_environment is not None:
        final_evaluation = evaluate_td3_policy(
            evaluation_environment,
            agent,
            episodes=training_config.evaluation_episodes,
            seed=training_config.seed + 999_999,
        )
    return TD3TrainingResult(
        history=history,
        replay_buffer=replay_buffer,
        final_evaluation=final_evaluation,
    )
