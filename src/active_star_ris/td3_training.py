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
                raise ValueError(
                    f"{name} must be positive"
                )

        nonnegative = {
            "random_action_steps": self.random_action_steps,
            "learning_starts": self.learning_starts,
            "evaluation_interval": self.evaluation_interval,
        }

        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )

        if self.replay_capacity < self.batch_size:
            raise ValueError(
                "replay_capacity must be at least batch_size"
            )


@dataclass(frozen=True)
class EvaluationSummary:
    """Deterministic TD3 policy evaluation summary."""

    mean_return: float

    mean_key_rate_bits_per_sample: float
    mean_key_rate_bits_per_second: float

    mean_key_disagreement_rate: float
    mean_reciprocity: float
    mean_surface_power: float

    mean_robust_reward: float
    mean_cvar_reward: float
    mean_worst_sample_reward: float

    robust_feasibility_rate: float
    mean_effective_active_elements: float
    mean_power_violation: float

    @property
    def mean_key_rate(self) -> float:
        """Backward-compatible alias using bit/sample."""

        return self.mean_key_rate_bits_per_sample


@dataclass
class TD3TrainingHistory:
    """All training and periodic-evaluation records."""

    environment_steps: list[int] = field(
        default_factory=list
    )
    episode_returns: list[float] = field(
        default_factory=list
    )
    episode_lengths: list[int] = field(
        default_factory=list
    )

    critic_update_indices: list[int] = field(
        default_factory=list
    )
    critic_losses: list[float] = field(
        default_factory=list
    )

    actor_update_indices: list[int] = field(
        default_factory=list
    )
    actor_losses: list[float] = field(
        default_factory=list
    )

    evaluation_steps: list[int] = field(
        default_factory=list
    )
    evaluation_returns: list[float] = field(
        default_factory=list
    )

    # 兼容旧接口：bit/sample。
    evaluation_key_rates: list[float] = field(
        default_factory=list
    )

    evaluation_key_rates_bits_per_second: list[float] = field(
        default_factory=list
    )
    evaluation_key_disagreement_rates: list[float] = field(
        default_factory=list
    )
    evaluation_reciprocities: list[float] = field(
        default_factory=list
    )
    evaluation_surface_powers: list[float] = field(
        default_factory=list
    )

    evaluation_robust_rewards: list[float] = field(
        default_factory=list
    )
    evaluation_cvar_rewards: list[float] = field(
        default_factory=list
    )
    evaluation_worst_sample_rewards: list[float] = field(
        default_factory=list
    )

    evaluation_feasibility_rates: list[float] = field(
        default_factory=list
    )
    evaluation_effective_active_elements: list[float] = field(
        default_factory=list
    )
    evaluation_power_violations: list[float] = field(
        default_factory=list
    )


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
    """Evaluate one deterministic TD3 policy."""

    if episodes <= 0:
        raise ValueError(
            "episodes must be positive"
        )

    returns: list[float] = []

    key_rates_per_sample: list[float] = []
    key_rates_per_second: list[float] = []

    kdrs: list[float] = []
    reciprocities: list[float] = []
    powers: list[float] = []

    robust_rewards: list[float] = []
    cvar_rewards: list[float] = []
    worst_rewards: list[float] = []

    feasibility: list[float] = []
    effective_active_elements: list[float] = []
    power_violations: list[float] = []

    for episode in range(episodes):
        state, _ = environment.reset(
            seed=seed + episode
        )

        episode_return = 0.0

        while True:
            action = agent.select_action(
                state,
                explore=False,
            )

            (
                state,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(action)

            episode_return += float(reward)

            key_rates_per_sample.append(
                float(
                    info[
                        "weighted_key_rate_bits_per_sample"
                    ]
                )
            )
            key_rates_per_second.append(
                float(
                    info[
                        "weighted_key_rate_bits_per_second"
                    ]
                )
            )

            kdrs.append(
                float(
                    info["raw_key_disagreement_rate"]
                )
            )
            reciprocities.append(
                float(
                    info["observation_reciprocity"]
                )
            )
            powers.append(
                float(
                    info["total_surface_power"]
                )
            )

            robust_rewards.append(
                float(
                    info["robust_reward"]
                )
            )
            cvar_rewards.append(
                float(
                    info["cvar_reward"]
                )
            )
            worst_rewards.append(
                float(
                    info["worst_sample_reward"]
                )
            )

            feasibility.append(
                float(
                    bool(info["robustly_feasible"])
                )
            )
            effective_active_elements.append(
                float(
                    info["effective_active_elements"]
                )
            )
            power_violations.append(
                float(
                    info["power_violation"]
                )
            )

            if terminated or truncated:
                break

        returns.append(
            float(episode_return)
        )

    return EvaluationSummary(
        mean_return=float(
            np.mean(returns)
        ),
        mean_key_rate_bits_per_sample=float(
            np.mean(key_rates_per_sample)
        ),
        mean_key_rate_bits_per_second=float(
            np.mean(key_rates_per_second)
        ),
        mean_key_disagreement_rate=float(
            np.mean(kdrs)
        ),
        mean_reciprocity=float(
            np.mean(reciprocities)
        ),
        mean_surface_power=float(
            np.mean(powers)
        ),
        mean_robust_reward=float(
            np.mean(robust_rewards)
        ),
        mean_cvar_reward=float(
            np.mean(cvar_rewards)
        ),
        mean_worst_sample_reward=float(
            np.mean(worst_rewards)
        ),
        robust_feasibility_rate=float(
            np.mean(feasibility)
        ),
        mean_effective_active_elements=float(
            np.mean(effective_active_elements)
        ),
        mean_power_violation=float(
            np.mean(power_violations)
        ),
    )


def train_td3(
    environment: RobustActiveStarRISEnv,
    agent: TD3Agent,
    config: TD3TrainingConfig | None = None,
    *,
    evaluation_environment: RobustActiveStarRISEnv | None = None,
    progress_callback: (
        Callable[
            [int, TD3TrainingHistory],
            None,
        ]
        | None
    ) = None,
) -> TD3TrainingResult:
    """Train TD3 against the robust STAR-RIS environment."""

    training_config = (
        TD3TrainingConfig()
        if config is None
        else config
    )
    training_config.validate()

    replay_buffer = ReplayBuffer(
        environment.state_dim,
        environment.action_dim,
        training_config.replay_capacity,
        seed=training_config.seed,
    )

    history = TD3TrainingHistory()

    state, _ = environment.reset(
        seed=training_config.seed
    )

    episode_return = 0.0
    episode_length = 0

    for step in range(
        1,
        training_config.total_environment_steps + 1,
    ):
        if step <= training_config.random_action_steps:
            action = environment.sample_action()
        else:
            action = agent.select_action(
                state,
                explore=True,
            )

        (
            next_state,
            reward,
            terminated,
            truncated,
            _,
        ) = environment.step(action)

        done = bool(
            terminated or truncated
        )

        replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

        state = next_state
        episode_return += float(reward)
        episode_length += 1

        if (
            step >= training_config.learning_starts
            and len(replay_buffer)
            >= training_config.batch_size
        ):
            for _ in range(
                training_config
                .gradient_steps_per_environment_step
            ):
                metrics: TD3UpdateMetrics | None = (
                    agent.train_step(
                        replay_buffer,
                        training_config.batch_size,
                    )
                )

                if metrics is None:
                    continue

                history.critic_update_indices.append(
                    int(metrics.update_index)
                )
                history.critic_losses.append(
                    float(metrics.critic_loss)
                )

                if metrics.actor_loss is not None:
                    history.actor_update_indices.append(
                        int(metrics.update_index)
                    )
                    history.actor_losses.append(
                        float(metrics.actor_loss)
                    )

        if done:
            history.environment_steps.append(
                int(step)
            )
            history.episode_returns.append(
                float(episode_return)
            )
            history.episode_lengths.append(
                int(episode_length)
            )

            state, _ = environment.reset()
            episode_return = 0.0
            episode_length = 0

        should_evaluate = (
            evaluation_environment is not None
            and training_config.evaluation_interval > 0
            and (
                step
                % training_config.evaluation_interval
                == 0
            )
        )

        if should_evaluate:
            summary = evaluate_td3_policy(
                evaluation_environment,
                agent,
                episodes=(
                    training_config.evaluation_episodes
                ),
                seed=(
                    training_config.seed
                    + 100_000
                    + step
                ),
            )

            history.evaluation_steps.append(
                int(step)
            )
            history.evaluation_returns.append(
                float(summary.mean_return)
            )

            history.evaluation_key_rates.append(
                float(
                    summary
                    .mean_key_rate_bits_per_sample
                )
            )
            history.evaluation_key_rates_bits_per_second.append(
                float(
                    summary
                    .mean_key_rate_bits_per_second
                )
            )

            history.evaluation_key_disagreement_rates.append(
                float(
                    summary
                    .mean_key_disagreement_rate
                )
            )
            history.evaluation_reciprocities.append(
                float(
                    summary.mean_reciprocity
                )
            )
            history.evaluation_surface_powers.append(
                float(
                    summary.mean_surface_power
                )
            )

            history.evaluation_robust_rewards.append(
                float(
                    summary.mean_robust_reward
                )
            )
            history.evaluation_cvar_rewards.append(
                float(
                    summary.mean_cvar_reward
                )
            )
            history.evaluation_worst_sample_rewards.append(
                float(
                    summary
                    .mean_worst_sample_reward
                )
            )

            history.evaluation_feasibility_rates.append(
                float(
                    summary
                    .robust_feasibility_rate
                )
            )
            history.evaluation_effective_active_elements.append(
                float(
                    summary
                    .mean_effective_active_elements
                )
            )
            history.evaluation_power_violations.append(
                float(
                    summary.mean_power_violation
                )
            )

        if progress_callback is not None:
            progress_callback(
                step,
                history,
            )

    # 训练可能在一个episode尚未自然结束时达到总步数。
    # 保存该未完成episode的累计回报和实际长度，
    # 避免短训练、调试训练或中断训练丢失最后一段数据。
    if episode_length > 0:
        history.environment_steps.append(
            int(
                training_config.total_environment_steps
            )
        )
        history.episode_returns.append(
            float(episode_return)
        )
        history.episode_lengths.append(
            int(episode_length)
        )

    final_evaluation = None

    if evaluation_environment is not None:
        final_evaluation = evaluate_td3_policy(
            evaluation_environment,
            agent,
            episodes=(
                training_config.evaluation_episodes
            ),
            seed=(
                training_config.seed
                + 999_999
            ),
        )

    return TD3TrainingResult(
        history=history,
        replay_buffer=replay_buffer,
        final_evaluation=final_evaluation,
    )