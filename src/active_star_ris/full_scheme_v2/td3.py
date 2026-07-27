from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .environment import RobustFullSchemeEnvironment


@dataclass(frozen=True)
class TD3Config:
    hidden_dimensions: tuple[int, ...] = (256, 256)
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    discount_factor: float = 0.99
    soft_update_rate: float = 0.005
    target_policy_noise: float = 0.20
    target_noise_clip: float = 0.50
    policy_delay: int = 2
    exploration_noise_std: float = 0.10

    def validate(self) -> None:
        if not self.hidden_dimensions or any(value < 1 for value in self.hidden_dimensions):
            raise ValueError("hidden_dimensions must contain positive integers")
        for name in ("actor_learning_rate", "critic_learning_rate"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must lie in [0, 1]")
        if not 0.0 < self.soft_update_rate <= 1.0:
            raise ValueError("soft_update_rate must lie in (0, 1]")
        if self.target_policy_noise < 0.0 or self.target_noise_clip < 0.0:
            raise ValueError("target noise values cannot be negative")
        if self.policy_delay < 1:
            raise ValueError("policy_delay must be positive")
        if self.exploration_noise_std < 0.0:
            raise ValueError("exploration_noise_std cannot be negative")


@dataclass(frozen=True)
class TrainingConfig:
    total_environment_steps: int = 100_000
    replay_capacity: int = 200_000
    random_action_steps: int = 5_000
    learning_starts: int = 5_000
    batch_size: int = 256
    gradient_steps_per_environment_step: int = 1
    evaluation_interval: int = 5_000
    evaluation_episodes: int = 10
    seed: int = 0

    def validate(self) -> None:
        for name in (
            "total_environment_steps",
            "replay_capacity",
            "batch_size",
            "gradient_steps_per_environment_step",
            "evaluation_interval",
            "evaluation_episodes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("random_action_steps", "learning_starts"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity")


@dataclass
class TrainingHistory:
    environment_steps: list[int] = field(
        default_factory=list
    )
    episode_returns: list[float] = field(
        default_factory=list
    )
    episode_lengths: list[int] = field(
        default_factory=list
    )
    actor_losses: list[float] = field(
        default_factory=list
    )
    critic_losses: list[float] = field(
        default_factory=list
    )

    evaluation_steps: list[int] = field(
        default_factory=list
    )
    evaluation_returns: list[float] = field(
        default_factory=list
    )

    # 旧字段保留，兼容旧 training_history.json
    evaluation_key_rates_bps: list[float] = field(
        default_factory=list
    )

    evaluation_training_key_rates_bps: list[float] = field(
        default_factory=list
    )
    evaluation_final_key_rates_bps: list[float] = field(
        default_factory=list
    )
    evaluation_system_training_key_rates_bps: list[
        float
    ] = field(default_factory=list)
    evaluation_system_final_key_rates_bps: list[
        float
    ] = field(default_factory=list)

    evaluation_raw_kdr: list[float] = field(
        default_factory=list
    )
    evaluation_reciprocity: list[float] = field(
        default_factory=list
    )
    evaluation_surface_power: list[float] = field(
        default_factory=list
    )
    evaluation_feasibility: list[float] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class EvaluationSummary:
    mean_return: float
    std_return: float

    mean_training_key_rate_bps: float
    mean_final_key_rate_bps: float

    mean_system_training_key_rate_bps: float
    mean_system_final_key_rate_bps: float

    mean_raw_kdr: float
    mean_post_reconciliation_kdr: float
    mean_reciprocity: float
    mean_surface_dc_power: float
    mean_feasibility_rate: float


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        seed: int,
    ) -> None:
        self.capacity = int(capacity)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.terminals = np.empty((capacity, 1), dtype=np.float32)
        self._size = 0
        self._position = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
    ) -> None:
        index = self._position
        self.states[index] = np.asarray(state, dtype=np.float32)
        self.actions[index] = np.asarray(action, dtype=np.float32)
        self.rewards[index, 0] = float(reward)
        self.next_states[index] = np.asarray(next_state, dtype=np.float32)
        self.terminals[index, 0] = float(terminal)
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        if self._size < batch_size:
            raise ValueError("not enough replay samples")
        indices = self._rng.integers(0, self._size, size=batch_size)
        return tuple(
            torch.as_tensor(array[indices], device=device)
            for array in (
                self.states,
                self.actions,
                self.rewards,
                self.next_states,
                self.terminals,
            )
        )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dimensions: tuple[int, ...],
        *,
        output_tanh: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for hidden in hidden_dimensions:
            layers.extend((nn.Linear(current, hidden), nn.ReLU()))
            current = hidden
        layers.append(nn.Linear(current, output_dim))
        if output_tanh:
            layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class TwinCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dimensions: tuple[int, ...],
    ) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1 = MLP(input_dim, 1, hidden_dimensions, output_tanh=False)
        self.q2 = MLP(input_dim, 1, hidden_dimensions, output_tanh=False)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat((state, action), dim=-1)
        return self.q1(values), self.q2(values)

    def q1_only(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.q1(torch.cat((state, action), dim=-1))


class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TD3Config,
        *,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        config.validate()
        self.config = config
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self.actor = MLP(
            state_dim,
            action_dim,
            config.hidden_dimensions,
            output_tanh=True,
        ).to(self.device)
        self.actor_target = MLP(
            state_dim,
            action_dim,
            config.hidden_dimensions,
            output_tanh=True,
        ).to(self.device)
        self.critic = TwinCritic(
            state_dim,
            action_dim,
            config.hidden_dimensions,
        ).to(self.device)
        self.critic_target = TwinCritic(
            state_dim,
            action_dim,
            config.hidden_dimensions,
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=config.actor_learning_rate,
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=config.critic_learning_rate,
        )
        self.update_count = 0

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        *,
        explore: bool,
    ) -> np.ndarray:
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32)[None, :],
            device=self.device,
        )
        action = self.actor(state_tensor).cpu().numpy()[0]
        if explore and self.config.exploration_noise_std > 0.0:
            action = action + self._rng.normal(
                0.0,
                self.config.exploration_noise_std,
                size=self.action_dim,
            )
        return np.asarray(np.clip(action, -1.0, 1.0), dtype=np.float32)

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(),
                target.parameters(),
                strict=True,
            ):
                target_parameter.mul_(1.0 - tau)
                target_parameter.add_(tau * source_parameter)

    def update(
        self,
        replay: ReplayBuffer,
        batch_size: int,
    ) -> tuple[float, float | None]:
        states, actions, rewards, next_states, terminals = replay.sample(
            batch_size,
            self.device,
        )

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.config.target_policy_noise
            noise = torch.clamp(
                noise,
                -self.config.target_noise_clip,
                self.config.target_noise_clip,
            )
            next_actions = torch.clamp(
                self.actor_target(next_states) + noise,
                -1.0,
                1.0,
            )
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.minimum(target_q1, target_q2)
            target = rewards + (
                1.0 - terminals
            ) * self.config.discount_factor * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q1, target) + nn.functional.mse_loss(
            current_q2,
            target,
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_count += 1
        actor_loss_value: float | None = None
        if self.update_count % self.config.policy_delay == 0:
            actor_loss = -self.critic.q1_only(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu())

            self._soft_update(
                self.actor,
                self.actor_target,
                self.config.soft_update_rate,
            )
            self._soft_update(
                self.critic,
                self.critic_target,
                self.config.soft_update_rate,
            )

        return float(critic_loss.detach().cpu()), actor_loss_value

    def save(self, path: str | Path, *, extra: dict | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "update_count": self.update_count,
                "extra": extra or {},
            },
            output,
        )

    def load(self, path: str | Path) -> dict:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload["state_dim"] != self.state_dim or payload["action_dim"] != self.action_dim:
            raise ValueError("checkpoint dimensions do not match this agent")
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.update_count = int(payload.get("update_count", 0))
        return dict(payload.get("extra", {}))


def evaluate_agent(
    environment: RobustFullSchemeEnvironment,
    agent: TD3Agent,
    episodes: int,
    *,
    seed: int,
) -> EvaluationSummary:
    returns: list[float] = []
    training_rates: list[float] = []
    final_rates: list[float] = []
    raw_kdrs: list[float] = []
    post_kdrs: list[float] = []
    reciprocities: list[float] = []
    powers: list[float] = []
    feasibilities: list[float] = []
    system_training_rates: list[float] = []
    system_final_rates: list[float] = []

    for episode in range(episodes):
        state, _ = environment.reset(seed=seed + episode)
        episode_return = 0.0
        while True:
            action = agent.select_action(state, explore=False)
            state, reward, terminated, truncated, info = environment.step(action)
            episode_return += reward
            training_rates.append(float(info["training_key_rate_bps"]))
            final_rates.append(float(info["final_key_rate_bps"]))
            raw_kdrs.append(float(info["raw_kdr"]))
            post_kdrs.append(float(info["post_reconciliation_kdr"]))
            reciprocities.append(float(info["reciprocity"]))
            powers.append(float(info["surface_dc_power"]))
            feasibilities.append(float(info["feasibility_rate"]))
            system_training_rates.append(
                float(info["system_training_key_rate_bps"])
            )
            system_final_rates.append(
                float(info["system_final_key_rate_bps"])
            )
            if terminated or truncated:
                break
        returns.append(episode_return)

    return EvaluationSummary(
        mean_return=float(np.mean(returns)),
        std_return=(
            float(np.std(returns, ddof=1))
            if len(returns) > 1
            else 0.0
        ),
        mean_training_key_rate_bps=float(
            np.mean(training_rates)
        ),
        mean_final_key_rate_bps=float(
            np.mean(final_rates)
        ),
        mean_system_training_key_rate_bps=float(
            np.mean(system_training_rates)
        ),
        mean_system_final_key_rate_bps=float(
            np.mean(system_final_rates)
        ),
        mean_raw_kdr=float(np.mean(raw_kdrs)),
        mean_post_reconciliation_kdr=float(
            np.mean(post_kdrs)
        ),
        mean_reciprocity=float(
            np.mean(reciprocities)
        ),
        mean_surface_dc_power=float(
            np.mean(powers)
        ),
        mean_feasibility_rate=float(
            np.mean(feasibilities)
        ),
    )

def train_td3(
    environment: RobustFullSchemeEnvironment,
    agent: TD3Agent,
    config: TrainingConfig,
    *,
    evaluation_environment: RobustFullSchemeEnvironment | None = None,
    progress_callback: Callable[[int, TrainingHistory], None] | None = None,
) -> TrainingHistory:
    config.validate()
    replay = ReplayBuffer(
        environment.state_dim,
        environment.action_dim,
        config.replay_capacity,
        config.seed,
    )
    rng = np.random.default_rng(config.seed)
    history = TrainingHistory()

    state, _ = environment.reset(seed=config.seed)
    episode_return = 0.0
    episode_length = 0

    for step in range(1, config.total_environment_steps + 1):
        if step <= config.random_action_steps:
            action = rng.uniform(-1.0, 1.0, size=environment.action_dim).astype(np.float32)
        else:
            action = agent.select_action(state, explore=True)

        next_state, reward, terminated, truncated, _ = environment.step(action)
        episode_finished = bool(terminated or truncated)

        # Critical fix: a time-limit truncation is not an absorbing MDP terminal.
        bootstrap_terminal = bool(terminated)
        replay.add(
            state,
            action,
            reward,
            next_state,
            bootstrap_terminal,
        )

        state = next_state
        episode_return += reward
        episode_length += 1

        if step >= config.learning_starts and len(replay) >= config.batch_size:
            for _ in range(config.gradient_steps_per_environment_step):
                critic_loss, actor_loss = agent.update(replay, config.batch_size)
                history.critic_losses.append(critic_loss)
                if actor_loss is not None:
                    history.actor_losses.append(actor_loss)

        if episode_finished:
            history.environment_steps.append(step)
            history.episode_returns.append(float(episode_return))
            history.episode_lengths.append(int(episode_length))
            state, _ = environment.reset()
            episode_return = 0.0
            episode_length = 0

        if (
            evaluation_environment is not None
            and step % config.evaluation_interval == 0
        ):
            summary = evaluate_agent(
                evaluation_environment,
                agent,
                config.evaluation_episodes,
                seed=config.seed + 100_000 + step,
            )
            history.evaluation_steps.append(step)
            history.evaluation_returns.append(
                summary.mean_return
            )

            # 旧字段兼容
            history.evaluation_key_rates_bps.append(
                summary.mean_final_key_rate_bps
            )

            history.evaluation_training_key_rates_bps.append(
                summary.mean_training_key_rate_bps
            )
            history.evaluation_final_key_rates_bps.append(
                summary.mean_final_key_rate_bps
            )

            history.evaluation_system_training_key_rates_bps.append(
                summary.mean_system_training_key_rate_bps
            )
            history.evaluation_system_final_key_rates_bps.append(
                summary.mean_system_final_key_rate_bps
            )

            history.evaluation_raw_kdr.append(
                summary.mean_raw_kdr
            )
            history.evaluation_reciprocity.append(
                summary.mean_reciprocity
            )
            history.evaluation_surface_power.append(
                summary.mean_surface_dc_power
            )
            history.evaluation_feasibility.append(
                summary.mean_feasibility_rate
            )
        if progress_callback is not None:
            progress_callback(step, history)

    return history
