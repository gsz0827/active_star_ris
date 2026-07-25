from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn
from torch.nn import functional as F

Float32Array = NDArray[np.float32]


@dataclass(frozen=True)
class TD3Config:
    """Hyperparameters for the Twin Delayed DDPG agent."""

    hidden_dimensions: tuple[int, int] = (256, 256)
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    discount_factor: float = 0.99
    soft_update_rate: float = 0.005
    target_policy_noise: float = 0.20
    target_noise_clip: float = 0.50
    policy_delay: int = 2
    exploration_noise_std: float = 0.10
    maximum_gradient_norm: float | None = 10.0

    def validate(self) -> None:
        if len(self.hidden_dimensions) != 2:
            raise ValueError("hidden_dimensions must contain exactly two widths")
        if any(width <= 0 for width in self.hidden_dimensions):
            raise ValueError("hidden layer widths must be positive")
        for name, value in {
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
        }.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must lie within [0, 1]")
        if not 0.0 < self.soft_update_rate <= 1.0:
            raise ValueError("soft_update_rate must lie within (0, 1]")
        if self.target_policy_noise < 0.0:
            raise ValueError("target_policy_noise cannot be negative")
        if self.target_noise_clip < 0.0:
            raise ValueError("target_noise_clip cannot be negative")
        if self.policy_delay <= 0:
            raise ValueError("policy_delay must be positive")
        if self.exploration_noise_std < 0.0:
            raise ValueError("exploration_noise_std cannot be negative")
        if (
            self.maximum_gradient_norm is not None
            and self.maximum_gradient_norm <= 0.0
        ):
            raise ValueError("maximum_gradient_norm must be positive or None")


@dataclass(frozen=True)
class ReplayBatch:
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor


class ReplayBuffer:
    """Preallocated replay buffer for continuous-control transitions."""

    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        capacity: int,
        *,
        seed: int | None = None,
    ) -> None:
        if state_dimension <= 0 or action_dimension <= 0:
            raise ValueError("state and action dimensions must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.state_dimension = int(state_dimension)
        self.action_dimension = int(action_dimension)
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._states = np.empty(
            (capacity, state_dimension), dtype=np.float32
        )
        self._actions = np.empty(
            (capacity, action_dimension), dtype=np.float32
        )
        self._rewards = np.empty((capacity, 1), dtype=np.float32)
        self._next_states = np.empty(
            (capacity, state_dimension), dtype=np.float32
        )
        self._dones = np.empty((capacity, 1), dtype=np.float32)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def _finite_vector(
        value: ArrayLike,
        expected_size: int,
        name: str,
    ) -> Float32Array:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != expected_size:
            raise ValueError(
                f"{name} must contain {expected_size} entries, got {array.size}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    def add(
        self,
        state: ArrayLike,
        action: ArrayLike,
        reward: float,
        next_state: ArrayLike,
        done: bool,
    ) -> None:
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        index = self._position
        self._states[index] = self._finite_vector(
            state, self.state_dimension, "state"
        )
        self._actions[index] = self._finite_vector(
            action, self.action_dimension, "action"
        )
        self._rewards[index, 0] = float(reward)
        self._next_states[index] = self._finite_vector(
            next_state, self.state_dimension, "next_state"
        )
        self._dones[index, 0] = float(bool(done))
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError(
                f"not enough transitions: requested {batch_size}, "
                f"available {self._size}"
            )
        indices = self._rng.integers(0, self._size, size=batch_size)
        return ReplayBatch(
            states=torch.as_tensor(
                self._states[indices], device=device
            ),
            actions=torch.as_tensor(
                self._actions[indices], device=device
            ),
            rewards=torch.as_tensor(
                self._rewards[indices], device=device
            ),
            next_states=torch.as_tensor(
                self._next_states[indices], device=device
            ),
            dones=torch.as_tensor(
                self._dones[indices], device=device
            ),
        )


class Actor(nn.Module):
    """TD3 actor with tanh-bounded output in [-1, 1]."""

    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimensions: Sequence[int],
    ) -> None:
        super().__init__()
        hidden_1, hidden_2 = (int(value) for value in hidden_dimensions)
        self.network = nn.Sequential(
            nn.Linear(state_dimension, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, action_dimension),
            nn.Tanh(),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        output = self.network[-2]
        if isinstance(output, nn.Linear):
            nn.init.uniform_(output.weight, -3.0e-3, 3.0e-3)
            nn.init.uniform_(output.bias, -3.0e-3, 3.0e-3)

    def forward(self, state: Tensor) -> Tensor:
        return self.network(state)


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimensions: Sequence[int],
    ) -> None:
        super().__init__()
        hidden_1, hidden_2 = (int(value) for value in hidden_dimensions)
        self.network = nn.Sequential(
            nn.Linear(state_dimension + action_dimension, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        output = self.network[-1]
        if isinstance(output, nn.Linear):
            nn.init.uniform_(output.weight, -3.0e-3, 3.0e-3)
            nn.init.uniform_(output.bias, -3.0e-3, 3.0e-3)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.network(torch.cat((state, action), dim=-1))


class TwinCritic(nn.Module):
    """Two independent Q-functions used by TD3 to reduce overestimation."""

    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimensions: Sequence[int],
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(
            state_dimension, action_dimension, hidden_dimensions
        )
        self.q2 = QNetwork(
            state_dimension, action_dimension, hidden_dimensions
        )

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def first(self, state: Tensor, action: Tensor) -> Tensor:
        return self.q1(state, action)


@dataclass(frozen=True)
class TD3UpdateMetrics:
    update_index: int
    critic_loss: float
    actor_loss: float | None
    mean_target_q: float
    mean_current_q1: float
    mean_current_q2: float
    actor_updated: bool


class TD3Agent:
    """PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradient."""

    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        config: TD3Config | None = None,
        *,
        device: str | torch.device | None = None,
        seed: int = 0,
    ) -> None:
        if state_dimension <= 0 or action_dimension <= 0:
            raise ValueError("state and action dimensions must be positive")
        self.state_dimension = int(state_dimension)
        self.action_dimension = int(action_dimension)
        self.config = TD3Config() if config is None else config
        self.config.validate()
        self.device = torch.device(
            "cuda" if device is None and torch.cuda.is_available() else
            "cpu" if device is None else device
        )
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        hidden = self.config.hidden_dimensions
        self.actor = Actor(
            self.state_dimension, self.action_dimension, hidden
        ).to(self.device)
        self.actor_target = Actor(
            self.state_dimension, self.action_dimension, hidden
        ).to(self.device)
        self.critic = TwinCritic(
            self.state_dimension, self.action_dimension, hidden
        ).to(self.device)
        self.critic_target = TwinCritic(
            self.state_dimension, self.action_dimension, hidden
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.requires_grad_(False)
        self.critic_target.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )
        self.update_count = 0

    def select_action(
        self,
        state: ArrayLike,
        *,
        explore: bool = False,
        noise_std: float | None = None,
    ) -> Float32Array:
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.size != self.state_dimension:
            raise ValueError(
                f"state must contain {self.state_dimension} entries"
            )
        if not np.all(np.isfinite(state_array)):
            raise ValueError("state must contain only finite values")
        state_tensor = torch.as_tensor(
            state_array, device=self.device
        ).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()
        self.actor.train()
        if explore:
            std = (
                self.config.exploration_noise_std
                if noise_std is None
                else float(noise_std)
            )
            if std < 0.0:
                raise ValueError("noise_std cannot be negative")
            if std > 0.0:
                action = action + self._rng.normal(
                    0.0, std, size=self.action_dimension
                )
        return np.asarray(np.clip(action, -1.0, 1.0), dtype=np.float32)

    @staticmethod
    def _soft_update(
        source: nn.Module,
        target: nn.Module,
        rate: float,
    ) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - rate)
                target_parameter.add_(source_parameter, alpha=rate)

    def _clip_gradients(self, parameters: Any) -> None:
        if self.config.maximum_gradient_norm is not None:
            nn.utils.clip_grad_norm_(
                parameters, self.config.maximum_gradient_norm
            )

    def train_step(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int,
    ) -> TD3UpdateMetrics | None:
        if len(replay_buffer) < batch_size:
            return None
        batch = replay_buffer.sample(batch_size, self.device)
        self.update_count += 1

        with torch.no_grad():
            noise = torch.randn_like(batch.actions)
            noise.mul_(self.config.target_policy_noise)
            noise.clamp_(
                -self.config.target_noise_clip,
                self.config.target_noise_clip,
            )
            next_actions = self.actor_target(batch.next_states) + noise
            next_actions.clamp_(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(
                batch.next_states, next_actions
            )
            minimum_target_q = torch.minimum(target_q1, target_q2)
            target_q = batch.rewards + (
                1.0 - batch.dones
            ) * self.config.discount_factor * minimum_target_q

        current_q1, current_q2 = self.critic(
            batch.states, batch.actions
        )
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip_gradients(self.critic.parameters())
        self.critic_optimizer.step()

        actor_loss_value: float | None = None
        actor_updated = self.update_count % self.config.policy_delay == 0
        if actor_updated:
            predicted_actions = self.actor(batch.states)
            actor_loss = -self.critic.first(
                batch.states, predicted_actions
            ).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip_gradients(self.actor.parameters())
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

        return TD3UpdateMetrics(
            update_index=self.update_count,
            critic_loss=float(critic_loss.detach().cpu()),
            actor_loss=actor_loss_value,
            mean_target_q=float(target_q.mean().detach().cpu()),
            mean_current_q1=float(current_q1.mean().detach().cpu()),
            mean_current_q2=float(current_q2.mean().detach().cpu()),
            actor_updated=actor_updated,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dimension": self.state_dimension,
                "action_dimension": self.action_dimension,
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "update_count": self.update_count,
                "extra": {} if extra is None else extra,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizers: bool = True,
    ) -> dict[str, Any]:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        if int(checkpoint["state_dimension"]) != self.state_dimension:
            raise ValueError("checkpoint state dimension does not match agent")
        if int(checkpoint["action_dimension"]) != self.action_dimension:
            raise ValueError("checkpoint action dimension does not match agent")
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer"]
            )
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer"]
            )
        self.update_count = int(checkpoint.get("update_count", 0))
        return dict(checkpoint.get("extra", {}))
