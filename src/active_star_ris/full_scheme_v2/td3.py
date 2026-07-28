from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import TD3Config


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: list[int], output_activation: nn.Module | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        if output_activation is not None:
            layers.append(output_activation)
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: list[int]):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden)
        self.q2 = MLP(state_dim + action_dim, 1, hidden)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joined = torch.cat([state, action], dim=-1)
        return self.q1(joined), self.q2(joined)

    def first(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index, 0] = reward
        self.next_states[index] = next_state
        self.dones[index, 0] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        arrays = (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
        return tuple(torch.as_tensor(array, device=device) for array in arrays)


@dataclass(frozen=True)
class TrainingLosses:
    critic_loss: float
    actor_loss: float | None


class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int, config: TD3Config, seed: int = 0, device: str | None = None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = MLP(state_dim, action_dim, config.hidden_dimensions, nn.Tanh()).to(self.device)
        self.actor_target = MLP(state_dim, action_dim, config.hidden_dimensions, nn.Tanh()).to(self.device)
        self.critic = TwinCritic(state_dim, action_dim, config.hidden_dimensions).to(self.device)
        self.critic_target = TwinCritic(state_dim, action_dim, config.hidden_dimensions).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_learning_rate)
        self.updates = 0
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, state: np.ndarray, exploration_noise: float = 0.0) -> np.ndarray:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(tensor).cpu().numpy()[0]
        if exploration_noise > 0.0:
            action += self.rng.normal(0.0, exploration_noise, self.action_dim)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def train(self, replay: ReplayBuffer) -> TrainingLosses | None:
        if replay.size < self.config.batch_size:
            return None
        state, action, reward, next_state, done = replay.sample(self.config.batch_size, self.device)
        with torch.no_grad():
            noise = torch.randn_like(action) * self.config.policy_noise
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + (1.0 - done) * self.config.discount_factor * torch.minimum(target_q1, target_q2)
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.updates += 1
        actor_loss_value: float | None = None
        if self.updates % self.config.policy_delay == 0:
            actor_loss = -self.critic.first(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            tau = self.config.target_update_rate
            with torch.no_grad():
                for target, source in zip(self.actor_target.parameters(), self.actor.parameters(), strict=True):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)
                for target, source in zip(self.critic_target.parameters(), self.critic.parameters(), strict=True):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)
        return TrainingLosses(float(critic_loss.detach().cpu()), actor_loss_value)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "updates": self.updates,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.actor_target.load_state_dict(data["actor_target"])
        self.critic_target.load_state_dict(data["critic_target"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        self.updates = int(data["updates"])
