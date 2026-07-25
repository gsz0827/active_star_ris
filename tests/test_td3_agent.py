from __future__ import annotations

import numpy as np
import torch

from active_star_ris.td3_agent import (
    ReplayBuffer,
    TD3Agent,
    TD3Config,
)


def fill_buffer(
    buffer: ReplayBuffer,
    *,
    transitions: int,
    seed: int = 1,
) -> None:
    rng = np.random.default_rng(seed)
    for index in range(transitions):
        state = rng.normal(size=buffer.state_dimension).astype(np.float32)
        action = rng.uniform(
            -1.0, 1.0, size=buffer.action_dimension
        ).astype(np.float32)
        next_state = (
            0.95 * state
            + 0.05 * rng.normal(size=buffer.state_dimension)
        ).astype(np.float32)
        reward = float(-np.mean(action**2) + 0.01 * index)
        buffer.add(state, action, reward, next_state, index % 7 == 0)


def test_replay_buffer_returns_torch_batches() -> None:
    buffer = ReplayBuffer(5, 3, capacity=32, seed=7)
    fill_buffer(buffer, transitions=20)
    batch = buffer.sample(8, torch.device("cpu"))
    assert batch.states.shape == (8, 5)
    assert batch.actions.shape == (8, 3)
    assert batch.rewards.shape == (8, 1)
    assert batch.next_states.shape == (8, 5)
    assert batch.dones.shape == (8, 1)
    assert batch.states.dtype == torch.float32


def test_actor_action_is_finite_and_bounded() -> None:
    agent = TD3Agent(
        7,
        4,
        TD3Config(hidden_dimensions=(32, 32)),
        device="cpu",
        seed=2,
    )
    action = agent.select_action(np.zeros(7), explore=False)
    assert action.shape == (4,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_td3_delays_actor_update_and_updates_critics() -> None:
    config = TD3Config(
        hidden_dimensions=(32, 32),
        policy_delay=2,
        target_policy_noise=0.1,
    )
    agent = TD3Agent(6, 3, config, device="cpu", seed=3)
    buffer = ReplayBuffer(6, 3, capacity=64, seed=3)
    fill_buffer(buffer, transitions=40)

    actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in agent.critic.parameters()]
    first = agent.train_step(buffer, batch_size=16)
    assert first is not None
    assert not first.actor_updated
    assert first.actor_loss is None
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, agent.critic.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(actor_before, agent.actor.parameters())
    )

    second = agent.train_step(buffer, batch_size=16)
    assert second is not None
    assert second.actor_updated
    assert second.actor_loss is not None
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, agent.actor.parameters())
    )


def test_checkpoint_round_trip_preserves_deterministic_action(tmp_path) -> None:
    config = TD3Config(hidden_dimensions=(32, 32))
    agent = TD3Agent(5, 2, config, device="cpu", seed=4)
    state = np.arange(5, dtype=np.float32) / 5.0
    expected = agent.select_action(state)
    checkpoint = agent.save_checkpoint(
        tmp_path / "agent.pt", extra={"step": 17}
    )

    restored = TD3Agent(5, 2, config, device="cpu", seed=99)
    extra = restored.load_checkpoint(checkpoint)
    actual = restored.select_action(state)
    assert np.allclose(expected, actual)
    assert extra == {"step": 17}
