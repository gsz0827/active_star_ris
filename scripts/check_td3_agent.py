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


def make_fast_environment(seed: int) -> RobustActiveStarRISEnv:
    config = RobustEnvironmentConfig(
        num_elements=8,
        num_active_elements=2,
        max_episode_steps=8,
        probing_samples_per_step=16,
    )
    return RobustActiveStarRISEnv(config, seed=seed)


def main() -> None:
    training_environment = make_fast_environment(20260724)
    evaluation_environment = make_fast_environment(20260725)
    agent = TD3Agent(
        training_environment.state_dim,
        training_environment.action_dim,
        TD3Config(
            hidden_dimensions=(64, 64),
            exploration_noise_std=0.10,
            policy_delay=2,
        ),
        device="cpu",
        seed=20260724,
    )

    before = evaluate_td3_policy(
        evaluation_environment, agent, episodes=1, seed=7000
    )
    result = train_td3(
        training_environment,
        agent,
        TD3TrainingConfig(
            total_environment_steps=48,
            replay_capacity=512,
            random_action_steps=12,
            learning_starts=12,
            batch_size=12,
            gradient_steps_per_environment_step=1,
            evaluation_interval=24,
            evaluation_episodes=1,
            seed=20260724,
        ),
        evaluation_environment=evaluation_environment,
    )
    after = result.final_evaluation
    assert after is not None

    test_state, _ = evaluation_environment.reset(seed=999)
    deterministic_action = agent.select_action(test_state, explore=False)

    print(f"Device: {agent.device}")
    print(f"State dimension: {training_environment.state_dim}")
    print(f"Action dimension: {training_environment.action_dim}")
    print(f"Replay size: {len(result.replay_buffer)}")
    print(f"Gradient updates: {agent.update_count}")
    print(f"Actor updates: {len(result.history.actor_losses)}")
    print(f"Critic updates: {len(result.history.critic_losses)}")
    print(f"Action finite: {bool(np.all(np.isfinite(deterministic_action)))}")
    print(
        "Action range: "
        f"[{float(np.min(deterministic_action)):.6f}, "
        f"{float(np.max(deterministic_action)):.6f}]"
    )
    print(
        "Evaluation before training: "
        f"return={before.mean_return:.6f}, "
        f"KGR={before.mean_key_rate:.6f}, "
        f"KDR={before.mean_key_disagreement_rate:.6f}, "
        f"rho={before.mean_reciprocity:.6f}, "
        f"P={before.mean_surface_power:.6f}"
    )
    print(
        "Evaluation after short smoke training: "
        f"return={after.mean_return:.6f}, "
        f"KGR={after.mean_key_rate:.6f}, "
        f"KDR={after.mean_key_disagreement_rate:.6f}, "
        f"rho={after.mean_reciprocity:.6f}, "
        f"P={after.mean_surface_power:.6f}, "
        f"feasible={after.robust_feasibility_rate:.3f}"
    )
    print("TD3 smoke check passed: True")


if __name__ == "__main__":
    main()
