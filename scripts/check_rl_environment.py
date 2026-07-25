from __future__ import annotations

import numpy as np

from active_star_ris.rl_environment import make_small_debug_environment


def main() -> None:
    env = make_small_debug_environment(
        seed=20260724,
        num_elements=8,
        num_active_elements=2,
    )
    observation, reset_info = env.reset()

    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Initial observation finite: {bool(np.all(np.isfinite(observation)))}")
    print(f"Episode NMSE (dB): {reset_info['nmse_db']:.4f}")
    print(f"Episode power budget: {reset_info['output_power_budget']:.4f}")

    for step_index in range(3):
        action = env.sample_action()
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            "Step "
            f"{step_index + 1}: reward={reward:.6f}, "
            f"KGR={info['weighted_key_rate']:.6f}, "
            f"KDR={info['weighted_key_disagreement_rate']:.6f}, "
            f"rho={info['weighted_reciprocity']:.6f}, "
            f"P_surface={info['total_surface_power']:.6f}, "
            f"active={info['effective_active_elements']}, "
            f"feasible={info['robustly_feasible']}"
        )
        assert np.all(np.isfinite(observation))
        assert not terminated
        assert not truncated

    print("Environment API check passed: True")


if __name__ == "__main__":
    main()
