import numpy as np

from active_star_ris.rl_environment import (
    DomainRandomizationConfig,
    RobustActiveStarRISEnv,
    RobustEnvironmentConfig,
    make_small_debug_environment,
)


def test_reset_returns_finite_observation_with_expected_spaces():
    env = make_small_debug_environment(seed=3)
    observation, info = env.reset()

    assert observation.shape == (env.state_dim,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert env.action_space.shape == (env.action_dim,)
    assert env.action_dim == 2 + 3 * 8
    assert info["requested_active_elements"] == 2


def test_step_connects_action_projection_and_joint_objective():
    env = make_small_debug_environment(seed=5)
    observation, _ = env.reset()
    action = np.zeros(env.action_dim, dtype=np.float32)

    next_observation, reward, terminated, truncated, info = env.step(action)

    assert observation.shape == next_observation.shape
    assert env.observation_space.contains(next_observation)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert 0.0 <= info["weighted_key_disagreement_rate"] <= 1.0
    assert 0.0 <= info["weighted_reciprocity"] <= 1.0
    assert info["total_surface_power"] >= 0.0
    assert info["star_energy_error"] < 1.0e-12
    assert info["robustly_feasible"]
    assert info["robust_output_upper"] <= info["output_power_budget"] + 1.0e-8
    assert env.last_diagnostics is not None


def test_seeded_environments_are_reproducible():
    config = RobustEnvironmentConfig(
        num_elements=6,
        num_active_elements=2,
        max_episode_steps=3,
        probing_samples_per_step=24,
    )
    env_a = RobustActiveStarRISEnv(config, seed=19)
    env_b = RobustActiveStarRISEnv(config, seed=19)

    obs_a, info_a = env_a.reset()
    obs_b, info_b = env_b.reset()
    assert np.allclose(obs_a, obs_b)
    assert info_a["nmse_db"] == info_b["nmse_db"]

    action = np.linspace(-1.0, 1.0, env_a.action_dim, dtype=np.float32)
    transition_a = env_a.step(action)
    transition_b = env_b.step(action)

    assert np.allclose(transition_a[0], transition_b[0])
    assert np.isclose(transition_a[1], transition_b[1])
    assert transition_a[2:4] == transition_b[2:4]
    assert np.isclose(
        transition_a[4]["weighted_key_rate"],
        transition_b[4]["weighted_key_rate"],
    )


def test_episode_is_truncated_at_configured_horizon():
    config = RobustEnvironmentConfig(
        num_elements=4,
        num_active_elements=1,
        max_episode_steps=2,
        probing_samples_per_step=16,
    )
    env = RobustActiveStarRISEnv(config, seed=23)
    env.reset()
    action = np.zeros(env.action_dim, dtype=np.float32)

    first = env.step(action)
    second = env.step(action)

    assert not first[3]
    assert second[3]
    assert not second[2]


def test_episode_domain_stays_inside_randomization_ranges():
    randomization = DomainRandomizationConfig(
        nmse_db_min=-18.0,
        nmse_db_max=-17.0,
        ris_internal_noise_variance_min=0.002,
        ris_internal_noise_variance_max=0.003,
        receiver_noise_variance_min=0.010,
        receiver_noise_variance_max=0.011,
        output_power_budget_min=12.0,
        output_power_budget_max=13.0,
        static_gain_error_std_db_min=0.15,
        static_gain_error_std_db_max=0.16,
        directional_gain_error_std_db_min=0.08,
        directional_gain_error_std_db_max=0.09,
        static_phase_error_std_rad_min=0.04,
        static_phase_error_std_rad_max=0.05,
        directional_phase_error_std_rad_min=0.02,
        directional_phase_error_std_rad_max=0.03,
    )
    env = RobustActiveStarRISEnv(
        RobustEnvironmentConfig(
            num_elements=4,
            num_active_elements=1,
            probing_samples_per_step=16,
            domain_randomization=randomization,
        ),
        seed=31,
    )
    _, info = env.reset()
    domain = env.current_domain

    assert -18.0 <= info["nmse_db"] <= -17.0
    assert 0.002 <= domain.ris_internal_noise_variance <= 0.003
    assert 0.010 <= domain.receiver_noise_variance <= 0.011
    assert 12.0 <= domain.output_power_budget <= 13.0
    assert 0.15 <= domain.hardware_parameters.static_gain_error_std_db <= 0.16
    assert 0.02 <= domain.hardware_parameters.directional_phase_error_std_rad <= 0.03


def test_hardware_mismatch_is_fixed_within_episode():
    randomization = DomainRandomizationConfig(
        nmse_db_min=-20.0,
        nmse_db_max=-20.0,
        ris_internal_noise_variance_min=0.002,
        ris_internal_noise_variance_max=0.002,
        receiver_noise_variance_min=0.01,
        receiver_noise_variance_max=0.01,
        output_power_budget_min=20.0,
        output_power_budget_max=20.0,
        static_gain_error_std_db_min=0.50,
        static_gain_error_std_db_max=0.50,
        directional_gain_error_std_db_min=0.25,
        directional_gain_error_std_db_max=0.25,
        static_phase_error_std_rad_min=0.20,
        static_phase_error_std_rad_max=0.20,
        directional_phase_error_std_rad_min=0.10,
        directional_phase_error_std_rad_max=0.10,
        fast_phase_jitter_std_rad_min=0.10,
        fast_phase_jitter_std_rad_max=0.10,
        transmission_amplitude_phase_coupling_rad_per_db_min=0.05,
        transmission_amplitude_phase_coupling_rad_per_db_max=0.05,
        reflection_amplitude_phase_coupling_rad_per_db_min=0.08,
        reflection_amplitude_phase_coupling_rad_per_db_max=0.08,
    )

    env = RobustActiveStarRISEnv(
        RobustEnvironmentConfig(
            num_elements=8,
            num_active_elements=2,
            max_episode_steps=4,
            probing_samples_per_step=32,
            domain_randomization=randomization,
        ),
        seed=123,
    )

    env.reset()
    action = np.zeros(
        env.action_dim,
        dtype=np.float32,
    )

    env.step(action)
    assert env.last_diagnostics is not None
    first = (
        env.last_diagnostics
        .objective
        .hardware_mismatch
    )

    env.step(action)
    assert env.last_diagnostics is not None
    second = (
        env.last_diagnostics
        .objective
        .hardware_mismatch
    )

    fixed_fields = (
        "static_gain_scale",
        "forward_directional_gain_scale",
        "reverse_directional_gain_scale",
        "static_phase_error_transmission",
        "static_phase_error_reflection",
        "forward_phase_jitter_transmission",
        "reverse_phase_jitter_transmission",
        "forward_phase_jitter_reflection",
        "reverse_phase_jitter_reflection",
        "forward_amplitude_phase_coupling_transmission",
        "reverse_amplitude_phase_coupling_transmission",
        "forward_amplitude_phase_coupling_reflection",
        "reverse_amplitude_phase_coupling_reflection",
    )

    # 同一episode内，硬件实现必须保持不变。
    for field_name in fixed_fields:
        np.testing.assert_allclose(
            getattr(first, field_name),
            getattr(second, field_name),
        )

    fast_fields = (
        "forward_fast_phase_jitter_transmission",
        "reverse_fast_phase_jitter_transmission",
        "forward_fast_phase_jitter_reflection",
        "reverse_fast_phase_jitter_reflection",
    )

    # 同一episode内，快速抖动应随step变化。
    assert any(
        not np.allclose(
            getattr(first, field_name),
            getattr(second, field_name),
        )
        for field_name in fast_fields
    )

    # 新episode应当对应新的设备误差实现。
    env.reset()
    env.step(action)

    assert env.last_diagnostics is not None
    third = (
        env.last_diagnostics
        .objective
        .hardware_mismatch
    )

    assert not np.allclose(
        first.static_gain_scale,
        third.static_gain_scale,
    )


def test_robust_reward_uses_lower_tail_cvar():
    config = RobustEnvironmentConfig(
        num_elements=6,
        num_active_elements=2,
        max_episode_steps=2,
        probing_samples_per_step=24,
        robust_objective_samples=4,
        robust_cvar_alpha=0.50,
        robust_mean_weight=0.0,
        robust_cvar_weight=1.0,
    )

    env = RobustActiveStarRISEnv(
        config,
        seed=321,
    )

    env.reset()

    action = np.zeros(
        env.action_dim,
        dtype=np.float32,
    )

    _, reward, _, _, info = env.step(
        action
    )

    assert np.isfinite(reward)

    assert np.isclose(
        reward,
        info["reward"],
    )

    assert np.isclose(
        reward,
        info["robust_reward"],
    )

    # 本测试将CVaR权重设为1，因此最终奖励应等于CVaR。
    assert np.isclose(
        reward,
        info["cvar_reward"],
    )

    # 下尾CVaR不应高于全部样本平均值。
    assert (
        info["cvar_reward"]
        <= info["mean_sample_reward"]
        + 1.0e-12
    )

    assert (
        info["worst_sample_reward"]
        <= info["cvar_reward"]
        + 1.0e-12
    )

    assert (
        info["robust_objective_samples"]
        == 4
    )

    assert env.last_diagnostics is not None

    assert (
        env
        .last_diagnostics
        .num_objective_samples
        == 4
    )
