import numpy as np

from active_star_ris.action_mapping import (
    ActionMappingConfig,
    action_dimension,
    build_action_layout,
    map_and_project_action,
)


def _channels(num_elements: int):
    g = np.linspace(0.7, 1.3, num_elements).astype(
        np.complex128
    )
    h_t = np.linspace(1.2, 0.8, num_elements).astype(
        np.complex128
    )
    h_r = np.linspace(0.9, 1.1, num_elements).astype(
        np.complex128
    )
    return g, h_t, h_r


def test_action_layout_and_physical_mapping():
    n = 5
    active_mask = np.array(
        [True, False, True, False, False]
    )
    layout = build_action_layout(active_mask)

    assert layout.num_active_elements == 2
    assert action_dimension(active_mask) == 2 + 3 * n

    action = np.zeros(layout.action_dimension)
    result = map_and_project_action(
        action,
        active_mask=active_mask,
        controller_to_ris_estimate=_channels(n)[0],
        transmission_user_to_ris_estimate=_channels(n)[1],
        reflection_user_to_ris_estimate=_channels(n)[2],
        config=ActionMappingConfig(
            maximum_active_amplitude=3.0,
            beta_min=0.1,
            beta_max=0.9,
            output_power_budget=1000.0,
            nmse_db=-100.0,
            robust_margin_multiplier=0.0,
        ),
    )

    # 归一化动作0映射到各区间中点。
    assert np.allclose(
        result.requested_amplitudes[active_mask],
        2.0,
    )
    assert np.allclose(
        result.projected_amplitudes[~active_mask],
        1.0,
    )
    assert np.allclose(
        result.phase_transmission_rad,
        np.pi,
    )
    assert np.allclose(
        result.phase_reflection_rad,
        np.pi,
    )
    assert np.allclose(
        result.beta_transmission,
        0.5,
    )
    assert result.surface.maximum_energy_error() < 1.0e-12


def test_action_entries_are_clipped_to_valid_physical_bounds():
    n = 4
    active_mask = np.array([True, True, False, False])
    dim = action_dimension(active_mask)
    action = np.linspace(-3.0, 3.0, dim)

    result = map_and_project_action(
        action,
        active_mask=active_mask,
        controller_to_ris_estimate=_channels(n)[0],
        transmission_user_to_ris_estimate=_channels(n)[1],
        reflection_user_to_ris_estimate=_channels(n)[2],
        config=ActionMappingConfig(
            maximum_active_amplitude=4.0,
            beta_min=0.05,
            beta_max=0.95,
            output_power_budget=1000.0,
            nmse_db=-100.0,
            robust_margin_multiplier=0.0,
        ),
    )

    assert np.all(result.clipped_action >= -1.0)
    assert np.all(result.clipped_action <= 1.0)
    assert np.all(result.projected_amplitudes >= 1.0)
    assert np.all(result.projected_amplitudes <= 4.0)
    assert np.all(result.beta_transmission >= 0.05)
    assert np.all(result.beta_transmission <= 0.95)


def test_robust_projection_reduces_requested_gain_to_power_budget():
    n = 4
    active_mask = np.ones(n, dtype=bool)
    dim = action_dimension(active_mask)
    action = np.zeros(dim)
    action[:n] = 1.0  # 请求最大增益3。

    result = map_and_project_action(
        action,
        active_mask=active_mask,
        controller_to_ris_estimate=np.ones(n),
        transmission_user_to_ris_estimate=np.ones(n),
        reflection_user_to_ris_estimate=np.ones(n),
        config=ActionMappingConfig(
            maximum_active_amplitude=3.0,
            output_power_budget=5.0,
            ris_internal_noise_variance=0.0,
            nmse_db=-100.0,
            robust_margin_multiplier=0.0,
            allow_active_bypass=False,
        ),
    )

    assert result.is_robustly_feasible
    assert 0.0 <= result.projection_scale < 1.0
    assert np.all(
        result.projected_amplitudes
        <= result.requested_amplitudes + 1.0e-12
    )
    assert result.maximum_robust_output_upper <= 5.0 + 1.0e-9


def test_infeasible_unit_gain_bypasses_active_elements():
    n = 3
    active_mask = np.ones(n, dtype=bool)
    action = np.zeros(action_dimension(active_mask))

    result = map_and_project_action(
        action,
        active_mask=active_mask,
        controller_to_ris_estimate=np.ones(n),
        transmission_user_to_ris_estimate=np.ones(n),
        reflection_user_to_ris_estimate=np.ones(n),
        config=ActionMappingConfig(
            maximum_active_amplitude=3.0,
            output_power_budget=0.5,
            ris_internal_noise_variance=0.0,
            nmse_db=-100.0,
            robust_margin_multiplier=0.0,
            allow_active_bypass=True,
        ),
    )

    assert result.is_robustly_feasible
    assert result.effective_active_elements == 0
    assert result.bypassed_indices.size == n
    assert np.allclose(result.projected_amplitudes, 1.0)
    assert result.maximum_robust_output_upper == 0.0
