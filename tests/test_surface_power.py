import numpy as np
import pytest

from active_star_ris.surface import (
    EnergySplit,
    build_surface_coefficients,
)
from active_star_ris.surface_power import (
    evaluate_bidirectional_surface_power,
    project_active_amplitude_vector_robust,
    project_common_active_amplitude_robust,
)
from active_star_ris.optimization import (
    design_active_surface_robust,
    design_active_surface_vector_robust,
    design_active_surface_vector_beta_robust,
    design_elementwise_transmission_split,
)


def _make_surface(
    amplitude: float,
):
    active_mask = np.array(
        [
            True,
            True,
            False,
            False,
        ],
        dtype=bool,
    )

    gain = np.ones(
        4,
        dtype=float,
    )

    gain[active_mask] = amplitude

    split = EnergySplit.from_transmission(
        beta_transmission=0.5,
        num_elements=4,
    )

    surface = build_surface_coefficients(
        energy_split=split,
        phase_transmission_rad=np.zeros(4),
        phase_reflection_rad=np.zeros(4),
        amplitude_gain=gain,
        active_mask=active_mask,
    )

    return surface


def test_bidirectional_power_uses_worst_direction():
    surface = _make_surface(
        amplitude=2.0
    )

    controller_channel = np.ones(
        4,
        dtype=np.complex128,
    )

    transmission_channel = (
        2.0
        * np.ones(
            4,
            dtype=np.complex128,
        )
    )

    reflection_channel = (
        0.5
        * np.ones(
            4,
            dtype=np.complex128,
        )
    )

    result = (
        evaluate_bidirectional_surface_power(
            controller_to_ris=(
                controller_channel
            ),
            transmission_user_to_ris=(
                transmission_channel
            ),
            reflection_user_to_ris=(
                reflection_channel
            ),
            surface=surface,
            controller_pilot_power=1.0,
            transmission_user_pilot_power=1.0,
            reflection_user_pilot_power=1.0,
            ris_internal_noise_variance=0.0,
            output_power_budget=35.0,
        )
    )

    # 两个有源单元，增益平方为4。
    assert (
        result.output_power_controller
        == pytest.approx(8.0)
    )

    assert (
        result.output_power_transmission_user
        == pytest.approx(32.0)
    )

    assert (
        result.output_power_reflection_user
        == pytest.approx(2.0)
    )

    assert (
        result.maximum_output_power
        == pytest.approx(32.0)
    )

    assert result.power_violation == 0.0


def test_bidirectional_power_detects_violation():
    surface = _make_surface(
        amplitude=3.0
    )

    result = (
        evaluate_bidirectional_surface_power(
            controller_to_ris=np.ones(4),
            transmission_user_to_ris=(
                2.0 * np.ones(4)
            ),
            reflection_user_to_ris=np.ones(4),
            surface=surface,
            controller_pilot_power=1.0,
            transmission_user_pilot_power=1.0,
            reflection_user_pilot_power=1.0,
            ris_internal_noise_variance=0.0,
            output_power_budget=35.0,
        )
    )

    # 透射用户方向：
    # 2个单元 × 3² × |2|² = 72
    assert (
        result.maximum_output_power
        == pytest.approx(72.0)
    )

    assert (
        result.power_violation
        == pytest.approx(37.0)
    )


def test_robust_projection_respects_budget_upper_bound():
    active_mask = np.array(
        [
            True,
            True,
            False,
            False,
        ],
        dtype=bool,
    )

    result = (
        project_common_active_amplitude_robust(
            controller_to_ris_estimate=(
                np.ones(4)
            ),
            transmission_user_to_ris_estimate=(
                2.0 * np.ones(4)
            ),
            reflection_user_to_ris_estimate=(
                np.ones(4)
            ),
            active_mask=active_mask,
            controller_pilot_power=1.0,
            transmission_user_pilot_power=1.0,
            reflection_user_pilot_power=1.0,
            ris_internal_noise_variance=0.0,
            output_power_budget=10.0,
            maximum_active_amplitude=3.0,
            nmse_db=-20.0,
            robust_margin_multiplier=0.0,
        )
    )

    # 最坏方向的单位增益输入功率为：
    # 2个单元 × |2|² = 8
    # 因此增益为sqrt(10/8)。
    assert (
        result.common_active_amplitude
        == pytest.approx(
            np.sqrt(10.0 / 8.0)
        )
    )

    assert (
        result.maximum_robust_output_upper
        <= 10.0 + 1.0e-12
    )


def test_poorer_csi_produces_more_conservative_gain():
    active_mask = np.array(
        [
            True,
            True,
            True,
            True,
        ],
        dtype=bool,
    )

    common_arguments = dict(
        controller_to_ris_estimate=np.ones(4),
        transmission_user_to_ris_estimate=np.ones(4),
        reflection_user_to_ris_estimate=np.ones(4),
        active_mask=active_mask,
        controller_pilot_power=1.0,
        transmission_user_pilot_power=1.0,
        reflection_user_pilot_power=1.0,
        ris_internal_noise_variance=0.002,
        output_power_budget=35.0,
        maximum_active_amplitude=3.0,
        robust_margin_multiplier=3.0,
    )

    accurate_csi = (
        project_common_active_amplitude_robust(
            nmse_db=-30.0,
            **common_arguments,
        )
    )

    poor_csi = (
        project_common_active_amplitude_robust(
            nmse_db=-5.0,
            **common_arguments,
        )
    )

    assert (
        poor_csi.common_active_amplitude
        <= accurate_csi.common_active_amplitude
    )


def test_surface_power_includes_static_power():
    surface = _make_surface(
        amplitude=1.0
    )

    result = (
        evaluate_bidirectional_surface_power(
            controller_to_ris=np.ones(4),
            transmission_user_to_ris=np.ones(4),
            reflection_user_to_ris=np.ones(4),
            surface=surface,
            controller_pilot_power=1.0,
            transmission_user_pilot_power=1.0,
            reflection_user_pilot_power=1.0,
            ris_internal_noise_variance=0.0,
            output_power_budget=35.0,
            amplifier_efficiency=0.5,
            controller_static_power=0.10,
            active_element_bias_power=0.01,
        )
    )

    # 增益为1，没有额外放大功耗。
    # 但仍有控制器功耗和两个有源单元偏置功耗。
    assert result.amplifier_dc_power == 0.0

    assert (
        result.total_surface_power
        == pytest.approx(0.12)
    )


def test_robust_design_reduces_active_count_when_needed():
    """不可行时应减少有源单元数量，而不是抛出异常。"""

    num_elements = 4

    design = design_active_surface_robust(
        alice_to_ris=np.ones(
            num_elements,
            dtype=np.complex128,
        ),
        ris_to_transmission_user=np.ones(
            num_elements,
            dtype=np.complex128,
        ),
        ris_to_reflection_user=np.ones(
            num_elements,
            dtype=np.complex128,
        ),
        beta_transmission=0.5,
        num_active_elements=4,
        transmit_power=1.0,
        ris_internal_noise_variance=0.0,
        ris_output_power_budget=1.0,
        maximum_active_amplitude=3.0,
        transmission_weight=0.5,
        reflection_weight=0.5,
        nmse_db=-20.0,
        robust_margin_multiplier=0.0,
        transmission_user_pilot_power=1.0,
        reflection_user_pilot_power=1.0,
    )

    # 每个单元在单位增益下消耗1单位功率，
    # 而总预算只有1，因此最多启用一个单元。
    assert design.selected_indices.size == 1
    assert np.sum(design.active_mask) == 1

    assert (
        design.common_active_amplitude
        == pytest.approx(1.0)
    )

    assert (
        design.requested_active_elements
        == 4
    )

    assert (
        design.effective_active_elements
        == 1
    )

    assert (
        design.disabled_active_elements
        == 3
    )

    assert (
        design.used_passive_fallback
        is False
    )


def test_vector_gain_projection_preserves_relative_order():
    active_mask = np.array(
        [
            True,
            True,
            True,
            False,
        ],
        dtype=bool,
    )

    result = (
        project_active_amplitude_vector_robust(
            controller_to_ris_estimate=(
                np.ones(4)
            ),
            transmission_user_to_ris_estimate=(
                np.ones(4)
            ),
            reflection_user_to_ris_estimate=(
                np.ones(4)
            ),
            requested_amplitudes=np.array(
                [
                    3.0,
                    2.0,
                    1.5,
                    1.0,
                ]
            ),
            active_mask=active_mask,
            controller_pilot_power=1.0,
            transmission_user_pilot_power=1.0,
            reflection_user_pilot_power=1.0,
            ris_internal_noise_variance=0.0,
            output_power_budget=8.0,
            maximum_active_amplitude=3.0,
            nmse_db=-20.0,
            robust_margin_multiplier=0.0,
        )
    )

    gains = (
        result.projected_amplitudes
    )

    assert (
        0.0
        < result.projection_scale
        < 1.0
    )

    assert gains[0] > gains[1]
    assert gains[1] > gains[2]
    assert gains[2] > 1.0
    assert gains[3] == pytest.approx(1.0)

    assert (
        result.maximum_robust_output_upper
        <= 8.0 + 1.0e-12
    )


def test_vector_gain_design_assigns_different_gains():
    design = (
        design_active_surface_vector_robust(
            alice_to_ris=np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
                dtype=np.complex128,
            ),
            ris_to_transmission_user=(
                np.ones(
                    4,
                    dtype=np.complex128,
                )
            ),
            ris_to_reflection_user=(
                np.ones(
                    4,
                    dtype=np.complex128,
                )
            ),
            beta_transmission=0.5,
            num_active_elements=3,
            transmit_power=1.0,
            ris_internal_noise_variance=0.0,
            ris_output_power_budget=1000.0,
            maximum_active_amplitude=3.0,
            transmission_weight=0.5,
            reflection_weight=0.5,
            nmse_db=-20.0,
            robust_margin_multiplier=0.0,
            transmission_user_pilot_power=1.0,
            reflection_user_pilot_power=1.0,
        )
    )

    active_gains = (
        design.surface.amplitude_gain[
            design.active_mask
        ]
    )

    assert active_gains.size == 3

    # 三个有源单元应当获得不同增益，
    # 而不是继续共享同一个公共值。
    assert (
        np.unique(
            np.round(
                active_gains,
                decimals=10,
            )
        ).size
        > 1
    )

    assert np.all(
        active_gains >= 1.0
    )

    assert np.all(
        active_gains <= 3.0
    )


def test_elementwise_beta_prefers_stronger_user_side():
    beta = (
        design_elementwise_transmission_split(
            alice_to_ris=np.ones(
                4,
                dtype=np.complex128,
            ),
            ris_to_transmission_user=np.array(
                [
                    4.0,
                    3.0,
                    1.0,
                    0.5,
                ],
                dtype=np.complex128,
            ),
            ris_to_reflection_user=np.array(
                [
                    0.5,
                    1.0,
                    3.0,
                    4.0,
                ],
                dtype=np.complex128,
            ),
            transmission_weight=0.5,
            reflection_weight=0.5,
            beta_min=0.05,
            beta_max=0.95,
            temperature=1.0,
        )
    )

    assert beta.shape == (4,)

    assert beta[0] > 0.5
    assert beta[1] > 0.5

    assert beta[2] < 0.5
    assert beta[3] < 0.5

    assert np.all(
        beta >= 0.05
    )

    assert np.all(
        beta <= 0.95
    )

    assert (
        np.std(beta)
        > 0.0
    )


def test_vector_beta_surface_design_is_nonuniform():
    beta_vector, design = (
        design_active_surface_vector_beta_robust(
            alice_to_ris=np.ones(
                4,
                dtype=np.complex128,
            ),
            ris_to_transmission_user=np.array(
                [
                    4.0,
                    3.0,
                    1.0,
                    0.5,
                ],
                dtype=np.complex128,
            ),
            ris_to_reflection_user=np.array(
                [
                    0.5,
                    1.0,
                    3.0,
                    4.0,
                ],
                dtype=np.complex128,
            ),
            num_active_elements=2,
            transmit_power=1.0,
            ris_internal_noise_variance=0.0,
            ris_output_power_budget=1000.0,
            maximum_active_amplitude=3.0,
            transmission_weight=0.5,
            reflection_weight=0.5,
            nmse_db=-20.0,
            robust_margin_multiplier=0.0,
            transmission_user_pilot_power=1.0,
            reflection_user_pilot_power=1.0,
            beta_min=0.05,
            beta_max=0.95,
            beta_temperature=1.0,
        )
    )

    assert beta_vector.shape == (4,)

    assert (
        np.std(beta_vector)
        > 0.0
    )

    assert (
        design.surface
        .amplitude_gain
        .shape
        == (4,)
    )


def test_surface_power_breakdown_sums_to_total():
    surface = _make_surface(
        amplitude=1.0
    )

    channel = np.ones(
        4,
        dtype=np.complex128,
    )

    result = evaluate_bidirectional_surface_power(
        controller_to_ris=channel,
        transmission_user_to_ris=channel,
        reflection_user_to_ris=channel,
        surface=surface,
        controller_pilot_power=1.0,
        transmission_user_pilot_power=1.0,
        reflection_user_pilot_power=1.0,
        ris_internal_noise_variance=0.0,
        output_power_budget=100.0,
        amplifier_efficiency=1.0,
        controller_static_power=0.40,
        passive_element_control_power=0.10,
        active_element_control_power=0.20,
        active_element_bias_power=0.05,
        switching_network_static_power=0.30,
    )

    expected_total = (
        result.controller_static_power
        + result.passive_element_control_power
        + result.active_element_control_power
        + result.active_element_bias_power
        + result.switching_network_power
        + result.amplifier_dc_power
    )

    assert np.isclose(
        result.total_surface_power,
        expected_total,
    )

    # _make_surface中包含两个有源和两个无源单元。
    assert np.isclose(
        result.passive_element_control_power,
        2 * 0.10,
    )

    assert np.isclose(
        result.active_element_control_power,
        2 * 0.20,
    )

    assert np.isclose(
        result.active_element_bias_power,
        2 * 0.05,
    )

    # 单位增益不产生额外放大RF功率。
    assert np.isclose(
        result.amplifier_additional_rf_power,
        0.0,
    )
