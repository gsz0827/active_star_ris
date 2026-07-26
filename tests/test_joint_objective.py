import numpy as np

from active_star_ris.channels import complex_normal
from active_star_ris.hardware_impairments import (
    HardwareMismatchParameters,
)
from active_star_ris.joint_objective import (
    JointObjectiveConfig,
    evaluate_joint_objective,
)
from active_star_ris.surface import (
    EnergySplit,
    build_surface_coefficients,
)


def _sample_channels(
    num_samples: int,
    num_elements: int,
    rng: np.random.Generator,
):
    return (
        complex_normal(
            (num_samples, num_elements),
            rng,
        ),
        complex_normal(
            (num_samples, num_elements),
            rng,
        ),
        complex_normal(
            (num_samples, num_elements),
            rng,
        ),
    )


def test_joint_objective_is_finite_and_uses_all_metrics():
    rng = np.random.default_rng(7)
    num_elements = 8
    channels = _sample_channels(
        128,
        num_elements,
        rng,
    )

    active_mask = np.zeros(
        num_elements,
        dtype=bool,
    )

    surface = build_surface_coefficients(
        energy_split=(
            EnergySplit.from_transmission(
                0.5,
                num_elements,
            )
        ),
        phase_transmission_rad=np.zeros(
            num_elements
        ),
        phase_reflection_rad=np.zeros(
            num_elements
        ),
        amplitude_gain=np.ones(
            num_elements
        ),
        active_mask=active_mask,
    )

    config = JointObjectiveConfig(
        probe_sample_rate_hz=200.0,
        key_rate_reference=1000.0,
        key_disagreement_reference=0.5,
        surface_power_reference=1.0,
    )

    result = evaluate_joint_objective(
        *channels,
        ideal_surface=surface,
        output_power_budget=10.0,
        objective_config=config,
        rng=np.random.default_rng(8),
    )

    assert np.isfinite(
        result.reward
    )

    assert np.isfinite(
        result.key_rate_bits_per_sample
    )

    assert np.isfinite(
        result.key_rate_bits_per_second
    )

    assert np.isclose(
        result.key_rate_bits_per_second,
        result.key_rate_bits_per_sample
        * config.probe_sample_rate_hz,
    )

    assert np.isclose(
        result.key_rate_bits_per_sample,
        result
        .key_generation
        .weighted_mutual_information,
    )

    assert np.isclose(
        result.raw_key_disagreement_rate,
        result
        .key_generation
        .weighted_key_disagreement_rate,
    )

    assert np.isclose(
        result.observation_reciprocity,
        result
        .key_generation
        .weighted_correlation,
    )

    assert (
        0.0
        <= result.raw_key_disagreement_rate
        <= 1.0
    )

    assert (
        0.0
        <= result.observation_reciprocity
        <= 1.0
    )

    assert (
        result.surface_power.total_surface_power
        >= 0.0
    )
    

def test_power_only_reward_penalizes_larger_active_gain():
    rng = np.random.default_rng(11)
    num_elements = 6
    channels = _sample_channels(128, num_elements, rng)
    active_mask = np.ones(num_elements, dtype=bool)
    split = EnergySplit.from_transmission(0.5, num_elements)

    low_gain_surface = build_surface_coefficients(
        energy_split=split,
        phase_transmission_rad=np.zeros(num_elements),
        phase_reflection_rad=np.zeros(num_elements),
        amplitude_gain=np.ones(num_elements),
        active_mask=active_mask,
    )
    high_gain_surface = build_surface_coefficients(
        energy_split=split,
        phase_transmission_rad=np.zeros(num_elements),
        phase_reflection_rad=np.zeros(num_elements),
        amplitude_gain=np.full(num_elements, 2.0),
        active_mask=active_mask,
    )

    config = JointObjectiveConfig(
        key_rate_weight=0.0,
        key_disagreement_weight=0.0,
        reciprocity_weight=0.0,
        surface_power_weight=1.0,
        constraint_violation_weight=0.0,
        surface_power_reference=1.0,
    )
    mismatch = HardwareMismatchParameters()

    low = evaluate_joint_objective(
        *channels,
        ideal_surface=low_gain_surface,
        output_power_budget=1000.0,
        hardware_parameters=mismatch,
        objective_config=config,
        rng=np.random.default_rng(12),
    )
    high = evaluate_joint_objective(
        *channels,
        ideal_surface=high_gain_surface,
        output_power_budget=1000.0,
        hardware_parameters=mismatch,
        objective_config=config,
        rng=np.random.default_rng(12),
    )

    assert high.surface_power.total_surface_power > low.surface_power.total_surface_power
    assert high.reward < low.reward


def test_power_violation_enters_reward():
    rng = np.random.default_rng(21)
    num_elements = 4
    channels = _sample_channels(64, num_elements, rng)
    active_mask = np.ones(num_elements, dtype=bool)

    surface = build_surface_coefficients(
        energy_split=EnergySplit.from_transmission(
            0.5,
            num_elements,
        ),
        phase_transmission_rad=np.zeros(num_elements),
        phase_reflection_rad=np.zeros(num_elements),
        amplitude_gain=np.full(num_elements, 3.0),
        active_mask=active_mask,
    )

    config = JointObjectiveConfig(
        key_rate_weight=0.0,
        key_disagreement_weight=0.0,
        reciprocity_weight=0.0,
        surface_power_weight=0.0,
        constraint_violation_weight=10.0,
    )

    result = evaluate_joint_objective(
        *channels,
        ideal_surface=surface,
        output_power_budget=0.01,
        objective_config=config,
        rng=np.random.default_rng(22),
    )

    assert result.surface_power.power_violation > 0.0
    assert result.normalized_power_violation > 0.0
    assert result.reward < 0.0
