from __future__ import annotations

import numpy as np

from active_star_ris.optimization import (
    design_active_surface,
    design_passive_surface,
    feasible_common_active_amplitude,
    optimize_scalar_energy_split,
    phase_align,
    select_active_elements,
)
from active_star_ris.system import evaluate_two_user_system


def test_phase_alignment_produces_coherent_sum() -> None:
    rng = np.random.default_rng(10)
    g = rng.normal(size=16) + 1j * rng.normal(size=16)
    h = rng.normal(size=16) + 1j * rng.normal(size=16)
    theta = phase_align(g, h)
    terms = h * np.exp(1j * theta) * g
    assert np.max(np.abs(np.angle(terms))) < 1e-12


def test_active_selection_count() -> None:
    g = np.array([1, 2, 3, 4], dtype=complex)
    h_t = np.ones(4, dtype=complex)
    h_r = np.ones(4, dtype=complex)
    mask, selected = select_active_elements(g, h_t, h_r, 2)
    assert mask.sum() == 2
    assert set(selected.tolist()) == {2, 3}


def test_feasible_amplitude_obeys_budget() -> None:
    g = np.ones(4, dtype=complex)
    mask = np.ones(4, dtype=bool)
    amplitude = feasible_common_active_amplitude(
        g,
        mask,
        transmit_power=1.0,
        ris_internal_noise_variance=0.0,
        ris_output_power_budget=16.0,
        maximum_active_amplitude=10.0,
    )
    assert amplitude == 2.0


def test_active_design_obeys_power_constraint() -> None:
    rng = np.random.default_rng(11)
    n = 12
    g = rng.normal(size=n) + 1j * rng.normal(size=n)
    h_t = rng.normal(size=n) + 1j * rng.normal(size=n)
    h_r = rng.normal(size=n) + 1j * rng.normal(size=n)

    design = design_active_surface(
        g,
        h_t,
        h_r,
        beta_transmission=0.5,
        num_active_elements=4,
        transmit_power=1.0,
        ris_internal_noise_variance=0.01,
        ris_output_power_budget=20.0,
        maximum_active_amplitude=3.0,
    )
    metrics = evaluate_two_user_system(
        g,
        h_t,
        h_r,
        design.surface,
        transmit_power=1.0,
        user_noise_variance=0.1,
        ris_internal_noise_variance=0.01,
        ris_output_power_budget=20.0,
    )
    assert metrics.ris_power_violation < 1e-10


def test_optimized_beta_is_not_worse_than_fixed_grid_member() -> None:
    rng = np.random.default_rng(12)
    n = 10
    g = rng.normal(size=n) + 1j * rng.normal(size=n)
    h_t = rng.normal(size=n) + 1j * rng.normal(size=n)
    h_r = rng.normal(size=n) + 1j * rng.normal(size=n)

    fixed_surface = design_passive_surface(g, h_t, h_r, 0.5)
    fixed_metrics = evaluate_two_user_system(
        g,
        h_t,
        h_r,
        fixed_surface,
        transmit_power=1.0,
        user_noise_variance=0.1,
        ris_internal_noise_variance=0.01,
        ris_output_power_budget=20.0,
    )

    beta, design, metrics = optimize_scalar_energy_split(
        g,
        h_t,
        h_r,
        num_active_elements=0,
        transmit_power=1.0,
        user_noise_variance=0.1,
        ris_internal_noise_variance=0.01,
        ris_output_power_budget=20.0,
        maximum_active_amplitude=3.0,
        beta_grid=np.array([0.25, 0.5, 0.75]),
    )

    assert beta in (0.25, 0.5, 0.75)
    assert design.surface.active_mask.sum() == 0
    assert metrics.weighted_sum_rate + 1e-12 >= fixed_metrics.weighted_sum_rate
