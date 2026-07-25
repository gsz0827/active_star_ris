from __future__ import annotations

import numpy as np

from active_star_ris.action_mapping import (
    ActionMappingConfig,
    action_dimension,
    map_and_project_action,
)


def main() -> None:
    rng = np.random.default_rng(20260724)
    num_elements = 8

    active_mask = np.zeros(num_elements, dtype=bool)
    active_mask[[0, 2, 4, 6]] = True

    g_hat = (
        rng.standard_normal(num_elements)
        + 1j * rng.standard_normal(num_elements)
    ) / np.sqrt(2.0)
    h_t_hat = (
        rng.standard_normal(num_elements)
        + 1j * rng.standard_normal(num_elements)
    ) / np.sqrt(2.0)
    h_r_hat = (
        rng.standard_normal(num_elements)
        + 1j * rng.standard_normal(num_elements)
    ) / np.sqrt(2.0)

    action = rng.uniform(
        -1.0,
        1.0,
        size=action_dimension(active_mask),
    )

    config = ActionMappingConfig(
        maximum_active_amplitude=3.0,
        beta_min=0.05,
        beta_max=0.95,
        output_power_budget=12.0,
        nmse_db=-15.0,
        robust_margin_multiplier=3.0,
        allow_active_bypass=True,
    )

    result = map_and_project_action(
        action,
        active_mask=active_mask,
        controller_to_ris_estimate=g_hat,
        transmission_user_to_ris_estimate=h_t_hat,
        reflection_user_to_ris_estimate=h_r_hat,
        config=config,
    )

    print("Action dimension:", result.layout.action_dimension)
    print(
        "Requested/effective active elements:",
        result.requested_active_elements,
        "/",
        result.effective_active_elements,
    )
    print("Projection scale:", result.projection_scale)
    print(
        "Maximum robust output upper:",
        result.maximum_robust_output_upper,
    )
    print("Power budget:", config.output_power_budget)
    print("Robustly feasible:", result.is_robustly_feasible)
    print(
        "Maximum STAR energy error:",
        result.surface.maximum_energy_error(),
    )


if __name__ == "__main__":
    main()
