from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from active_star_ris.star_key_system import (  # noqa: E402
    build_star_coefficients,
    energy_splitting_residual,
    simulate_dual_side_key_generation,
)


def main() -> None:
    rng = np.random.default_rng(
        20260719
    )

    num_samples = 4000
    num_elements = 16

    channel_controller_to_ris = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    channel_ris_to_transmission_user = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    channel_ris_to_reflection_user = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    active_mask = np.zeros(
        num_elements,
        dtype=bool,
    )

    active_mask[:4] = True

    amplitudes = np.ones(
        num_elements
    )

    amplitudes[active_mask] = 1.5

    beta_transmission = np.full(
        num_elements,
        0.65,
    )

    beta_reflection = (
        1.0 - beta_transmission
    )

    phase_transmission = rng.uniform(
        -np.pi,
        np.pi,
        size=num_elements,
    )

    phase_reflection = rng.uniform(
        -np.pi,
        np.pi,
        size=num_elements,
    )

    coefficients = build_star_coefficients(
        amplitudes=amplitudes,
        beta_transmission=(
            beta_transmission
        ),
        beta_reflection=(
            beta_reflection
        ),
        phase_transmission=(
            phase_transmission
        ),
        phase_reflection=(
            phase_reflection
        ),
    )

    result = simulate_dual_side_key_generation(
        channel_controller_to_ris=(
            channel_controller_to_ris
        ),
        channel_ris_to_transmission_user=(
            channel_ris_to_transmission_user
        ),
        channel_ris_to_reflection_user=(
            channel_ris_to_reflection_user
        ),
        coefficients=coefficients,
        active_mask=active_mask,
        pilot_power_controller=1.0,
        pilot_power_transmission_user=1.0,
        pilot_power_reflection_user=1.0,
        active_noise_variance=0.002,
        receiver_noise_variance_controller=0.01,
        receiver_noise_variance_transmission_user=0.01,
        receiver_noise_variance_reflection_user=0.01,
        transmission_weight=0.5,
        reflection_weight=0.5,
        rng=rng,
    )

    maximum_energy_error = float(
        np.max(
            np.abs(
                energy_splitting_residual(
                    coefficients
                )
            )
        )
    )

    transmission_metrics = (
        result.transmission.metrics
    )

    reflection_metrics = (
        result.reflection.metrics
    )

    print(
        "Step 3 dual-side key-generation check"
    )
    print(
        f"Number of STAR-RIS elements = "
        f"{num_elements}"
    )
    print(
        f"Number of active elements   = "
        f"{np.sum(active_mask)}"
    )
    print(
        f"Maximum energy-split error  = "
        f"{maximum_energy_error:.12e}"
    )

    print()
    print("Transmission-side link")
    print(
        f"  |rho_T|                   = "
        f"{transmission_metrics.correlation_magnitude:.6f}"
    )
    print(
        f"  Gaussian MI_T             = "
        f"{transmission_metrics.mutual_information_bits_per_sample:.6f} "
        "bit/sample"
    )
    print(
        f"  Raw KDR_T                 = "
        f"{transmission_metrics.key_disagreement_rate:.6f}"
    )

    print()
    print("Reflection-side link")
    print(
        f"  |rho_R|                   = "
        f"{reflection_metrics.correlation_magnitude:.6f}"
    )
    print(
        f"  Gaussian MI_R             = "
        f"{reflection_metrics.mutual_information_bits_per_sample:.6f} "
        "bit/sample"
    )
    print(
        f"  Raw KDR_R                 = "
        f"{reflection_metrics.key_disagreement_rate:.6f}"
    )

    print()
    print("Joint weighted metrics")
    print(
        f"  Weighted correlation      = "
        f"{result.weighted_correlation:.6f}"
    )
    print(
        f"  Weighted Gaussian MI      = "
        f"{result.weighted_mutual_information:.6f} "
        "bit/sample"
    )
    print(
        f"  Weighted raw KDR          = "
        f"{result.weighted_key_disagreement_rate:.6f}"
    )

    assert maximum_energy_error < 1.0e-10

    assert (
        transmission_metrics
        .correlation_magnitude
        > 0.9
    )

    assert (
        reflection_metrics
        .correlation_magnitude
        > 0.9
    )

    assert (
        transmission_metrics
        .key_disagreement_rate
        < 0.2
    )

    assert (
        reflection_metrics
        .key_disagreement_rate
        < 0.2
    )

    print()
    print(
        "STEP 3 MANUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()