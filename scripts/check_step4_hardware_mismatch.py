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

from active_star_ris.hardware_impairments import (  # noqa: E402
    HardwareMismatchParameters,
    apply_hardware_mismatch,
)
from active_star_ris.star_key_system import (  # noqa: E402
    build_star_coefficients,
    simulate_dual_side_key_generation,
)


def main() -> None:
    channel_rng = np.random.default_rng(
        20260719
    )

    num_samples = 4000
    num_elements = 16

    g = (
        channel_rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * channel_rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    h_t = (
        channel_rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * channel_rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    h_r = (
        channel_rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * channel_rng.normal(
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

    beta_t = np.full(
        num_elements,
        0.65,
    )
    beta_r = 1.0 - beta_t

    theta_t = channel_rng.uniform(
        -np.pi,
        np.pi,
        size=num_elements,
    )
    theta_r = channel_rng.uniform(
        -np.pi,
        np.pi,
        size=num_elements,
    )

    ideal_coefficients = (
        build_star_coefficients(
            amplitudes=amplitudes,
            beta_transmission=beta_t,
            beta_reflection=beta_r,
            phase_transmission=theta_t,
            phase_reflection=theta_r,
        )
    )

    ideal_result = (
        simulate_dual_side_key_generation(
            channel_controller_to_ris=g,
            channel_ris_to_transmission_user=h_t,
            channel_ris_to_reflection_user=h_r,
            coefficients=ideal_coefficients,
            active_mask=active_mask,
            active_noise_variance=0.002,
            receiver_noise_variance_controller=0.01,
            receiver_noise_variance_transmission_user=0.01,
            receiver_noise_variance_reflection_user=0.01,
            rng=np.random.default_rng(100),
        )
    )

    mismatch = apply_hardware_mismatch(
        ideal_coefficients=ideal_coefficients,
        active_mask=active_mask,
        parameters=HardwareMismatchParameters(
            static_gain_error_std_db=0.5,
            directional_gain_error_std_db=1.5,
            static_phase_error_std_rad=0.1,
            directional_phase_error_std_rad=0.5,
            gain_scale_min=0.25,
            gain_scale_max=4.0,
        ),
        rng=np.random.default_rng(200),
    )

    impaired_result = (
        simulate_dual_side_key_generation(
            channel_controller_to_ris=g,
            channel_ris_to_transmission_user=h_t,
            channel_ris_to_reflection_user=h_r,
            coefficients=(
                mismatch.forward_coefficients
            ),
            reverse_coefficients=(
                mismatch.reverse_coefficients
            ),
            active_mask=active_mask,
            active_noise_variance=0.002,
            receiver_noise_variance_controller=0.01,
            receiver_noise_variance_transmission_user=0.01,
            receiver_noise_variance_reflection_user=0.01,
            # 与理想情况使用相同噪声随机种子
            rng=np.random.default_rng(100),
        )
    )

    mean_amplitude_difference = float(
        np.mean(
            np.abs(
                mismatch
                .forward_coefficients
                .amplitudes
                - mismatch
                .reverse_coefficients
                .amplitudes
            )
        )
    )

    mean_transmission_coefficient_difference = float(
        np.mean(
            np.abs(
                mismatch
                .forward_coefficients
                .transmission
                - mismatch
                .reverse_coefficients
                .transmission
            )
        )
    )

    print(
        "Step 4 hardware-mismatch check"
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
        f"Mean forward/reverse "
        f"amplitude difference = "
        f"{mean_amplitude_difference:.6f}"
    )
    print(
        f"Mean forward/reverse "
        f"coefficient difference = "
        f"{mean_transmission_coefficient_difference:.6f}"
    )

    print()
    print("Ideal hardware")
    print(
        f"  Weighted correlation = "
        f"{ideal_result.weighted_correlation:.6f}"
    )
    print(
        f"  Weighted Gaussian MI = "
        f"{ideal_result.weighted_mutual_information:.6f}"
    )
    print(
        f"  Weighted raw KDR     = "
        f"{ideal_result.weighted_key_disagreement_rate:.6f}"
    )

    print()
    print("Impaired hardware")
    print(
        f"  Weighted correlation = "
        f"{impaired_result.weighted_correlation:.6f}"
    )
    print(
        f"  Weighted Gaussian MI = "
        f"{impaired_result.weighted_mutual_information:.6f}"
    )
    print(
        f"  Weighted raw KDR     = "
        f"{impaired_result.weighted_key_disagreement_rate:.6f}"
    )

    assert (
        mean_amplitude_difference > 0.0
    )

    assert (
        mean_transmission_coefficient_difference
        > 0.0
    )

    assert (
        impaired_result.weighted_correlation
        < ideal_result.weighted_correlation
    )

    assert (
        impaired_result
        .weighted_key_disagreement_rate
        > ideal_result
        .weighted_key_disagreement_rate
    )

    print()
    print(
        "STEP 4 MANUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()