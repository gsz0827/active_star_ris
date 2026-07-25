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

from active_star_ris.secure_key_generation import (  # noqa: E402
    simulate_dual_side_secure_key_generation,
)
from active_star_ris.star_key_system import (  # noqa: E402
    build_star_coefficients,
)


def complex_gaussian(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    scale: float = 1.0,
) -> np.ndarray:
    return (
        scale
        * (
            rng.normal(size=shape)
            + 1j * rng.normal(size=shape)
        )
        / np.sqrt(2.0)
    )


def main() -> None:
    channel_rng = np.random.default_rng(
        20260719
    )

    num_samples = 5000
    num_elements = 16

    g = complex_gaussian(
        channel_rng,
        (num_samples, num_elements),
    )

    h_t = complex_gaussian(
        channel_rng,
        (num_samples, num_elements),
    )

    h_r = complex_gaussian(
        channel_rng,
        (num_samples, num_elements),
    )

    # 距离较远且信道相对较弱的Eve。
    h_e_t_far = complex_gaussian(
        channel_rng,
        (num_samples, num_elements),
        scale=0.2,
    )

    h_e_r_far = complex_gaussian(
        channel_rng,
        (num_samples, num_elements),
        scale=0.2,
    )

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

    coefficients = build_star_coefficients(
        amplitudes=amplitudes,
        beta_transmission=beta_t,
        beta_reflection=beta_r,
        phase_transmission=theta_t,
        phase_reflection=theta_r,
    )

    far_eve_result = (
        simulate_dual_side_secure_key_generation(
            channel_controller_to_ris=g,
            channel_ris_to_transmission_user=h_t,
            channel_ris_to_reflection_user=h_r,
            channel_ris_to_eve_transmission=(
                h_e_t_far
            ),
            channel_ris_to_eve_reflection=(
                h_e_r_far
            ),
            coefficients=coefficients,
            active_mask=active_mask,
            active_noise_variance=0.002,
            receiver_noise_variance_controller=0.01,
            receiver_noise_variance_transmission_user=0.01,
            receiver_noise_variance_reflection_user=0.01,
            receiver_noise_variance_eve_transmission=0.5,
            receiver_noise_variance_eve_reflection=0.5,
            public_leakage_transmission_bits_per_sample=0.05,
            public_leakage_reflection_bits_per_sample=0.05,
            transmission_weight=0.5,
            reflection_weight=0.5,
            rng=np.random.default_rng(100),
        )
    )

    # 最不利情况：Eve与对应合法用户共址，
    # 并且Eve接收机没有额外噪声。
    colocated_eve_result = (
        simulate_dual_side_secure_key_generation(
            channel_controller_to_ris=g,
            channel_ris_to_transmission_user=h_t,
            channel_ris_to_reflection_user=h_r,
            channel_ris_to_eve_transmission=h_t,
            channel_ris_to_eve_reflection=h_r,
            coefficients=coefficients,
            active_mask=active_mask,
            active_noise_variance=0.002,
            receiver_noise_variance_controller=0.01,
            receiver_noise_variance_transmission_user=0.01,
            receiver_noise_variance_reflection_user=0.01,
            receiver_noise_variance_eve_transmission=0.0,
            receiver_noise_variance_eve_reflection=0.0,
            public_leakage_transmission_bits_per_sample=0.0,
            public_leakage_reflection_bits_per_sample=0.0,
            transmission_weight=0.5,
            reflection_weight=0.5,
            # 使用相同随机种子，保证合法链路噪声一致。
            rng=np.random.default_rng(100),
        )
    )

    print(
        "Step 5 secure key-generation check"
    )
    print(
        f"Number of STAR-RIS elements = "
        f"{num_elements}"
    )
    print(
        f"Number of active elements   = "
        f"{np.sum(active_mask)}"
    )

    print()
    print("Far/noisy Eve")

    print(
        f"  Legitimate weighted MI = "
        f"{far_eve_result.legitimate.weighted_mutual_information:.6f}"
    )

    print(
        f"  Weighted Eve leakage   = "
        f"{far_eve_result.weighted_eve_leakage_bits_per_sample:.6f}"
    )

    print(
        f"  Weighted secret rate   = "
        f"{far_eve_result.weighted_secret_key_rate_bits_per_sample:.6f}"
    )

    print(
        f"  Transmission rate      = "
        f"{far_eve_result.transmission_secrecy.secret_key_rate_bits_per_sample:.6f}"
    )

    print(
        f"  Reflection rate        = "
        f"{far_eve_result.reflection_secrecy.secret_key_rate_bits_per_sample:.6f}"
    )

    print()
    print("Co-located clean Eve")

    print(
        f"  Legitimate weighted MI = "
        f"{colocated_eve_result.legitimate.weighted_mutual_information:.6f}"
    )

    print(
        f"  Weighted Eve leakage   = "
        f"{colocated_eve_result.weighted_eve_leakage_bits_per_sample:.6f}"
    )

    print(
        f"  Weighted secret rate   = "
        f"{colocated_eve_result.weighted_secret_key_rate_bits_per_sample:.6f}"
    )

    assert np.isclose(
        far_eve_result
        .legitimate
        .weighted_mutual_information,
        colocated_eve_result
        .legitimate
        .weighted_mutual_information,
    )

    assert (
        far_eve_result
        .weighted_secret_key_rate_bits_per_sample
        > 0.0
    )

    assert (
        colocated_eve_result
        .weighted_secret_key_rate_bits_per_sample
        == 0.0
    )

    assert (
        colocated_eve_result
        .weighted_eve_leakage_bits_per_sample
        > far_eve_result
        .weighted_eve_leakage_bits_per_sample
    )

    print()
    print(
        "STEP 5 MANUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()