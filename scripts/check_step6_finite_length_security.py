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

from active_star_ris.finite_length_security import (  # noqa: E402
    FiniteLengthSecurityParameters,
    evaluate_dual_side_finite_length_security,
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

    num_samples = 10000
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

    h_e_t = complex_gaussian(
        channel_rng,
        (num_samples, num_elements),
        scale=0.2,
    )

    h_e_r = complex_gaussian(
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

    secure_result = (
        simulate_dual_side_secure_key_generation(
            channel_controller_to_ris=g,
            channel_ris_to_transmission_user=h_t,
            channel_ris_to_reflection_user=h_r,
            channel_ris_to_eve_transmission=h_e_t,
            channel_ris_to_eve_reflection=h_e_r,
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

    short_block_parameters = (
        FiniteLengthSecurityParameters(
            block_length=2000,
            parameter_estimation_samples=200,
            reconciliation_efficiency=0.95,
            epsilon_smoothing=1.0e-10,
            epsilon_parameter_estimation=1.0e-10,
            epsilon_privacy_amplification=1.0e-10,
            authentication_leakage_bits=128.0,
            implementation_margin_bits_per_sample=0.01,
        )
    )

    long_block_parameters = (
        FiniteLengthSecurityParameters(
            block_length=10000,
            parameter_estimation_samples=1000,
            reconciliation_efficiency=0.95,
            epsilon_smoothing=1.0e-10,
            epsilon_parameter_estimation=1.0e-10,
            epsilon_privacy_amplification=1.0e-10,
            authentication_leakage_bits=128.0,
            implementation_margin_bits_per_sample=0.01,
        )
    )

    short_block_result = (
        evaluate_dual_side_finite_length_security(
            transmission_secrecy=(
                secure_result
                .transmission_secrecy
            ),
            reflection_secrecy=(
                secure_result
                .reflection_secrecy
            ),
            transmission_parameters=(
                short_block_parameters
            ),
            reflection_parameters=(
                short_block_parameters
            ),
            transmission_weight=0.5,
            reflection_weight=0.5,
        )
    )

    long_block_result = (
        evaluate_dual_side_finite_length_security(
            transmission_secrecy=(
                secure_result
                .transmission_secrecy
            ),
            reflection_secrecy=(
                secure_result
                .reflection_secrecy
            ),
            transmission_parameters=(
                long_block_parameters
            ),
            reflection_parameters=(
                long_block_parameters
            ),
            transmission_weight=0.5,
            reflection_weight=0.5,
        )
    )

    print(
        "Step 6 finite-length security check"
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
    print("Asymptotic Gaussian proxy")

    print(
        f"  Weighted legitimate MI  = "
        f"{secure_result.legitimate.weighted_mutual_information:.6f}"
    )

    print(
        f"  Weighted Eve leakage    = "
        f"{secure_result.weighted_eve_leakage_bits_per_sample:.6f}"
    )

    print(
        f"  Weighted secret rate    = "
        f"{secure_result.weighted_secret_key_rate_bits_per_sample:.6f}"
    )

    print()
    print("Short finite block")

    print(
        f"  Total samples           = "
        f"{short_block_parameters.block_length}"
    )

    print(
        f"  Key-generation samples  = "
        f"{short_block_result.transmission.key_generation_samples}"
    )

    print(
        f"  Weighted finite rate    = "
        f"{short_block_result.weighted_finite_length_rate_bits_per_sample:.6f}"
    )

    print(
        f"  Aggregate secret bits   = "
        f"{short_block_result.aggregate_extractable_secret_bits}"
    )

    print()
    print("Long finite block")

    print(
        f"  Total samples           = "
        f"{long_block_parameters.block_length}"
    )

    print(
        f"  Key-generation samples  = "
        f"{long_block_result.transmission.key_generation_samples}"
    )

    print(
        f"  Weighted finite rate    = "
        f"{long_block_result.weighted_finite_length_rate_bits_per_sample:.6f}"
    )

    print(
        f"  Aggregate secret bits   = "
        f"{long_block_result.aggregate_extractable_secret_bits}"
    )

    print()
    print("Transmission-side penalty decomposition")

    transmission_metrics = (
        long_block_result.transmission
    )

    print(
        f"  Reconciliation loss     = "
        f"{transmission_metrics.reconciliation_loss_bits_per_sample:.6f}"
    )

    print(
        f"  AEP penalty             = "
        f"{transmission_metrics.aep_penalty_bits_per_sample:.6f}"
    )

    print(
        f"  Parameter-estimation    = "
        f"{transmission_metrics.parameter_estimation_penalty_bits_per_sample:.6f}"
    )

    print(
        f"  Privacy amplification   = "
        f"{transmission_metrics.privacy_amplification_penalty_bits_per_sample:.6f}"
    )

    print(
        f"  Authentication penalty  = "
        f"{transmission_metrics.authentication_penalty_bits_per_sample:.6f}"
    )

    assert (
        short_block_result
        .weighted_finite_length_rate_bits_per_sample
        < secure_result
        .weighted_secret_key_rate_bits_per_sample
    )

    assert (
        long_block_result
        .weighted_finite_length_rate_bits_per_sample
        < secure_result
        .weighted_secret_key_rate_bits_per_sample
    )

    assert (
        long_block_result
        .weighted_finite_length_rate_bits_per_sample
        > short_block_result
        .weighted_finite_length_rate_bits_per_sample
    )

    assert (
        long_block_result
        .aggregate_extractable_secret_bits
        > short_block_result
        .aggregate_extractable_secret_bits
    )

    print()
    print(
        "STEP 6 MANUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()