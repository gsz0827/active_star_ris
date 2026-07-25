from __future__ import annotations

import sys
from hashlib import sha256
from math import floor
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
from active_star_ris.practical_key_generation import (  # noqa: E402
    EndToEndKeyGenerationResult,
    generate_end_to_end_key,
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


def key_fingerprint(
    key_hex: str,
) -> str:
    return sha256(
        bytes.fromhex(
            key_hex
        )
    ).hexdigest()[:16]


def print_branch_result(
    name: str,
    result: EndToEndKeyGenerationResult,
) -> None:
    print()
    print(name)

    print(
        f"  Retained raw bits       = "
        f"{result.quantization.retained_samples}"
    )

    print(
        f"  Retention ratio         = "
        f"{result.quantization.retention_ratio:.6f}"
    )

    print(
        f"  Raw KDR                 = "
        f"{result.quantization.raw_key_disagreement_rate:.6f}"
    )

    print(
        f"  Corrections applied     = "
        f"{result.reconciliation.corrections_applied}"
    )

    print(
        f"  Post-reconciliation KDR = "
        f"{result.reconciliation.post_reconciliation_kdr:.6f}"
    )

    print(
        f"  Parity leakage          = "
        f"{result.reconciliation.parity_leakage_bits} bits"
    )

    print(
        f"  Verification passed     = "
        f"{result.reconciliation.verification_passed}"
    )

    print(
        f"  Operational bit bound   = "
        f"{result.operational_secret_bit_bound}"
    )

    print(
        f"  Final key length        = "
        f"{result.final_key_length_bits} bits"
    )

    print(
        f"  End-to-end success      = "
        f"{result.success}"
    )

    if result.privacy_amplification is not None:
        fingerprint = key_fingerprint(
            result
            .privacy_amplification
            .final_key_at_a_hex
        )

        print(
            f"  Key fingerprint         = "
            f"{fingerprint}"
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

    amplitudes[
        active_mask
    ] = 1.5

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

    finite_parameters = (
        FiniteLengthSecurityParameters(
            block_length=num_samples,
            parameter_estimation_samples=1000,
            reconciliation_efficiency=0.95,
            epsilon_smoothing=1.0e-10,
            epsilon_parameter_estimation=1.0e-10,
            epsilon_privacy_amplification=1.0e-10,
            authentication_leakage_bits=128.0,
            implementation_margin_bits_per_sample=0.01,
        )
    )

    finite_result = (
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
                finite_parameters
            ),
            reflection_parameters=(
                finite_parameters
            ),
            transmission_weight=0.5,
            reflection_weight=0.5,
        )
    )

    parameter_estimation_samples = (
        finite_parameters
        .parameter_estimation_samples
    )

    transmission_metrics = (
        finite_result.transmission
    )

    reflection_metrics = (
        finite_result.reflection
    )

    # 第6步已经扣除了0.05 bit/sample的占位公开泄漏。
    # 此处将其恢复，然后由第7步实际Cascade泄漏替代。
    transmission_entropy_bound = (
        transmission_metrics
        .extractable_secret_bits
        + floor(
            transmission_metrics
            .key_generation_samples
            * transmission_metrics
            .public_leakage_bits_per_sample
        )
    )

    reflection_entropy_bound = (
        reflection_metrics
        .extractable_secret_bits
        + floor(
            reflection_metrics
            .key_generation_samples
            * reflection_metrics
            .public_leakage_bits_per_sample
        )
    )

    transmission_probing = (
        secure_result
        .legitimate
        .transmission
        .probing
    )

    reflection_probing = (
        secure_result
        .legitimate
        .reflection
        .probing
    )

    # 参数估计样本不进入实际密钥。
    transmission_observation_a = (
        transmission_probing
        .observation_at_a[
            parameter_estimation_samples:
        ]
    )

    transmission_observation_b = (
        transmission_probing
        .observation_at_b[
            parameter_estimation_samples:
        ]
    )

    reflection_observation_a = (
        reflection_probing
        .observation_at_a[
            parameter_estimation_samples:
        ]
    )

    reflection_observation_b = (
        reflection_probing
        .observation_at_b[
            parameter_estimation_samples:
        ]
    )

    transmission_key_result = (
        generate_end_to_end_key(
            transmission_observation_a,
            transmission_observation_b,
            pre_reconciliation_entropy_bound_bits=(
                transmission_entropy_bound
            ),
            feature="real",
            guard_band_sigma=0.15,
            selection_policy="intersection",
            initial_block_size=8,
            number_of_reconciliation_passes=8,
            verification_tag_bits=32,
            privacy_margin_bits=64,
            maximum_final_key_bits=256,
            rng=np.random.default_rng(201),
        )
    )

    reflection_key_result = (
        generate_end_to_end_key(
            reflection_observation_a,
            reflection_observation_b,
            pre_reconciliation_entropy_bound_bits=(
                reflection_entropy_bound
            ),
            feature="real",
            guard_band_sigma=0.15,
            selection_policy="intersection",
            initial_block_size=8,
            number_of_reconciliation_passes=8,
            verification_tag_bits=32,
            privacy_margin_bits=64,
            maximum_final_key_bits=256,
            rng=np.random.default_rng(202),
        )
    )

    print(
        "Step 7 practical key-generation check"
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
        f"Parameter-estimation samples = "
        f"{parameter_estimation_samples}"
    )

    print_branch_result(
        "Transmission-side key",
        transmission_key_result,
    )

    print_branch_result(
        "Reflection-side key",
        reflection_key_result,
    )

    assert transmission_key_result.success
    assert reflection_key_result.success

    assert (
        transmission_key_result
        .reconciliation
        .post_reconciliation_kdr
        == 0.0
    )

    assert (
        reflection_key_result
        .reconciliation
        .post_reconciliation_kdr
        == 0.0
    )

    assert (
        transmission_key_result
        .final_key_length_bits
        == 256
    )

    assert (
        reflection_key_result
        .final_key_length_bits
        == 256
    )

    assert (
        transmission_key_result
        .privacy_amplification
        is not None
    )

    assert (
        reflection_key_result
        .privacy_amplification
        is not None
    )

    assert (
        transmission_key_result
        .privacy_amplification
        .keys_match
    )

    assert (
        reflection_key_result
        .privacy_amplification
        .keys_match
    )

    print()
    print(
        "STEP 7 MANUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()