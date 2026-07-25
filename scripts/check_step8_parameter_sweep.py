from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from active_star_ris.parameter_sweep import (  # noqa: E402
    SweepExperimentConfig,
    SweepPoint,
    build_one_factor_sweep,
    run_parameter_sweep,
    write_sweep_results_csv,
)


def main() -> None:
    config = SweepExperimentConfig(
        # 第8步功能验证使用1800个样本。
        # 正式论文实验建议提高至10000以上。
        num_samples=1800,
        num_elements=16,
        parameter_estimation_fraction=0.1,
        beta_transmission=0.65,
        active_noise_variance=0.002,
        reconciliation_efficiency=0.95,
        authentication_leakage_bits=128.0,
        implementation_margin_bits_per_sample=0.01,
        maximum_final_key_bits=128,
        channel_seed=20260719,
        coefficient_seed=20260720,
        observation_seed=20260721,
        practical_seed=20260722,
    )

    baseline = SweepPoint(
        sweep_name="baseline",
        scenario="baseline",
        active_fraction=0.25,
        active_gain=1.5,
        directional_gain_error_std_db=0.0,
        directional_phase_error_std_rad=0.0,
        eve_channel_scale=0.2,
        eve_channel_correlation=0.0,
        legitimate_receiver_noise_variance=0.01,
        eve_receiver_noise_variance=0.5,
        guard_band_sigma=0.05,
        # 使用Alice单侧可靠性选择，
        # 便于观察非零原始KDR和实际纠错过程。
        selection_policy="alice",
    )

    points = []

    points.extend(
        build_one_factor_sweep(
            baseline,
            "active_fraction",
            [
                0.0,
                0.125,
                0.25,
                0.5,
                0.75,
                1.0,
            ],
        )
    )

    points.extend(
        build_one_factor_sweep(
            baseline,
            "active_gain",
            [
                1.0,
                1.25,
                1.5,
                2.0,
            ],
        )
    )

    points.extend(
        build_one_factor_sweep(
            baseline,
            "directional_gain_error_std_db",
            [
                0.0,
                0.25,
                0.5,
                1.0,
                1.5,
            ],
        )
    )

    points.extend(
        build_one_factor_sweep(
            baseline,
            "directional_phase_error_std_rad",
            [
                0.0,
                0.05,
                0.1,
                0.2,
                0.5,
            ],
        )
    )

    points.extend(
        build_one_factor_sweep(
            baseline,
            "eve_channel_correlation",
            [
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
            ],
        )
    )

    points.extend(
        build_one_factor_sweep(
            baseline,
            "legitimate_receiver_noise_variance",
            [
                0.001,
                0.01,
                0.03,
                0.1,
            ],
        )
    )

    points.extend(
        build_one_factor_sweep(
            baseline,
            "guard_band_sigma",
            [
                0.0,
                0.05,
                0.1,
                0.15,
                0.25,
            ],
        )
    )

    print(
        "Step 8 parameter-sweep check"
    )

    print(
        f"Number of parameter points = "
        f"{len(points)}"
    )

    print(
        f"Samples per point          = "
        f"{config.num_samples}"
    )

    print(
        "Running parameter sweep..."
    )

    # 功能验证先使用1次重复。
    # 正式论文统计可改为20或30。
    results = run_parameter_sweep(
        config,
        points,
        num_repetitions=1,
    )

    output_path = (
        PROJECT_ROOT
        / "results"
        / "step8_parameter_sweep.csv"
    )

    write_sweep_results_csv(
        results,
        output_path,
    )

    print()
    print(
        f"CSV output = {output_path}"
    )

    print()
    print(
        "Sweep summary"
    )

    for result in results:
        print(
            f"{result.sweep_name:42s} "
            f"{result.scenario:48s} "
            f"MI={result.weighted_legitimate_mi_bits_per_sample:8.4f} "
            f"Eve={result.weighted_eve_leakage_bits_per_sample:8.4f} "
            f"Finite={result.weighted_finite_length_rate_bits_per_sample:8.4f} "
            f"KDR-T={result.transmission_raw_kdr:7.4f} "
            f"KDR-R={result.reflection_raw_kdr:7.4f} "
            f"Bits={result.aggregate_final_key_bits:4d} "
            f"Success={result.dual_side_success}"
        )

    ideal_mismatch_result = next(
        result
        for result in results
        if (
            result.sweep_name
            == "directional_phase_error_std_rad"
            and result.scenario
            == "directional_phase_error_std_rad=0.0"
        )
    )

    strong_mismatch_result = next(
        result
        for result in results
        if (
            result.sweep_name
            == "directional_phase_error_std_rad"
            and result.scenario
            == "directional_phase_error_std_rad=0.5"
        )
    )

    independent_eve_result = next(
        result
        for result in results
        if (
            result.sweep_name
            == "eve_channel_correlation"
            and result.scenario
            == "eve_channel_correlation=0.0"
        )
    )

    correlated_eve_result = next(
        result
        for result in results
        if (
            result.sweep_name
            == "eve_channel_correlation"
            and result.scenario
            == "eve_channel_correlation=1.0"
        )
    )

    assert output_path.exists()

    assert len(results) == len(
        points
    )

    assert (
        strong_mismatch_result
        .weighted_legitimate_mi_bits_per_sample
        < ideal_mismatch_result
        .weighted_legitimate_mi_bits_per_sample
    )

    assert (
        correlated_eve_result
        .weighted_eve_leakage_bits_per_sample
        > independent_eve_result
        .weighted_eve_leakage_bits_per_sample
    )

    print()
    print(
        "STEP 8 MANUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()