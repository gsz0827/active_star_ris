from __future__ import annotations

from pathlib import Path

from active_star_ris.imperfect_csi_sweep import (
    ImperfectCSISweepRecord,
    run_imperfect_csi_sweep,
    write_imperfect_csi_sweep_csv,
)
from active_star_ris.simulation import (
    load_config,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "default.yaml"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "step4_imperfect_csi_sweep.csv"
)


def _record_by_scenario(
    records: list[
        ImperfectCSISweepRecord
    ],
    scenario: str,
) -> list[ImperfectCSISweepRecord]:
    return sorted(
        [
            record
            for record in records
            if record.scenario == scenario
        ],
        key=lambda item: item.nmse_db,
    )


def main() -> None:
    config = load_config(
        CONFIG_PATH
    )

    csi_config = config["csi"]

    nmse_values = [
        float(value)
        for value in csi_config[
            "evaluation_nmse_db"
        ]
    ]

    num_trials = int(
        csi_config[
            "evaluation_trials"
        ]
    )

    seed = int(
        config.get(
            "seed",
            20260719,
        )
    )

    print("=" * 78)
    print(
        "Step 4: Imperfect CSI impact "
        "and power-violation diagnosis"
    )
    print("=" * 78)
    print(
        f"NMSE values: {nmse_values}"
    )
    print(
        f"Trials per NMSE: {num_trials}"
    )
    print()

    records = run_imperfect_csi_sweep(
        config=config,
        nmse_values_db=nmse_values,
        num_trials=num_trials,
        seed=seed,
    )

    output_path = (
        write_imperfect_csi_sweep_csv(
            records=records,
            output_path=OUTPUT_PATH,
        )
    )

    scenarios = sorted(
        {
            record.scenario
            for record in records
        }
    )

    for scenario in scenarios:
        print("-" * 78)
        print(f"Scenario: {scenario}")
        print("-" * 78)

        print(
            f"{'NMSE(dB)':>10}"
            f"{'Mean rate':>14}"
            f"{'Mean P_RIS':>14}"
            f"{'K_req':>10}"
            f"{'K_eff':>10}"
            f"{'Reduce P':>12}"
            f"{'Fallback P':>12}"
            f"{'Violation P':>14}"
        )

        scenario_records = (
            _record_by_scenario(
                records,
                scenario,
            )
        )

        for record in scenario_records:
            print(
                f"{record.nmse_db:>10.1f}"
                f"{record.mean_weighted_sum_rate:>14.6f}"
                f"{record.mean_ris_output_power:>14.6f}"
                f"{record.mean_requested_active_elements:>10.2f}"
                f"{record.mean_effective_active_elements:>10.2f}"
                f"{record.active_reduction_probability:>12.4f}"
                f"{record.passive_fallback_probability:>12.4f}"
                f"{record.power_violation_probability:>14.4f}"
            )

        print()

    # -------------------------------------------------
    # 检查1：随机无源方案不应受CSI误差影响。
    # 因为该方案根本不使用CSI进行设计。
    # -------------------------------------------------
    random_records = (
        _record_by_scenario(
            records,
            "passive_random",
        )
    )

    random_rates = [
        record.mean_weighted_sum_rate
        for record in random_records
    ]

    random_rate_range = (
        max(random_rates)
        - min(random_rates)
    )

    print("=" * 78)
    print("Diagnostic conclusions")
    print("=" * 78)

    print(
        "Random-passive mean-rate range "
        f"across NMSE levels: "
        f"{random_rate_range:.12e}"
    )

    if random_rate_range <= 1.0e-10:
        print(
            "[PASS] Random passive design is "
            "independent of CSI quality."
        )
    else:
        print(
            "[WARNING] Random passive results "
            "changed with NMSE. Check paired "
            "random seeds."
        )

    # -------------------------------------------------
    # 检查2：比较最好CSI和最差CSI下的相位对齐性能。
    #
    # 数值更小的NMSE dB，例如-30 dB，代表CSI更准确；
    # 数值更大的NMSE dB，例如0 dB，代表CSI更差。
    # -------------------------------------------------
    passive_records = (
        _record_by_scenario(
            records,
            "passive_phase_aligned",
        )
    )

    best_csi_record = min(
        passive_records,
        key=lambda item: item.nmse_db,
    )

    worst_csi_record = max(
        passive_records,
        key=lambda item: item.nmse_db,
    )

    rate_loss = (
        best_csi_record
        .mean_weighted_sum_rate
        - worst_csi_record
        .mean_weighted_sum_rate
    )

    relative_rate_loss = (
        rate_loss
        / max(
            best_csi_record
            .mean_weighted_sum_rate,
            1.0e-12,
        )
    )

    print(
        "Passive phase-aligned rate:"
    )
    print(
        f"  Accurate CSI "
        f"({best_csi_record.nmse_db:.1f} dB): "
        f"{best_csi_record.mean_weighted_sum_rate:.6f}"
    )
    print(
        f"  Poor CSI "
        f"({worst_csi_record.nmse_db:.1f} dB): "
        f"{worst_csi_record.mean_weighted_sum_rate:.6f}"
    )
    print(
        f"  Relative loss: "
        f"{100.0 * relative_rate_loss:.2f}%"
    )

    if rate_loss > 0.0:
        print(
            "[PASS] Poorer CSI reduces the "
            "average phase-aligned performance."
        )
    else:
        print(
            "[WARNING] No average performance "
            "loss was observed. Increase the "
            "number of trials and inspect the "
            "CSI connection."
        )

    # -------------------------------------------------
    # 检查3：汇总部分有源方案的真实功率越界。
    # -------------------------------------------------
    active_scenarios = [
        "partial_active_fixed_beta",
        "partial_active_optimized_beta",
    ]

    any_power_violation = False

    for scenario in active_scenarios:
        scenario_records = (
            _record_by_scenario(
                records,
                scenario,
            )
        )

        maximum_probability = max(
            record
            .power_violation_probability
            for record in scenario_records
        )

        maximum_violation = max(
            record.maximum_power_violation
            for record in scenario_records
        )

        print(
            f"{scenario}:"
        )
        print(
            "  Maximum violation probability: "
            f"{maximum_probability:.4f}"
        )
        print(
            "  Maximum violation amount: "
            f"{maximum_violation:.6f}"
        )

        if maximum_violation > 1.0e-10:
            any_power_violation = True

    if any_power_violation:
        print(
            "[EXPECTED] Power violations occur "
            "because gains are designed using "
            "estimated CSI but evaluated using "
            "true channels."
        )
        print(
            "The next step will add a robust "
            "bidirectional power projection."
        )
    else:
        print(
            "[PASS] No power violation was "
            "observed for the current samples."
        )
        print(
            "A robust power projection is still "
            "needed for unseen CSI errors."
        )

    print()
    print(
        f"CSV saved to: {output_path}"
    )


if __name__ == "__main__":
    main()