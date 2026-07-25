from __future__ import annotations

import csv
from dataclasses import (
    asdict,
    dataclass,
)
from math import sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

from active_star_ris.parameter_sweep import (
    SweepPointResult,
)


@dataclass(frozen=True)
class SweepAggregateResult:
    """
    同一参数点在多次独立Monte Carlo实验下的统计结果。

    连续指标报告：

        mean
        sample standard deviation
        two-sided 95% Student-t confidence interval half-width

    当重复次数为1时：

        std = 0
        ci95 = 0
    """

    sweep_name: str
    scenario: str
    num_repetitions: int

    active_fraction: float
    active_count: int
    active_gain: float

    directional_gain_error_std_db: float
    directional_phase_error_std_rad: float

    eve_channel_scale: float
    eve_channel_correlation: float

    legitimate_receiver_noise_variance: float
    eve_receiver_noise_variance: float

    guard_band_sigma: float
    selection_policy: str

    # 连续复信道域指标。
    legitimate_mi_mean: float
    legitimate_mi_std: float
    legitimate_mi_ci95: float

    eve_leakage_mean: float
    eve_leakage_std: float
    eve_leakage_ci95: float

    asymptotic_secret_rate_mean: float
    asymptotic_secret_rate_std: float
    asymptotic_secret_rate_ci95: float

    finite_proxy_rate_mean: float
    finite_proxy_rate_std: float
    finite_proxy_rate_ci95: float

    # 量化域Eve安全指标。
    quantized_eve_mi_mean: float
    quantized_eve_mi_std: float
    quantized_eve_mi_ci95: float

    quantized_min_entropy_mean: float
    quantized_min_entropy_std: float
    quantized_min_entropy_ci95: float

    transmission_eve_guessing_probability_mean: float
    transmission_eve_guessing_probability_std: float
    transmission_eve_guessing_probability_ci95: float

    reflection_eve_guessing_probability_mean: float
    reflection_eve_guessing_probability_std: float
    reflection_eve_guessing_probability_ci95: float

    # 量化域熵界与实际Cascade泄漏共同决定的运行指标。
    operational_bound_rate_mean: float
    operational_bound_rate_std: float
    operational_bound_rate_ci95: float

    aggregate_operational_secret_bits_mean: float
    aggregate_operational_secret_bits_std: float
    aggregate_operational_secret_bits_ci95: float

    aggregate_final_key_bits_mean: float
    aggregate_final_key_bits_std: float
    aggregate_final_key_bits_ci95: float

    # 量化与信息协调指标。
    transmission_raw_kdr_mean: float
    transmission_raw_kdr_std: float
    transmission_raw_kdr_ci95: float

    reflection_raw_kdr_mean: float
    reflection_raw_kdr_std: float
    reflection_raw_kdr_ci95: float

    transmission_post_reconciliation_kdr_mean: float
    reflection_post_reconciliation_kdr_mean: float

    transmission_retention_ratio_mean: float
    transmission_retention_ratio_std: float
    transmission_retention_ratio_ci95: float

    reflection_retention_ratio_mean: float
    reflection_retention_ratio_std: float
    reflection_retention_ratio_ci95: float

    transmission_parity_leakage_bits_mean: float
    reflection_parity_leakage_bits_mean: float

    # 协议成功概率。
    transmission_success_rate: float
    reflection_success_rate: float
    dual_side_success_rate: float


_STUDENT_T_CRITICAL_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _student_t_critical_95(
    sample_count: int,
) -> float:
    """
    返回双侧95% Student-t区间的临界值。

    sample_count为样本数，故自由度为：

        df = sample_count - 1
    """
    if sample_count < 2:
        raise ValueError(
            "sample_count must be at least 2"
        )

    degrees_of_freedom = (
        sample_count
        - 1
    )

    if degrees_of_freedom <= 30:
        return _STUDENT_T_CRITICAL_95[
            degrees_of_freedom
        ]

    if degrees_of_freedom <= 40:
        lower_value = _STUDENT_T_CRITICAL_95[30]
        upper_value = 2.021

        fraction = (
            degrees_of_freedom
            - 30
        ) / 10.0

        return float(
            lower_value
            + fraction
            * (
                upper_value
                - lower_value
            )
        )

    if degrees_of_freedom <= 60:
        lower_value = 2.021
        upper_value = 2.000

        fraction = (
            degrees_of_freedom
            - 40
        ) / 20.0

        return float(
            lower_value
            + fraction
            * (
                upper_value
                - lower_value
            )
        )

    if degrees_of_freedom <= 120:
        lower_value = 2.000
        upper_value = 1.980

        fraction = (
            degrees_of_freedom
            - 60
        ) / 60.0

        return float(
            lower_value
            + fraction
            * (
                upper_value
                - lower_value
            )
        )

    return 1.96


def _summary_triplet(
    values: Sequence[float],
) -> tuple[
    float,
    float,
    float,
]:
    """
    返回：

        mean
        sample standard deviation
        95% Student-t confidence interval half-width
    """
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        raise ValueError(
            "values cannot be empty"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "values contain non-finite entries"
        )

    mean_value = float(
        np.mean(array)
    )

    if array.size == 1:
        return (
            mean_value,
            0.0,
            0.0,
        )

    standard_deviation = float(
        np.std(
            array,
            ddof=1,
        )
    )

    critical_value = (
        _student_t_critical_95(
            int(array.size)
        )
    )

    ci95_half_width = float(
        critical_value
        * standard_deviation
        / sqrt(array.size)
    )

    return (
        mean_value,
        standard_deviation,
        ci95_half_width,
    )


def _extract_triplet(
    group: Sequence[SweepPointResult],
    attribute_name: str,
) -> tuple[
    float,
    float,
    float,
]:
    values = [
        float(
            getattr(
                item,
                attribute_name,
            )
        )
        for item in group
    ]

    return _summary_triplet(
        values
    )


def _mean_attribute(
    group: Sequence[SweepPointResult],
    attribute_name: str,
) -> float:
    return float(
        np.mean(
            [
                float(
                    getattr(
                        item,
                        attribute_name,
                    )
                )
                for item in group
            ]
        )
    )


def _validate_group_consistency(
    group: Sequence[SweepPointResult],
) -> None:
    """
    防止相同(sweep_name, scenario)下混入不同参数点。
    """
    if len(group) == 0:
        raise ValueError(
            "group cannot be empty"
        )

    first = group[0]

    fields_to_compare = (
        "active_fraction",
        "active_count",
        "active_gain",
        "directional_gain_error_std_db",
        "directional_phase_error_std_rad",
        "eve_channel_scale",
        "eve_channel_correlation",
        "legitimate_receiver_noise_variance",
        "eve_receiver_noise_variance",
        "guard_band_sigma",
        "selection_policy",
    )

    for item in group[1:]:
        for field_name in fields_to_compare:
            if (
                getattr(
                    item,
                    field_name,
                )
                != getattr(
                    first,
                    field_name,
                )
            ):
                raise ValueError(
                    "inconsistent parameter values "
                    f"inside scenario {first.scenario}: "
                    f"{field_name}"
                )


def aggregate_sweep_results(
    results: Sequence[SweepPointResult],
) -> tuple[SweepAggregateResult, ...]:
    """
    按照：

        (sweep_name, scenario)

    对重复实验进行聚合。
    """
    if len(results) == 0:
        raise ValueError(
            "results cannot be empty"
        )

    grouped: dict[
        tuple[str, str],
        list[SweepPointResult],
    ] = {}

    for result in results:
        key = (
            result.sweep_name,
            result.scenario,
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            result
        )

    summaries: list[
        SweepAggregateResult
    ] = []

    for group in grouped.values():
        _validate_group_consistency(
            group
        )

        first = group[0]

        legitimate_mi = _extract_triplet(
            group,
            "weighted_legitimate_mi_bits_per_sample",
        )

        eve_leakage = _extract_triplet(
            group,
            "weighted_eve_leakage_bits_per_sample",
        )

        asymptotic_rate = _extract_triplet(
            group,
            "weighted_asymptotic_secret_rate_bits_per_sample",
        )

        finite_proxy_rate = _extract_triplet(
            group,
            "weighted_finite_length_rate_bits_per_sample",
        )

        quantized_eve_mi = _extract_triplet(
            group,
            "weighted_quantized_eve_mutual_information_bits_per_retained_bit",
        )

        quantized_min_entropy = _extract_triplet(
            group,
            "weighted_quantized_conditional_min_entropy_bits_per_retained_bit",
        )

        transmission_guessing_probability = (
            _extract_triplet(
                group,
                "transmission_quantized_eve_guessing_probability",
            )
        )

        reflection_guessing_probability = (
            _extract_triplet(
                group,
                "reflection_quantized_eve_guessing_probability",
            )
        )

        operational_rate = _extract_triplet(
            group,
            "weighted_operational_bound_bits_per_sample",
        )

        operational_bits = _extract_triplet(
            group,
            "aggregate_operational_secret_bit_bound",
        )

        final_key_bits = _extract_triplet(
            group,
            "aggregate_final_key_bits",
        )

        transmission_raw_kdr = _extract_triplet(
            group,
            "transmission_raw_kdr",
        )

        reflection_raw_kdr = _extract_triplet(
            group,
            "reflection_raw_kdr",
        )

        transmission_retention = _extract_triplet(
            group,
            "transmission_retention_ratio",
        )

        reflection_retention = _extract_triplet(
            group,
            "reflection_retention_ratio",
        )

        summaries.append(
            SweepAggregateResult(
                sweep_name=first.sweep_name,
                scenario=first.scenario,
                num_repetitions=len(group),
                active_fraction=float(
                    first.active_fraction
                ),
                active_count=int(
                    first.active_count
                ),
                active_gain=float(
                    first.active_gain
                ),
                directional_gain_error_std_db=float(
                    first
                    .directional_gain_error_std_db
                ),
                directional_phase_error_std_rad=float(
                    first
                    .directional_phase_error_std_rad
                ),
                eve_channel_scale=float(
                    first.eve_channel_scale
                ),
                eve_channel_correlation=float(
                    first.eve_channel_correlation
                ),
                legitimate_receiver_noise_variance=float(
                    first
                    .legitimate_receiver_noise_variance
                ),
                eve_receiver_noise_variance=float(
                    first
                    .eve_receiver_noise_variance
                ),
                guard_band_sigma=float(
                    first.guard_band_sigma
                ),
                selection_policy=str(
                    first.selection_policy
                ),
                legitimate_mi_mean=legitimate_mi[0],
                legitimate_mi_std=legitimate_mi[1],
                legitimate_mi_ci95=legitimate_mi[2],
                eve_leakage_mean=eve_leakage[0],
                eve_leakage_std=eve_leakage[1],
                eve_leakage_ci95=eve_leakage[2],
                asymptotic_secret_rate_mean=(
                    asymptotic_rate[0]
                ),
                asymptotic_secret_rate_std=(
                    asymptotic_rate[1]
                ),
                asymptotic_secret_rate_ci95=(
                    asymptotic_rate[2]
                ),
                finite_proxy_rate_mean=(
                    finite_proxy_rate[0]
                ),
                finite_proxy_rate_std=(
                    finite_proxy_rate[1]
                ),
                finite_proxy_rate_ci95=(
                    finite_proxy_rate[2]
                ),
                quantized_eve_mi_mean=(
                    quantized_eve_mi[0]
                ),
                quantized_eve_mi_std=(
                    quantized_eve_mi[1]
                ),
                quantized_eve_mi_ci95=(
                    quantized_eve_mi[2]
                ),
                quantized_min_entropy_mean=(
                    quantized_min_entropy[0]
                ),
                quantized_min_entropy_std=(
                    quantized_min_entropy[1]
                ),
                quantized_min_entropy_ci95=(
                    quantized_min_entropy[2]
                ),
                transmission_eve_guessing_probability_mean=(
                    transmission_guessing_probability[0]
                ),
                transmission_eve_guessing_probability_std=(
                    transmission_guessing_probability[1]
                ),
                transmission_eve_guessing_probability_ci95=(
                    transmission_guessing_probability[2]
                ),
                reflection_eve_guessing_probability_mean=(
                    reflection_guessing_probability[0]
                ),
                reflection_eve_guessing_probability_std=(
                    reflection_guessing_probability[1]
                ),
                reflection_eve_guessing_probability_ci95=(
                    reflection_guessing_probability[2]
                ),
                operational_bound_rate_mean=(
                    operational_rate[0]
                ),
                operational_bound_rate_std=(
                    operational_rate[1]
                ),
                operational_bound_rate_ci95=(
                    operational_rate[2]
                ),
                aggregate_operational_secret_bits_mean=(
                    operational_bits[0]
                ),
                aggregate_operational_secret_bits_std=(
                    operational_bits[1]
                ),
                aggregate_operational_secret_bits_ci95=(
                    operational_bits[2]
                ),
                aggregate_final_key_bits_mean=(
                    final_key_bits[0]
                ),
                aggregate_final_key_bits_std=(
                    final_key_bits[1]
                ),
                aggregate_final_key_bits_ci95=(
                    final_key_bits[2]
                ),
                transmission_raw_kdr_mean=(
                    transmission_raw_kdr[0]
                ),
                transmission_raw_kdr_std=(
                    transmission_raw_kdr[1]
                ),
                transmission_raw_kdr_ci95=(
                    transmission_raw_kdr[2]
                ),
                reflection_raw_kdr_mean=(
                    reflection_raw_kdr[0]
                ),
                reflection_raw_kdr_std=(
                    reflection_raw_kdr[1]
                ),
                reflection_raw_kdr_ci95=(
                    reflection_raw_kdr[2]
                ),
                transmission_post_reconciliation_kdr_mean=(
                    _mean_attribute(
                        group,
                        "transmission_post_reconciliation_kdr",
                    )
                ),
                reflection_post_reconciliation_kdr_mean=(
                    _mean_attribute(
                        group,
                        "reflection_post_reconciliation_kdr",
                    )
                ),
                transmission_retention_ratio_mean=(
                    transmission_retention[0]
                ),
                transmission_retention_ratio_std=(
                    transmission_retention[1]
                ),
                transmission_retention_ratio_ci95=(
                    transmission_retention[2]
                ),
                reflection_retention_ratio_mean=(
                    reflection_retention[0]
                ),
                reflection_retention_ratio_std=(
                    reflection_retention[1]
                ),
                reflection_retention_ratio_ci95=(
                    reflection_retention[2]
                ),
                transmission_parity_leakage_bits_mean=(
                    _mean_attribute(
                        group,
                        "transmission_parity_leakage_bits",
                    )
                ),
                reflection_parity_leakage_bits_mean=(
                    _mean_attribute(
                        group,
                        "reflection_parity_leakage_bits",
                    )
                ),
                transmission_success_rate=(
                    _mean_attribute(
                        group,
                        "transmission_success",
                    )
                ),
                reflection_success_rate=(
                    _mean_attribute(
                        group,
                        "reflection_success",
                    )
                ),
                dual_side_success_rate=(
                    _mean_attribute(
                        group,
                        "dual_side_success",
                    )
                ),
            )
        )

    return tuple(
        summaries
    )


def write_sweep_summary_csv(
    summaries: Sequence[SweepAggregateResult],
    output_path: str | Path,
) -> Path:
    """
    将统计汇总结果写入CSV。
    """
    if len(summaries) == 0:
        raise ValueError(
            "summaries cannot be empty"
        )

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        asdict(
            summary
        )
        for summary in summaries
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    return path