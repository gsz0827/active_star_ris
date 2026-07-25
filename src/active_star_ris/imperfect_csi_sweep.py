from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .simulation import (
    ScenarioResult,
    evaluate_one_realization_imperfect_csi,
)


@dataclass(frozen=True)
class ImperfectCSISweepRecord:
    """一个NMSE水平下某个方案的统计结果。"""

    nmse_db: float
    scenario: str
    num_trials: int

    mean_weighted_sum_rate: float
    std_weighted_sum_rate: float

    mean_transmission_rate: float
    mean_reflection_rate: float

    mean_ris_output_power: float
    maximum_ris_output_power: float

    mean_power_violation: float
    maximum_power_violation: float
    power_violation_probability: float

    mean_beta_transmission: float
    mean_active_amplitude: float

    mean_requested_active_elements: float
    mean_effective_active_elements: float

    minimum_effective_active_elements: int
    maximum_effective_active_elements: int

    mean_disabled_active_elements: float

    active_reduction_probability: float
    passive_fallback_probability: float

    # 一次实现内部的逐单元增益离散程度，
    # 再对所有蒙特卡洛实现取平均。
    mean_active_gain_std: float
    mean_active_gain_min: float
    mean_active_gain_max: float
    mean_beta_transmission_std: float
    mean_beta_transmission_min: float
    mean_beta_transmission_max: float


def _summarize_scenario(
    nmse_db: float,
    scenario: str,
    entries: list[ScenarioResult],
) -> ImperfectCSISweepRecord:
    """汇总一个NMSE水平下某个方案的结果。"""

    if not entries:
        raise ValueError(
            "entries must not be empty"
        )

    weighted_rates = np.asarray(
        [
            entry.weighted_sum_rate
            for entry in entries
        ],
        dtype=np.float64,
    )

    transmission_rates = np.asarray(
        [
            entry.transmission_rate
            for entry in entries
        ],
        dtype=np.float64,
    )

    reflection_rates = np.asarray(
        [
            entry.reflection_rate
            for entry in entries
        ],
        dtype=np.float64,
    )

    output_powers = np.asarray(
        [
            entry.ris_output_power
            for entry in entries
        ],
        dtype=np.float64,
    )

    violations = np.asarray(
        [
            entry.ris_power_violation
            for entry in entries
        ],
        dtype=np.float64,
    )

    beta_values = np.asarray(
        [
            entry.beta_transmission
            for entry in entries
        ],
        dtype=np.float64,
    )

    amplitudes = np.asarray(
        [
            entry.active_amplitude
            for entry in entries
        ],
        dtype=np.float64,
    )

    active_gain_stds = np.asarray(
        [
            entry.active_gain_std
            for entry in entries
        ],
        dtype=np.float64,
    )

    active_gain_mins = np.asarray(
        [
            entry.active_gain_min
            for entry in entries
        ],
        dtype=np.float64,
    )

    active_gain_maxs = np.asarray(
        [
            entry.active_gain_max
            for entry in entries
        ],
        dtype=np.float64,
    )

    requested_active = np.asarray(
        [
            entry.requested_active_elements
            for entry in entries
        ],
        dtype=np.int64,
    )

    effective_active = np.asarray(
        [
            entry.effective_active_elements
            for entry in entries
        ],
        dtype=np.int64,
    )

    disabled_active = np.asarray(
        [
            entry.disabled_active_elements
            for entry in entries
        ],
        dtype=np.int64,
    )

    passive_fallback = np.asarray(
        [
            entry.passive_fallback
            for entry in entries
        ],
        dtype=bool,
    )

    active_reduction = (
        effective_active
        < requested_active
    )

    if len(entries) > 1:
        rate_std = float(
            np.std(
                weighted_rates,
                ddof=1,
            )
        )
    else:
        rate_std = 0.0

    violation_probability = float(
        np.mean(
            violations > 1.0e-12
        )
    )

    beta_stds = np.asarray(
        [
            entry.beta_transmission_std
            for entry in entries
        ],
        dtype=np.float64,
    )

    beta_mins = np.asarray(
        [
            entry.beta_transmission_min
            for entry in entries
        ],
        dtype=np.float64,
    )

    beta_maxs = np.asarray(
        [
            entry.beta_transmission_max
            for entry in entries
        ],
        dtype=np.float64,
    )

    return ImperfectCSISweepRecord(
        nmse_db=float(nmse_db),
        scenario=scenario,
        num_trials=len(entries),
        mean_weighted_sum_rate=float(
            np.mean(weighted_rates)
        ),
        std_weighted_sum_rate=rate_std,
        mean_transmission_rate=float(
            np.mean(transmission_rates)
        ),
        mean_reflection_rate=float(
            np.mean(reflection_rates)
        ),
        mean_ris_output_power=float(
            np.mean(output_powers)
        ),
        maximum_ris_output_power=float(
            np.max(output_powers)
        ),
        mean_power_violation=float(
            np.mean(violations)
        ),
        maximum_power_violation=float(
            np.max(violations)
        ),
        power_violation_probability=(
            violation_probability
        ),
        mean_beta_transmission=float(
            np.mean(beta_values)
        ),
        mean_active_amplitude=float(
            np.mean(amplitudes)
        ),
        mean_requested_active_elements=float(
            np.mean(requested_active)
        ),
        mean_effective_active_elements=float(
            np.mean(effective_active)
        ),
        minimum_effective_active_elements=int(
            np.min(effective_active)
        ),
        maximum_effective_active_elements=int(
            np.max(effective_active)
        ),
        mean_disabled_active_elements=float(
            np.mean(disabled_active)
        ),
        active_reduction_probability=float(
            np.mean(active_reduction)
        ),
        passive_fallback_probability=float(
            np.mean(passive_fallback)
        ),
        mean_active_gain_std=float(
            np.mean(
                active_gain_stds
            )
        ),
        mean_active_gain_min=float(
            np.mean(
                active_gain_mins
            )
        ),
        mean_active_gain_max=float(
            np.mean(
                active_gain_maxs
            )
        ),
        mean_beta_transmission_std=float(
            np.mean(beta_stds)
        ),
        mean_beta_transmission_min=float(
            np.mean(beta_mins)
        ),
        mean_beta_transmission_max=float(
            np.mean(beta_maxs)
        ),
    )


def run_imperfect_csi_sweep(
    config: dict[str, Any],
    nmse_values_db: Iterable[float],
    num_trials: int,
    seed: int,
    num_elements: int | None = None,
    num_active_elements: int | None = None,
) -> list[ImperfectCSISweepRecord]:
    """进行不完美CSI的配对蒙特卡洛扫描。

    对每一个NMSE水平使用完全相同的trial seeds。

    因此，不同NMSE水平之间：
    1. 真实信道实现相同；
    2. 随机无源STAR-RIS相位相同；
    3. 只有CSI误差强度发生变化。

    这种配对设计可以避免不同信道样本掩盖CSI误差影响。
    """

    if num_trials <= 0:
        raise ValueError(
            "num_trials must be positive"
        )

    nmse_values = np.asarray(
        list(nmse_values_db),
        dtype=np.float64,
    ).reshape(-1)

    if nmse_values.size == 0:
        raise ValueError(
            "nmse_values_db must not be empty"
        )

    if not np.all(
        np.isfinite(nmse_values)
    ):
        raise ValueError(
            "all NMSE values must be finite"
        )

    # 只生成一次trial seeds。
    # 每个NMSE水平重复使用这些种子。
    seed_rng = np.random.default_rng(
        seed
    )

    trial_seeds = seed_rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=num_trials,
        dtype=np.uint32,
    )

    collected: dict[
        tuple[float, str],
        list[ScenarioResult],
    ] = {}

    for nmse_db in nmse_values:
        for trial_seed in trial_seeds:
            trial_rng = np.random.default_rng(
                int(trial_seed)
            )

            actual_nmse_db, results = (
                evaluate_one_realization_imperfect_csi(
                    config=config,
                    rng=trial_rng,
                    num_elements=num_elements,
                    num_active_elements=(
                        num_active_elements
                    ),
                    nmse_db=float(nmse_db),
                )
            )

            if actual_nmse_db != float(
                nmse_db
            ):
                raise RuntimeError(
                    "evaluation returned an "
                    "unexpected NMSE value"
                )

            for scenario, result in (
                results.items()
            ):
                key = (
                    float(nmse_db),
                    scenario,
                )

                collected.setdefault(
                    key,
                    [],
                ).append(result)

    records: list[
        ImperfectCSISweepRecord
    ] = []

    for nmse_db in nmse_values:
        scenario_names = sorted(
            scenario
            for stored_nmse, scenario
            in collected
            if stored_nmse == float(
                nmse_db
            )
        )

        for scenario in scenario_names:
            records.append(
                _summarize_scenario(
                    nmse_db=float(
                        nmse_db
                    ),
                    scenario=scenario,
                    entries=collected[
                        (
                            float(nmse_db),
                            scenario,
                        )
                    ],
                )
            )

    return records


def write_imperfect_csi_sweep_csv(
    records: list[
        ImperfectCSISweepRecord
    ],
    output_path: str | Path,
) -> Path:
    """将扫描统计结果写入CSV文件。"""

    if not records:
        raise ValueError(
            "records must not be empty"
        )

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        asdict(record)
        for record in records
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
        writer.writerows(rows)

    return path