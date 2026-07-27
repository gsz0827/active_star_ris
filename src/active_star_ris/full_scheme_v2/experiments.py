from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .config import EnvironmentConfig
from .environment import RobustFullSchemeEnvironment
from .td3 import TD3Agent


Policy = Callable[[RobustFullSchemeEnvironment, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ExperimentSummary:
    method: str
    episodes: int

    mean_return: float
    std_return: float
    ci95_return: float

    mean_training_key_rate_bps: float
    std_training_key_rate_bps: float
    ci95_training_key_rate_bps: float

    mean_final_key_rate_bps: float
    std_final_key_rate_bps: float
    ci95_final_key_rate_bps: float

    mean_system_training_key_rate_bps: float
    std_system_training_key_rate_bps: float
    ci95_system_training_key_rate_bps: float

    mean_system_final_key_rate_bps: float
    std_system_final_key_rate_bps: float
    ci95_system_final_key_rate_bps: float

    mean_raw_kdr: float
    std_raw_kdr: float
    ci95_raw_kdr: float

    mean_post_reconciliation_kdr: float
    std_post_reconciliation_kdr: float
    ci95_post_reconciliation_kdr: float

    mean_reciprocity: float
    std_reciprocity: float
    ci95_reciprocity: float

    mean_surface_dc_power: float
    std_surface_dc_power: float
    ci95_surface_dc_power: float

    mean_feasibility_rate: float
    std_feasibility_rate: float
    ci95_feasibility_rate: float


def _summary_statistics(
    values: list[float],
) -> tuple[float, float, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        raise ValueError(
            "statistics values cannot be empty"
        )

    mean = float(np.mean(array))

    if array.size < 2:
        return mean, 0.0, 0.0

    std = float(
        np.std(array, ddof=1)
    )

    ci95 = float(
        1.96
        * std
        / np.sqrt(array.size)
    )

    return mean, std, ci95


def passive_policy(
    environment: RobustFullSchemeEnvironment,
    state: np.ndarray,
) -> np.ndarray:
    del state
    return environment.passive_action()


def random_policy(
    environment: RobustFullSchemeEnvironment,
    state: np.ndarray,
) -> np.ndarray:
    del state
    return environment.random_action()


def heuristic_policy(
    environment: RobustFullSchemeEnvironment,
    state: np.ndarray,
) -> np.ndarray:
    del state
    return environment.heuristic_action()


def agent_policy(agent: TD3Agent) -> Policy:
    def policy(
        environment: RobustFullSchemeEnvironment,
        state: np.ndarray,
    ) -> np.ndarray:
        del environment
        return agent.select_action(state, explore=False)

    return policy


def evaluate_policy(
    environment: RobustFullSchemeEnvironment,
    policy: Policy,
    *,
    method: str,
    episodes: int,
    seed: int,
) -> ExperimentSummary:
    if episodes < 1:
        raise ValueError(
            "episodes must be positive"
        )

    episode_returns: list[float] = []

    training_rates: list[float] = []
    final_rates: list[float] = []

    system_training_rates: list[float] = []
    system_final_rates: list[float] = []

    raw_kdrs: list[float] = []
    post_kdrs: list[float] = []
    reciprocities: list[float] = []
    powers: list[float] = []
    feasibilities: list[float] = []

    for episode in range(episodes):
        state, _ = environment.reset(
            seed=seed + episode
        )

        episode_return = 0.0

        step_training_rates: list[float] = []
        step_final_rates: list[float] = []

        step_system_training_rates: list[
            float
        ] = []
        step_system_final_rates: list[
            float
        ] = []

        step_raw_kdrs: list[float] = []
        step_post_kdrs: list[float] = []
        step_reciprocities: list[float] = []
        step_powers: list[float] = []
        step_feasibilities: list[float] = []

        while True:
            action = policy(
                environment,
                state,
            )

            (
                state,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(action)

            episode_return += reward

            step_training_rates.append(
                float(
                    info[
                        "training_key_rate_bps"
                    ]
                )
            )
            step_final_rates.append(
                float(
                    info[
                        "final_key_rate_bps"
                    ]
                )
            )

            step_system_training_rates.append(
                float(
                    info[
                        "system_training_key_rate_bps"
                    ]
                )
            )
            step_system_final_rates.append(
                float(
                    info[
                        "system_final_key_rate_bps"
                    ]
                )
            )

            step_raw_kdrs.append(
                float(info["raw_kdr"])
            )
            step_post_kdrs.append(
                float(
                    info[
                        "post_reconciliation_kdr"
                    ]
                )
            )
            step_reciprocities.append(
                float(info["reciprocity"])
            )
            step_powers.append(
                float(
                    info["surface_dc_power"]
                )
            )
            step_feasibilities.append(
                float(
                    info["feasibility_rate"]
                )
            )

            if terminated or truncated:
                break

        episode_returns.append(
            float(episode_return)
        )

        training_rates.append(
            float(
                np.mean(step_training_rates)
            )
        )
        final_rates.append(
            float(
                np.mean(step_final_rates)
            )
        )

        system_training_rates.append(
            float(
                np.mean(
                    step_system_training_rates
                )
            )
        )
        system_final_rates.append(
            float(
                np.mean(
                    step_system_final_rates
                )
            )
        )

        raw_kdrs.append(
            float(np.mean(step_raw_kdrs))
        )
        post_kdrs.append(
            float(np.mean(step_post_kdrs))
        )
        reciprocities.append(
            float(
                np.mean(step_reciprocities)
            )
        )
        powers.append(
            float(np.mean(step_powers))
        )
        feasibilities.append(
            float(
                np.mean(step_feasibilities)
            )
        )

    return_mean, return_std, return_ci = (
        _summary_statistics(
            episode_returns
        )
    )

    training_mean, training_std, training_ci = (
        _summary_statistics(
            training_rates
        )
    )
    final_mean, final_std, final_ci = (
        _summary_statistics(
            final_rates
        )
    )

    (
        system_training_mean,
        system_training_std,
        system_training_ci,
    ) = _summary_statistics(
        system_training_rates
    )

    (
        system_final_mean,
        system_final_std,
        system_final_ci,
    ) = _summary_statistics(
        system_final_rates
    )

    raw_kdr_mean, raw_kdr_std, raw_kdr_ci = (
        _summary_statistics(raw_kdrs)
    )

    (
        post_kdr_mean,
        post_kdr_std,
        post_kdr_ci,
    ) = _summary_statistics(post_kdrs)

    (
        reciprocity_mean,
        reciprocity_std,
        reciprocity_ci,
    ) = _summary_statistics(
        reciprocities
    )

    power_mean, power_std, power_ci = (
        _summary_statistics(powers)
    )

    (
        feasibility_mean,
        feasibility_std,
        feasibility_ci,
    ) = _summary_statistics(
        feasibilities
    )

    return ExperimentSummary(
        method=method,
        episodes=episodes,

        mean_return=return_mean,
        std_return=return_std,
        ci95_return=return_ci,

        mean_training_key_rate_bps=(
            training_mean
        ),
        std_training_key_rate_bps=(
            training_std
        ),
        ci95_training_key_rate_bps=(
            training_ci
        ),

        mean_final_key_rate_bps=final_mean,
        std_final_key_rate_bps=final_std,
        ci95_final_key_rate_bps=final_ci,

        mean_system_training_key_rate_bps=(
            system_training_mean
        ),
        std_system_training_key_rate_bps=(
            system_training_std
        ),
        ci95_system_training_key_rate_bps=(
            system_training_ci
        ),

        mean_system_final_key_rate_bps=(
            system_final_mean
        ),
        std_system_final_key_rate_bps=(
            system_final_std
        ),
        ci95_system_final_key_rate_bps=(
            system_final_ci
        ),

        mean_raw_kdr=raw_kdr_mean,
        std_raw_kdr=raw_kdr_std,
        ci95_raw_kdr=raw_kdr_ci,

        mean_post_reconciliation_kdr=(
            post_kdr_mean
        ),
        std_post_reconciliation_kdr=(
            post_kdr_std
        ),
        ci95_post_reconciliation_kdr=(
            post_kdr_ci
        ),

        mean_reciprocity=reciprocity_mean,
        std_reciprocity=reciprocity_std,
        ci95_reciprocity=reciprocity_ci,

        mean_surface_dc_power=power_mean,
        std_surface_dc_power=power_std,
        ci95_surface_dc_power=power_ci,

        mean_feasibility_rate=(
            feasibility_mean
        ),
        std_feasibility_rate=(
            feasibility_std
        ),
        ci95_feasibility_rate=(
            feasibility_ci
        ),
    )


def write_summaries_csv(
    path: str | Path,
    summaries: list[ExperimentSummary],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        raise ValueError("summaries cannot be empty")
    rows = [asdict(summary) for summary in summaries]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_baseline_comparison(
    environment: RobustFullSchemeEnvironment,
    *,
    agent: TD3Agent | None,
    episodes: int,
    seed: int,
) -> list[ExperimentSummary]:
    """公平比较无源、部分有源启发式和鲁棒TD3。"""

    # 严格无源 STAR-RIS：
    # 所有 active_mask=False，因此：
    #   - 没有内部放大噪声；
    #   - 没有 active bias power；
    #   - 没有 active control power；
    #   - 所有单元增益固定为 1。
    passive_mask = np.zeros(
        environment.config.channel.num_elements,
        dtype=bool,
    )

    passive_environment = RobustFullSchemeEnvironment(
        environment.config,
        active_mask=passive_mask,
        seed=seed,
    )

    # 主无源baseline使用相位对齐，而不是故意使用零相位。
    methods: list[
        tuple[
            str,
            RobustFullSchemeEnvironment,
            Policy,
        ]
    ] = [
        (
            "passive_star_ris",
            passive_environment,
            heuristic_policy,
        ),
        (
            "random_partially_active",
            environment,
            random_policy,
        ),
        (
            "phase_aligned_partially_active",
            environment,
            heuristic_policy,
        ),
    ]

    if agent is not None:
        methods.append(
            (
                "robust_td3",
                environment,
                agent_policy(agent),
            )
        )

    # 所有方法使用相同 episode seed：
    # seed, seed+1, seed+2, ...
    #
    # 这样信道/domain realization具有更好的可比性。
    return [
        evaluate_policy(
            method_environment,
            policy,
            method=name,
            episodes=episodes,
            seed=seed,
        )
        for (
            name,
            method_environment,
            policy,
        ) in methods
    ]


def run_active_ratio_sweep(
    base_config: EnvironmentConfig,
    policy_factory: Callable[[RobustFullSchemeEnvironment], Policy],
    active_ratios: list[float],
    *,
    episodes: int,
    seed: int,
) -> list[dict[str, float | str | int]]:
    records: list[dict[str, float | str | int]] = []
    for index, ratio in enumerate(active_ratios):
        channel = replace(base_config.channel, active_ratio=float(ratio))
        config = replace(base_config, channel=channel)
        environment = RobustFullSchemeEnvironment(config, seed=seed + index)
        policy = policy_factory(environment)
        summary = evaluate_policy(
            environment,
            policy,
            method="active_ratio_sweep",
            episodes=episodes,
            seed=seed + 1000 * index,
        )
        record = asdict(summary)
        record["active_ratio"] = float(ratio)
        records.append(record)
    return records


def write_records_csv(
    path: str | Path,
    records: list[dict[str, float | str | int]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("records cannot be empty")
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_named_config_sweep(
    cases: list[tuple[str, float | str, EnvironmentConfig]],
    policy_factory: Callable[[RobustFullSchemeEnvironment], Policy],
    *,
    parameter_name: str,
    episodes: int,
    seed: int,
) -> list[dict[str, float | str | int]]:
    """对保持状态/动作维度不变的配置项执行统一扫描。"""
    records: list[dict[str, float | str | int]] = []
    for index, (label, value, config) in enumerate(cases):
        environment = RobustFullSchemeEnvironment(config, seed=seed + index)
        policy = policy_factory(environment)
        summary = evaluate_policy(
            environment,
            policy,
            method=label,
            episodes=episodes,
            seed=seed + 1000 * index,
        )
        record = asdict(summary)
        record[parameter_name] = value
        records.append(record)
    return records
