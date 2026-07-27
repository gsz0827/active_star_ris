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
    mean_training_key_rate_bps: float
    mean_final_key_rate_bps: float
    mean_raw_kdr: float
    mean_post_reconciliation_kdr: float
    mean_reciprocity: float
    mean_surface_dc_power: float
    mean_feasibility_rate: float


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
    episode_returns: list[float] = []
    training_rates: list[float] = []
    final_rates: list[float] = []
    raw_kdrs: list[float] = []
    post_kdrs: list[float] = []
    reciprocities: list[float] = []
    powers: list[float] = []
    feasibilities: list[float] = []

    for episode in range(episodes):
        state, _ = environment.reset(seed=seed + episode)
        episode_return = 0.0
        while True:
            action = policy(environment, state)
            state, reward, terminated, truncated, info = environment.step(action)
            episode_return += reward
            training_rates.append(float(info["training_key_rate_bps"]))
            final_rates.append(float(info["final_key_rate_bps"]))
            raw_kdrs.append(float(info["raw_kdr"]))
            post_kdrs.append(float(info["post_reconciliation_kdr"]))
            reciprocities.append(float(info["reciprocity"]))
            powers.append(float(info["surface_dc_power"]))
            feasibilities.append(float(info["feasibility_rate"]))
            if terminated or truncated:
                break
        episode_returns.append(episode_return)

    return ExperimentSummary(
        method=method,
        episodes=episodes,
        mean_return=float(np.mean(episode_returns)),
        std_return=(
            float(np.std(episode_returns, ddof=1))
            if len(episode_returns) > 1
            else 0.0
        ),
        mean_training_key_rate_bps=float(np.mean(training_rates)),
        mean_final_key_rate_bps=float(np.mean(final_rates)),
        mean_raw_kdr=float(np.mean(raw_kdrs)),
        mean_post_reconciliation_kdr=float(np.mean(post_kdrs)),
        mean_reciprocity=float(np.mean(reciprocities)),
        mean_surface_dc_power=float(np.mean(powers)),
        mean_feasibility_rate=float(np.mean(feasibilities)),
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
