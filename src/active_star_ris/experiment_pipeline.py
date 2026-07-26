from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .channels import ChannelConfig
from .joint_objective import JointObjectiveConfig
from .rl_environment import (
    DomainRandomizationConfig,
    RobustActiveStarRISEnv,
    RobustEnvironmentConfig,
)
from .td3_agent import TD3Agent, TD3Config
from .td3_training import (
    TD3TrainingConfig,
    TD3TrainingHistory,
    train_td3,
)


ABLATIONS = (
    "full_model",
    "no_internal_noise",
    "perfect_csi",
    "no_hardware_mismatch",
    "no_amplitude_phase_coupling",
    "no_cvar",
    "no_surface_power_penalty",
)


def _filtered_kwargs(
    dataclass_type: type,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = {
        item.name
        for item in fields(dataclass_type)
    }

    return {
        key: value
        for key, value in values.items()
        if key in allowed
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item)
            for item in value
        ]

    return value


def _write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = list(rows)

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    field_names: list[str] = []

    for row in rows:
        for key in row:
            if key not in field_names:
                field_names.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=field_names,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: _jsonable(
                        row.get(key, "")
                    )
                    for key in field_names
                }
            )


def build_environment_config(
    raw_config: Mapping[str, Any],
    *,
    num_active_elements: int,
    ablation: str = "full_model",
    quick: bool = False,
) -> RobustEnvironmentConfig:
    if ablation not in ABLATIONS:
        raise ValueError(
            f"unknown ablation: {ablation}"
        )

    system = raw_config["system"]
    channels = raw_config["channel"]
    surface_power = raw_config["surface_power"]
    action_mapping = raw_config["action_mapping"]
    rl = raw_config["rl_environment"]

    objective = JointObjectiveConfig(
        **_filtered_kwargs(
            JointObjectiveConfig,
            raw_config["joint_objective"],
        )
    )

    domain = DomainRandomizationConfig(
        **_filtered_kwargs(
            DomainRandomizationConfig,
            rl["domain_randomization"],
        )
    )

    if ablation == "no_internal_noise":
        domain = replace(
            domain,
            ris_internal_noise_variance_min=0.0,
            ris_internal_noise_variance_max=0.0,
        )

    elif ablation == "perfect_csi":
        # 使用极低有限NMSE近似理想CSI，
        # 避免-inf进入数值运算。
        domain = replace(
            domain,
            nmse_db_min=-100.0,
            nmse_db_max=-100.0,
        )

    elif ablation == "no_hardware_mismatch":
        domain = replace(
            domain,
            static_gain_error_std_db_min=0.0,
            static_gain_error_std_db_max=0.0,
            directional_gain_error_std_db_min=0.0,
            directional_gain_error_std_db_max=0.0,
            static_phase_error_std_rad_min=0.0,
            static_phase_error_std_rad_max=0.0,
            directional_phase_error_std_rad_min=0.0,
            directional_phase_error_std_rad_max=0.0,
            fast_phase_jitter_std_rad_min=0.0,
            fast_phase_jitter_std_rad_max=0.0,
            transmission_amplitude_phase_coupling_rad_per_db_min=0.0,
            transmission_amplitude_phase_coupling_rad_per_db_max=0.0,
            reflection_amplitude_phase_coupling_rad_per_db_min=0.0,
            reflection_amplitude_phase_coupling_rad_per_db_max=0.0,
        )

    elif ablation == "no_amplitude_phase_coupling":
        domain = replace(
            domain,
            transmission_amplitude_phase_coupling_rad_per_db_min=0.0,
            transmission_amplitude_phase_coupling_rad_per_db_max=0.0,
            reflection_amplitude_phase_coupling_rad_per_db_min=0.0,
            reflection_amplitude_phase_coupling_rad_per_db_max=0.0,
        )

    elif ablation == "no_surface_power_penalty":
        objective = replace(
            objective,
            surface_power_weight=0.0,
        )

    probing_samples = int(
        rl["probing_samples_per_step"]
    )
    robust_samples = int(
        rl["robust_objective_samples"]
    )
    max_episode_steps = int(
        rl["max_episode_steps"]
    )

    if quick:
        probing_samples = min(
            probing_samples,
            32,
        )
        robust_samples = min(
            robust_samples,
            2,
        )
        max_episode_steps = min(
            max_episode_steps,
            16,
        )

    robust_mean_weight = float(
        rl["robust_mean_weight"]
    )
    robust_cvar_weight = float(
        rl["robust_cvar_weight"]
    )

    if ablation == "no_cvar":
        robust_mean_weight = 1.0
        robust_cvar_weight = 0.0

    config = RobustEnvironmentConfig(
        num_elements=int(
            system["num_elements"]
        ),
        num_active_elements=int(
            num_active_elements
        ),
        max_episode_steps=max_episode_steps,
        probing_samples_per_step=probing_samples,

        channel_temporal_correlation=float(
            rl["channel_temporal_correlation"]
        ),
        probing_sample_correlation=float(
            rl["probing_sample_correlation"]
        ),

        controller_to_ris=ChannelConfig(
            **channels["alice_to_ris"]
        ),
        ris_to_transmission_user=ChannelConfig(
            **channels[
                "ris_to_transmission_user"
            ]
        ),
        ris_to_reflection_user=ChannelConfig(
            **channels[
                "ris_to_reflection_user"
            ]
        ),
        direct_transmission=ChannelConfig(
            **channels["direct_transmission"]
        ),
        direct_reflection=ChannelConfig(
            **channels["direct_reflection"]
        ),

        controller_pilot_power=float(
            surface_power[
                "controller_pilot_power"
            ]
        ),
        transmission_user_pilot_power=float(
            surface_power[
                "transmission_user_pilot_power"
            ]
        ),
        reflection_user_pilot_power=float(
            surface_power[
                "reflection_user_pilot_power"
            ]
        ),

        amplifier_efficiency=float(
            surface_power[
                "amplifier_efficiency"
            ]
        ),
        controller_static_power=float(
            surface_power[
                "controller_static_power"
            ]
        ),
        passive_element_control_power=float(
            surface_power[
                "passive_element_control_power"
            ]
        ),
        active_element_control_power=float(
            surface_power[
                "active_element_control_power"
            ]
        ),
        active_element_bias_power=float(
            surface_power[
                "active_element_bias_power"
            ]
        ),
        switching_network_static_power=float(
            surface_power[
                "switching_network_static_power"
            ]
        ),

        transmission_weight=float(
            system["transmission_weight"]
        ),
        reflection_weight=float(
            system["reflection_weight"]
        ),

        maximum_active_amplitude=float(
            system["maximum_active_amplitude"]
        ),
        beta_min=float(
            action_mapping["beta_min"]
        ),
        beta_max=float(
            action_mapping["beta_max"]
        ),
        robust_margin_multiplier=float(
            surface_power[
                "robust_margin_multiplier"
            ]
        ),
        allow_active_bypass=bool(
            action_mapping[
                "allow_active_bypass"
            ]
        ),

        robust_objective_samples=robust_samples,
        robust_cvar_alpha=float(
            rl["robust_cvar_alpha"]
        ),
        robust_mean_weight=robust_mean_weight,
        robust_cvar_weight=robust_cvar_weight,

        domain_randomization=domain,
        objective=objective,

        state_clip=float(
            rl["state_clip"]
        ),
        include_impairment_context=bool(
            rl[
                "include_impairment_context"
            ]
        ),
        include_previous_metrics=bool(
            rl[
                "include_previous_metrics"
            ]
        ),
    )

    config.validate()
    return config


def build_td3_config(
    raw_config: Mapping[str, Any],
) -> TD3Config:
    values = dict(
        raw_config["td3"]
    )

    values["hidden_dimensions"] = tuple(
        int(value)
        for value in values[
            "hidden_dimensions"
        ]
    )

    config = TD3Config(
        **_filtered_kwargs(
            TD3Config,
            values,
        )
    )
    config.validate()
    return config


def build_training_config(
    raw_config: Mapping[str, Any],
    *,
    seed: int,
    quick: bool = False,
    total_steps_override: int | None = None,
) -> TD3TrainingConfig:
    values = dict(
        raw_config["td3_training"]
    )
    values["seed"] = int(seed)

    if quick:
        values.update(
            {
                "total_environment_steps": 2_000,
                "replay_capacity": 5_000,
                "random_action_steps": 200,
                "learning_starts": 200,
                "batch_size": 64,
                "gradient_steps_per_environment_step": 1,
                "evaluation_interval": 500,
                "evaluation_episodes": 2,
            }
        )

    if total_steps_override is not None:
        total_steps = int(
            total_steps_override
        )

        if total_steps <= 0:
            raise ValueError(
                "total_steps_override must be positive"
            )

        warmup = max(
            1,
            total_steps // 4,
        )
        batch_size = max(
            2,
            min(
                int(
                    values.get(
                        "batch_size",
                        256,
                    )
                ),
                warmup,
            ),
        )

        values.update(
            {
                "total_environment_steps": total_steps,
                "random_action_steps": min(
                    int(
                        values.get(
                            "random_action_steps",
                            warmup,
                        )
                    ),
                    warmup,
                ),
                "learning_starts": min(
                    int(
                        values.get(
                            "learning_starts",
                            warmup,
                        )
                    ),
                    warmup,
                ),
                "batch_size": batch_size,
                "replay_capacity": max(
                    int(
                        values.get(
                            "replay_capacity",
                            total_steps * 2,
                        )
                    ),
                    batch_size,
                    total_steps * 2,
                ),
                "evaluation_interval": max(
                    1,
                    total_steps // 2,
                ),
                "evaluation_episodes": 1,
            }
        )

    config = TD3TrainingConfig(
        **_filtered_kwargs(
            TD3TrainingConfig,
            values,
        )
    )
    config.validate()
    return config


def _save_training_history(
    history: TD3TrainingHistory,
    output_directory: Path,
) -> None:
    csv_directory = (
        output_directory
        / "csv"
    )

    episode_rows = [
        {
            "episode": index + 1,
            "environment_step": step,
            "episode_return": episode_return,
            "episode_length": episode_length,
        }
        for (
            index,
            (
                step,
                episode_return,
                episode_length,
            ),
        ) in enumerate(
            zip(
                history.environment_steps,
                history.episode_returns,
                history.episode_lengths,
                strict=True,
            )
        )
    ]

    critic_rows = [
        {
            "update_index": update_index,
            "critic_loss": loss,
        }
        for update_index, loss in zip(
            history.critic_update_indices,
            history.critic_losses,
            strict=True,
        )
    ]

    actor_rows = [
        {
            "update_index": update_index,
            "actor_loss": loss,
        }
        for update_index, loss in zip(
            history.actor_update_indices,
            history.actor_losses,
            strict=True,
        )
    ]

    evaluation_rows = [
        {
            "environment_step": step,
            "mean_return": mean_return,
            "key_rate_bits_per_sample": kgr_sample,
            "key_rate_bits_per_second": kgr_second,
            "raw_key_disagreement_rate": kdr,
            "observation_reciprocity": reciprocity,
            "surface_power_w": power,
            "robust_reward": robust_reward,
            "cvar_reward": cvar_reward,
            "worst_sample_reward": worst_reward,
            "robust_feasibility_rate": feasibility,
            "effective_active_elements": active_elements,
            "power_violation": power_violation,
        }
        for (
            step,
            mean_return,
            kgr_sample,
            kgr_second,
            kdr,
            reciprocity,
            power,
            robust_reward,
            cvar_reward,
            worst_reward,
            feasibility,
            active_elements,
            power_violation,
        ) in zip(
            history.evaluation_steps,
            history.evaluation_returns,
            history.evaluation_key_rates,
            history.evaluation_key_rates_bits_per_second,
            history.evaluation_key_disagreement_rates,
            history.evaluation_reciprocities,
            history.evaluation_surface_powers,
            history.evaluation_robust_rewards,
            history.evaluation_cvar_rewards,
            history.evaluation_worst_sample_rewards,
            history.evaluation_feasibility_rates,
            history.evaluation_effective_active_elements,
            history.evaluation_power_violations,
            strict=True,
        )
    ]

    _write_csv(
        csv_directory / "episodes.csv",
        episode_rows,
    )
    _write_csv(
        csv_directory / "critic_losses.csv",
        critic_rows,
    )
    _write_csv(
        csv_directory / "actor_losses.csv",
        actor_rows,
    )
    _write_csv(
        csv_directory / "periodic_evaluations.csv",
        evaluation_rows,
    )


def _plot_line(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    if len(x_values) == 0:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 5)
    )
    plt.plot(
        x_values,
        y_values,
        linewidth=1.5,
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def _save_convergence_figures(
    history: TD3TrainingHistory,
    output_directory: Path,
) -> None:
    figure_directory = (
        output_directory
        / "figures"
    )

    _plot_line(
        history.environment_steps,
        history.episode_returns,
        x_label="Environment step",
        y_label="Episode return",
        title="TD3 training return",
        output_path=(
            figure_directory
            / "episode_return.png"
        ),
    )

    _plot_line(
        history.critic_update_indices,
        history.critic_losses,
        x_label="Gradient update",
        y_label="Critic loss",
        title="TD3 critic convergence",
        output_path=(
            figure_directory
            / "critic_loss.png"
        ),
    )

    _plot_line(
        history.actor_update_indices,
        history.actor_losses,
        x_label="Gradient update",
        y_label="Actor loss",
        title="TD3 actor convergence",
        output_path=(
            figure_directory
            / "actor_loss.png"
        ),
    )

    evaluation_series = (
        (
            history.evaluation_returns,
            "Mean evaluation return",
            "evaluation_return.png",
        ),
        (
            history
            .evaluation_key_rates_bits_per_second,
            "Key generation rate (bit/s)",
            "evaluation_kgr_bps.png",
        ),
        (
            history
            .evaluation_key_disagreement_rates,
            "Raw KDR",
            "evaluation_kdr.png",
        ),
        (
            history.evaluation_reciprocities,
            "Observation reciprocity",
            "evaluation_reciprocity.png",
        ),
        (
            history.evaluation_surface_powers,
            "Surface power (W)",
            "evaluation_surface_power.png",
        ),
        (
            history.evaluation_cvar_rewards,
            "CVaR reward",
            "evaluation_cvar_reward.png",
        ),
    )

    for values, label, filename in evaluation_series:
        _plot_line(
            history.evaluation_steps,
            values,
            x_label="Environment step",
            y_label=label,
            title=label,
            output_path=(
                figure_directory
                / filename
            ),
        )


def evaluate_policy_detailed(
    environment: RobustActiveStarRISEnv,
    agent: TD3Agent,
    *,
    episodes: int,
    evaluation_seed: int,
    scenario: str,
    training_seed: int,
    output_directory: Path,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError(
            "episodes must be positive"
        )

    step_rows: list[
        dict[str, Any]
    ] = []
    episode_rows: list[
        dict[str, Any]
    ] = []

    metric_mapping = {
        "weighted_key_rate_bits_per_sample": (
            "key_rate_bits_per_sample"
        ),
        "weighted_key_rate_bits_per_second": (
            "key_rate_bits_per_second"
        ),
        "raw_key_disagreement_rate": (
            "raw_key_disagreement_rate"
        ),
        "observation_reciprocity": (
            "observation_reciprocity"
        ),
        "total_surface_power": (
            "surface_power_w"
        ),
        "robust_reward": (
            "robust_reward"
        ),
        "mean_sample_reward": (
            "mean_sample_reward"
        ),
        "cvar_reward": (
            "cvar_reward"
        ),
        "worst_sample_reward": (
            "worst_sample_reward"
        ),
        "maximum_output_power": (
            "maximum_output_power"
        ),
        "power_violation": (
            "power_violation"
        ),
        "effective_active_elements": (
            "effective_active_elements"
        ),
        "projection_scale": (
            "projection_scale"
        ),
    }

    for episode in range(episodes):
        state, _ = environment.reset(
            seed=(
                evaluation_seed
                + episode
            )
        )

        episode_return = 0.0
        episode_metrics: dict[
            str,
            list[float],
        ] = {
            target: []
            for target in metric_mapping.values()
        }
        feasibility_values: list[
            float
        ] = []

        step_index = 0

        while True:
            action = agent.select_action(
                state,
                explore=False,
            )

            (
                state,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(action)

            step_index += 1
            episode_return += float(reward)

            row: dict[str, Any] = {
                "scenario": scenario,
                "training_seed": training_seed,
                "episode": episode,
                "step": step_index,
                "reward": float(reward),
            }

            for source, target in metric_mapping.items():
                value = float(
                    info[source]
                )
                row[target] = value
                episode_metrics[
                    target
                ].append(value)

            feasible = float(
                bool(
                    info[
                        "robustly_feasible"
                    ]
                )
            )
            row[
                "robustly_feasible"
            ] = feasible
            feasibility_values.append(
                feasible
            )

            step_rows.append(row)

            if terminated or truncated:
                break

        episode_row: dict[str, Any] = {
            "scenario": scenario,
            "training_seed": training_seed,
            "episode": episode,
            "episode_return": float(
                episode_return
            ),
            "episode_length": int(
                step_index
            ),
            "robust_feasibility_rate": float(
                np.mean(
                    feasibility_values
                )
            ),
        }

        for metric_name, values in episode_metrics.items():
            episode_row[
                f"mean_{metric_name}"
            ] = float(
                np.mean(values)
            )

        episode_rows.append(
            episode_row
        )

    csv_directory = (
        output_directory
        / "csv"
    )

    _write_csv(
        csv_directory
        / "final_evaluation_steps.csv",
        step_rows,
    )
    _write_csv(
        csv_directory
        / "final_evaluation_episodes.csv",
        episode_rows,
    )

    summary: dict[str, Any] = {
        "scenario": scenario,
        "training_seed": int(
            training_seed
        ),
        "evaluation_episodes": int(
            episodes
        ),
    }

    numeric_keys = [
        key
        for key in episode_rows[0]
        if key.startswith("mean_")
        or key in {
            "episode_return",
            "episode_length",
            "robust_feasibility_rate",
        }
    ]

    for key in numeric_keys:
        values = np.asarray(
            [
                float(row[key])
                for row in episode_rows
            ],
            dtype=np.float64,
        )

        summary[
            f"{key}_mean"
        ] = float(
            np.mean(values)
        )
        summary[
            f"{key}_std"
        ] = float(
            np.std(
                values,
                ddof=(
                    1
                    if values.size > 1
                    else 0
                ),
            )
        )

    _write_csv(
        csv_directory
        / "final_evaluation_summary.csv",
        [summary],
    )
    _write_json(
        output_directory
        / "final_evaluation_summary.json",
        summary,
    )

    return summary


def run_single_experiment(
    raw_config: Mapping[str, Any],
    *,
    output_directory: Path,
    scenario: str,
    num_active_elements: int,
    ablation: str,
    seed: int,
    evaluation_seed: int,
    final_evaluation_episodes: int,
    device: str | None,
    quick: bool,
    total_steps_override: int | None = None,
) -> dict[str, Any]:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 固定验证集用于周期模型选择。
    validation_seed = int(
        evaluation_seed
    )

    # 独立测试集只用于训练完成后的最终详细评价，
    # 避免验证场景和最终测试场景重合。
    final_test_seed = int(
        evaluation_seed + 1_000_000
    )

    environment_config = (
        build_environment_config(
            raw_config,
            num_active_elements=(
                num_active_elements
            ),
            ablation=ablation,
            quick=quick,
        )
    )

    td3_config = build_td3_config(
        raw_config
    )

    training_config = (
        build_training_config(
            raw_config,
            seed=seed,
            quick=quick,
            total_steps_override=(
                total_steps_override
            ),
        )
    )

    training_environment = (
        RobustActiveStarRISEnv(
            environment_config,
            seed=seed,
        )
    )

    evaluation_environment = (
        RobustActiveStarRISEnv(
            environment_config,
            seed=validation_seed,
        )
    )

    agent = TD3Agent(
        training_environment.state_dim,
        training_environment.action_dim,
        td3_config,
        device=device,
        seed=seed,
    )

    checkpoint_directory = (
        output_directory
        / "checkpoints"
    )
    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_evaluation_return = (
        -np.inf
    )
    observed_evaluations = 0

    def progress_callback(
        step: int,
        history: TD3TrainingHistory,
    ) -> None:
        nonlocal best_evaluation_return
        nonlocal observed_evaluations

        evaluation_count = len(
            history.evaluation_returns
        )

        if (
            evaluation_count
            <= observed_evaluations
        ):
            return

        observed_evaluations = (
            evaluation_count
        )
        evaluation_return = float(
            history
            .evaluation_returns[-1]
        )

        print(
            f"[{scenario} | seed={seed}] "
            f"step={step} "
            f"evaluation_return="
            f"{evaluation_return:.6f}"
        )

        if (
            evaluation_return
            > best_evaluation_return
        ):
            best_evaluation_return = (
                evaluation_return
            )

            agent.save_checkpoint(
                checkpoint_directory
                / "best.pt",
                extra={
                    "scenario": scenario,
                    "ablation": ablation,
                    "seed": int(seed),
                    "environment_step": int(
                        step
                    ),
                    "evaluation_return": (
                        evaluation_return
                    ),
                },
            )

    result = train_td3(
        training_environment,
        agent,
        training_config,
        evaluation_environment=(
            evaluation_environment
        ),
        evaluation_seed=validation_seed,
        progress_callback=(
            progress_callback
        ),
    )

    final_checkpoint = (
        agent.save_checkpoint(
            checkpoint_directory
            / "final.pt",
            extra={
                "scenario": scenario,
                "ablation": ablation,
                "seed": int(seed),
                "total_environment_steps": (
                    training_config
                    .total_environment_steps
                ),
            },
        )
    )

    best_checkpoint = (
        checkpoint_directory
        / "best.pt"
    )

    if not best_checkpoint.exists():
        agent.save_checkpoint(
            best_checkpoint,
            extra={
                "scenario": scenario,
                "ablation": ablation,
                "seed": int(seed),
                "fallback_from_final": True,
            },
        )

    _save_training_history(
        result.history,
        output_directory,
    )
    _save_convergence_figures(
        result.history,
        output_directory,
    )

    # 正式最终评价使用验证集上表现最好的检查点，
    # 而不是训练最后一步的模型。
    best_checkpoint_extra = (
        agent.load_checkpoint(
            best_checkpoint,
            load_optimizers=False,
        )
    )

    # 使用独立测试随机种子，不与周期验证集重合。
    detailed_environment = (
        RobustActiveStarRISEnv(
            environment_config,
            seed=final_test_seed,
        )
    )

    summary = evaluate_policy_detailed(
        detailed_environment,
        agent,
        episodes=(
            final_evaluation_episodes
        ),
        evaluation_seed=final_test_seed,
        scenario=scenario,
        training_seed=seed,
        output_directory=(
            output_directory
        ),
    )

    summary.update(
        {
            "ablation": ablation,
            "num_elements": (
                environment_config
                .num_elements
            ),
            "requested_active_elements": (
                num_active_elements
            ),
            "state_dimension": (
                training_environment
                .state_dim
            ),
            "action_dimension": (
                training_environment
                .action_dim
            ),
            "training_steps": (
                training_config
                .total_environment_steps
            ),
            "best_evaluation_return": (
                float(
                    best_evaluation_return
                )
                if np.isfinite(
                    best_evaluation_return
                )
                else None
            ),
            "best_checkpoint": str(
                best_checkpoint
            ),
            "final_checkpoint": str(
                final_checkpoint
            ),
            "validation_seed": int(
                validation_seed
            ),
            "final_test_seed": int(
                final_test_seed
            ),
            "evaluated_checkpoint": str(
                best_checkpoint
            ),
            "evaluated_checkpoint_extra": (
                best_checkpoint_extra
            ),
        }
    )

    _write_json(
        output_directory
        / "run_metadata.json",
        {
            "summary": summary,
            "environment_config": asdict(
                environment_config
            ),
            "td3_config": asdict(
                td3_config
            ),
            "training_config": asdict(
                training_config
            ),
        },
    )

    return summary


def _aggregate_summaries(
    summaries: Sequence[
        Mapping[str, Any]
    ],
    *,
    grouping_key: str,
    output_path: Path,
) -> list[dict[str, Any]]:
    groups: dict[
        str,
        list[Mapping[str, Any]],
    ] = {}

    for summary in summaries:
        label = str(
            summary[grouping_key]
        )
        groups.setdefault(
            label,
            [],
        ).append(summary)

    metric_keys = (
        "episode_return_mean",
        "mean_key_rate_bits_per_second_mean",
        "mean_raw_key_disagreement_rate_mean",
        "mean_observation_reciprocity_mean",
        "mean_surface_power_w_mean",
        "mean_cvar_reward_mean",
        "robust_feasibility_rate_mean",
        "mean_effective_active_elements_mean",
        "mean_power_violation_mean",
    )

    rows: list[
        dict[str, Any]
    ] = []

    for label, group in groups.items():
        row: dict[str, Any] = {
            grouping_key: label,
            "runs": len(group),
        }

        for metric in metric_keys:
            values = np.asarray(
                [
                    float(item[metric])
                    for item in group
                ],
                dtype=np.float64,
            )

            row[
                f"{metric}_across_seed_mean"
            ] = float(
                np.mean(values)
            )
            row[
                f"{metric}_across_seed_std"
            ] = float(
                np.std(
                    values,
                    ddof=(
                        1
                        if values.size > 1
                        else 0
                    ),
                )
            )

        rows.append(row)

    _write_csv(
        output_path,
        rows,
    )
    return rows


def _plot_comparison(
    rows: Sequence[
        Mapping[str, Any]
    ],
    *,
    grouping_key: str,
    metric_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    if not rows:
        return

    labels = [
        str(row[grouping_key])
        for row in rows
    ]
    means = [
        float(
            row[
                f"{metric_key}"
                "_across_seed_mean"
            ]
        )
        for row in rows
    ]
    errors = [
        float(
            row[
                f"{metric_key}"
                "_across_seed_std"
            ]
        )
        for row in rows
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    positions = np.arange(
        len(labels)
    )

    plt.figure(
        figsize=(9, 5)
    )
    plt.bar(
        positions,
        means,
        yerr=errors,
        capsize=4,
    )
    plt.xticks(
        positions,
        labels,
        rotation=20,
        ha="right",
    )
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(
        axis="y",
        alpha=0.3,
    )
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def _save_comparison_figures(
    rows: Sequence[
        Mapping[str, Any]
    ],
    *,
    grouping_key: str,
    output_directory: Path,
    prefix: str,
) -> None:
    metrics = (
        (
            "mean_key_rate_bits_per_second_mean",
            "Key generation rate (bit/s)",
            "kgr_bps",
        ),
        (
            "mean_raw_key_disagreement_rate_mean",
            "Raw KDR",
            "kdr",
        ),
        (
            "mean_observation_reciprocity_mean",
            "Observation reciprocity",
            "reciprocity",
        ),
        (
            "mean_surface_power_w_mean",
            "Surface power (W)",
            "surface_power",
        ),
        (
            "episode_return_mean",
            "Episode return",
            "episode_return",
        ),
    )

    for metric, label, filename in metrics:
        _plot_comparison(
            rows,
            grouping_key=grouping_key,
            metric_key=metric,
            y_label=label,
            title=f"{prefix}: {label}",
            output_path=(
                output_directory
                / f"{prefix}_{filename}.png"
            ),
        )


def run_full_experiment_suite(
    raw_config: Mapping[str, Any],
    *,
    output_directory: Path,
    seeds: Sequence[int],
    evaluation_seed: int,
    final_evaluation_episodes: int,
    device: str | None,
    quick: bool = False,
    total_steps_override: int | None = None,
    run_baselines: bool = True,
    run_ablations: bool = True,
) -> dict[str, Any]:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_elements = int(
        raw_config["system"][
            "num_elements"
        ]
    )

    experiment_config = (
        raw_config.get(
            "experiment_suite",
            {},
        )
    )

    partial_active_elements = int(
        experiment_config.get(
            "partial_active_elements",
            raw_config["system"][
                "num_active_elements"
            ],
        )
    )

    baseline_summaries: list[
        dict[str, Any]
    ] = []

    ablation_summaries: list[
        dict[str, Any]
    ] = []

    baseline_definitions = (
        ("passive", 0),
        (
            "partially_active",
            partial_active_elements,
        ),
        (
            "fully_active",
            num_elements,
        ),
    )

    if run_baselines:
        for (
            scenario,
            active_elements,
        ) in baseline_definitions:
            for seed in seeds:
                run_directory = (
                    output_directory
                    / "baselines"
                    / scenario
                    / f"seed_{seed}"
                )

                summary = (
                    run_single_experiment(
                        raw_config,
                        output_directory=(
                            run_directory
                        ),
                        scenario=scenario,
                        num_active_elements=(
                            active_elements
                        ),
                        ablation="full_model",
                        seed=int(seed),
                        evaluation_seed=(
                            evaluation_seed
                        ),
                        final_evaluation_episodes=(
                            final_evaluation_episodes
                        ),
                        device=device,
                        quick=quick,
                        total_steps_override=(
                            total_steps_override
                        ),
                    )
                )

                baseline_summaries.append(
                    summary
                )

                if (
                    scenario
                    == "partially_active"
                ):
                    full_summary = dict(
                        summary
                    )
                    full_summary[
                        "ablation"
                    ] = "full_model"
                    ablation_summaries.append(
                        full_summary
                    )

    if run_ablations:
        for ablation in ABLATIONS:
            if ablation == "full_model":
                # 已复用部分有源基线，避免重复训练。
                continue

            for seed in seeds:
                scenario = (
                    "partially_active"
                    f"__{ablation}"
                )

                run_directory = (
                    output_directory
                    / "ablations"
                    / ablation
                    / f"seed_{seed}"
                )

                summary = (
                    run_single_experiment(
                        raw_config,
                        output_directory=(
                            run_directory
                        ),
                        scenario=scenario,
                        num_active_elements=(
                            partial_active_elements
                        ),
                        ablation=ablation,
                        seed=int(seed),
                        evaluation_seed=(
                            evaluation_seed
                        ),
                        final_evaluation_episodes=(
                            final_evaluation_episodes
                        ),
                        device=device,
                        quick=quick,
                        total_steps_override=(
                            total_steps_override
                        ),
                    )
                )

                ablation_summaries.append(
                    summary
                )

    aggregate_directory = (
        output_directory
        / "aggregate"
    )
    aggregate_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_rows: list[
        dict[str, Any]
    ] = []

    if baseline_summaries:
        baseline_rows = (
            _aggregate_summaries(
                baseline_summaries,
                grouping_key="scenario",
                output_path=(
                    aggregate_directory
                    / "baseline_summary.csv"
                ),
            )
        )

        _save_comparison_figures(
            baseline_rows,
            grouping_key="scenario",
            output_directory=(
                aggregate_directory
            ),
            prefix="baseline",
        )

    ablation_rows: list[
        dict[str, Any]
    ] = []

    if ablation_summaries:
        ablation_rows = (
            _aggregate_summaries(
                ablation_summaries,
                grouping_key="ablation",
                output_path=(
                    aggregate_directory
                    / "ablation_summary.csv"
                ),
            )
        )

        _save_comparison_figures(
            ablation_rows,
            grouping_key="ablation",
            output_directory=(
                aggregate_directory
            ),
            prefix="ablation",
        )

    manifest = {
        "output_directory": str(
            output_directory
        ),
        "seeds": [
            int(seed)
            for seed in seeds
        ],
        "evaluation_seed": int(
            evaluation_seed
        ),
        "final_evaluation_episodes": int(
            final_evaluation_episodes
        ),
        "quick": bool(quick),
        "total_steps_override": (
            total_steps_override
        ),
        "baseline_runs": len(
            baseline_summaries
        ),
        "ablation_runs": len(
            ablation_summaries
        ),
        "baseline_summary": (
            baseline_rows
        ),
        "ablation_summary": (
            ablation_rows
        ),
    }

    _write_json(
        output_directory
        / "experiment_manifest.json",
        manifest,
    )

    return manifest