from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from active_star_ris.full_scheme_v2 import (
    RobustFullSchemeEnvironment,
    TD3Agent,
    TD3Config,
    load_environment_config,
)
from active_star_ris.full_scheme_v2.experiments import (
    agent_policy,
    heuristic_policy,
    run_active_ratio_sweep,
    run_baseline_comparison,
    run_named_config_sweep,
    write_records_csv,
    write_summaries_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行完整方案基线和参数扫描。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full_scheme_v2.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/full_scheme_v2/experiments"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_environment_config(args.config)

    evaluation_samples = 128 if args.quick else 2048
    objective_samples = 4 if args.quick else 128
    cvar_alpha = 1.0 if args.quick else 0.10
    episodes = 1 if args.quick else args.episodes

    if args.quick:
        config = replace(
            config,
            max_episode_steps=2,
            key_generation=replace(
                config.key_generation,
                privacy_margin_bits=16,
                maximum_final_key_bits=64,
            ),
        )

    environment = RobustFullSchemeEnvironment(config, seed=args.seed)
    evaluation_environment = environment.evaluation_copy(
        samples_per_step=evaluation_samples,
        objective_samples=objective_samples,
        cvar_alpha=cvar_alpha,
        seed=args.seed + 10_000,
    )

    agent = None
    if args.checkpoint is not None:
        agent = TD3Agent(
            evaluation_environment.state_dim,
            evaluation_environment.action_dim,
            TD3Config(),
            device=args.device,
            seed=args.seed,
        )
        agent.load(args.checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = run_baseline_comparison(
        evaluation_environment,
        agent=agent,
        episodes=episodes,
        seed=args.seed,
    )
    write_summaries_csv(
        args.output_dir / "baseline_comparison.csv",
        summaries,
    )

    sweep_config = replace(
        evaluation_environment.config,
        max_episode_steps=min(10, evaluation_environment.config.max_episode_steps),
    )

    # 有源比例会改变动作维度；不能直接复用只在一个比例上训练的TD3。
    # 该扫描固定使用相位对齐启发式。要比较TD3，必须为每个比例单独训练。
    records = run_active_ratio_sweep(
        sweep_config,
        lambda env: heuristic_policy,
        ([0.0, 0.25, 1.0] if args.quick else [0.0, 0.125, 0.25, 0.50, 1.0]),
        episodes=episodes,
        seed=args.seed + 50_000,
    )
    write_records_csv(
        args.output_dir / "active_ratio_sweep.csv",
        records,
    )

    fixed_dimension_policy_factory = (
        (lambda env: heuristic_policy)
        if agent is None
        else (lambda env: agent_policy(agent))
    )

    nmse_values = [-20.0, -10.0] if args.quick else [-30.0, -20.0, -15.0, -10.0, -5.0]
    nmse_cases = []
    for value in nmse_values:
        robust = replace(
            sweep_config.robust,
            nmse_db_min=value,
            nmse_db_max=value,
        )
        nmse_cases.append((f"nmse_{value:g}_db", value, replace(sweep_config, robust=robust)))
    write_records_csv(
        args.output_dir / "nmse_sweep.csv",
        run_named_config_sweep(
            nmse_cases,
            fixed_dimension_policy_factory,
            parameter_name="nmse_db",
            episodes=episodes,
            seed=args.seed + 60_000,
        ),
    )

    delay_values = [0.0, 0.005] if args.quick else [0.0, 0.0005, 0.001, 0.002, 0.005, 0.010]
    delay_cases = [
        (
            f"delay_{value:g}_s",
            value,
            replace(
                sweep_config,
                channel=replace(
                    sweep_config.channel,
                    forward_reverse_delay_seconds=value,
                ),
            ),
        )
        for value in delay_values
    ]
    write_records_csv(
        args.output_dir / "delay_sweep.csv",
        run_named_config_sweep(
            delay_cases,
            fixed_dimension_policy_factory,
            parameter_name="delay_seconds",
            episodes=episodes,
            seed=args.seed + 70_000,
        ),
    )

    rf_values = [20.0, 35.0] if args.quick else [10.0, 20.0, 35.0, 50.0]
    rf_cases = []
    for value in rf_values:
        power = replace(sweep_config.power, maximum_rf_output_power=value)
        robust = replace(
            sweep_config.robust,
            rf_budget_scale_min=1.0,
            rf_budget_scale_max=1.0,
        )
        rf_cases.append((f"rf_budget_{value:g}", value, replace(sweep_config, power=power, robust=robust)))
    write_records_csv(
        args.output_dir / "rf_power_budget_sweep.csv",
        run_named_config_sweep(
            rf_cases,
            fixed_dimension_policy_factory,
            parameter_name="rf_budget",
            episodes=episodes,
            seed=args.seed + 80_000,
        ),
    )

    dc_values = [2.0, 5.0] if args.quick else [1.0, 2.0, 3.0, 5.0, 8.0]
    dc_cases = []
    for value in dc_values:
        power = replace(sweep_config.power, maximum_total_dc_power=value)
        robust = replace(
            sweep_config.robust,
            dc_budget_scale_min=1.0,
            dc_budget_scale_max=1.0,
        )
        dc_cases.append((f"dc_budget_{value:g}", value, replace(sweep_config, power=power, robust=robust)))
    write_records_csv(
        args.output_dir / "dc_power_budget_sweep.csv",
        run_named_config_sweep(
            dc_cases,
            fixed_dimension_policy_factory,
            parameter_name="dc_budget",
            episodes=episodes,
            seed=args.seed + 90_000,
        ),
    )

    ablation_cases = [
        ("complete_model", "complete", sweep_config),
        (
            "without_internal_noise",
            "without_internal_noise",
            replace(
                sweep_config,
                probing=replace(
                    sweep_config.probing,
                    input_referred_amplifier_noise_variance=0.0,
                ),
            ),
        ),
        (
            "perfect_control_csi",
            "perfect_control_csi",
            replace(
                sweep_config,
                robust=replace(
                    sweep_config.robust,
                    nmse_db_min=-100.0,
                    nmse_db_max=-100.0,
                ),
            ),
        ),
        (
            "without_delay_mismatch",
            "without_delay_mismatch",
            replace(
                sweep_config,
                channel=replace(
                    sweep_config.channel,
                    forward_reverse_delay_seconds=0.0,
                ),
            ),
        ),
        (
            "without_hardware_mismatch",
            "without_hardware_mismatch",
            replace(
                sweep_config,
                hardware=replace(
                    sweep_config.hardware,
                    static_gain_error_std_db=0.0,
                    directional_gain_error_std_db=0.0,
                    static_phase_error_std_rad=0.0,
                    directional_phase_error_std_rad=0.0,
                    fast_phase_jitter_std_rad=0.0,
                    transmission_split_error_std=0.0,
                    endpoint_gain_error_std_db=0.0,
                    endpoint_phase_error_std_rad=0.0,
                    phase_quantization_bits=None,
                    gain_quantization_bits=None,
                ),
            ),
        ),
        (
            "without_cvar",
            "without_cvar",
            replace(
                sweep_config,
                robust=replace(
                    sweep_config.robust,
                    mean_weight=1.0,
                    cvar_weight=0.0,
                ),
            ),
        ),
    ]
    if args.quick:
        ablation_cases = ablation_cases[:3]
    write_records_csv(
        args.output_dir / "ablation_comparison.csv",
        run_named_config_sweep(
            ablation_cases,
            fixed_dimension_policy_factory,
            parameter_name="ablation",
            episodes=episodes,
            seed=args.seed + 100_000,
        ),
    )

    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
