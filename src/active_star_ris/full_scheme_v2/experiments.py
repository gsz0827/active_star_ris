from __future__ import annotations

import time
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .config import FullSchemeConfig
from .environment import ActiveStarRisKeyEnvironment
from .td3 import ReplayBuffer, TD3Agent


def random_policy(action_dimension: int, seed: int) -> Callable[[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    return lambda state: rng.uniform(-1.0, 1.0, action_dimension).astype(np.float32)


def heuristic_policy(environment: ActiveStarRisKeyEnvironment) -> Callable[[np.ndarray], np.ndarray]:
    n = environment.num_elements

    def policy(state: np.ndarray) -> np.ndarray:
        del state
        _, csi, _ = environment._require_initialized()
        cascaded_t = csi.controller_ris.estimate * csi.ris_transmission.estimate
        cascaded_r = csi.controller_ris.estimate * csi.ris_reflection.estimate
        phase_t = np.mod(-np.angle(cascaded_t), 2.0 * np.pi)
        phase_r = np.mod(-np.angle(cascaded_r), 2.0 * np.pi)
        phase_t_action = phase_t / np.pi - 1.0
        phase_r_action = phase_r / np.pi - 1.0
        gain_action = np.zeros(n, dtype=np.float64)
        split_action = np.zeros(n, dtype=np.float64)
        utility = np.abs(cascaded_t) + np.abs(cascaded_r)
        if np.max(utility) > np.min(utility):
            gates = 2.0 * (utility - np.min(utility)) / (np.max(utility) - np.min(utility)) - 1.0
        else:
            gates = np.zeros(n, dtype=np.float64)
        return np.concatenate(
            [gain_action, phase_t_action, phase_r_action, split_action, gates]
        ).astype(np.float32)

    return policy


def train_td3(
    config: FullSchemeConfig,
    *,
    steps: int,
    output_dir: str | Path,
    seed: int,
) -> tuple[TD3Agent, list[dict[str, float]]]:
    environment_config = replace(config.environment, seed=seed)
    config = replace(config, environment=environment_config)
    env = ActiveStarRisKeyEnvironment(config)
    state, _ = env.reset(seed=seed)
    agent = TD3Agent(env.state_dimension, env.action_dimension, config.td3, seed=seed)
    replay = ReplayBuffer(
        env.state_dimension,
        env.action_dimension,
        config.td3.replay_capacity,
        seed,
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    # 每个模型大约输出 100 次进度。
    progress_interval = max(1, steps // 100)

    print(
        "\n"
        f"[训练开始] "
        f"architecture={config.environment.architecture}, "
        f"seed={seed}, "
        f"steps={steps}, "
        f"device={agent.device}",
        flush=True,
    )

    for step in range(steps):
        if step < config.td3.warmup_steps:
            action = rng.uniform(-1.0, 1.0, env.action_dimension).astype(np.float32)
        else:
            action = agent.act(state, config.td3.exploration_noise)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay.add(state, action, reward, next_state, done)
        losses = agent.train(replay)
        completed_steps = step + 1

        if (
            completed_steps == 1
            or completed_steps % progress_interval == 0
            or completed_steps == steps
        ):
            elapsed_seconds = time.perf_counter() - start_time

            steps_per_second = (
                completed_steps / max(elapsed_seconds, 1.0e-12)
            )

            remaining_seconds = (
                (steps - completed_steps)
                / max(steps_per_second, 1.0e-12)
            )

            critic_text = (
                f"{losses.critic_loss:.6g}"
                if losses is not None
                else "warmup"
            )

            actor_text = (
                f"{losses.actor_loss:.6g}"
                if (
                    losses is not None
                    and losses.actor_loss is not None
                )
                else "-"
            )

            print(
                f"[训练进度] "
                f"{config.environment.architecture} "
                f"seed={seed} | "
                f"{completed_steps}/{steps} "
                f"({100.0 * completed_steps / steps:6.2f}%) | "
                f"reward={reward:.5f} | "
                f"key_rate={info['mean_secure_key_rate_bps']:.3f} | "
                f"raw_kdr={info['mean_raw_kdr']:.5f} | "
                f"power={info['mean_surface_power_watt']:.6g} W | "
                f"critic={critic_text} | "
                f"actor={actor_text} | "
                f"速度={steps_per_second:.3f} step/s | "
                f"预计剩余={remaining_seconds / 60.0:.1f} min",
                flush=True,
            )
        if step % 100 == 0 or step == steps - 1:
            row = {
                "step": float(step),
                "reward": float(reward),
                "mean_secure_key_rate_bps": float(info["mean_secure_key_rate_bps"]),
                "mean_raw_kdr": float(info["mean_raw_kdr"]),
                "mean_surface_power_watt": float(info["mean_surface_power_watt"]),
                "power_violation_probability": float(info["power_violation_probability"]),
                "critic_loss": float(losses.critic_loss) if losses else float("nan"),
                "actor_loss": float(losses.actor_loss) if losses and losses.actor_loss is not None else float("nan"),
            }
            history.append(row)
        state = next_state
        if done:
            state, _ = env.reset()
    agent.save(output / "td3_checkpoint.pt")
    write_csv(output / "training_history.csv", history)

    total_seconds = time.perf_counter() - start_time

    print(
        f"[训练完成] "
        f"architecture={config.environment.architecture}, "
        f"seed={seed}, "
        f"耗时={total_seconds / 60.0:.1f} min, "
        f"输出={output}",
        flush=True,
    )

    return agent, history
    write_csv(output / "training_history.csv", history)
    return agent, history


def evaluate_policy(
    config: FullSchemeConfig,
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    episodes: int,
    seed: int,
    full_protocol: bool,
    objective_samples: int = 64,
) -> list[dict[str, float | str]]:
    env = ActiveStarRisKeyEnvironment(replace(config, environment=replace(config.environment, seed=seed)))
    rows: list[dict[str, float | str]] = []

    evaluation_start_time = time.perf_counter()
    evaluation_interval = max(1, episodes // 10)

    print(
        f"[评估开始] "
        f"architecture={config.environment.architecture}, "
        f"seed={seed}, "
        f"episodes={episodes}, "
        f"objective_samples={objective_samples}",
        flush=True,
    )

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)

        # 先运行代理协议，使上一时隙指标和时变信道进入状态。
        burn_in_steps = max(
            env.config.environment.episode_length - 1,
            0,
        )

        for _ in range(burn_in_steps):
            action = policy(state)

            (
                state,
                _,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            if terminated or truncated:
                break

        # 在具有历史状态的最终时隙执行正式协议评价。
        action = policy(state)

        summary = env.evaluate_action(
            action,
            full_protocol=full_protocol,
            objective_samples=objective_samples,
        )
        rows.append(
            {
                "episode": float(episode),
                "architecture": config.environment.architecture,
                "robust_reward": summary.robust_reward,
                "mean_reward": summary.mean_reward,
                "cvar_reward": summary.cvar_reward,
                "worst_reward": summary.worst_reward,
                "mean_secure_key_rate_bps": summary.mean_secure_key_rate_bps,
                "cvar_secure_key_rate_bps": summary.cvar_secure_key_rate_bps,
                "worst_secure_key_rate_bps": summary.worst_secure_key_rate_bps,
                "mean_raw_kdr": summary.mean_raw_kdr,
                "cvar_raw_kdr": summary.cvar_raw_kdr,
                "worst_raw_kdr": summary.worst_raw_kdr,
                "mean_reciprocity": summary.mean_reciprocity,
                "mean_surface_power_watt": summary.mean_surface_power_watt,
                "power_violation_probability": summary.power_violation_probability,
                "mean_active_elements": summary.mean_active_elements,
                "mean_projection_scale": summary.mean_projection_scale,
            }
        )
        completed_episodes = episode + 1

        if (
            completed_episodes == 1
            or completed_episodes % evaluation_interval == 0
            or completed_episodes == episodes
        ):
            elapsed_seconds = (
                time.perf_counter() - evaluation_start_time
            )

            print(
                f"[评估进度] "
                f"{config.environment.architecture} | "
                f"{completed_episodes}/{episodes} "
                f"({100.0 * completed_episodes / episodes:6.2f}%) | "
                f"reward={summary.robust_reward:.5f} | "
                f"key_rate={summary.mean_secure_key_rate_bps:.3f} | "
                f"raw_kdr={summary.mean_raw_kdr:.5f} | "
                f"耗时={elapsed_seconds / 60.0:.1f} min",
                flush=True,
            ) 
        print(
            f"[评估完成] "
            f"architecture={config.environment.architecture}, "
            f"episodes={episodes}, "
            f"耗时="
            f"{(time.perf_counter() - evaluation_start_time) / 60.0:.1f} min",
            flush=True,
        )
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_rows(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    if not rows:
        return {}
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != "episode"]
    summary: dict[str, float | str] = {"architecture": str(rows[0]["architecture"])}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary[f"{key}_ci95_half_width"] = float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return summary


def run_architecture_suite(
    config: FullSchemeConfig,
    *,
    output_dir: str | Path,
    training_steps: int,
    evaluation_episodes: int,
    seeds: list[int],
    objective_samples: int,
    evaluation_probing_samples: int | None = None,
) -> list[dict[str, float | str]]:
    output = Path(output_dir)
    all_summaries: list[dict[str, float | str]] = []

    total_runs = (
        len(config.experiment.architectures)
        * len(seeds)
    )
    run_number = 0

    print(
        f"[实验套件开始] "
        f"总任务数={total_runs}, "
        f"每个任务训练步数={training_steps}, "
        f"每个任务评估回合={evaluation_episodes}",
        flush=True,
    )

    for architecture in config.experiment.architectures:
        for seed in seeds:
            run_number += 1

            print(
                "\n"
                f"{'=' * 72}\n"
                f"[总任务 {run_number}/{total_runs}] "
                f"architecture={architecture}, seed={seed}\n"
                f"{'=' * 72}",
                flush=True,
            )
            architecture_config = replace(
                config,
                environment=replace(config.environment, architecture=architecture, seed=seed),
            )
            run_dir = output / architecture / f"seed_{seed}"
            agent, _ = train_td3(
                architecture_config,
                steps=training_steps,
                output_dir=run_dir,
                seed=seed,
            )
            evaluation_config = architecture_config
            if evaluation_probing_samples is not None:
                evaluation_config = replace(
                    architecture_config,
                    probing=replace(
                        architecture_config.probing,
                        samples_per_step=evaluation_probing_samples,
                    ),
                )
            td3_policy = lambda state, agent=agent: agent.act(state, 0.0)
            rows = evaluate_policy(
                evaluation_config,
                td3_policy,
                episodes=evaluation_episodes,
                seed=seed + 100_000,
                full_protocol=True,
                objective_samples=objective_samples,
            )
            write_csv(run_dir / "evaluation.csv", rows)
            summary = summarize_rows(rows)
            summary["seed"] = float(seed)
            all_summaries.append(summary)
    write_csv(output / "all_seed_summaries.csv", all_summaries)
    write_json(output / "config_snapshot.json", asdict(config))
    return all_summaries
