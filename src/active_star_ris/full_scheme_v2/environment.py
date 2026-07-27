from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .channels import (
    build_bidirectional_block,
    estimate_channel,
    evolve_snapshot,
    sample_channel_snapshot,
)
from .config import EnvironmentConfig
from .hardware import (
    action_dimension,
    build_active_mask,
    decode_action,
    sample_static_hardware,
)
from .models import ChannelSnapshot, HardwareStaticRealization, ObjectiveResult
from .objective import evaluate_objective
from .power import conservative_input_powers, project_command_to_power_constraints


@dataclass(frozen=True)
class EpisodeDomain:
    control_csi_nmse_db: float
    amplifier_noise_scale: float
    receiver_noise_scale: float
    rf_budget: float
    dc_budget: float


class RobustFullSchemeEnvironment:
    """不依赖Gym的STAR-RIS鲁棒TD3环境。

    reset() -> (state, info)
    step(action) -> (next_state, reward, terminated, truncated, info)
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        *,
        active_mask: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._rng = np.random.default_rng(seed)

        if active_mask is None:
            self.active_mask = build_active_mask(
                config.channel.num_elements,
                config.channel.active_ratio,
            )
        else:
            mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if mask.size != config.channel.num_elements:
                raise ValueError("active_mask size mismatch")
            self.active_mask = mask

        self.action_dim = action_dimension(
            config.channel.num_elements,
            self.active_mask,
        )
        self.state_dim = self._calculate_state_dimension()

        self._snapshot: ChannelSnapshot | None = None
        self._estimated_snapshot: ChannelSnapshot | None = None
        self._hardware: HardwareStaticRealization | None = None
        self._domain: EpisodeDomain | None = None
        self._step_index = 0
        self._last_metrics = np.zeros(7, dtype=np.float64)

    def _calculate_state_dimension(self) -> int:
        n = self.config.channel.num_elements
        # g, h_T, h_R: each real+imag => 6N; two direct links => 4.
        dimension = 6 * n + 4 + 7
        if self.config.robust.include_known_system_context:
            dimension += 2  # normalized RF and DC budgets.
        if self.config.robust.include_oracle_impairment_context:
            dimension += 3  # NMSE, amplifier-noise scale, receiver-noise scale.
        return dimension

    def _sample_domain(self) -> EpisodeDomain:
        robust = self.config.robust
        return EpisodeDomain(
            control_csi_nmse_db=float(
                self._rng.uniform(robust.nmse_db_min, robust.nmse_db_max)
            ),
            amplifier_noise_scale=float(
                self._rng.uniform(
                    robust.amplifier_noise_scale_min,
                    robust.amplifier_noise_scale_max,
                )
            ),
            receiver_noise_scale=float(
                self._rng.uniform(
                    robust.receiver_noise_scale_min,
                    robust.receiver_noise_scale_max,
                )
            ),
            rf_budget=float(
                self.config.power.maximum_rf_output_power
                * self._rng.uniform(
                    robust.rf_budget_scale_min,
                    robust.rf_budget_scale_max,
                )
            ),
            dc_budget=float(
                self.config.power.maximum_total_dc_power
                * self._rng.uniform(
                    robust.dc_budget_scale_min,
                    robust.dc_budget_scale_max,
                )
            ),
        )

    def _estimate_snapshot(
        self,
        snapshot: ChannelSnapshot,
        nmse_db: float,
    ) -> ChannelSnapshot:
        return ChannelSnapshot(
            controller_to_ris=estimate_channel(
                snapshot.controller_to_ris,
                nmse_db,
                self._rng,
            ),
            ris_to_transmission=estimate_channel(
                snapshot.ris_to_transmission,
                nmse_db,
                self._rng,
            ),
            ris_to_reflection=estimate_channel(
                snapshot.ris_to_reflection,
                nmse_db,
                self._rng,
            ),
            direct_transmission=complex(
                estimate_channel(
                    snapshot.direct_transmission,
                    nmse_db,
                    self._rng,
                ).reshape(-1)[0]
            ),
            direct_reflection=complex(
                estimate_channel(
                    snapshot.direct_reflection,
                    nmse_db,
                    self._rng,
                ).reshape(-1)[0]
            ),
        )

    @staticmethod
    def _normalize_complex(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.complex128).reshape(-1)
        scale = max(float(np.sqrt(np.mean(np.abs(array) ** 2))), 1.0e-8)
        normalized = array / scale
        return np.concatenate((normalized.real, normalized.imag)).astype(np.float64)

    def _build_state(self) -> np.ndarray:
        if self._estimated_snapshot is None or self._domain is None:
            raise RuntimeError("environment must be reset first")

        estimate = self._estimated_snapshot
        components = [
            self._normalize_complex(estimate.controller_to_ris),
            self._normalize_complex(estimate.ris_to_transmission),
            self._normalize_complex(estimate.ris_to_reflection),
            np.asarray(
                [
                    estimate.direct_transmission.real,
                    estimate.direct_transmission.imag,
                    estimate.direct_reflection.real,
                    estimate.direct_reflection.imag,
                ],
                dtype=np.float64,
            ),
            np.asarray(self._last_metrics, dtype=np.float64),
        ]

        if self.config.robust.include_known_system_context:
            components.append(
                np.asarray(
                    [
                        self._domain.rf_budget
                        / self.config.power.maximum_rf_output_power,
                        self._domain.dc_budget
                        / self.config.power.maximum_total_dc_power,
                    ],
                    dtype=np.float64,
                )
            )

        if self.config.robust.include_oracle_impairment_context:
            components.append(
                np.asarray(
                    [
                        self._domain.control_csi_nmse_db / 30.0,
                        self._domain.amplifier_noise_scale,
                        self._domain.receiver_noise_scale,
                    ],
                    dtype=np.float64,
                )
            )

        state = np.concatenate(components).astype(np.float32)
        if state.size != self.state_dim:
            raise RuntimeError(
                f"state dimension mismatch: got {state.size}, expected {self.state_dim}"
            )
        return state

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._domain = self._sample_domain()
        self._snapshot = sample_channel_snapshot(self.config.channel, self._rng)
        self._estimated_snapshot = self._estimate_snapshot(
            self._snapshot,
            self._domain.control_csi_nmse_db,
        )
        self._hardware = sample_static_hardware(
            self.config.channel.num_elements,
            self.config.hardware,
            self._rng,
        )
        self._step_index = 0
        self._last_metrics = np.zeros(7, dtype=np.float64)

        return self._build_state(), {
            "domain": self._domain,
            "active_elements": int(np.count_nonzero(self.active_mask)),
        }

    def _require_initialized(
        self,
    ) -> tuple[
        ChannelSnapshot,
        ChannelSnapshot,
        HardwareStaticRealization,
        EpisodeDomain,
    ]:
        if (
            self._snapshot is None
            or self._estimated_snapshot is None
            or self._hardware is None
            or self._domain is None
        ):
            raise RuntimeError("environment must be reset before step")
        return (
            self._snapshot,
            self._estimated_snapshot,
            self._hardware,
            self._domain,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        snapshot, estimate, hardware, domain = self._require_initialized()

        command = decode_action(action, self.active_mask, self.config.hardware)
        input_c, input_t, input_r = conservative_input_powers(
            estimate.controller_to_ris,
            estimate.ris_to_transmission,
            estimate.ris_to_reflection,
            nmse_db=domain.control_csi_nmse_db,
            probing=self.config.probing,
            power=self.config.power,
        )
        projected_command, projected_power = project_command_to_power_constraints(
            command,
            input_c,
            input_t,
            input_r,
            power_config=self.config.power,
            hardware_config=self.config.hardware,
            rf_budget=domain.rf_budget,
            dc_budget=domain.dc_budget,
        )

        objective_samples: list[ObjectiveResult] = []
        for _ in range(self.config.robust.objective_samples):
            block = build_bidirectional_block(
                snapshot,
                self.config.channel,
                self.config.probing.samples_per_step,
                self._rng,
            )
            result = evaluate_objective(
                block,
                projected_command,
                hardware,
                self.config,
                self._rng,
                full_protocol=(self.config.objective.key_rate_mode == "final_key"),
                amplifier_noise_scale=domain.amplifier_noise_scale,
                receiver_noise_scale=domain.receiver_noise_scale,
                rf_budget=domain.rf_budget,
                dc_budget=domain.dc_budget,
            )
            objective_samples.append(result)

        rewards = np.asarray(
            [sample.reward for sample in objective_samples],
            dtype=np.float64,
        )
        mean_reward = float(np.mean(rewards))
        tail_count = max(
            self.config.robust.minimum_tail_samples,
            int(np.ceil(self.config.robust.cvar_alpha * rewards.size)),
        )
        tail_count = min(tail_count, rewards.size)
        cvar_reward = float(np.mean(np.sort(rewards)[:tail_count]))
        robust_weight_sum = (
            self.config.robust.mean_weight + self.config.robust.cvar_weight
        )
        reward = float(
            (
                self.config.robust.mean_weight * mean_reward
                + self.config.robust.cvar_weight * cvar_reward
            )
            / robust_weight_sum
        )

        def average(attribute: str) -> float:
            return float(
                np.mean([getattr(sample, attribute) for sample in objective_samples])
            )

        mean_training_rate = average("training_key_rate_bps")
        mean_final_rate = average("final_key_rate_bps")
        mean_raw_kdr = average("raw_kdr")
        mean_post_kdr = average("post_reconciliation_kdr")
        mean_reciprocity = average("reciprocity")
        mean_power = float(
            np.mean(
                [sample.power.total_surface_dc_power for sample in objective_samples]
            )
        )
        feasibility_rate = float(
            np.mean([sample.power.fully_feasible for sample in objective_samples])
        )

        self._last_metrics = np.asarray(
            [
                np.log1p(mean_training_rate) / 10.0,
                np.log1p(mean_final_rate) / 10.0,
                mean_raw_kdr,
                mean_post_kdr,
                mean_reciprocity,
                mean_power / max(domain.dc_budget, 1.0e-12),
                feasibility_rate,
            ],
            dtype=np.float64,
        )

        self._snapshot = evolve_snapshot(snapshot, self.config.channel, self._rng)
        self._estimated_snapshot = self._estimate_snapshot(
            self._snapshot,
            domain.control_csi_nmse_db,
        )
        self._step_index += 1

        terminated = False
        truncated = self._step_index >= self.config.max_episode_steps

        info: dict[str, Any] = {
            "mean_reward": mean_reward,
            "cvar_reward": cvar_reward,
            "tail_count": tail_count,
            "training_key_rate_bps": mean_training_rate,
            "final_key_rate_bps": mean_final_rate,
            "raw_kdr": mean_raw_kdr,
            "post_reconciliation_kdr": mean_post_kdr,
            "reciprocity": mean_reciprocity,
            "surface_dc_power": mean_power,
            "feasibility_rate": feasibility_rate,
            "projection_fully_feasible": projected_power.fully_feasible,
            "projected_gain": projected_command.gain.copy(),
            "domain": domain,
        }
        return self._build_state(), reward, terminated, truncated, info


    def passive_action(self) -> np.ndarray:
        """返回单位增益、零相位、均匀能量分配动作。"""
        n = self.config.channel.num_elements
        num_active = int(np.count_nonzero(self.active_mask))
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[:num_active] = -1.0
        # phase action -1 corresponds to phase 0.
        action[num_active : num_active + 2 * n] = -1.0
        # beta action 0 corresponds to beta_T=0.5.
        return action

    def random_action(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

    def heuristic_action(self) -> np.ndarray:
        """基于估计级联信道相位对齐的确定性基线动作。"""
        if self._estimated_snapshot is None:
            raise RuntimeError("environment must be reset first")
        estimate = self._estimated_snapshot
        n = self.config.channel.num_elements
        num_active = int(np.count_nonzero(self.active_mask))

        desired_t = np.mod(
            -np.angle(estimate.controller_to_ris * estimate.ris_to_transmission),
            2.0 * np.pi,
        )
        desired_r = np.mod(
            -np.angle(estimate.controller_to_ris * estimate.ris_to_reflection),
            2.0 * np.pi,
        )
        phase_t_action = desired_t / np.pi - 1.0
        if self.config.hardware.phase_coupling_mode == "independent":
            phase_r_action = desired_r / np.pi - 1.0
        else:
            wrapped_difference = np.angle(np.exp(1j * (desired_r - desired_t)))
            phase_r_action = np.where(wrapped_difference >= 0.0, 1.0, -1.0)

        strength_t = float(np.mean(np.abs(estimate.ris_to_transmission) ** 2))
        strength_r = float(np.mean(np.abs(estimate.ris_to_reflection) ** 2))
        beta_t = strength_r / max(strength_t + strength_r, 1.0e-12)
        beta_action = np.full(n, 2.0 * beta_t - 1.0, dtype=np.float64)

        action = np.concatenate(
            (
                np.ones(num_active, dtype=np.float64),
                phase_t_action,
                phase_r_action,
                beta_action,
            )
        )
        return np.asarray(np.clip(action, -1.0, 1.0), dtype=np.float32)

    def evaluation_copy(
        self,
        *,
        samples_per_step: int = 2048,
        objective_samples: int = 128,
        cvar_alpha: float = 0.10,
        seed: int | None = None,
    ) -> "RobustFullSchemeEnvironment":
        robust = replace(
            self.config.robust,
            objective_samples=objective_samples,
            cvar_alpha=cvar_alpha,
            minimum_tail_samples=max(
                4,
                int(np.ceil(objective_samples * cvar_alpha)),
            ),
        )
        objective = replace(self.config.objective, key_rate_mode="final_key")
        probing = replace(self.config.probing, samples_per_step=samples_per_step)
        evaluation_config = replace(
            self.config,
            robust=robust,
            objective=objective,
            probing=probing,
        )
        return RobustFullSchemeEnvironment(
            evaluation_config,
            active_mask=self.active_mask,
            seed=seed,
        )
