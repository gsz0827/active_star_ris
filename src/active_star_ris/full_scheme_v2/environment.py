from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .channels import (
    advance_static_channels,
    estimate_control_csi,
    sample_static_channels,
)
from .config import FullSchemeConfig
from .hardware import (
    action_dimension,
    decode_action,
    realize_coefficients,
    sample_static_hardware,
)
from .key_protocol import evaluate_joint_key_metrics
from .models import CSIState, ObjectiveSample, RobustSummary, StaticChannels, StaticHardwareState
from .objective import aggregate_robust_samples, objective_reward
from .power import actual_power_metrics, project_gains, replace_gain
from .probing import simulate_dual_side_probing


class ActiveStarRisKeyEnvironment:
    def __init__(self, config: FullSchemeConfig):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.environment.seed)
        self.channels: StaticChannels | None = None
        self.csi: CSIState | None = None
        self.hardware: StaticHardwareState | None = None
        self.step_index = 0
        self.previous_summary = np.zeros(6, dtype=np.float64)

    @property
    def num_elements(self) -> int:
        return self.config.geometry.num_elements

    @property
    def action_dimension(self) -> int:
        return action_dimension(self.num_elements)

    @property
    def state_dimension(self) -> int:
        return 6 * self.num_elements + 11

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.channels = sample_static_channels(
            self.config.geometry,
            self.config.channel,
            self.rng,
        )
        self.csi = estimate_control_csi(self.channels, self.config.channel, self.rng)
        self.hardware = sample_static_hardware(
            self.num_elements,
            self.config.hardware,
            self.rng,
        )
        self.step_index = 0
        self.previous_summary = np.zeros(6, dtype=np.float64)
        return self._state(), {"architecture": self.config.environment.architecture}

    def _require_initialized(self) -> tuple[StaticChannels, CSIState, StaticHardwareState]:
        if self.channels is None or self.csi is None or self.hardware is None:
            raise RuntimeError("reset() must be called before step()")
        return self.channels, self.csi, self.hardware

    def _normalize_complex(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = max(float(np.sqrt(np.mean(np.abs(values) ** 2))), 1.0e-15)
        return np.real(values) / scale, np.imag(values) / scale

    def _state(self) -> np.ndarray:
        channels, csi, _ = self._require_initialized()
        features: list[np.ndarray] = []
        for result in (csi.controller_ris, csi.ris_transmission, csi.ris_reflection):
            real, imag = self._normalize_complex(result.estimate)
            features.extend([real, imag])
        direct_t = csi.direct_transmission.estimate[0]
        direct_r = csi.direct_reflection.estimate[0]
        scalar_features = np.asarray(
            [
                np.real(direct_t),
                np.imag(direct_t),
                np.real(direct_r),
                np.imag(direct_r),
                self.step_index / max(self.config.environment.episode_length, 1),
            ],
            dtype=np.float64,
        )
        state = np.concatenate(features + [scalar_features, self.previous_summary])
        if state.size != self.state_dimension:
            raise RuntimeError(f"state size mismatch: {state.size} != {self.state_dimension}")
        return np.clip(state, -10.0, 10.0).astype(np.float32)

    def _prepare_action(self, action: np.ndarray) -> tuple[Any, Any]:
        """Decode and project one action exactly once per robust evaluation.

        Action decoding and gain projection depend on the action and the current
        control CSI, but not on the Monte-Carlo hardware/channel realization.
        Reusing them therefore removes deterministic duplicate work without
        reducing the number of robust samples or changing the uncertainty model.
        """
        _, csi, _ = self._require_initialized()
        ideal = decode_action(
            np.asarray(action, dtype=np.float64),
            num_elements=self.num_elements,
            architecture=self.config.environment.architecture,
            config=self.config.hardware,
        )
        projection = project_gains(
            ideal,
            csi,
            self.config.probing,
            self.config.hardware,
            self.config.power,
        )
        return replace_gain(ideal, projection), projection

    def _evaluate_once(
        self,
        projected: Any,
        projection: Any,
        *,
        full_protocol: bool,
    ) -> ObjectiveSample:
        channels, _, hardware_state = self._require_initialized()
        coefficients = realize_coefficients(
            projected,
            hardware_state,
            samples=self.config.probing.samples_per_step,
            config=self.config.hardware,
            rng=self.rng,
        )
        transmission, reflection = simulate_dual_side_probing(
            channels,
            coefficients,
            projected.active_mask,
            self.config.channel,
            self.config.probing,
            self.rng,
        )
        key_metrics = evaluate_joint_key_metrics(
            transmission,
            reflection,
            self.config.key_generation,
            self.config.probing,
            self.config.objective,
            self.rng,
            full_protocol=full_protocol,
        )
        power_metrics = actual_power_metrics(
            channels,
            coefficients.actual_gain_forward,
            coefficients.actual_gain_reverse,
            projected.active_mask,
            self.config.probing,
            self.config.hardware,
            self.config.power,
        )
        reward = objective_reward(
            key_metrics,
            power_metrics,
            self.config.objective,
            power_config=self.config.power,
            hardware_config=self.config.hardware,
        )
        return ObjectiveSample(
            reward=reward,
            key_metrics=key_metrics,
            power_metrics=power_metrics,
            active_elements=int(np.count_nonzero(projected.active_mask)),
            projection_scale=projection.projection_scale,
            architecture_feasible=projection.unit_gain_feasible,
        )

    def evaluate_action_with_samples(
        self,
        action: np.ndarray,
        *,
        full_protocol: bool = False,
        objective_samples: int | None = None,
    ) -> tuple[RobustSummary, list[ObjectiveSample]]:
        """Evaluate one action and retain the underlying robust samples.

        Training only needs the aggregated RobustSummary, while protocol
        diagnosis also needs each Monte-Carlo sample's branch key metrics.
        Keeping this as a separate method preserves the training interface.
        """
        count = (
            self.config.robust.objective_samples
            if objective_samples is None
            else int(objective_samples)
        )
        if count < 1:
            raise ValueError("objective_samples must be positive")

        projected, projection = self._prepare_action(action)
        samples = [
            self._evaluate_once(
                projected,
                projection,
                full_protocol=full_protocol,
            )
            for _ in range(count)
        ]
        summary = aggregate_robust_samples(samples, self.config.robust)
        return summary, samples

    def evaluate_action(
        self,
        action: np.ndarray,
        *,
        full_protocol: bool = False,
        objective_samples: int | None = None,
    ) -> RobustSummary:
        summary, _ = self.evaluate_action_with_samples(
            action,
            full_protocol=full_protocol,
            objective_samples=objective_samples,
        )
        return summary

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        summary = self.evaluate_action(action, full_protocol=False)
        self.previous_summary = np.asarray(
            [
                np.clip(summary.robust_reward / 5.0, -1.0, 1.0),
                np.clip(summary.mean_secure_key_rate_bps / self.config.objective.key_rate_reference_bps, 0.0, 5.0) / 5.0,
                2.0 * np.clip(summary.mean_raw_kdr, 0.0, 1.0) - 1.0,
                2.0 * np.clip(summary.mean_reciprocity, 0.0, 1.0) - 1.0,
                np.clip(summary.mean_surface_power_watt / self.config.objective.surface_power_reference_watt, 0.0, 2.0) - 1.0,
                2.0 * summary.power_violation_probability - 1.0,
            ],
            dtype=np.float64,
        )
        self.step_index += 1
        terminated = self.step_index >= self.config.environment.episode_length
        channels, _, _ = self._require_initialized()
        self.channels = advance_static_channels(
            channels,
            self.config.channel.between_step_correlation,
            self.rng,
        )
        self.csi = estimate_control_csi(self.channels, self.config.channel, self.rng)
        info = {
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
            "cvar_reciprocity": summary.cvar_reciprocity,
            "worst_reciprocity": summary.worst_reciprocity,
            "mean_surface_power_watt": summary.mean_surface_power_watt,
            "cvar_surface_power_watt": summary.cvar_surface_power_watt,
            "worst_surface_power_watt": summary.worst_surface_power_watt,
            "power_violation_probability": summary.power_violation_probability,
            "mean_active_elements": summary.mean_active_elements,
            "mean_projection_scale": summary.mean_projection_scale,
            "architecture": self.config.environment.architecture,
        }
        return self._state(), summary.robust_reward, terminated, False, info

    def with_architecture(self, architecture: str, seed: int | None = None) -> "ActiveStarRisKeyEnvironment":
        environment = replace(
            self.config.environment,
            architecture=architecture,
            seed=self.config.environment.seed if seed is None else seed,
        )
        return ActiveStarRisKeyEnvironment(replace(self.config, environment=environment))
