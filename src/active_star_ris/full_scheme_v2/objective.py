from __future__ import annotations

import math

import numpy as np

from .config import (
    HardwareConfig,
    ObjectiveConfig,
    PowerConfig,
    RobustConfig,
)
from .models import JointKeyMetrics, ObjectiveSample, PowerMetrics, RobustSummary


def objective_reward_components(
    key_metrics: JointKeyMetrics,
    power_metrics: PowerMetrics,
    config: ObjectiveConfig,
    *,
    power_config: PowerConfig,
    hardware_config: HardwareConfig,
    projection_scale: float = 1.0,
) -> dict[str, float]:
    key_term = (
        key_metrics.weighted_secure_key_rate_bps
        / config.key_rate_reference_bps
    )

    transmission_margin = float(key_metrics.transmission.key_margin_bits)
    reflection_margin = float(key_metrics.reflection.key_margin_bits)
    mean_margin_raw = (
        key_metrics.weighted_key_margin_bits
        / config.key_margin_reference_bits
    )
    worst_branch_margin_raw = (
        min(transmission_margin, reflection_margin)
        / config.key_margin_reference_bits
    )
    branch_imbalance_raw = (
        abs(transmission_margin - reflection_margin)
        / config.key_margin_reference_bits
    )
    mean_margin_term = float(np.clip(mean_margin_raw, -2.0, 2.0))
    worst_branch_margin_term = float(
        np.clip(worst_branch_margin_raw, -2.0, 2.0)
    )
    branch_imbalance_term = float(
        np.clip(branch_imbalance_raw, 0.0, 2.0)
    )

    raw_kdr_term = key_metrics.weighted_raw_kdr / config.raw_kdr_reference
    post_kdr_term = (
        key_metrics.weighted_post_reconciliation_kdr
        / config.post_reconciliation_kdr_reference
    )
    power_term = (
        power_metrics.total_dc_power
        / config.surface_power_reference_watt
    )
    projection_penalty = 1.0 - float(
        np.clip(projection_scale, 0.0, 1.0)
    )

    rf_scale = max(power_config.maximum_rf_output_power, 1.0e-30)
    dc_scale = max(power_config.maximum_total_dc_power, 1.0e-30)
    saturation_scale = max(
        hardware_config.per_active_element_saturation_power,
        1.0e-30,
    )
    normalized_violation = (
        power_metrics.rf_violation / rf_scale
        + power_metrics.dc_violation / dc_scale
        + power_metrics.saturation_violation / saturation_scale
    )

    components = {
        "key_rate_component": config.key_rate_weight * key_term,
        "mean_margin_component": config.key_margin_weight * mean_margin_term,
        "worst_branch_margin_component": (
            config.worst_branch_key_margin_weight * worst_branch_margin_term
        ),
        "branch_imbalance_penalty_component": (
            -config.branch_imbalance_penalty_weight * branch_imbalance_term
        ),
        "raw_kdr_penalty_component": -config.raw_kdr_weight * raw_kdr_term,
        "post_kdr_penalty_component": (
            -config.post_reconciliation_kdr_weight * post_kdr_term
        ),
        "reciprocity_component": (
            config.reciprocity_weight * key_metrics.weighted_reciprocity
        ),
        "surface_power_penalty_component": (
            -config.surface_power_weight * power_term
        ),
        "projection_penalty_component": (
            -config.projection_penalty_weight * projection_penalty
        ),
        "constraint_penalty_component": (
            -config.constraint_violation_weight * normalized_violation**2
        ),
        "transmission_margin_normalized": (
            transmission_margin / config.key_margin_reference_bits
        ),
        "reflection_margin_normalized": (
            reflection_margin / config.key_margin_reference_bits
        ),
        "mean_margin_normalized_raw": float(mean_margin_raw),
        "worst_margin_normalized_raw": float(worst_branch_margin_raw),
        "branch_imbalance_normalized_raw": float(branch_imbalance_raw),
        "mean_margin_clipped": float(abs(mean_margin_raw) > 2.0),
        "worst_margin_clipped": float(abs(worst_branch_margin_raw) > 2.0),
        "branch_imbalance_clipped": float(branch_imbalance_raw > 2.0),
    }
    components["reward_reconstructed"] = float(
        sum(
            value
            for name, value in components.items()
            if name.endswith("_component")
        )
    )
    return components


def objective_reward(
    key_metrics: JointKeyMetrics,
    power_metrics: PowerMetrics,
    config: ObjectiveConfig,
    *,
    power_config: PowerConfig,
    hardware_config: HardwareConfig,
    projection_scale: float = 1.0,
) -> float:
    components = objective_reward_components(
        key_metrics,
        power_metrics,
        config,
        power_config=power_config,
        hardware_config=hardware_config,
        projection_scale=projection_scale,
    )
    return float(components["reward_reconstructed"])


def aggregate_robust_samples(
    samples: list[ObjectiveSample],
    config: RobustConfig,
) -> RobustSummary:
    if not samples:
        raise ValueError("at least one objective sample is required")
    rewards = np.asarray([sample.reward for sample in samples], dtype=np.float64)
    tail_count = max(1, int(math.ceil(config.cvar_alpha * len(samples))))
    order = np.argsort(rewards)
    tail = order[:tail_count]
    worst = int(order[0])

    def values(getter) -> np.ndarray:
        return np.asarray([getter(sample) for sample in samples], dtype=np.float64)

    def stats(getter) -> tuple[float, float, float]:
        array = values(getter)
        return float(np.mean(array)), float(np.mean(array[tail])), float(array[worst])

    mean_rate, cvar_rate, worst_rate = stats(
        lambda sample: sample.key_metrics.weighted_secure_key_rate_bps
    )
    mean_kdr, cvar_kdr, worst_kdr = stats(
        lambda sample: sample.key_metrics.weighted_raw_kdr
    )
    mean_post_kdr, cvar_post_kdr, worst_post_kdr = stats(
        lambda sample: sample.key_metrics.weighted_post_reconciliation_kdr
    )
    mean_reciprocity, cvar_reciprocity, worst_reciprocity = stats(
        lambda sample: sample.key_metrics.weighted_reciprocity
    )
    mean_power, cvar_power, worst_power = stats(
        lambda sample: sample.power_metrics.total_dc_power
    )
    mean_reward = float(np.mean(rewards))
    cvar_reward = float(np.mean(rewards[tail]))
    denominator = config.mean_weight + config.cvar_weight
    robust_reward = (
        config.mean_weight * mean_reward + config.cvar_weight * cvar_reward
    ) / denominator
    return RobustSummary(
        robust_reward=float(robust_reward),
        mean_reward=mean_reward,
        cvar_reward=cvar_reward,
        worst_reward=float(rewards[worst]),
        mean_secure_key_rate_bps=mean_rate,
        cvar_secure_key_rate_bps=cvar_rate,
        worst_secure_key_rate_bps=worst_rate,
        mean_raw_kdr=mean_kdr,
        cvar_raw_kdr=cvar_kdr,
        worst_raw_kdr=worst_kdr,
        mean_reciprocity=mean_reciprocity,
        cvar_reciprocity=cvar_reciprocity,
        worst_reciprocity=worst_reciprocity,
        mean_surface_power_watt=mean_power,
        cvar_surface_power_watt=cvar_power,
        worst_surface_power_watt=worst_power,
        power_violation_probability=float(
            np.mean([sample.power_metrics.any_violation for sample in samples])
        ),
        mean_active_elements=float(np.mean([sample.active_elements for sample in samples])),
        mean_projection_scale=float(np.mean([sample.projection_scale for sample in samples])),
        mean_post_reconciliation_kdr=mean_post_kdr,
        cvar_post_reconciliation_kdr=cvar_post_kdr,
        worst_post_reconciliation_kdr=worst_post_kdr,
    )
