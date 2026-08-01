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


def objective_reward(
    key_metrics: JointKeyMetrics,
    power_metrics: PowerMetrics,
    config: ObjectiveConfig,
    *,
    power_config: PowerConfig,
    hardware_config: HardwareConfig,
    projection_scale: float = 1.0,
) -> float:
    key_term = (
        key_metrics.weighted_secure_key_rate_bps
        / config.key_rate_reference_bps
    )
    key_margin_term = float(
        np.clip(
            key_metrics.weighted_key_margin_bits
            / config.key_margin_reference_bits,
            -2.0,
            2.0,
        )
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

    return float(
        config.key_rate_weight * key_term
        + config.key_margin_weight * key_margin_term
        - config.raw_kdr_weight * raw_kdr_term
        - config.post_reconciliation_kdr_weight * post_kdr_term
        + config.reciprocity_weight * key_metrics.weighted_reciprocity
        - config.surface_power_weight * power_term
        - config.projection_penalty_weight * projection_penalty
        - config.constraint_violation_weight * normalized_violation**2
    )

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
