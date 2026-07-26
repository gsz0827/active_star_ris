from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .action_mapping import (
    ActionMappingConfig,
    ActionProjectionResult,
    action_dimension,
    map_and_project_action,
)
from .channels import ChannelConfig, complex_normal, generate_channel
from .csi_estimation import generate_imperfect_csi
from .hardware_impairments import HardwareMismatchParameters
from .joint_objective import (
    JointObjectiveConfig,
    JointObjectiveResult,
    evaluate_joint_objective,
)

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ContinuousBox:
    """Minimal continuous box space used by the TD3 environment.

    The class intentionally avoids a hard dependency on Gymnasium while
    providing the small API needed by a TD3 implementation: ``shape``,
    ``sample`` and ``contains``.
    """

    low: Float32Array
    high: Float32Array

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=np.float32)
        high = np.asarray(self.high, dtype=np.float32)
        if low.shape != high.shape:
            raise ValueError("low and high must have the same shape")
        if low.size == 0:
            raise ValueError("box space cannot be empty")
        if np.any(low > high):
            raise ValueError("each low bound must not exceed high")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape

    def sample(self, rng: np.random.Generator | None = None) -> Float32Array:
        generator = np.random.default_rng() if rng is None else rng
        return np.asarray(
            generator.uniform(self.low, self.high),
            dtype=np.float32,
        )

    def contains(self, value: ArrayLike) -> bool:
        array = np.asarray(value, dtype=np.float32)
        return bool(
            array.shape == self.shape
            and np.all(np.isfinite(array))
            and np.all(array >= self.low)
            and np.all(array <= self.high)
        )


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Episode-level uncertainty ranges used for robust training."""

    nmse_db_min: float = -25.0
    nmse_db_max: float = -10.0

    ris_internal_noise_variance_min: float = 0.001
    ris_internal_noise_variance_max: float = 0.004
    receiver_noise_variance_min: float = 0.005
    receiver_noise_variance_max: float = 0.020
    output_power_budget_min: float = 30.0
    output_power_budget_max: float = 40.0

    static_gain_error_std_db_min: float = 0.10
    static_gain_error_std_db_max: float = 0.30
    directional_gain_error_std_db_min: float = 0.05
    directional_gain_error_std_db_max: float = 0.15
    static_phase_error_std_rad_min: float = 0.02
    static_phase_error_std_rad_max: float = 0.08
    directional_phase_error_std_rad_min: float = 0.01
    directional_phase_error_std_rad_max: float = 0.05

    def validate(self) -> None:
        pairs = {
            "nmse_db": (self.nmse_db_min, self.nmse_db_max),
            "ris_internal_noise_variance": (
                self.ris_internal_noise_variance_min,
                self.ris_internal_noise_variance_max,
            ),
            "receiver_noise_variance": (
                self.receiver_noise_variance_min,
                self.receiver_noise_variance_max,
            ),
            "output_power_budget": (
                self.output_power_budget_min,
                self.output_power_budget_max,
            ),
            "static_gain_error_std_db": (
                self.static_gain_error_std_db_min,
                self.static_gain_error_std_db_max,
            ),
            "directional_gain_error_std_db": (
                self.directional_gain_error_std_db_min,
                self.directional_gain_error_std_db_max,
            ),
            "static_phase_error_std_rad": (
                self.static_phase_error_std_rad_min,
                self.static_phase_error_std_rad_max,
            ),
            "directional_phase_error_std_rad": (
                self.directional_phase_error_std_rad_min,
                self.directional_phase_error_std_rad_max,
            ),
        }
        for name, (lower, upper) in pairs.items():
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"{name} bounds must be finite")
            if lower > upper:
                raise ValueError(f"{name}_min must not exceed {name}_max")

        nonnegative = {
            "ris_internal_noise_variance_min": (
                self.ris_internal_noise_variance_min
            ),
            "receiver_noise_variance_min": self.receiver_noise_variance_min,
            "output_power_budget_min": self.output_power_budget_min,
            "static_gain_error_std_db_min": self.static_gain_error_std_db_min,
            "directional_gain_error_std_db_min": (
                self.directional_gain_error_std_db_min
            ),
            "static_phase_error_std_rad_min": (
                self.static_phase_error_std_rad_min
            ),
            "directional_phase_error_std_rad_min": (
                self.directional_phase_error_std_rad_min
            ),
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.output_power_budget_min <= 0.0:
            raise ValueError("output_power_budget_min must be positive")


@dataclass(frozen=True)
class EpisodeDomain:
    """One sampled uncertainty realization held fixed within an episode."""

    nmse_db: float
    ris_internal_noise_variance: float
    receiver_noise_variance: float
    output_power_budget: float
    hardware_parameters: HardwareMismatchParameters


@dataclass(frozen=True)
class RobustEnvironmentConfig:
    """Configuration for the robust partially-active STAR-RIS environment."""

    num_elements: int = 32
    num_active_elements: int = 8
    max_episode_steps: int = 64
    probing_samples_per_step: int = 128

    # Correlation between consecutive environment decisions and between
    # samples inside one bidirectional probing block.
    channel_temporal_correlation: float = 0.95
    probing_sample_correlation: float = 0.995

    controller_to_ris: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(
            model="rician",
            large_scale_power=1.0,
            k_factor_db=5.0,
        )
    )
    ris_to_transmission_user: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(
            model="rician",
            large_scale_power=1.0,
            k_factor_db=3.0,
        )
    )
    ris_to_reflection_user: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(
            model="rician",
            large_scale_power=1.0,
            k_factor_db=3.0,
        )
    )
    direct_transmission: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(
            model="rayleigh",
            large_scale_power=0.05,
        )
    )
    direct_reflection: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(
            model="rayleigh",
            large_scale_power=0.05,
        )
    )

    controller_pilot_power: float = 1.0
    transmission_user_pilot_power: float = 1.0
    reflection_user_pilot_power: float = 1.0

    amplifier_efficiency: float = 0.35
    controller_static_power: float = 0.10
    active_element_bias_power: float = 0.01

    transmission_weight: float = 0.5
    reflection_weight: float = 0.5

    maximum_active_amplitude: float = 3.0
    beta_min: float = 0.05
    beta_max: float = 0.95
    robust_margin_multiplier: float = 3.0
    allow_active_bypass: bool = True

    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    objective: JointObjectiveConfig = field(
        default_factory=JointObjectiveConfig
    )

    # Observation composition.
    state_clip: float = 5.0
    include_impairment_context: bool = True
    include_previous_metrics: bool = True

    def validate(self) -> None:
        if self.num_elements <= 0:
            raise ValueError("num_elements must be positive")
        if not 0 <= self.num_active_elements <= self.num_elements:
            raise ValueError(
                "num_active_elements must lie within [0, num_elements]"
            )
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.probing_samples_per_step < 2:
            raise ValueError("probing_samples_per_step must be at least 2")
        for name, correlation in {
            "channel_temporal_correlation": self.channel_temporal_correlation,
            "probing_sample_correlation": self.probing_sample_correlation,
        }.items():
            if not 0.0 <= correlation <= 1.0:
                raise ValueError(f"{name} must lie within [0, 1]")
        for channel in (
            self.controller_to_ris,
            self.ris_to_transmission_user,
            self.ris_to_reflection_user,
            self.direct_transmission,
            self.direct_reflection,
        ):
            channel.validate()
        positive = {
            "controller_pilot_power": self.controller_pilot_power,
            "transmission_user_pilot_power": self.transmission_user_pilot_power,
            "reflection_user_pilot_power": self.reflection_user_pilot_power,
            "amplifier_efficiency": self.amplifier_efficiency,
            "state_clip": self.state_clip,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        nonnegative = {
            "controller_static_power": self.controller_static_power,
            "active_element_bias_power": self.active_element_bias_power,
            "transmission_weight": self.transmission_weight,
            "reflection_weight": self.reflection_weight,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.transmission_weight + self.reflection_weight <= 0.0:
            raise ValueError("at least one branch weight must be positive")
        if self.maximum_active_amplitude < 1.0:
            raise ValueError("maximum_active_amplitude must be at least 1")
        if not 0.0 <= self.beta_min < self.beta_max <= 1.0:
            raise ValueError(
                "beta bounds must satisfy 0 <= beta_min < beta_max <= 1"
            )
        if self.robust_margin_multiplier < 0.0:
            raise ValueError("robust_margin_multiplier cannot be negative")
        self.domain_randomization.validate()
        self.objective.validate()


@dataclass(frozen=True)
class EnvironmentChannels:
    controller_to_ris: ComplexArray
    ris_to_transmission_user: ComplexArray
    ris_to_reflection_user: ComplexArray
    direct_transmission: complex
    direct_reflection: complex


@dataclass(frozen=True)
class EnvironmentEstimates:
    controller_to_ris: ComplexArray
    ris_to_transmission_user: ComplexArray
    ris_to_reflection_user: ComplexArray
    direct_transmission: complex
    direct_reflection: complex


@dataclass(frozen=True)
class EnvironmentStepDiagnostics:
    projection: ActionProjectionResult
    objective: JointObjectiveResult
    episode_domain: EpisodeDomain


class RobustActiveStarRISEnv:
    """Gymnasium-style environment directly usable by a TD3 implementation.

    ``reset`` returns ``(observation, info)`` and ``step`` returns
    ``(observation, reward, terminated, truncated, info)``.  No external RL
    library is required at this stage.
    """

    metadata: Mapping[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: RobustEnvironmentConfig | None = None,
        *,
        active_mask: ArrayLike | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = RobustEnvironmentConfig() if config is None else config
        self.config.validate()
        self._initial_seed = seed
        self._rng = np.random.default_rng(seed)
        self.active_mask = self._prepare_active_mask(active_mask)

        self.action_dim = action_dimension(self.active_mask)
        self.state_dim = self._calculate_state_dimension()
        self.action_space = ContinuousBox(
            low=np.full(self.action_dim, -1.0, dtype=np.float32),
            high=np.full(self.action_dim, 1.0, dtype=np.float32),
        )
        self.observation_space = ContinuousBox(
            low=np.full(
                self.state_dim,
                -self.config.state_clip,
                dtype=np.float32,
            ),
            high=np.full(
                self.state_dim,
                self.config.state_clip,
                dtype=np.float32,
            ),
        )

        self._channels: EnvironmentChannels | None = None
        self._estimates: EnvironmentEstimates | None = None
        self._episode_domain: EpisodeDomain | None = None

        # 同一episode内固定硬件失配 realization。
        self._hardware_seed: int | None = None

        self._step_index = 0
        self._previous_metrics = np.zeros(5, dtype=np.float64)
        self._last_diagnostics: EnvironmentStepDiagnostics | None = None

    @property
    def unwrapped(self) -> "RobustActiveStarRISEnv":
        return self

    @property
    def current_domain(self) -> EpisodeDomain:
        if self._episode_domain is None:
            raise RuntimeError("environment must be reset before use")
        return self._episode_domain

    @property
    def last_diagnostics(self) -> EnvironmentStepDiagnostics | None:
        return self._last_diagnostics

    def _prepare_active_mask(self, active_mask: ArrayLike | None) -> BoolArray:
        n = self.config.num_elements
        if active_mask is not None:
            mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if mask.size != n:
                raise ValueError(
                    f"active_mask must contain {n} entries"
                )
            return mask

        mask = np.zeros(n, dtype=bool)
        count = self.config.num_active_elements
        if count > 0:
            # Even spacing avoids introducing an artificial preference for
            # the first elements when no array geometry is modeled.
            indices = np.floor(
                np.arange(count, dtype=np.float64) * n / count
            ).astype(np.int64)
            mask[np.unique(indices)] = True
            # The formula is exact for count <= n, but keep this defensive
            # fallback for future changes.
            if int(np.sum(mask)) < count:
                remaining = np.flatnonzero(~mask)[: count - int(np.sum(mask))]
                mask[remaining] = True
        return mask

    def _calculate_state_dimension(self) -> int:
        # Three N-element complex estimated RIS links and two scalar complex
        # direct links.
        dimension = 6 * self.config.num_elements + 4
        if self.config.include_impairment_context:
            dimension += 8
        if self.config.include_previous_metrics:
            dimension += 5
        return dimension

    @staticmethod
    def _sample_uniform(
        rng: np.random.Generator,
        lower: float,
        upper: float,
    ) -> float:
        if lower == upper:
            return float(lower)
        return float(rng.uniform(lower, upper))

    def _sample_episode_domain(self) -> EpisodeDomain:
        dr = self.config.domain_randomization
        hardware = HardwareMismatchParameters(
            static_gain_error_std_db=self._sample_uniform(
                self._rng,
                dr.static_gain_error_std_db_min,
                dr.static_gain_error_std_db_max,
            ),
            directional_gain_error_std_db=self._sample_uniform(
                self._rng,
                dr.directional_gain_error_std_db_min,
                dr.directional_gain_error_std_db_max,
            ),
            static_phase_error_std_rad=self._sample_uniform(
                self._rng,
                dr.static_phase_error_std_rad_min,
                dr.static_phase_error_std_rad_max,
            ),
            directional_phase_error_std_rad=self._sample_uniform(
                self._rng,
                dr.directional_phase_error_std_rad_min,
                dr.directional_phase_error_std_rad_max,
            ),
        )
        return EpisodeDomain(
            nmse_db=self._sample_uniform(
                self._rng,
                dr.nmse_db_min,
                dr.nmse_db_max,
            ),
            ris_internal_noise_variance=self._sample_uniform(
                self._rng,
                dr.ris_internal_noise_variance_min,
                dr.ris_internal_noise_variance_max,
            ),
            receiver_noise_variance=self._sample_uniform(
                self._rng,
                dr.receiver_noise_variance_min,
                dr.receiver_noise_variance_max,
            ),
            output_power_budget=self._sample_uniform(
                self._rng,
                dr.output_power_budget_min,
                dr.output_power_budget_max,
            ),
            hardware_parameters=hardware,
        )

    def _sample_initial_channels(self) -> EnvironmentChannels:
        n = self.config.num_elements
        return EnvironmentChannels(
            controller_to_ris=generate_channel(
                n, self._rng, self.config.controller_to_ris
            ),
            ris_to_transmission_user=generate_channel(
                n, self._rng, self.config.ris_to_transmission_user
            ),
            ris_to_reflection_user=generate_channel(
                n, self._rng, self.config.ris_to_reflection_user
            ),
            direct_transmission=complex(
                generate_channel(1, self._rng, self.config.direct_transmission)[0]
            ),
            direct_reflection=complex(
                generate_channel(1, self._rng, self.config.direct_reflection)[0]
            ),
        )

    @staticmethod
    def _rician_mean_and_innovation_variance(
        channel_config: ChannelConfig,
        shape: tuple[int, ...],
    ) -> tuple[ComplexArray, float]:
        power = float(channel_config.large_scale_power)
        if channel_config.model == "rayleigh":
            return np.zeros(shape, dtype=np.complex128), power
        k_linear = float(10.0 ** (channel_config.k_factor_db / 10.0))
        mean_amplitude = np.sqrt(power * k_linear / (k_linear + 1.0))
        innovation_variance = power / (k_linear + 1.0)
        return (
            np.full(shape, mean_amplitude, dtype=np.complex128),
            innovation_variance,
        )

    def _evolve_channel(
        self,
        current: ArrayLike,
        channel_config: ChannelConfig,
        correlation: float,
    ) -> ComplexArray:
        values = np.asarray(current, dtype=np.complex128)
        mean, innovation_variance = self._rician_mean_and_innovation_variance(
            channel_config, values.shape
        )
        innovation = complex_normal(
            values.shape,
            self._rng,
            variance=innovation_variance,
        )
        next_values = (
            mean
            + correlation * (values - mean)
            + np.sqrt(max(0.0, 1.0 - correlation**2)) * innovation
        )
        return np.asarray(next_values, dtype=np.complex128)

    def _generate_probe_sequence(
        self,
        initial: ArrayLike,
        channel_config: ChannelConfig,
    ) -> ComplexArray:
        samples = self.config.probing_samples_per_step
        first = np.asarray(initial, dtype=np.complex128)
        sequence = np.empty((samples,) + first.shape, dtype=np.complex128)
        sequence[0] = first
        for index in range(1, samples):
            sequence[index] = self._evolve_channel(
                sequence[index - 1],
                channel_config,
                self.config.probing_sample_correlation,
            )
        return sequence

    def _make_estimates(self) -> EnvironmentEstimates:
        if self._channels is None or self._episode_domain is None:
            raise RuntimeError("environment must be reset before use")
        nmse_db = self._episode_domain.nmse_db

        def estimate_vector(values: ArrayLike) -> ComplexArray:
            return generate_imperfect_csi(
                values, nmse_db, self._rng
            ).estimated_channel

        return EnvironmentEstimates(
            controller_to_ris=estimate_vector(
                self._channels.controller_to_ris
            ),
            ris_to_transmission_user=estimate_vector(
                self._channels.ris_to_transmission_user
            ),
            ris_to_reflection_user=estimate_vector(
                self._channels.ris_to_reflection_user
            ),
            direct_transmission=complex(
                estimate_vector([self._channels.direct_transmission])[0]
            ),
            direct_reflection=complex(
                estimate_vector([self._channels.direct_reflection])[0]
            ),
        )

    @staticmethod
    def _range_normalize(value: float, lower: float, upper: float) -> float:
        if upper == lower:
            return 0.0
        return float(2.0 * (value - lower) / (upper - lower) - 1.0)

    def _normalized_complex_vector(
        self,
        values: ArrayLike,
        channel_config: ChannelConfig,
    ) -> FloatArray:
        array = np.asarray(values, dtype=np.complex128).reshape(-1)
        reference = max(np.sqrt(channel_config.large_scale_power), 1.0e-12)
        normalized = array / reference
        return np.concatenate((normalized.real, normalized.imag))

    def _build_observation(self) -> Float32Array:
        if self._estimates is None or self._episode_domain is None:
            raise RuntimeError("environment must be reset before use")
        pieces: list[FloatArray] = [
            self._normalized_complex_vector(
                self._estimates.controller_to_ris,
                self.config.controller_to_ris,
            ),
            self._normalized_complex_vector(
                self._estimates.ris_to_transmission_user,
                self.config.ris_to_transmission_user,
            ),
            self._normalized_complex_vector(
                self._estimates.ris_to_reflection_user,
                self.config.ris_to_reflection_user,
            ),
            self._normalized_complex_vector(
                [self._estimates.direct_transmission],
                self.config.direct_transmission,
            ),
            self._normalized_complex_vector(
                [self._estimates.direct_reflection],
                self.config.direct_reflection,
            ),
        ]

        if self.config.include_impairment_context:
            dr = self.config.domain_randomization
            hp = self._episode_domain.hardware_parameters
            context = np.asarray(
                [
                    self._range_normalize(
                        self._episode_domain.nmse_db,
                        dr.nmse_db_min,
                        dr.nmse_db_max,
                    ),
                    self._range_normalize(
                        self._episode_domain.ris_internal_noise_variance,
                        dr.ris_internal_noise_variance_min,
                        dr.ris_internal_noise_variance_max,
                    ),
                    self._range_normalize(
                        self._episode_domain.receiver_noise_variance,
                        dr.receiver_noise_variance_min,
                        dr.receiver_noise_variance_max,
                    ),
                    self._range_normalize(
                        self._episode_domain.output_power_budget,
                        dr.output_power_budget_min,
                        dr.output_power_budget_max,
                    ),
                    self._range_normalize(
                        hp.static_gain_error_std_db,
                        dr.static_gain_error_std_db_min,
                        dr.static_gain_error_std_db_max,
                    ),
                    self._range_normalize(
                        hp.directional_gain_error_std_db,
                        dr.directional_gain_error_std_db_min,
                        dr.directional_gain_error_std_db_max,
                    ),
                    self._range_normalize(
                        hp.static_phase_error_std_rad,
                        dr.static_phase_error_std_rad_min,
                        dr.static_phase_error_std_rad_max,
                    ),
                    self._range_normalize(
                        hp.directional_phase_error_std_rad,
                        dr.directional_phase_error_std_rad_min,
                        dr.directional_phase_error_std_rad_max,
                    ),
                ],
                dtype=np.float64,
            )
            pieces.append(context)

        if self.config.include_previous_metrics:
            pieces.append(self._previous_metrics)

        observation = np.concatenate(pieces)
        if observation.size != self.state_dim:
            raise RuntimeError(
                f"internal state dimension mismatch: expected {self.state_dim}, "
                f"got {observation.size}"
            )
        observation = np.nan_to_num(
            observation,
            nan=0.0,
            posinf=self.config.state_clip,
            neginf=-self.config.state_clip,
        )
        return np.asarray(
            np.clip(
                observation,
                -self.config.state_clip,
                self.config.state_clip,
            ),
            dtype=np.float32,
        )

    def _mapping_config(self) -> ActionMappingConfig:
        domain = self.current_domain
        return ActionMappingConfig(
            maximum_active_amplitude=self.config.maximum_active_amplitude,
            beta_min=self.config.beta_min,
            beta_max=self.config.beta_max,
            controller_pilot_power=self.config.controller_pilot_power,
            transmission_user_pilot_power=(
                self.config.transmission_user_pilot_power
            ),
            reflection_user_pilot_power=(
                self.config.reflection_user_pilot_power
            ),
            ris_internal_noise_variance=domain.ris_internal_noise_variance,
            output_power_budget=domain.output_power_budget,
            nmse_db=domain.nmse_db,
            robust_margin_multiplier=self.config.robust_margin_multiplier,
            transmission_weight=self.config.transmission_weight,
            reflection_weight=self.config.reflection_weight,
            allow_active_bypass=self.config.allow_active_bypass,
        )

    def _probing_blocks(self) -> tuple[
        ComplexArray,
        ComplexArray,
        ComplexArray,
        ComplexArray,
        ComplexArray,
    ]:
        if self._channels is None:
            raise RuntimeError("environment must be reset before use")
        g = self._generate_probe_sequence(
            self._channels.controller_to_ris,
            self.config.controller_to_ris,
        )
        h_t = self._generate_probe_sequence(
            self._channels.ris_to_transmission_user,
            self.config.ris_to_transmission_user,
        )
        h_r = self._generate_probe_sequence(
            self._channels.ris_to_reflection_user,
            self.config.ris_to_reflection_user,
        )
        d_t = self._generate_probe_sequence(
            np.asarray(self._channels.direct_transmission),
            self.config.direct_transmission,
        ).reshape(-1)
        d_r = self._generate_probe_sequence(
            np.asarray(self._channels.direct_reflection),
            self.config.direct_reflection,
        ).reshape(-1)
        return g, h_t, h_r, d_t, d_r

    def _advance_channels(self) -> None:
        if self._channels is None:
            raise RuntimeError("environment must be reset before use")
        rho = self.config.channel_temporal_correlation
        self._channels = EnvironmentChannels(
            controller_to_ris=self._evolve_channel(
                self._channels.controller_to_ris,
                self.config.controller_to_ris,
                rho,
            ),
            ris_to_transmission_user=self._evolve_channel(
                self._channels.ris_to_transmission_user,
                self.config.ris_to_transmission_user,
                rho,
            ),
            ris_to_reflection_user=self._evolve_channel(
                self._channels.ris_to_reflection_user,
                self.config.ris_to_reflection_user,
                rho,
            ),
            direct_transmission=complex(
                self._evolve_channel(
                    np.asarray(self._channels.direct_transmission),
                    self.config.direct_transmission,
                    rho,
                )
            ),
            direct_reflection=complex(
                self._evolve_channel(
                    np.asarray(self._channels.direct_reflection),
                    self.config.direct_reflection,
                    rho,
                )
            ),
        )
        self._estimates = self._make_estimates()

    def _base_info(self) -> dict[str, Any]:
        domain = self.current_domain
        return {
            "step_index": int(self._step_index),
            "nmse_db": float(domain.nmse_db),
            "ris_internal_noise_variance": float(
                domain.ris_internal_noise_variance
            ),
            "receiver_noise_variance": float(
                domain.receiver_noise_variance
            ),
            "output_power_budget": float(domain.output_power_budget),
            "requested_active_elements": int(np.sum(self.active_mask)),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Float32Array, dict[str, Any]]:
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._channels is None and self._initial_seed is not None:
            # The constructor seed controls the first reset. Later resets keep
            # advancing the same RNG stream, matching common RL behavior.
            self._rng = np.random.default_rng(self._initial_seed)
            self._initial_seed = None

        self._episode_domain = self._sample_episode_domain()

        self._hardware_seed = int(
            self._rng.integers(
                low=0,
                high=np.iinfo(np.int64).max,
                dtype=np.int64,
            )
        )

        self._channels = self._sample_initial_channels()
        self._estimates = self._make_estimates()
        self._step_index = 0
        self._previous_metrics = np.zeros(5, dtype=np.float64)
        self._last_diagnostics = None
        observation = self._build_observation()
        return observation, self._base_info()

    def step(
        self,
        action: ArrayLike,
    ) -> tuple[Float32Array, float, bool, bool, dict[str, Any]]:
        if (
            self._channels is None
            or self._estimates is None
            or self._hardware_seed is None
        ):
            raise RuntimeError(
                "reset must be called before step"
            )

        action_array = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)

        if action_array.size != self.action_dim:
            raise ValueError(
                f"action must contain {self.action_dim} entries"
            )

        if not np.all(np.isfinite(action_array)):
            raise ValueError(
                "action must contain only finite values"
            )
        projection = map_and_project_action(
            action_array,
            active_mask=self.active_mask,
            controller_to_ris_estimate=self._estimates.controller_to_ris,
            transmission_user_to_ris_estimate=(
                self._estimates.ris_to_transmission_user
            ),
            reflection_user_to_ris_estimate=(
                self._estimates.ris_to_reflection_user
            ),
            config=self._mapping_config(),
        )

        g, h_t, h_r, d_t, d_r = self._probing_blocks()
        domain = self.current_domain
        objective = evaluate_joint_objective(
            channel_controller_to_ris=g,
            channel_ris_to_transmission_user=h_t,
            channel_ris_to_reflection_user=h_r,
            ideal_surface=projection.surface,
            direct_channel_transmission=d_t,
            direct_channel_reflection=d_r,
            pilot_power_controller=self.config.controller_pilot_power,
            pilot_power_transmission_user=(
                self.config.transmission_user_pilot_power
            ),
            pilot_power_reflection_user=(
                self.config.reflection_user_pilot_power
            ),
            ris_internal_noise_variance=(
                domain.ris_internal_noise_variance
            ),
            receiver_noise_variance_controller=domain.receiver_noise_variance,
            receiver_noise_variance_transmission_user=(
                domain.receiver_noise_variance
            ),
            receiver_noise_variance_reflection_user=(
                domain.receiver_noise_variance
            ),
            output_power_budget=domain.output_power_budget,
            amplifier_efficiency=self.config.amplifier_efficiency,
            controller_static_power=self.config.controller_static_power,
            active_element_bias_power=self.config.active_element_bias_power,
            transmission_weight=self.config.transmission_weight,
            reflection_weight=self.config.reflection_weight,
            hardware_parameters=domain.hardware_parameters,
            hardware_rng=np.random.default_rng(
                self._hardware_seed
            ),
            objective_config=self.config.objective,
            rng=self._rng,
        )

        self._last_diagnostics = EnvironmentStepDiagnostics(
            projection=projection,
            objective=objective,
            episode_domain=domain,
        )
        self._previous_metrics = np.asarray(
            [
                np.clip(objective.reward, -5.0, 5.0) / 5.0,
                np.clip(objective.normalized_key_rate, 0.0, 5.0) / 5.0,
                np.clip(objective.normalized_key_disagreement, 0.0, 2.0)
                - 1.0,
                2.0 * np.clip(objective.normalized_reciprocity, 0.0, 1.0)
                - 1.0,
                np.clip(objective.normalized_surface_power, 0.0, 5.0) / 2.5
                - 1.0,
            ],
            dtype=np.float64,
        )

        self._step_index += 1
        truncated = self._step_index >= self.config.max_episode_steps
        terminated = False
        self._advance_channels()
        observation = self._build_observation()

        info = self._base_info()
        info.update(
            {
                "reward": float(objective.reward),
                "weighted_key_rate": float(
                    objective.key_generation.weighted_mutual_information
                ),
                "weighted_key_disagreement_rate": float(
                    objective.key_generation.weighted_key_disagreement_rate
                ),
                "weighted_reciprocity": float(
                    objective.key_generation.weighted_correlation
                ),
                "total_surface_power": float(
                    objective.surface_power.total_surface_power
                ),
                "maximum_output_power": float(
                    objective.surface_power.maximum_output_power
                ),
                "power_violation": float(
                    objective.surface_power.power_violation
                ),
                "projection_scale": float(projection.projection_scale),
                "robust_output_upper": float(
                    projection.maximum_robust_output_upper
                ),
                "robustly_feasible": bool(
                    projection.is_robustly_feasible
                ),
                "effective_active_elements": int(
                    projection.effective_active_elements
                ),
                "bypassed_indices": projection.bypassed_indices.copy(),
                "star_energy_error": float(
                    projection.surface.maximum_energy_error()
                ),
            }
        )
        return observation, float(objective.reward), terminated, truncated, info

    def sample_action(self) -> Float32Array:
        return self.action_space.sample(self._rng)

    def close(self) -> None:
        return None


def make_small_debug_environment(
    *,
    seed: int = 0,
    num_elements: int = 8,
    num_active_elements: int = 2,
) -> RobustActiveStarRISEnv:
    """Create a fast environment for smoke tests and TD3 debugging."""

    config = RobustEnvironmentConfig(
        num_elements=num_elements,
        num_active_elements=num_active_elements,
        max_episode_steps=8,
        probing_samples_per_step=32,
    )
    return RobustActiveStarRISEnv(config, seed=seed)
