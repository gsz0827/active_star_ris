from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .surface import (
    EnergySplit,
    SurfaceCoefficients,
    build_surface_coefficients,
)
from .surface_power import (
    RobustAmplitudeVectorProjection,
    project_active_amplitude_vector_robust,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ActionLayout:
    """连续动作向量在各控制量中的切片布局。

    对于N个STAR-RIS单元和N_a个预装有源单元，动作维数为：

        N_a + 3N

    顺序依次为：

    1. 有源单元候选放大增益；
    2. N个透射相移；
    3. N个反射相移；
    4. N个透射能量分配系数。

    所有动作在进入本模块前均约定为[-1, 1]。
    """

    num_elements: int
    active_indices: IntArray

    @property
    def num_active_elements(self) -> int:
        return int(self.active_indices.size)

    @property
    def gain_slice(self) -> slice:
        return slice(0, self.num_active_elements)

    @property
    def phase_transmission_slice(self) -> slice:
        start = self.gain_slice.stop
        return slice(start, start + self.num_elements)

    @property
    def phase_reflection_slice(self) -> slice:
        start = self.phase_transmission_slice.stop
        return slice(start, start + self.num_elements)

    @property
    def beta_transmission_slice(self) -> slice:
        start = self.phase_reflection_slice.stop
        return slice(start, start + self.num_elements)

    @property
    def action_dimension(self) -> int:
        return int(self.beta_transmission_slice.stop)


@dataclass(frozen=True)
class ActionMappingConfig:
    """动作物理映射与鲁棒功率投影配置。"""

    maximum_active_amplitude: float = 3.0
    beta_min: float = 0.05
    beta_max: float = 0.95

    controller_pilot_power: float = 1.0
    transmission_user_pilot_power: float = 1.0
    reflection_user_pilot_power: float = 1.0
    ris_internal_noise_variance: float = 0.002
    output_power_budget: float = 35.0

    nmse_db: float = -15.0
    robust_margin_multiplier: float = 3.0

    transmission_weight: float = 0.5
    reflection_weight: float = 0.5

    # 当预装有源集合在单位增益下都不满足功率约束时，
    # 是否允许将部分有源单元旁路为无源模式。
    allow_active_bypass: bool = True

    def validate(self) -> None:
        if self.maximum_active_amplitude < 1.0:
            raise ValueError(
                "maximum_active_amplitude must be at least 1"
            )
        if not 0.0 <= self.beta_min < self.beta_max <= 1.0:
            raise ValueError(
                "beta bounds must satisfy "
                "0 <= beta_min < beta_max <= 1"
            )

        nonnegative = {
            "controller_pilot_power": self.controller_pilot_power,
            "transmission_user_pilot_power": (
                self.transmission_user_pilot_power
            ),
            "reflection_user_pilot_power": (
                self.reflection_user_pilot_power
            ),
            "ris_internal_noise_variance": (
                self.ris_internal_noise_variance
            ),
            "output_power_budget": self.output_power_budget,
            "robust_margin_multiplier": (
                self.robust_margin_multiplier
            ),
            "transmission_weight": self.transmission_weight,
            "reflection_weight": self.reflection_weight,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")

        if (
            self.transmission_weight
            + self.reflection_weight
            <= 0.0
        ):
            raise ValueError(
                "at least one user weight must be positive"
            )
        if not np.isfinite(self.nmse_db):
            raise ValueError("nmse_db must be finite")


@dataclass(frozen=True)
class ActionProjectionResult:
    """归一化动作映射到可执行STAR-RIS控制量的结果。"""

    surface: SurfaceCoefficients
    layout: ActionLayout

    clipped_action: FloatArray
    requested_amplitudes: FloatArray
    projected_amplitudes: FloatArray

    requested_active_mask: BoolArray
    effective_active_mask: BoolArray
    bypassed_indices: IntArray

    phase_transmission_rad: FloatArray
    phase_reflection_rad: FloatArray
    beta_transmission: FloatArray

    projection_scale: float
    maximum_robust_output_upper: float
    is_robustly_feasible: bool

    @property
    def requested_active_elements(self) -> int:
        return int(np.sum(self.requested_active_mask))

    @property
    def effective_active_elements(self) -> int:
        return int(np.sum(self.effective_active_mask))



def build_action_layout(active_mask: ArrayLike) -> ActionLayout:
    """根据预装有源单元掩码构造连续动作布局。"""
    mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        raise ValueError("active_mask cannot be empty")

    return ActionLayout(
        num_elements=int(mask.size),
        active_indices=np.flatnonzero(mask).astype(np.int64),
    )



def action_dimension(active_mask: ArrayLike) -> int:
    """返回给定有源掩码对应的连续动作维数。"""
    return build_action_layout(active_mask).action_dimension



def _prepare_channel(
    channel: ArrayLike,
    expected_size: int,
    name: str,
) -> ComplexArray:
    values = np.asarray(
        channel,
        dtype=np.complex128,
    ).reshape(-1)
    if values.size != expected_size:
        raise ValueError(
            f"{name} must contain {expected_size} entries"
        )
    return values



def _decode_normalized_action(
    action: ArrayLike,
    layout: ActionLayout,
    requested_active_mask: BoolArray,
    config: ActionMappingConfig,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size != layout.action_dimension:
        raise ValueError(
            "action has incorrect length: "
            f"expected {layout.action_dimension}, "
            f"got {values.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("action entries must be finite")

    clipped = np.clip(values, -1.0, 1.0)
    n = layout.num_elements

    requested_amplitudes = np.ones(n, dtype=np.float64)
    normalized_gain = clipped[layout.gain_slice]
    requested_amplitudes[layout.active_indices] = (
        1.0
        + 0.5
        * (normalized_gain + 1.0)
        * (config.maximum_active_amplitude - 1.0)
    )
    requested_amplitudes[~requested_active_mask] = 1.0

    phase_t = np.mod(
        np.pi
        * (
            clipped[layout.phase_transmission_slice]
            + 1.0
        ),
        2.0 * np.pi,
    )
    phase_r = np.mod(
        np.pi
        * (
            clipped[layout.phase_reflection_slice]
            + 1.0
        ),
        2.0 * np.pi,
    )

    normalized_beta = clipped[
        layout.beta_transmission_slice
    ]
    beta_t = (
        config.beta_min
        + 0.5
        * (normalized_beta + 1.0)
        * (config.beta_max - config.beta_min)
    )

    return (
        clipped,
        requested_amplitudes,
        np.asarray(phase_t, dtype=np.float64),
        np.asarray(phase_r, dtype=np.float64),
        np.asarray(beta_t, dtype=np.float64),
    )



def _active_value_to_burden_ratio(
    controller_to_ris_estimate: ComplexArray,
    transmission_user_to_ris_estimate: ComplexArray,
    reflection_user_to_ris_estimate: ComplexArray,
    active_mask: BoolArray,
    config: ActionMappingConfig,
) -> FloatArray:
    """计算有源单元旁路优先级。

    该分数近似表示单元的合法链路价值与三方向输入功率负担之比。
    当单位增益下仍超出预算时，优先旁路分数最低的有源单元。
    """
    total_weight = (
        config.transmission_weight
        + config.reflection_weight
    )
    wt = config.transmission_weight / total_weight
    wr = config.reflection_weight / total_weight

    utility = (
        np.abs(controller_to_ris_estimate) ** 2
        * (
            wt
            * np.abs(
                transmission_user_to_ris_estimate
            ) ** 2
            + wr
            * np.abs(
                reflection_user_to_ris_estimate
            ) ** 2
        )
    )

    burden = np.maximum.reduce(
        (
            config.controller_pilot_power
            * np.abs(controller_to_ris_estimate) ** 2,
            config.transmission_user_pilot_power
            * np.abs(
                transmission_user_to_ris_estimate
            ) ** 2,
            config.reflection_user_pilot_power
            * np.abs(
                reflection_user_to_ris_estimate
            ) ** 2,
        )
    )
    burden = burden + config.ris_internal_noise_variance

    ratio = np.full(
        active_mask.size,
        np.inf,
        dtype=np.float64,
    )
    ratio[active_mask] = (
        utility[active_mask]
        / np.maximum(burden[active_mask], 1.0e-15)
    )
    return ratio



def _project_with_optional_bypass(
    controller_to_ris_estimate: ComplexArray,
    transmission_user_to_ris_estimate: ComplexArray,
    reflection_user_to_ris_estimate: ComplexArray,
    requested_amplitudes: FloatArray,
    requested_active_mask: BoolArray,
    config: ActionMappingConfig,
) -> tuple[
    RobustAmplitudeVectorProjection,
    BoolArray,
    IntArray,
]:
    effective_mask = np.asarray(
        requested_active_mask,
        dtype=bool,
    ).copy()

    bypassed: list[int] = []
    ratio = _active_value_to_burden_ratio(
        controller_to_ris_estimate,
        transmission_user_to_ris_estimate,
        reflection_user_to_ris_estimate,
        effective_mask,
        config,
    )

    while True:
        projection = project_active_amplitude_vector_robust(
            controller_to_ris_estimate=(
                controller_to_ris_estimate
            ),
            transmission_user_to_ris_estimate=(
                transmission_user_to_ris_estimate
            ),
            reflection_user_to_ris_estimate=(
                reflection_user_to_ris_estimate
            ),
            requested_amplitudes=requested_amplitudes,
            active_mask=effective_mask,
            controller_pilot_power=(
                config.controller_pilot_power
            ),
            transmission_user_pilot_power=(
                config.transmission_user_pilot_power
            ),
            reflection_user_pilot_power=(
                config.reflection_user_pilot_power
            ),
            ris_internal_noise_variance=(
                config.ris_internal_noise_variance
            ),
            output_power_budget=(
                config.output_power_budget
            ),
            maximum_active_amplitude=(
                config.maximum_active_amplitude
            ),
            nmse_db=config.nmse_db,
            robust_margin_multiplier=(
                config.robust_margin_multiplier
            ),
        )

        if projection.is_feasible_at_unit_gain:
            return (
                projection,
                effective_mask,
                np.asarray(bypassed, dtype=np.int64),
            )

        if (
            not config.allow_active_bypass
            or not np.any(effective_mask)
        ):
            return (
                projection,
                effective_mask,
                np.asarray(bypassed, dtype=np.int64),
            )

        active_indices = np.flatnonzero(effective_mask)
        if active_indices.size == 0:
            return (
                projection,
                effective_mask,
                np.asarray(bypassed, dtype=np.int64),
            )

        # 旁路价值/功率比最低的单元。
        remove_index = int(
            active_indices[
                np.argmin(ratio[active_indices])
            ]
        )
        effective_mask[remove_index] = False
        bypassed.append(remove_index)



def map_and_project_action(
    action: ArrayLike,
    *,
    active_mask: ArrayLike,
    controller_to_ris_estimate: ArrayLike,
    transmission_user_to_ris_estimate: ArrayLike,
    reflection_user_to_ris_estimate: ArrayLike,
    config: ActionMappingConfig | None = None,
) -> ActionProjectionResult:
    """将TD3/SAC输出映射为满足约束的STAR-RIS控制量。

    该函数只使用估计CSI执行鲁棒功率投影。后续环境应把返回的
    ``surface`` 传入 ``evaluate_joint_objective``，由真实信道、
    实际硬件失配和前后向独立噪声计算奖励。
    """
    mapping_config = (
        ActionMappingConfig()
        if config is None
        else config
    )
    mapping_config.validate()

    requested_mask = np.asarray(
        active_mask,
        dtype=bool,
    ).reshape(-1)
    if requested_mask.size == 0:
        raise ValueError("active_mask cannot be empty")

    layout = build_action_layout(requested_mask)
    n = layout.num_elements

    g_hat = _prepare_channel(
        controller_to_ris_estimate,
        n,
        "controller_to_ris_estimate",
    )
    h_t_hat = _prepare_channel(
        transmission_user_to_ris_estimate,
        n,
        "transmission_user_to_ris_estimate",
    )
    h_r_hat = _prepare_channel(
        reflection_user_to_ris_estimate,
        n,
        "reflection_user_to_ris_estimate",
    )

    (
        clipped_action,
        requested_amplitudes,
        phase_t,
        phase_r,
        beta_t,
    ) = _decode_normalized_action(
        action,
        layout,
        requested_mask,
        mapping_config,
    )

    (
        projection,
        effective_mask,
        bypassed_indices,
    ) = _project_with_optional_bypass(
        controller_to_ris_estimate=g_hat,
        transmission_user_to_ris_estimate=h_t_hat,
        reflection_user_to_ris_estimate=h_r_hat,
        requested_amplitudes=requested_amplitudes,
        requested_active_mask=requested_mask,
        config=mapping_config,
    )

    projected_amplitudes = np.asarray(
        projection.projected_amplitudes,
        dtype=np.float64,
    ).copy()
    projected_amplitudes[~effective_mask] = 1.0

    split = EnergySplit.from_transmission(
        beta_t,
        n,
    )
    surface = build_surface_coefficients(
        energy_split=split,
        phase_transmission_rad=phase_t,
        phase_reflection_rad=phase_r,
        amplitude_gain=projected_amplitudes,
        active_mask=effective_mask,
    )

    feasible = bool(
        projection.is_feasible_at_unit_gain
        and projection.maximum_robust_output_upper
        <= mapping_config.output_power_budget
        + 1.0e-9
    )

    return ActionProjectionResult(
        surface=surface,
        layout=layout,
        clipped_action=np.asarray(
            clipped_action,
            dtype=np.float64,
        ),
        requested_amplitudes=np.asarray(
            requested_amplitudes,
            dtype=np.float64,
        ),
        projected_amplitudes=(
            projected_amplitudes
        ),
        requested_active_mask=requested_mask,
        effective_active_mask=np.asarray(
            effective_mask,
            dtype=bool,
        ),
        bypassed_indices=np.asarray(
            bypassed_indices,
            dtype=np.int64,
        ),
        phase_transmission_rad=phase_t,
        phase_reflection_rad=phase_r,
        beta_transmission=beta_t,
        projection_scale=float(
            projection.projection_scale
        ),
        maximum_robust_output_upper=float(
            projection.maximum_robust_output_upper
        ),
        is_robustly_feasible=feasible,
    )
