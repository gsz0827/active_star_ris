from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .surface import (
    EnergySplit,
    SurfaceCoefficients,
    build_surface_coefficients,
    random_phases,
    wrap_phase,
)
from .system import TwoUserMetrics, evaluate_two_user_system
from .surface_power import (
    project_active_amplitude_vector_robust,
    project_common_active_amplitude_robust,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ActiveDesign:
    """部分有源STAR-RIS的一次设计结果。"""

    surface: SurfaceCoefficients
    active_mask: BoolArray
    common_active_amplitude: float
    selected_indices: NDArray[np.int64]

    # 用户或配置中原本要求启用的有源单元数量。
    requested_active_elements: int = 0

    # 经过功率可行域投影后真正启用的数量。
    effective_active_elements: int = 0

    @property
    def disabled_active_elements(self) -> int:
        """因功率约束而关闭的有源单元数量。"""

        return max(
            0,
            int(self.requested_active_elements)
            - int(self.effective_active_elements),
        )

    @property
    def used_passive_fallback(self) -> bool:
        """请求过有源单元，但最终全部退化为无源。"""

        return (
            self.requested_active_elements > 0
            and self.effective_active_elements == 0
        )


def phase_align(
    alice_to_ris: ArrayLike,
    ris_to_user: ArrayLike,
    direct_channel: complex = 0.0j,
) -> FloatArray:
    g = np.asarray(alice_to_ris, dtype=np.complex128).reshape(-1)
    h = np.asarray(ris_to_user, dtype=np.complex128).reshape(-1)
    if g.size != h.size:
        raise ValueError("channel vectors must have equal length")

    target_phase = 0.0 if abs(direct_channel) == 0.0 else np.angle(direct_channel)
    return wrap_phase(target_phase - np.angle(h * g))


def select_active_elements(
    alice_to_ris: ArrayLike,
    ris_to_transmission_user: ArrayLike,
    ris_to_reflection_user: ArrayLike,
    num_active_elements: int,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
) -> tuple[BoolArray, NDArray[np.int64]]:
    g = np.asarray(alice_to_ris, dtype=np.complex128).reshape(-1)
    h_t = np.asarray(
        ris_to_transmission_user,
        dtype=np.complex128,
    ).reshape(-1)
    h_r = np.asarray(
        ris_to_reflection_user,
        dtype=np.complex128,
    ).reshape(-1)

    if not (g.size == h_t.size == h_r.size):
        raise ValueError("channel vectors must have equal length")
    if not 0 <= num_active_elements <= g.size:
        raise ValueError("num_active_elements is outside the valid range")

    total_weight = transmission_weight + reflection_weight
    if total_weight <= 0:
        raise ValueError("at least one weight must be positive")
    wt = transmission_weight / total_weight
    wr = reflection_weight / total_weight

    score = np.abs(g) ** 2 * (
        wt * np.abs(h_t) ** 2
        + wr * np.abs(h_r) ** 2
    )
    mask = np.zeros(g.size, dtype=bool)
    if num_active_elements == 0:
        return mask, np.empty(0, dtype=np.int64)

    selected = np.argsort(score)[-num_active_elements:]
    mask[selected] = True
    return mask, np.asarray(selected, dtype=np.int64)


def feasible_common_active_amplitude(
    alice_to_ris: ArrayLike,
    active_mask: ArrayLike,
    transmit_power: float,
    ris_internal_noise_variance: float,
    ris_output_power_budget: float,
    maximum_active_amplitude: float,
) -> float:
    if maximum_active_amplitude < 1.0:
        raise ValueError("maximum_active_amplitude must be at least 1")
    if ris_output_power_budget < 0:
        raise ValueError("ris_output_power_budget must be non-negative")

    g = np.asarray(alice_to_ris, dtype=np.complex128).reshape(-1)
    mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    if g.size != mask.size:
        raise ValueError("channel and mask lengths must match")

    if not np.any(mask):
        return 1.0

    input_sum = float(
        np.sum(
            transmit_power * np.abs(g[mask]) ** 2
            + ris_internal_noise_variance
        )
    )
    if input_sum <= 0:
        return 1.0

    budget_limited_amplitude = np.sqrt(ris_output_power_budget / input_sum)
    return float(
        max(
            1.0,
            min(maximum_active_amplitude, budget_limited_amplitude),
        )
    )


def design_passive_surface(
    alice_to_ris: ArrayLike,
    ris_to_transmission_user: ArrayLike,
    ris_to_reflection_user: ArrayLike,
    beta_transmission: float,
    direct_transmission: complex = 0.0j,
    direct_reflection: complex = 0.0j,
) -> SurfaceCoefficients:
    g = np.asarray(alice_to_ris, dtype=np.complex128).reshape(-1)
    split = EnergySplit.from_transmission(beta_transmission, g.size)
    theta_t = phase_align(
        g,
        ris_to_transmission_user,
        direct_transmission,
    )
    theta_r = phase_align(
        g,
        ris_to_reflection_user,
        direct_reflection,
    )
    return build_surface_coefficients(
        split,
        theta_t,
        theta_r,
        amplitude_gain=1.0,
        active_mask=np.zeros(g.size, dtype=bool),
    )


def design_random_passive_surface(
    num_elements: int,
    beta_transmission: float,
    rng: np.random.Generator,
) -> SurfaceCoefficients:
    split = EnergySplit.from_transmission(beta_transmission, num_elements)
    return build_surface_coefficients(
        split,
        random_phases(num_elements, rng),
        random_phases(num_elements, rng),
        amplitude_gain=1.0,
        active_mask=np.zeros(num_elements, dtype=bool),
    )


def design_active_surface(
    alice_to_ris: ArrayLike,
    ris_to_transmission_user: ArrayLike,
    ris_to_reflection_user: ArrayLike,
    beta_transmission: float,
    num_active_elements: int,
    transmit_power: float,
    ris_internal_noise_variance: float,
    ris_output_power_budget: float,
    maximum_active_amplitude: float,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    direct_transmission: complex = 0.0j,
    direct_reflection: complex = 0.0j,
) -> ActiveDesign:
    g = np.asarray(alice_to_ris, dtype=np.complex128).reshape(-1)
    mask, selected = select_active_elements(
        g,
        ris_to_transmission_user,
        ris_to_reflection_user,
        num_active_elements,
        transmission_weight,
        reflection_weight,
    )
    amplitude = feasible_common_active_amplitude(
        g,
        mask,
        transmit_power,
        ris_internal_noise_variance,
        ris_output_power_budget,
        maximum_active_amplitude,
    )
    gains = np.ones(g.size, dtype=float)
    gains[mask] = amplitude

    split = EnergySplit.from_transmission(beta_transmission, g.size)
    theta_t = phase_align(
        g,
        ris_to_transmission_user,
        direct_transmission,
    )
    theta_r = phase_align(
        g,
        ris_to_reflection_user,
        direct_reflection,
    )
    surface = build_surface_coefficients(
        split,
        theta_t,
        theta_r,
        amplitude_gain=gains,
        active_mask=mask,
    )
    return ActiveDesign(
        surface=surface,
        active_mask=mask,
        common_active_amplitude=amplitude,
        selected_indices=selected,
        requested_active_elements=int(
            num_active_elements
        ),
        effective_active_elements=int(
            selected.size
        ),
    )


def design_active_surface_robust(
    alice_to_ris: ArrayLike,
    ris_to_transmission_user: ArrayLike,
    ris_to_reflection_user: ArrayLike,
    beta_transmission: float,
    num_active_elements: int,
    transmit_power: float,
    ris_internal_noise_variance: float,
    ris_output_power_budget: float,
    maximum_active_amplitude: float,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    direct_transmission: complex = 0.0j,
    direct_reflection: complex = 0.0j,
    *,
    nmse_db: float,
    robust_margin_multiplier: float = 3.0,
    transmission_user_pilot_power: float | None = None,
    reflection_user_pilot_power: float | None = None,
) -> ActiveDesign:
    """使用估计CSI设计满足三方向鲁棒功率约束的表面。

    如果请求启用的有源单元数量在单位增益下仍不可行，
    则依次减少实际启用数量，直到满足功率约束。

    极端情况下允许退化为全无源表面，避免仿真程序崩溃。
    """

    g_hat = np.asarray(
        alice_to_ris,
        dtype=np.complex128,
    ).reshape(-1)

    h_t_hat = np.asarray(
        ris_to_transmission_user,
        dtype=np.complex128,
    ).reshape(-1)

    h_r_hat = np.asarray(
        ris_to_reflection_user,
        dtype=np.complex128,
    ).reshape(-1)

    if not (
        g_hat.size
        == h_t_hat.size
        == h_r_hat.size
    ):
        raise ValueError(
            "channel vectors must have equal length"
        )

    if not 0 <= num_active_elements <= g_hat.size:
        raise ValueError(
            "num_active_elements is outside the valid range"
        )

    # 未单独指定用户导频功率时，
    # 默认与控制端导频功率相同。
    p_t_user = (
        transmit_power
        if transmission_user_pilot_power is None
        else float(
            transmission_user_pilot_power
        )
    )

    p_r_user = (
        transmit_power
        if reflection_user_pilot_power is None
        else float(
            reflection_user_pilot_power
        )
    )

    # -------------------------------------------------
    # 首先尝试用户要求的有源单元数量。
    #
    # 若在单位增益下仍无法满足鲁棒功率预算，
    # 则逐步减少实际启用的有源单元数量。
    # -------------------------------------------------
    effective_active_elements = int(
        num_active_elements
    )

    projection = None

    mask = np.zeros(
        g_hat.size,
        dtype=bool,
    )

    selected = np.empty(
        0,
        dtype=np.int64,
    )

    while effective_active_elements >= 0:
        mask, selected = select_active_elements(
            alice_to_ris=g_hat,
            ris_to_transmission_user=h_t_hat,
            ris_to_reflection_user=h_r_hat,
            num_active_elements=(
                effective_active_elements
            ),
            transmission_weight=(
                transmission_weight
            ),
            reflection_weight=(
                reflection_weight
            ),
        )

        projection = (
            project_common_active_amplitude_robust(
                controller_to_ris_estimate=(
                    g_hat
                ),
                transmission_user_to_ris_estimate=(
                    h_t_hat
                ),
                reflection_user_to_ris_estimate=(
                    h_r_hat
                ),
                active_mask=mask,
                controller_pilot_power=(
                    transmit_power
                ),
                transmission_user_pilot_power=(
                    p_t_user
                ),
                reflection_user_pilot_power=(
                    p_r_user
                ),
                ris_internal_noise_variance=(
                    ris_internal_noise_variance
                ),
                output_power_budget=(
                    ris_output_power_budget
                ),
                maximum_active_amplitude=(
                    maximum_active_amplitude
                ),
                nmse_db=nmse_db,
                robust_margin_multiplier=(
                    robust_margin_multiplier
                ),
            )
        )

        if projection.is_feasible_at_unit_gain:
            break

        effective_active_elements -= 1

    # 正常情况下，零个有源单元一定应当可行。
    # 此异常只用于捕捉功率模型内部逻辑错误。
    if (
        projection is None
        or not projection.is_feasible_at_unit_gain
    ):
        raise RuntimeError(
            "Failed to construct a feasible "
            "robust active-surface design"
        )

    amplitude = (
        projection.common_active_amplitude
    )

    gains = np.ones(
        g_hat.size,
        dtype=float,
    )

    gains[mask] = amplitude

    split = EnergySplit.from_transmission(
        beta_transmission,
        g_hat.size,
    )

    theta_t = phase_align(
        g_hat,
        h_t_hat,
        direct_transmission,
    )

    theta_r = phase_align(
        g_hat,
        h_r_hat,
        direct_reflection,
    )

    surface = build_surface_coefficients(
        energy_split=split,
        phase_transmission_rad=theta_t,
        phase_reflection_rad=theta_r,
        amplitude_gain=gains,
        active_mask=mask,
    )

    return ActiveDesign(
        surface=surface,
        active_mask=mask,
        common_active_amplitude=amplitude,
        selected_indices=selected,
        requested_active_elements=int(
            num_active_elements
        ),
        effective_active_elements=int(
            selected.size
        ),
    )


def design_elementwise_transmission_split(
    alice_to_ris: ArrayLike,
    ris_to_transmission_user: ArrayLike,
    ris_to_reflection_user: ArrayLike,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    beta_min: float = 0.05,
    beta_max: float = 0.95,
    temperature: float = 1.0,
) -> FloatArray:
    """根据估计CSI生成逐单元透射能量分配系数。

    对第n个单元，分别计算其对透射用户和反射用户的
    加权级联信道贡献：

        s_T,n = w_T |g_n h_T,n|²
        s_R,n = w_R |g_n h_R,n|²

    再根据二者的对数比值产生平滑的能量分配：

        beta_T,n = sigmoid(
            [log(s_T,n) - log(s_R,n)] / temperature
        )

    beta_T,n越大，表示该单元更偏向透射侧；
    beta_T,n越小，表示该单元更偏向反射侧。
    """

    g = np.asarray(
        alice_to_ris,
        dtype=np.complex128,
    ).reshape(-1)

    h_t = np.asarray(
        ris_to_transmission_user,
        dtype=np.complex128,
    ).reshape(-1)

    h_r = np.asarray(
        ris_to_reflection_user,
        dtype=np.complex128,
    ).reshape(-1)

    if not (
        g.size
        == h_t.size
        == h_r.size
    ):
        raise ValueError(
            "channel vectors must have equal length"
        )

    if not (
        0.0
        <= beta_min
        < beta_max
        <= 1.0
    ):
        raise ValueError(
            "beta bounds must satisfy "
            "0 <= beta_min < beta_max <= 1"
        )

    if temperature <= 0.0:
        raise ValueError(
            "temperature must be positive"
        )

    total_weight = (
        transmission_weight
        + reflection_weight
    )

    if total_weight <= 0.0:
        raise ValueError(
            "at least one user weight "
            "must be positive"
        )

    wt = (
        transmission_weight
        / total_weight
    )

    wr = (
        reflection_weight
        / total_weight
    )

    transmission_scores = (
        wt
        * np.abs(g * h_t) ** 2
    )

    reflection_scores = (
        wr
        * np.abs(g * h_r) ** 2
    )

    epsilon = np.finfo(
        np.float64
    ).eps

    log_score_ratio = (
        np.log(
            transmission_scores
            + epsilon
        )
        - np.log(
            reflection_scores
            + epsilon
        )
    )

    normalized_log_ratio = np.clip(
        log_score_ratio
        / temperature,
        -60.0,
        60.0,
    )

    beta_transmission = (
        1.0
        / (
            1.0
            + np.exp(
                -normalized_log_ratio
            )
        )
    )

    # 当两个方向的贡献都几乎为零时，
    # 使用完全均衡的能量分配。
    zero_score_mask = (
        transmission_scores
        + reflection_scores
        <= epsilon
    )

    beta_transmission[
        zero_score_mask
    ] = 0.5

    beta_transmission = np.clip(
        beta_transmission,
        beta_min,
        beta_max,
    )

    return np.asarray(
        beta_transmission,
        dtype=np.float64,
    )


def design_active_surface_vector_robust(
    alice_to_ris: ArrayLike,
    ris_to_transmission_user: ArrayLike,
    ris_to_reflection_user: ArrayLike,
    beta_transmission: ArrayLike,
    num_active_elements: int,
    transmit_power: float,
    ris_internal_noise_variance: float,
    ris_output_power_budget: float,
    maximum_active_amplitude: float,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    direct_transmission: complex = 0.0j,
    direct_reflection: complex = 0.0j,
    *,
    nmse_db: float,
    robust_margin_multiplier: float = 3.0,
    transmission_user_pilot_power: float | None = None,
    reflection_user_pilot_power: float | None = None,
    gain_shape_exponent: float = 0.5,
) -> ActiveDesign:
    """逐有源单元独立增益的鲁棒设计。

    当前候选增益按照级联信道得分分配。

    后续强化学习将直接输出候选增益向量，
    并继续复用相同的鲁棒投影函数。
    """

    g_hat = np.asarray(
        alice_to_ris,
        dtype=np.complex128,
    ).reshape(-1)

    h_t_hat = np.asarray(
        ris_to_transmission_user,
        dtype=np.complex128,
    ).reshape(-1)

    h_r_hat = np.asarray(
        ris_to_reflection_user,
        dtype=np.complex128,
    ).reshape(-1)

    if not (
        g_hat.size
        == h_t_hat.size
        == h_r_hat.size
    ):
        raise ValueError(
            "channel vectors must have equal length"
        )

    if not 0 <= num_active_elements <= g_hat.size:
        raise ValueError(
            "num_active_elements is outside "
            "the valid range"
        )

    if gain_shape_exponent <= 0.0:
        raise ValueError(
            "gain_shape_exponent must be positive"
        )

    total_weight = (
        transmission_weight
        + reflection_weight
    )

    if total_weight <= 0.0:
        raise ValueError(
            "at least one weight must be positive"
        )

    wt = (
        transmission_weight
        / total_weight
    )

    wr = (
        reflection_weight
        / total_weight
    )

    # 每个单元的级联信道贡献得分。
    element_scores = (
        np.abs(g_hat) ** 2
        * (
            wt * np.abs(h_t_hat) ** 2
            + wr * np.abs(h_r_hat) ** 2
        )
    )

    p_t_user = (
        transmit_power
        if transmission_user_pilot_power
        is None
        else float(
            transmission_user_pilot_power
        )
    )

    p_r_user = (
        transmit_power
        if reflection_user_pilot_power
        is None
        else float(
            reflection_user_pilot_power
        )
    )

    effective_active_elements = int(
        num_active_elements
    )

    projection = None

    mask = np.zeros(
        g_hat.size,
        dtype=bool,
    )

    selected = np.empty(
        0,
        dtype=np.int64,
    )

    while effective_active_elements >= 0:
        mask, selected = select_active_elements(
            alice_to_ris=g_hat,
            ris_to_transmission_user=h_t_hat,
            ris_to_reflection_user=h_r_hat,
            num_active_elements=(
                effective_active_elements
            ),
            transmission_weight=(
                transmission_weight
            ),
            reflection_weight=(
                reflection_weight
            ),
        )

        requested_amplitudes = np.ones(
            g_hat.size,
            dtype=np.float64,
        )

        if np.any(mask):
            selected_scores = (
                element_scores[mask]
            )

            maximum_score = float(
                np.max(selected_scores)
            )

            if maximum_score > 0.0:
                normalized_scores = (
                    selected_scores
                    / maximum_score
                )

                requested_amplitudes[mask] = (
                    1.0
                    + (
                        maximum_active_amplitude
                        - 1.0
                    )
                    * normalized_scores
                    ** gain_shape_exponent
                )

        projection = (
            project_active_amplitude_vector_robust(
                controller_to_ris_estimate=(
                    g_hat
                ),
                transmission_user_to_ris_estimate=(
                    h_t_hat
                ),
                reflection_user_to_ris_estimate=(
                    h_r_hat
                ),
                requested_amplitudes=(
                    requested_amplitudes
                ),
                active_mask=mask,
                controller_pilot_power=(
                    transmit_power
                ),
                transmission_user_pilot_power=(
                    p_t_user
                ),
                reflection_user_pilot_power=(
                    p_r_user
                ),
                ris_internal_noise_variance=(
                    ris_internal_noise_variance
                ),
                output_power_budget=(
                    ris_output_power_budget
                ),
                maximum_active_amplitude=(
                    maximum_active_amplitude
                ),
                nmse_db=nmse_db,
                robust_margin_multiplier=(
                    robust_margin_multiplier
                ),
            )
        )

        if projection.is_feasible_at_unit_gain:
            break

        effective_active_elements -= 1

    if (
        projection is None
        or not projection
        .is_feasible_at_unit_gain
    ):
        raise RuntimeError(
            "Failed to construct a feasible "
            "vector-gain surface design"
        )

    gains = np.asarray(
        projection.projected_amplitudes,
        dtype=np.float64,
    )

    if np.any(mask):
        mean_active_amplitude = float(
            np.mean(
                gains[mask]
            )
        )
    else:
        mean_active_amplitude = 1.0

    split = EnergySplit.from_transmission(
        beta_transmission,
        g_hat.size,
    )

    theta_t = phase_align(
        g_hat,
        h_t_hat,
        direct_transmission,
    )

    theta_r = phase_align(
        g_hat,
        h_r_hat,
        direct_reflection,
    )

    surface = build_surface_coefficients(
        energy_split=split,
        phase_transmission_rad=theta_t,
        phase_reflection_rad=theta_r,
        amplitude_gain=gains,
        active_mask=mask,
    )

    return ActiveDesign(
        surface=surface,
        active_mask=mask,
        # 为兼容原有结果字段，这里记录实际有源增益均值。
        common_active_amplitude=(
            mean_active_amplitude
        ),
        selected_indices=selected,
        requested_active_elements=int(
            num_active_elements
        ),
        effective_active_elements=int(
            selected.size
        ),
    )


def design_active_surface_vector_beta_robust(
    alice_to_ris: ArrayLike,
    ris_to_transmission_user: ArrayLike,
    ris_to_reflection_user: ArrayLike,
    num_active_elements: int,
    transmit_power: float,
    ris_internal_noise_variance: float,
    ris_output_power_budget: float,
    maximum_active_amplitude: float,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    direct_transmission: complex = 0.0j,
    direct_reflection: complex = 0.0j,
    *,
    nmse_db: float,
    robust_margin_multiplier: float = 3.0,
    transmission_user_pilot_power: float | None = None,
    reflection_user_pilot_power: float | None = None,
    gain_shape_exponent: float = 0.5,
    beta_min: float = 0.05,
    beta_max: float = 0.95,
    beta_temperature: float = 1.0,
) -> tuple[
    FloatArray,
    ActiveDesign,
]:
    """联合设计逐单元放大增益和逐单元能量分配。

    当前阶段：

    1. 根据估计CSI生成逐单元beta向量；
    2. 根据级联信道得分生成逐单元候选增益；
    3. 将增益向量投影到三方向鲁棒功率可行域；
    4. 返回beta向量和最终表面设计。

    后续强化学习将替代第1步和第2步，
    但继续复用相同的功率安全投影。
    """

    beta_vector = (
        design_elementwise_transmission_split(
            alice_to_ris=alice_to_ris,
            ris_to_transmission_user=(
                ris_to_transmission_user
            ),
            ris_to_reflection_user=(
                ris_to_reflection_user
            ),
            transmission_weight=(
                transmission_weight
            ),
            reflection_weight=(
                reflection_weight
            ),
            beta_min=beta_min,
            beta_max=beta_max,
            temperature=beta_temperature,
        )
    )

    design = (
        design_active_surface_vector_robust(
            alice_to_ris=alice_to_ris,
            ris_to_transmission_user=(
                ris_to_transmission_user
            ),
            ris_to_reflection_user=(
                ris_to_reflection_user
            ),
            beta_transmission=beta_vector,
            num_active_elements=(
                num_active_elements
            ),
            transmit_power=transmit_power,
            ris_internal_noise_variance=(
                ris_internal_noise_variance
            ),
            ris_output_power_budget=(
                ris_output_power_budget
            ),
            maximum_active_amplitude=(
                maximum_active_amplitude
            ),
            transmission_weight=(
                transmission_weight
            ),
            reflection_weight=(
                reflection_weight
            ),
            direct_transmission=(
                direct_transmission
            ),
            direct_reflection=(
                direct_reflection
            ),
            nmse_db=nmse_db,
            robust_margin_multiplier=(
                robust_margin_multiplier
            ),
            transmission_user_pilot_power=(
                transmission_user_pilot_power
            ),
            reflection_user_pilot_power=(
                reflection_user_pilot_power
            ),
            gain_shape_exponent=(
                gain_shape_exponent
            ),
        )
    )

    return (
        beta_vector,
        design,
    )


def optimize_scalar_energy_split(
    alice_to_ris: ArrayLike,
    ris_to_transmission_user: ArrayLike,
    ris_to_reflection_user: ArrayLike,
    num_active_elements: int,
    transmit_power: float,
    user_noise_variance: float,
    ris_internal_noise_variance: float,
    ris_output_power_budget: float,
    maximum_active_amplitude: float,
    beta_grid: ArrayLike,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    direct_transmission: complex = 0.0j,
    direct_reflection: complex = 0.0j,
) -> tuple[float, ActiveDesign, TwoUserMetrics]:
    beta_values = np.asarray(beta_grid, dtype=float).reshape(-1)
    if beta_values.size == 0:
        raise ValueError("beta_grid cannot be empty")
    if np.any((beta_values < 0.0) | (beta_values > 1.0)):
        raise ValueError("all beta values must lie within [0, 1]")

    best_beta = float(beta_values[0])
    best_design: ActiveDesign | None = None
    best_metrics: TwoUserMetrics | None = None

    for beta in beta_values:
        design = design_active_surface(
            alice_to_ris,
            ris_to_transmission_user,
            ris_to_reflection_user,
            beta_transmission=float(beta),
            num_active_elements=num_active_elements,
            transmit_power=transmit_power,
            ris_internal_noise_variance=ris_internal_noise_variance,
            ris_output_power_budget=ris_output_power_budget,
            maximum_active_amplitude=maximum_active_amplitude,
            transmission_weight=transmission_weight,
            reflection_weight=reflection_weight,
            direct_transmission=direct_transmission,
            direct_reflection=direct_reflection,
        )
        metrics = evaluate_two_user_system(
            alice_to_ris,
            ris_to_transmission_user,
            ris_to_reflection_user,
            design.surface,
            transmit_power,
            user_noise_variance,
            ris_internal_noise_variance,
            ris_output_power_budget,
            transmission_weight,
            reflection_weight,
            direct_transmission,
            direct_reflection,
        )
        if (
            best_metrics is None
            or metrics.weighted_sum_rate > best_metrics.weighted_sum_rate
        ):
            best_beta = float(beta)
            best_design = design
            best_metrics = metrics

    assert best_design is not None
    assert best_metrics is not None
    return best_beta, best_design, best_metrics


def optimize_scalar_energy_split_robust(
    alice_to_ris: ArrayLike,
    ris_to_transmission_user: ArrayLike,
    ris_to_reflection_user: ArrayLike,
    num_active_elements: int,
    transmit_power: float,
    user_noise_variance: float,
    ris_internal_noise_variance: float,
    ris_output_power_budget: float,
    maximum_active_amplitude: float,
    beta_grid: ArrayLike,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    direct_transmission: complex = 0.0j,
    direct_reflection: complex = 0.0j,
    *,
    nmse_db: float,
    robust_margin_multiplier: float = 3.0,
    transmission_user_pilot_power: float | None = None,
    reflection_user_pilot_power: float | None = None,
) -> tuple[
    float,
    ActiveDesign,
    TwoUserMetrics,
]:
    """在估计CSI上进行带鲁棒功率投影的标量β搜索。"""

    beta_values = np.asarray(
        beta_grid,
        dtype=float,
    ).reshape(-1)

    if beta_values.size == 0:
        raise ValueError(
            "beta_grid cannot be empty"
        )

    if np.any(
        (beta_values < 0.0)
        | (beta_values > 1.0)
    ):
        raise ValueError(
            "all beta values must lie "
            "within [0, 1]"
        )

    best_beta = float(
        beta_values[0]
    )

    best_design: ActiveDesign | None = None
    best_metrics: TwoUserMetrics | None = None

    for beta in beta_values:
        design = design_active_surface_robust(
            alice_to_ris=alice_to_ris,
            ris_to_transmission_user=(
                ris_to_transmission_user
            ),
            ris_to_reflection_user=(
                ris_to_reflection_user
            ),
            beta_transmission=float(beta),
            num_active_elements=(
                num_active_elements
            ),
            transmit_power=transmit_power,
            ris_internal_noise_variance=(
                ris_internal_noise_variance
            ),
            ris_output_power_budget=(
                ris_output_power_budget
            ),
            maximum_active_amplitude=(
                maximum_active_amplitude
            ),
            transmission_weight=(
                transmission_weight
            ),
            reflection_weight=(
                reflection_weight
            ),
            direct_transmission=(
                direct_transmission
            ),
            direct_reflection=(
                direct_reflection
            ),
            nmse_db=nmse_db,
            robust_margin_multiplier=(
                robust_margin_multiplier
            ),
            transmission_user_pilot_power=(
                transmission_user_pilot_power
            ),
            reflection_user_pilot_power=(
                reflection_user_pilot_power
            ),
        )

        estimated_metrics = (
            evaluate_two_user_system(
                alice_to_ris,
                ris_to_transmission_user,
                ris_to_reflection_user,
                design.surface,
                transmit_power,
                user_noise_variance,
                ris_internal_noise_variance,
                ris_output_power_budget,
                transmission_weight,
                reflection_weight,
                direct_transmission,
                direct_reflection,
            )
        )

        if (
            best_metrics is None
            or estimated_metrics.weighted_sum_rate
            > best_metrics.weighted_sum_rate
        ):
            best_beta = float(beta)
            best_design = design
            best_metrics = (
                estimated_metrics
            )

    assert best_design is not None
    assert best_metrics is not None

    return (
        best_beta,
        best_design,
        best_metrics,
    )


def optimize_scalar_energy_split_vector_robust(
    alice_to_ris: ArrayLike,
    ris_to_transmission_user: ArrayLike,
    ris_to_reflection_user: ArrayLike,
    num_active_elements: int,
    transmit_power: float,
    user_noise_variance: float,
    ris_internal_noise_variance: float,
    ris_output_power_budget: float,
    maximum_active_amplitude: float,
    beta_grid: ArrayLike,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    direct_transmission: complex = 0.0j,
    direct_reflection: complex = 0.0j,
    *,
    nmse_db: float,
    robust_margin_multiplier: float = 3.0,
    transmission_user_pilot_power: float | None = None,
    reflection_user_pilot_power: float | None = None,
    gain_shape_exponent: float = 0.5,
) -> tuple[
    float,
    ActiveDesign,
    TwoUserMetrics,
]:
    """逐单元增益条件下的标量beta搜索。"""

    beta_values = np.asarray(
        beta_grid,
        dtype=float,
    ).reshape(-1)

    if beta_values.size == 0:
        raise ValueError(
            "beta_grid cannot be empty"
        )

    if np.any(
        (beta_values < 0.0)
        | (beta_values > 1.0)
    ):
        raise ValueError(
            "all beta values must lie "
            "within [0, 1]"
        )

    best_beta = float(
        beta_values[0]
    )

    best_design: ActiveDesign | None = None
    best_metrics: TwoUserMetrics | None = None

    for beta in beta_values:
        design = (
            design_active_surface_vector_robust(
                alice_to_ris=(
                    alice_to_ris
                ),
                ris_to_transmission_user=(
                    ris_to_transmission_user
                ),
                ris_to_reflection_user=(
                    ris_to_reflection_user
                ),
                beta_transmission=float(
                    beta
                ),
                num_active_elements=(
                    num_active_elements
                ),
                transmit_power=(
                    transmit_power
                ),
                ris_internal_noise_variance=(
                    ris_internal_noise_variance
                ),
                ris_output_power_budget=(
                    ris_output_power_budget
                ),
                maximum_active_amplitude=(
                    maximum_active_amplitude
                ),
                transmission_weight=(
                    transmission_weight
                ),
                reflection_weight=(
                    reflection_weight
                ),
                direct_transmission=(
                    direct_transmission
                ),
                direct_reflection=(
                    direct_reflection
                ),
                nmse_db=nmse_db,
                robust_margin_multiplier=(
                    robust_margin_multiplier
                ),
                transmission_user_pilot_power=(
                    transmission_user_pilot_power
                ),
                reflection_user_pilot_power=(
                    reflection_user_pilot_power
                ),
                gain_shape_exponent=(
                    gain_shape_exponent
                ),
            )
        )

        estimated_metrics = (
            evaluate_two_user_system(
                alice_to_ris,
                ris_to_transmission_user,
                ris_to_reflection_user,
                design.surface,
                transmit_power,
                user_noise_variance,
                ris_internal_noise_variance,
                ris_output_power_budget,
                transmission_weight,
                reflection_weight,
                direct_transmission,
                direct_reflection,
            )
        )

        if (
            best_metrics is None
            or estimated_metrics
            .weighted_sum_rate
            > best_metrics
            .weighted_sum_rate
        ):
            best_beta = float(beta)
            best_design = design
            best_metrics = (
                estimated_metrics
            )

    assert best_design is not None
    assert best_metrics is not None

    return (
        best_beta,
        best_design,
        best_metrics,
    )
