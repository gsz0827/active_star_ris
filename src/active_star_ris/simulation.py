from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .channels import ChannelConfig, generate_channel
from .csi_estimation import (
    CSIErrorConfig,
    generate_imperfect_csi,
    sample_nmse_db,
)
from .optimization import (
    design_active_surface,
    design_active_surface_robust,
    design_active_surface_vector_robust,
    design_passive_surface,
    design_random_passive_surface,
    optimize_scalar_energy_split,
    optimize_scalar_energy_split_robust,
    optimize_scalar_energy_split_vector_robust,
    design_active_surface_vector_beta_robust,
)
from .surface_power import (
    BidirectionalSurfacePower,
    evaluate_bidirectional_surface_power,
)
from .system import (
    TwoUserMetrics,
    evaluate_two_user_system,
)


@dataclass(frozen=True)
class ChannelSnapshot:
    """一个时刻的完整STAR-RIS信道快照。"""

    alice_to_ris: np.ndarray
    ris_to_transmission_user: np.ndarray
    ris_to_reflection_user: np.ndarray

    direct_transmission: complex
    direct_reflection: complex


@dataclass(frozen=True)
class ImperfectCSIChannelSnapshot:
    """真实信道及其对应的不完美估计CSI。"""

    true: ChannelSnapshot
    estimated: ChannelSnapshot

    nmse_db: float


@dataclass(frozen=True)
class ScenarioResult:
    """一次方案仿真所得到的主要性能指标。"""

    transmission_snr: float
    reflection_snr: float

    transmission_rate: float
    reflection_rate: float
    weighted_sum_rate: float

    ris_output_power: float
    ris_power_violation: float

    beta_transmission: float
    active_amplitude: float

    # 三种双向探测方向对应的实际输出功率。
    ris_output_power_controller: float = 0.0
    ris_output_power_transmission_user: float = 0.0
    ris_output_power_reflection_user: float = 0.0

    # 三个方向中的最大输出功率和功率越界量。
    ris_max_bidirectional_output_power: float = 0.0
    ris_bidirectional_power_violation: float = 0.0

    # 包含放大器、控制器和有源单元偏置功耗的总表面功耗。
    surface_total_power: float = 0.0

    # 有源单元数量记录。
    requested_active_elements: int = 0
    effective_active_elements: int = 0
    disabled_active_elements: int = 0

    # 是否因功率约束完全退化为无源方案。
    passive_fallback: bool = False

    # 实际启用有源单元的增益离散程度。
    #
    # 没有实际启用有源单元时：
    # mean = 1，std = 0，min = 1，max = 1。
    active_gain_std: float = 0.0
    active_gain_min: float = 1.0
    active_gain_max: float = 1.0

    # 一次实现内部逐单元透射能量分配的离散程度。
    beta_transmission_std: float = 0.0
    beta_transmission_min: float = 0.5
    beta_transmission_max: float = 0.5


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    """读取YAML配置文件。"""

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def _channel_config(
    data: dict[str, Any],
) -> ChannelConfig:
    """将YAML中的信道配置转换为ChannelConfig。"""

    return ChannelConfig(
        model=str(
            data["model"]
        ),
        large_scale_power=float(
            data.get(
                "large_scale_power",
                1.0,
            )
        ),
        k_factor_db=float(
            data.get(
                "k_factor_db",
                0.0,
            )
        ),
    )


def _csi_error_config(
    data: dict[str, Any],
) -> CSIErrorConfig:
    """从YAML配置中读取CSI误差范围。"""

    return CSIErrorConfig(
        nmse_db_min=float(
            data["nmse_db_min"]
        ),
        nmse_db_max=float(
            data["nmse_db_max"]
        ),
    )


def sample_channels(
    config: dict[str, Any],
    rng: np.random.Generator,
    num_elements: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    complex,
    complex,
]:
    """生成一组真实STAR-RIS信道。"""

    channel_cfg = config["channel"]

    g = generate_channel(
        num_elements,
        rng,
        _channel_config(
            channel_cfg[
                "alice_to_ris"
            ]
        ),
    )

    h_t = generate_channel(
        num_elements,
        rng,
        _channel_config(
            channel_cfg[
                "ris_to_transmission_user"
            ]
        ),
    )

    h_r = generate_channel(
        num_elements,
        rng,
        _channel_config(
            channel_cfg[
                "ris_to_reflection_user"
            ]
        ),
    )

    d_t = complex(
        generate_channel(
            1,
            rng,
            _channel_config(
                channel_cfg[
                    "direct_transmission"
                ]
            ),
        )[0]
    )

    d_r = complex(
        generate_channel(
            1,
            rng,
            _channel_config(
                channel_cfg[
                    "direct_reflection"
                ]
            ),
        )[0]
    )

    return (
        g,
        h_t,
        h_r,
        d_t,
        d_r,
    )


def sample_channels_with_imperfect_csi(
    config: dict[str, Any],
    rng: np.random.Generator,
    num_elements: int,
    nmse_db: float | None = None,
) -> ImperfectCSIChannelSnapshot:
    """同时生成真实信道和不完美估计CSI。

    当nmse_db为None时，会从配置文件规定的NMSE范围中
    随机采样一个误差水平。

    同一次信道快照中的所有链路采用相同的目标NMSE，
    但各链路的复高斯估计误差相互独立。
    """

    (
        g_true,
        h_t_true,
        h_r_true,
        d_t_true,
        d_r_true,
    ) = sample_channels(
        config=config,
        rng=rng,
        num_elements=num_elements,
    )

    csi_config = _csi_error_config(
        config["csi"]
    )

    if nmse_db is None:
        selected_nmse_db = sample_nmse_db(
            config=csi_config,
            rng=rng,
        )
    else:
        selected_nmse_db = float(
            nmse_db
        )

    if not np.isfinite(
        selected_nmse_db
    ):
        raise ValueError(
            "nmse_db must be finite"
        )

    g_csi = generate_imperfect_csi(
        true_channel=g_true,
        nmse_db=selected_nmse_db,
        rng=rng,
    )

    h_t_csi = generate_imperfect_csi(
        true_channel=h_t_true,
        nmse_db=selected_nmse_db,
        rng=rng,
    )

    h_r_csi = generate_imperfect_csi(
        true_channel=h_r_true,
        nmse_db=selected_nmse_db,
        rng=rng,
    )

    # 直接链路是复数标量，因此先转成长度为1的数组。
    d_t_csi = generate_imperfect_csi(
        true_channel=np.asarray(
            [d_t_true],
            dtype=np.complex128,
        ),
        nmse_db=selected_nmse_db,
        rng=rng,
    )

    d_r_csi = generate_imperfect_csi(
        true_channel=np.asarray(
            [d_r_true],
            dtype=np.complex128,
        ),
        nmse_db=selected_nmse_db,
        rng=rng,
    )

    true_snapshot = ChannelSnapshot(
        alice_to_ris=np.asarray(
            g_true,
            dtype=np.complex128,
        ),
        ris_to_transmission_user=np.asarray(
            h_t_true,
            dtype=np.complex128,
        ),
        ris_to_reflection_user=np.asarray(
            h_r_true,
            dtype=np.complex128,
        ),
        direct_transmission=complex(
            d_t_true
        ),
        direct_reflection=complex(
            d_r_true
        ),
    )

    estimated_snapshot = ChannelSnapshot(
        alice_to_ris=np.asarray(
            g_csi.estimated_channel,
            dtype=np.complex128,
        ),
        ris_to_transmission_user=np.asarray(
            h_t_csi.estimated_channel,
            dtype=np.complex128,
        ),
        ris_to_reflection_user=np.asarray(
            h_r_csi.estimated_channel,
            dtype=np.complex128,
        ),
        direct_transmission=complex(
            d_t_csi.estimated_channel[0]
        ),
        direct_reflection=complex(
            d_r_csi.estimated_channel[0]
        ),
    )

    return ImperfectCSIChannelSnapshot(
        true=true_snapshot,
        estimated=estimated_snapshot,
        nmse_db=selected_nmse_db,
    )


def summarize_active_gains(
    amplitude_gain: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """统计实际启用有源单元的增益。

    返回：
        mean_gain,
        std_gain,
        minimum_gain,
        maximum_gain

    当没有启用有源单元时，返回：
        1.0, 0.0, 1.0, 1.0
    """

    gains = np.asarray(
        amplitude_gain,
        dtype=np.float64,
    ).reshape(-1)

    mask = np.asarray(
        active_mask,
        dtype=bool,
    ).reshape(-1)

    if gains.size != mask.size:
        raise ValueError(
            "amplitude_gain and active_mask "
            "must have equal length"
        )

    if not np.all(
        np.isfinite(gains)
    ):
        raise ValueError(
            "amplitude_gain must be finite"
        )

    active_gains = gains[mask]

    if active_gains.size == 0:
        return (
            1.0,
            0.0,
            1.0,
            1.0,
        )

    return (
        float(
            np.mean(active_gains)
        ),
        float(
            np.std(
                active_gains,
                ddof=0,
            )
        ),
        float(
            np.min(active_gains)
        ),
        float(
            np.max(active_gains)
        ),
    )


def summarize_transmission_split(
    beta_transmission: np.ndarray | float,
    num_elements: int,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """统计逐单元透射能量分配系数。

    返回：
        mean_beta,
        std_beta,
        minimum_beta,
        maximum_beta
    """

    beta = np.asarray(
        beta_transmission,
        dtype=np.float64,
    )

    if beta.ndim == 0:
        beta = np.full(
            num_elements,
            float(beta),
            dtype=np.float64,
        )
    else:
        beta = beta.reshape(-1)

    if beta.size != num_elements:
        raise ValueError(
            "beta_transmission must be "
            "a scalar or have num_elements entries"
        )

    if not np.all(
        np.isfinite(beta)
    ):
        raise ValueError(
            "beta_transmission must be finite"
        )

    if np.any(
        (beta < 0.0)
        | (beta > 1.0)
    ):
        raise ValueError(
            "beta_transmission entries "
            "must lie within [0, 1]"
        )

    return (
        float(
            np.mean(beta)
        ),
        float(
            np.std(
                beta,
                ddof=0,
            )
        ),
        float(
            np.min(beta)
        ),
        float(
            np.max(beta)
        ),
    )


def metrics_to_result(
    metrics: TwoUserMetrics,
    beta_transmission: float,
    active_amplitude: float,
    bidirectional_power: (
        BidirectionalSurfacePower
        | None
    ) = None,
    requested_active_elements: int = 0,
    effective_active_elements: int = 0,
    active_gain_std: float = 0.0,
    active_gain_min: float = 1.0,
    active_gain_max: float = 1.0,
    beta_transmission_std: float = 0.0,
    beta_transmission_min: float | None = None,
    beta_transmission_max: float | None = None,
) -> ScenarioResult:
    """将系统内部指标转换为可保存的ScenarioResult。"""

    resolved_beta_min = (
        float(beta_transmission)
        if beta_transmission_min is None
        else float(beta_transmission_min)
    )

    resolved_beta_max = (
        float(beta_transmission)
        if beta_transmission_max is None
        else float(beta_transmission_max)
    )

    if bidirectional_power is None:
        # 保持原有完美CSI通信基线的兼容性。
        output_power = (
            metrics.ris_output_power
        )

        violation = (
            metrics.ris_power_violation
        )

        output_controller = (
            metrics.ris_output_power
        )

        output_transmission = 0.0
        output_reflection = 0.0

        maximum_bidirectional = (
            metrics.ris_output_power
        )

        bidirectional_violation = (
            metrics.ris_power_violation
        )

        total_surface_power = 0.0

    else:
        # 不完美CSI鲁棒方案使用三方向功率模型。
        output_power = (
            bidirectional_power
            .maximum_output_power
        )

        violation = (
            bidirectional_power
            .power_violation
        )

        output_controller = (
            bidirectional_power
            .output_power_controller
        )

        output_transmission = (
            bidirectional_power
            .output_power_transmission_user
        )

        output_reflection = (
            bidirectional_power
            .output_power_reflection_user
        )

        maximum_bidirectional = (
            bidirectional_power
            .maximum_output_power
        )

        bidirectional_violation = (
            bidirectional_power
            .power_violation
        )

        total_surface_power = (
            bidirectional_power
            .total_surface_power
        )

    return ScenarioResult(
        transmission_snr=(
            metrics.transmission
            .snr_linear
        ),
        reflection_snr=(
            metrics.reflection
            .snr_linear
        ),
        transmission_rate=(
            metrics.transmission
            .rate_bps_hz
        ),
        reflection_rate=(
            metrics.reflection
            .rate_bps_hz
        ),
        weighted_sum_rate=(
            metrics.weighted_sum_rate
        ),
        ris_output_power=float(
            output_power
        ),
        ris_power_violation=float(
            violation
        ),
        beta_transmission=float(
            beta_transmission
        ),
        active_amplitude=float(
            active_amplitude
        ),
        ris_output_power_controller=float(
            output_controller
        ),
        ris_output_power_transmission_user=float(
            output_transmission
        ),
        ris_output_power_reflection_user=float(
            output_reflection
        ),
        ris_max_bidirectional_output_power=float(
            maximum_bidirectional
        ),
        ris_bidirectional_power_violation=float(
            bidirectional_violation
        ),
        surface_total_power=float(
            total_surface_power
        ),
        requested_active_elements=int(
            requested_active_elements
        ),
        effective_active_elements=int(
            effective_active_elements
        ),
        disabled_active_elements=max(
            0,
            int(requested_active_elements)
            - int(effective_active_elements),
        ),
        passive_fallback=bool(
            requested_active_elements > 0
            and effective_active_elements == 0
        ),
        active_gain_std=float(
            active_gain_std
        ),
        active_gain_min=float(
            active_gain_min
        ),
        active_gain_max=float(
            active_gain_max
        ),
        beta_transmission_std=float(
            beta_transmission_std
        ),
        beta_transmission_min=(
            resolved_beta_min
        ),
        beta_transmission_max=(
            resolved_beta_max
        ),
    )


def evaluate_one_realization(
    config: dict[str, Any],
    rng: np.random.Generator,
    num_elements: int | None = None,
    num_active_elements: int | None = None,
) -> dict[str, ScenarioResult]:
    """原有完美CSI通信速率基线。

    该函数继续保留，不引入不完美CSI和鲁棒功率投影，
    用于后续与不完美CSI方案进行对比。
    """

    system_cfg = config["system"]
    opt_cfg = config["optimization"]

    n = int(
        system_cfg["num_elements"]
        if num_elements is None
        else num_elements
    )

    k_active = int(
        system_cfg[
            "num_active_elements"
        ]
        if num_active_elements is None
        else num_active_elements
    )

    p_tx = float(
        system_cfg["transmit_power"]
    )

    sigma_user = float(
        system_cfg[
            "user_noise_variance"
        ]
    )

    sigma_ris = float(
        system_cfg[
            "ris_internal_noise_variance"
        ]
    )

    p_ris_max = float(
        system_cfg[
            "ris_output_power_budget"
        ]
    )

    a_max = float(
        system_cfg[
            "maximum_active_amplitude"
        ]
    )

    beta = float(
        system_cfg[
            "beta_transmission"
        ]
    )

    wt = float(
        system_cfg[
            "transmission_weight"
        ]
    )

    wr = float(
        system_cfg[
            "reflection_weight"
        ]
    )

    (
        g,
        h_t,
        h_r,
        d_t,
        d_r,
    ) = sample_channels(
        config=config,
        rng=rng,
        num_elements=n,
    )

    random_surface = (
        design_random_passive_surface(
            num_elements=n,
            beta_transmission=beta,
            rng=rng,
        )
    )

    random_metrics = (
        evaluate_two_user_system(
            g,
            h_t,
            h_r,
            random_surface,
            p_tx,
            sigma_user,
            sigma_ris,
            p_ris_max,
            wt,
            wr,
            d_t,
            d_r,
        )
    )

    passive_surface = (
        design_passive_surface(
            g,
            h_t,
            h_r,
            beta,
            d_t,
            d_r,
        )
    )

    passive_metrics = (
        evaluate_two_user_system(
            g,
            h_t,
            h_r,
            passive_surface,
            p_tx,
            sigma_user,
            sigma_ris,
            p_ris_max,
            wt,
            wr,
            d_t,
            d_r,
        )
    )

    active_design = (
        design_active_surface(
            g,
            h_t,
            h_r,
            beta,
            k_active,
            p_tx,
            sigma_ris,
            p_ris_max,
            a_max,
            wt,
            wr,
            d_t,
            d_r,
        )
    )

    active_metrics = (
        evaluate_two_user_system(
            g,
            h_t,
            h_r,
            active_design.surface,
            p_tx,
            sigma_user,
            sigma_ris,
            p_ris_max,
            wt,
            wr,
            d_t,
            d_r,
        )
    )

    beta_grid = np.linspace(
        float(
            opt_cfg["beta_grid_min"]
        ),
        float(
            opt_cfg["beta_grid_max"]
        ),
        int(
            opt_cfg["beta_grid_points"]
        ),
    )

    (
        best_beta,
        optimized_design,
        optimized_metrics,
    ) = optimize_scalar_energy_split(
        g,
        h_t,
        h_r,
        k_active,
        p_tx,
        sigma_user,
        sigma_ris,
        p_ris_max,
        a_max,
        beta_grid,
        wt,
        wr,
        d_t,
        d_r,
    )

    (
        active_gain_mean,
        active_gain_std,
        active_gain_min,
        active_gain_max,
    ) = summarize_active_gains(
        amplitude_gain=(
            active_design
            .surface
            .amplitude_gain
        ),
        active_mask=(
            active_design.active_mask
        ),
    )

    (
        optimized_gain_mean,
        optimized_gain_std,
        optimized_gain_min,
        optimized_gain_max,
    ) = summarize_active_gains(
        amplitude_gain=(
            optimized_design
            .surface
            .amplitude_gain
        ),
        active_mask=(
            optimized_design.active_mask
        ),
    )

    return {
        "passive_random": (
            metrics_to_result(
                random_metrics,
                beta,
                1.0,
            )
        ),
        "passive_phase_aligned": (
            metrics_to_result(
                passive_metrics,
                beta,
                1.0,
            )
        ),
        "partial_active_fixed_beta": (
            metrics_to_result(
                active_metrics,
                beta,
                active_gain_mean,
                requested_active_elements=(
                    active_design
                    .requested_active_elements
                ),
                effective_active_elements=(
                    active_design
                    .effective_active_elements
                ),
                active_gain_std=(
                    active_gain_std
                ),
                active_gain_min=(
                    active_gain_min
                ),
                active_gain_max=(
                    active_gain_max
                ),
            )
        ),
        "partial_active_optimized_beta": (
            metrics_to_result(
                optimized_metrics,
                best_beta,
                optimized_gain_mean,
                requested_active_elements=(
                    optimized_design
                    .requested_active_elements
                ),
                effective_active_elements=(
                    optimized_design
                    .effective_active_elements
                ),
                active_gain_std=(
                    optimized_gain_std
                ),
                active_gain_min=(
                    optimized_gain_min
                ),
                active_gain_max=(
                    optimized_gain_max
                ),
            )
        ),
    }


def evaluate_one_realization_imperfect_csi(
    config: dict[str, Any],
    rng: np.random.Generator,
    num_elements: int | None = None,
    num_active_elements: int | None = None,
    nmse_db: float | None = None,
) -> tuple[
    float,
    dict[str, ScenarioResult],
]:
    """在不完美CSI和鲁棒功率约束下评价一次系统。

    处理原则：

    1. 估计CSI用于设计STAR-RIS参数；
    2. 真实信道用于计算真实性能；
    3. 有源增益同时满足三个探测方向的鲁棒功率约束；
    4. 最终报告真实三方向输出功率和总表面功耗。
    """

    system_cfg = config["system"]
    opt_cfg = config["optimization"]
    csi_cfg = config["csi"]
    power_cfg = config[
        "surface_power"
    ]

    elementwise_beta_min = float(
        opt_cfg.get(
            "elementwise_beta_min",
            0.05,
        )
    )

    elementwise_beta_max = float(
        opt_cfg.get(
            "elementwise_beta_max",
            0.95,
        )
    )

    elementwise_beta_temperature = float(
        opt_cfg.get(
            "elementwise_beta_temperature",
            1.0,
        )
    )

    n = int(
        system_cfg["num_elements"]
        if num_elements is None
        else num_elements
    )

    k_active = int(
        system_cfg[
            "num_active_elements"
        ]
        if num_active_elements is None
        else num_active_elements
    )

    p_tx = float(
        system_cfg["transmit_power"]
    )

    sigma_user = float(
        system_cfg[
            "user_noise_variance"
        ]
    )

    sigma_ris = float(
        system_cfg[
            "ris_internal_noise_variance"
        ]
    )

    p_ris_max = float(
        system_cfg[
            "ris_output_power_budget"
        ]
    )

    a_max = float(
        system_cfg[
            "maximum_active_amplitude"
        ]
    )

    beta = float(
        system_cfg[
            "beta_transmission"
        ]
    )

    wt = float(
        system_cfg[
            "transmission_weight"
        ]
    )

    wr = float(
        system_cfg[
            "reflection_weight"
        ]
    )

    controller_pilot_power = float(
        power_cfg[
            "controller_pilot_power"
        ]
    )

    transmission_user_pilot_power = float(
        power_cfg[
            "transmission_user_pilot_power"
        ]
    )

    reflection_user_pilot_power = float(
        power_cfg[
            "reflection_user_pilot_power"
        ]
    )

    robust_margin_multiplier = float(
        power_cfg[
            "robust_margin_multiplier"
        ]
    )

    amplifier_efficiency = float(
        power_cfg[
            "amplifier_efficiency"
        ]
    )

    controller_static_power = float(
        power_cfg[
            "controller_static_power"
        ]
    )

    active_element_bias_power = float(
        power_cfg[
            "active_element_bias_power"
        ]
    )

    selected_nmse_db = (
        float(
            csi_cfg[
                "default_nmse_db"
            ]
        )
        if nmse_db is None
        else float(nmse_db)
    )

    if not np.isfinite(
        selected_nmse_db
    ):
        raise ValueError(
            "nmse_db must be finite"
        )

    channel_snapshot = (
        sample_channels_with_imperfect_csi(
            config=config,
            rng=rng,
            num_elements=n,
            nmse_db=selected_nmse_db,
        )
    )

    # -------------------------------------------------
    # 真实信道：只用于计算真实性能和实际功耗。
    # -------------------------------------------------
    true_channel = (
        channel_snapshot.true
    )

    g_true = (
        true_channel.alice_to_ris
    )

    h_t_true = (
        true_channel
        .ris_to_transmission_user
    )

    h_r_true = (
        true_channel
        .ris_to_reflection_user
    )

    d_t_true = (
        true_channel
        .direct_transmission
    )

    d_r_true = (
        true_channel
        .direct_reflection
    )

    # -------------------------------------------------
    # 估计信道：用于设计STAR-RIS参数。
    # -------------------------------------------------
    estimated_channel = (
        channel_snapshot.estimated
    )

    g_hat = (
        estimated_channel.alice_to_ris
    )

    h_t_hat = (
        estimated_channel
        .ris_to_transmission_user
    )

    h_r_hat = (
        estimated_channel
        .ris_to_reflection_user
    )

    d_t_hat = (
        estimated_channel
        .direct_transmission
    )

    d_r_hat = (
        estimated_channel
        .direct_reflection
    )

    # -------------------------------------------------
    # 方案1：随机无源STAR-RIS。
    #
    # 随机无源方案不使用CSI，因此不同NMSE条件下，
    # 在配对随机种子下应得到相同随机表面。
    # -------------------------------------------------
    random_surface = (
        design_random_passive_surface(
            num_elements=n,
            beta_transmission=beta,
            rng=rng,
        )
    )

    random_metrics = (
        evaluate_two_user_system(
            g_true,
            h_t_true,
            h_r_true,
            random_surface,
            p_tx,
            sigma_user,
            sigma_ris,
            p_ris_max,
            wt,
            wr,
            d_t_true,
            d_r_true,
        )
    )

    # -------------------------------------------------
    # 方案2：无源相位对齐。
    #
    # 用估计CSI设计相移，用真实信道计算真实性能。
    # -------------------------------------------------
    passive_surface = (
        design_passive_surface(
            g_hat,
            h_t_hat,
            h_r_hat,
            beta,
            d_t_hat,
            d_r_hat,
        )
    )

    passive_metrics = (
        evaluate_two_user_system(
            g_true,
            h_t_true,
            h_r_true,
            passive_surface,
            p_tx,
            sigma_user,
            sigma_ris,
            p_ris_max,
            wt,
            wr,
            d_t_true,
            d_r_true,
        )
    )

    # -------------------------------------------------
    # 方案3：部分有源STAR-RIS，固定beta。
    #
    # 有源单元选择、相移和逐单元增益均使用估计CSI。
    # 逐单元增益使用三方向鲁棒功率投影。
    # -------------------------------------------------
    active_design = (
        design_active_surface_vector_robust(
            alice_to_ris=g_hat,
            ris_to_transmission_user=(
                h_t_hat
            ),
            ris_to_reflection_user=(
                h_r_hat
            ),
            beta_transmission=beta,
            num_active_elements=(
                k_active
            ),
            transmit_power=(
                controller_pilot_power
            ),
            ris_internal_noise_variance=(
                sigma_ris
            ),
            ris_output_power_budget=(
                p_ris_max
            ),
            maximum_active_amplitude=(
                a_max
            ),
            transmission_weight=wt,
            reflection_weight=wr,
            direct_transmission=(
                d_t_hat
            ),
            direct_reflection=(
                d_r_hat
            ),
            nmse_db=(
                selected_nmse_db
            ),
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
    )

    active_metrics = (
        evaluate_two_user_system(
            g_true,
            h_t_true,
            h_r_true,
            active_design.surface,
            p_tx,
            sigma_user,
            sigma_ris,
            p_ris_max,
            wt,
            wr,
            d_t_true,
            d_r_true,
        )
    )

    # -------------------------------------------------
    # 方案4：部分有源STAR-RIS，标量beta搜索。
    #
    # 标量beta搜索使用估计CSI和逐单元增益鲁棒投影。
    # 得到最优设计后，再在真实信道上评价。
    # -------------------------------------------------
    beta_grid = np.linspace(
        float(
            opt_cfg[
                "beta_grid_min"
            ]
        ),
        float(
            opt_cfg[
                "beta_grid_max"
            ]
        ),
        int(
            opt_cfg[
                "beta_grid_points"
            ]
        ),
    )

    (
        best_beta,
        optimized_design,
        _estimated_optimized_metrics,
    ) = (
        optimize_scalar_energy_split_vector_robust(
            alice_to_ris=g_hat,
            ris_to_transmission_user=(
                h_t_hat
            ),
            ris_to_reflection_user=(
                h_r_hat
            ),
            num_active_elements=(
                k_active
            ),
            transmit_power=(
                controller_pilot_power
            ),
            user_noise_variance=(
                sigma_user
            ),
            ris_internal_noise_variance=(
                sigma_ris
            ),
            ris_output_power_budget=(
                p_ris_max
            ),
            maximum_active_amplitude=(
                a_max
            ),
            beta_grid=beta_grid,
            transmission_weight=wt,
            reflection_weight=wr,
            direct_transmission=(
                d_t_hat
            ),
            direct_reflection=(
                d_r_hat
            ),
            nmse_db=(
                selected_nmse_db
            ),
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
    )

    optimized_metrics = (
        evaluate_two_user_system(
            g_true,
            h_t_true,
            h_r_true,
            optimized_design.surface,
            p_tx,
            sigma_user,
            sigma_ris,
            p_ris_max,
            wt,
            wr,
            d_t_true,
            d_r_true,
        )
    )

    (
        vector_beta,
        vector_beta_design,
    ) = (
        design_active_surface_vector_beta_robust(
            alice_to_ris=g_hat,
            ris_to_transmission_user=(
                h_t_hat
            ),
            ris_to_reflection_user=(
                h_r_hat
            ),
            num_active_elements=(
                k_active
            ),
            transmit_power=(
                controller_pilot_power
            ),
            ris_internal_noise_variance=(
                sigma_ris
            ),
            ris_output_power_budget=(
                p_ris_max
            ),
            maximum_active_amplitude=(
                a_max
            ),
            transmission_weight=wt,
            reflection_weight=wr,
            direct_transmission=(
                d_t_hat
            ),
            direct_reflection=(
                d_r_hat
            ),
            nmse_db=(
                selected_nmse_db
            ),
            robust_margin_multiplier=(
                robust_margin_multiplier
            ),
            transmission_user_pilot_power=(
                transmission_user_pilot_power
            ),
            reflection_user_pilot_power=(
                reflection_user_pilot_power
            ),
            beta_min=(
                elementwise_beta_min
            ),
            beta_max=(
                elementwise_beta_max
            ),
            beta_temperature=(
                elementwise_beta_temperature
            ),
        )
    )

    vector_beta_metrics = (
        evaluate_two_user_system(
            g_true,
            h_t_true,
            h_r_true,
            vector_beta_design.surface,
            p_tx,
            sigma_user,
            sigma_ris,
            p_ris_max,
            wt,
            wr,
            d_t_true,
            d_r_true,
        )
    )

    # -------------------------------------------------
    # 使用真实信道计算三个探测方向的实际功率。
    # -------------------------------------------------
    common_power_arguments = {
        "controller_to_ris": (
            g_true
        ),
        "transmission_user_to_ris": (
            h_t_true
        ),
        "reflection_user_to_ris": (
            h_r_true
        ),
        "controller_pilot_power": (
            controller_pilot_power
        ),
        "transmission_user_pilot_power": (
            transmission_user_pilot_power
        ),
        "reflection_user_pilot_power": (
            reflection_user_pilot_power
        ),
        "ris_internal_noise_variance": (
            sigma_ris
        ),
        "output_power_budget": (
            p_ris_max
        ),
        "amplifier_efficiency": (
            amplifier_efficiency
        ),
        "controller_static_power": (
            controller_static_power
        ),
        "active_element_bias_power": (
            active_element_bias_power
        ),
    }

    random_power = (
        evaluate_bidirectional_surface_power(
            surface=random_surface,
            **common_power_arguments,
        )
    )

    passive_power = (
        evaluate_bidirectional_surface_power(
            surface=passive_surface,
            **common_power_arguments,
        )
    )

    active_power = (
        evaluate_bidirectional_surface_power(
            surface=active_design.surface,
            **common_power_arguments,
        )
    )

    optimized_power = (
        evaluate_bidirectional_surface_power(
            surface=(
                optimized_design.surface
            ),
            **common_power_arguments,
        )
    )

    vector_beta_power = (
        evaluate_bidirectional_surface_power(
            surface=(
                vector_beta_design.surface
            ),
            **common_power_arguments,
        )
    )

    (
        active_gain_mean,
        active_gain_std,
        active_gain_min,
        active_gain_max,
    ) = summarize_active_gains(
        amplitude_gain=(
            active_design
            .surface
            .amplitude_gain
        ),
        active_mask=(
            active_design.active_mask
        ),
    )

    (
        optimized_gain_mean,
        optimized_gain_std,
        optimized_gain_min,
        optimized_gain_max,
    ) = summarize_active_gains(
        amplitude_gain=(
            optimized_design
            .surface
            .amplitude_gain
        ),
        active_mask=(
            optimized_design.active_mask
        ),
    )

    (
        vector_beta_gain_mean,
        vector_beta_gain_std,
        vector_beta_gain_min,
        vector_beta_gain_max,
    ) = summarize_active_gains(
        amplitude_gain=(
            vector_beta_design
            .surface
            .amplitude_gain
        ),
        active_mask=(
            vector_beta_design
            .active_mask
        ),
    )

    (
        vector_beta_mean,
        vector_beta_std,
        vector_beta_minimum,
        vector_beta_maximum,
    ) = summarize_transmission_split(
        beta_transmission=(
            vector_beta
        ),
        num_elements=n,
    )

    results = {
        "passive_random": (
            metrics_to_result(
                random_metrics,
                beta,
                1.0,
                bidirectional_power=(
                    random_power
                ),
            )
        ),
        "passive_phase_aligned": (
            metrics_to_result(
                passive_metrics,
                beta,
                1.0,
                bidirectional_power=(
                    passive_power
                ),
            )
        ),
        "partial_active_fixed_beta": (
            metrics_to_result(
                active_metrics,
                beta,
                active_gain_mean,
                bidirectional_power=(
                    active_power
                ),
                requested_active_elements=(
                    active_design
                    .requested_active_elements
                ),
                effective_active_elements=(
                    active_design
                    .effective_active_elements
                ),
                active_gain_std=(
                    active_gain_std
                ),
                active_gain_min=(
                    active_gain_min
                ),
                active_gain_max=(
                    active_gain_max
                ),
            )
        ),
        "partial_active_optimized_beta": (
            metrics_to_result(
                optimized_metrics,
                best_beta,
                optimized_gain_mean,
                bidirectional_power=(
                    optimized_power
                ),
                requested_active_elements=(
                    optimized_design
                    .requested_active_elements
                ),
                effective_active_elements=(
                    optimized_design
                    .effective_active_elements
                ),
                active_gain_std=(
                    optimized_gain_std
                ),
                active_gain_min=(
                    optimized_gain_min
                ),
                active_gain_max=(
                    optimized_gain_max
                ),
            )
        ),
        "partial_active_vector_beta": (
            metrics_to_result(
                vector_beta_metrics,
                vector_beta_mean,
                vector_beta_gain_mean,
                bidirectional_power=(
                    vector_beta_power
                ),
                requested_active_elements=(
                    vector_beta_design
                    .requested_active_elements
                ),
                effective_active_elements=(
                    vector_beta_design
                    .effective_active_elements
                ),
                active_gain_std=(
                    vector_beta_gain_std
                ),
                active_gain_min=(
                    vector_beta_gain_min
                ),
                active_gain_max=(
                    vector_beta_gain_max
                ),
                beta_transmission_std=(
                    vector_beta_std
                ),
                beta_transmission_min=(
                    vector_beta_minimum
                ),
                beta_transmission_max=(
                    vector_beta_maximum
                ),
            )
        ),
    }

    return (
        channel_snapshot.nmse_db,
        results,
    )


def monte_carlo(
    config: dict[str, Any],
    num_trials: int,
    seed: int,
    num_elements: int | None = None,
    num_active_elements: int | None = None,
) -> dict[
    str,
    list[ScenarioResult],
]:
    """运行原有完美CSI蒙特卡洛实验。"""

    rng = np.random.default_rng(
        seed
    )

    collected: dict[
        str,
        list[ScenarioResult],
    ] = {}

    for _ in range(
        num_trials
    ):
        realization = (
            evaluate_one_realization(
                config=config,
                rng=rng,
                num_elements=(
                    num_elements
                ),
                num_active_elements=(
                    num_active_elements
                ),
            )
        )

        for (
            name,
            result,
        ) in realization.items():
            collected.setdefault(
                name,
                [],
            ).append(result)

    return collected


def summarize_results(
    results: dict[
        str,
        list[ScenarioResult],
    ],
) -> dict[
    str,
    dict[str, float],
]:
    """计算每个方案各指标的均值和标准差。"""

    summary: dict[
        str,
        dict[str, float],
    ] = {}

    for (
        name,
        entries,
    ) in results.items():
        fields = (
            ScenarioResult
            .__dataclass_fields__
            .keys()
        )

        summary[name] = {}

        for field in fields:
            values = np.asarray(
                [
                    getattr(
                        entry,
                        field,
                    )
                    for entry in entries
                ],
                dtype=float,
            )

            summary[name][
                f"mean_{field}"
            ] = float(
                np.mean(values)
            )

            # 避免只有一次试验时，ddof=1产生NaN。
            if values.size > 1:
                std_value = float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
            else:
                std_value = 0.0

            summary[name][
                f"std_{field}"
            ] = std_value

        summary[name][
            "num_trials"
        ] = float(
            len(entries)
        )

    return summary


def results_as_serializable(
    results: dict[
        str,
        list[ScenarioResult],
    ],
) -> dict[
    str,
    list[dict[str, float]],
]:
    """将仿真结果转换为可写入JSON的字典。"""

    return {
        name: [
            asdict(entry)
            for entry in entries
        ]
        for name, entries
        in results.items()
    }