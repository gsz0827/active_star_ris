from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from active_star_ris.key_generation import (
    KeyGenerationMetrics,
    evaluate_key_generation,
)
from active_star_ris.probing import (
    BidirectionalProbingResult,
    simulate_bidirectional_probing,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StarCoefficientPair:
    """
    STAR-RIS透射侧和反射侧复系数。

    transmission:
        透射侧复系数：

            phi_T,n
            =
            a_n * sqrt(beta_T,n)
            * exp(j * theta_T,n)

    reflection:
        反射侧复系数：

            phi_R,n
            =
            a_n * sqrt(beta_R,n)
            * exp(j * theta_R,n)

    amplitudes:
        每个单元的总幅度增益a_n。

        无源单元通常满足：

            a_n = 1

        有源单元可以满足：

            a_n > 1

    beta_transmission:
        每个单元分配给透射侧的能量比例。

    beta_reflection:
        每个单元分配给反射侧的能量比例。
    """

    transmission: ComplexArray
    reflection: ComplexArray

    amplitudes: FloatArray
    beta_transmission: FloatArray
    beta_reflection: FloatArray


@dataclass(frozen=True)
class BranchKeyGenerationResult:
    """
    单个STAR-RIS分支的密钥生成结果。

    一个分支可以是：

    1. 透射侧链路；
    2. 反射侧链路。
    """

    probing: BidirectionalProbingResult
    metrics: KeyGenerationMetrics


@dataclass(frozen=True)
class DualSideKeyGenerationResult:
    """
    STAR-RIS透射侧与反射侧联合密钥生成结果。
    """

    transmission: BranchKeyGenerationResult
    reflection: BranchKeyGenerationResult

    weighted_mutual_information: float
    weighted_key_disagreement_rate: float
    weighted_correlation: float


def _real_vector(
    values: ArrayLike,
    name: str,
) -> FloatArray:
    """
    将输入转换为一维float64数组并进行基本检查。
    """
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if array.size == 0:
        raise ValueError(
            f"{name} cannot be empty"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains non-finite values"
        )

    return array


def _validate_equal_lengths(
    arrays: dict[str, np.ndarray],
) -> int:
    """
    检查多个STAR-RIS参数向量长度是否一致。

    返回：
        STAR-RIS单元数量。
    """
    lengths = {
        name: array.size
        for name, array in arrays.items()
    }

    unique_lengths = set(
        lengths.values()
    )

    if len(unique_lengths) != 1:
        description = ", ".join(
            f"{name}={length}"
            for name, length in lengths.items()
        )

        raise ValueError(
            "all STAR-RIS parameter vectors must "
            f"have equal lengths: {description}"
        )

    return next(
        iter(unique_lengths)
    )


def _validate_coefficient_pair(
    coefficients: StarCoefficientPair,
    name: str,
) -> int:
    """
    检查一个StarCoefficientPair内部各向量长度是否一致。
    """
    return _validate_equal_lengths(
        {
            f"{name}.transmission": (
                coefficients.transmission
            ),
            f"{name}.reflection": (
                coefficients.reflection
            ),
            f"{name}.amplitudes": (
                coefficients.amplitudes
            ),
            f"{name}.beta_transmission": (
                coefficients.beta_transmission
            ),
            f"{name}.beta_reflection": (
                coefficients.beta_reflection
            ),
        }
    )


def build_star_coefficients(
    amplitudes: ArrayLike,
    beta_transmission: ArrayLike,
    beta_reflection: ArrayLike,
    phase_transmission: ArrayLike,
    phase_reflection: ArrayLike,
    *,
    energy_tolerance: float = 1.0e-9,
) -> StarCoefficientPair:
    """
    根据幅度增益、能量分配和相移构造STAR-RIS复系数。

    透射侧系数：

        phi_T,n
        =
        a_n * sqrt(beta_T,n)
        * exp(j * theta_T,n)

    反射侧系数：

        phi_R,n
        =
        a_n * sqrt(beta_R,n)
        * exp(j * theta_R,n)

    能量分配约束：

        beta_T,n + beta_R,n = 1

    注意：
        对有源单元而言，a_n可以大于1，因此：

            |phi_T,n|^2 + |phi_R,n|^2
            =
            a_n^2

        这里约束的是放大后输出能量在透射和反射两侧的
        分配比例，而不是要求总输出幅度等于1。
    """
    if energy_tolerance < 0.0:
        raise ValueError(
            "energy_tolerance cannot be negative"
        )

    amplitudes_array = _real_vector(
        amplitudes,
        "amplitudes",
    )

    beta_t = _real_vector(
        beta_transmission,
        "beta_transmission",
    )

    beta_r = _real_vector(
        beta_reflection,
        "beta_reflection",
    )

    theta_t = _real_vector(
        phase_transmission,
        "phase_transmission",
    )

    theta_r = _real_vector(
        phase_reflection,
        "phase_reflection",
    )

    _validate_equal_lengths(
        {
            "amplitudes": amplitudes_array,
            "beta_transmission": beta_t,
            "beta_reflection": beta_r,
            "phase_transmission": theta_t,
            "phase_reflection": theta_r,
        }
    )

    if np.any(amplitudes_array < 0.0):
        raise ValueError(
            "amplitudes cannot be negative"
        )

    if np.any(
        (beta_t < 0.0)
        | (beta_t > 1.0)
    ):
        raise ValueError(
            "beta_transmission must lie within [0, 1]"
        )

    if np.any(
        (beta_r < 0.0)
        | (beta_r > 1.0)
    ):
        raise ValueError(
            "beta_reflection must lie within [0, 1]"
        )

    if not np.allclose(
        beta_t + beta_r,
        1.0,
        atol=energy_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            "each STAR-RIS element must satisfy "
            "beta_transmission + beta_reflection = 1"
        )

    phi_t = (
        amplitudes_array
        * np.sqrt(beta_t)
        * np.exp(1j * theta_t)
    )

    phi_r = (
        amplitudes_array
        * np.sqrt(beta_r)
        * np.exp(1j * theta_r)
    )

    return StarCoefficientPair(
        transmission=np.asarray(
            phi_t,
            dtype=np.complex128,
        ),
        reflection=np.asarray(
            phi_r,
            dtype=np.complex128,
        ),
        amplitudes=np.asarray(
            amplitudes_array,
            dtype=np.float64,
        ),
        beta_transmission=np.asarray(
            beta_t,
            dtype=np.float64,
        ),
        beta_reflection=np.asarray(
            beta_r,
            dtype=np.float64,
        ),
    )


def energy_splitting_residual(
    coefficients: StarCoefficientPair,
) -> FloatArray:
    """
    计算STAR-RIS逐单元能量分配残差。

    残差定义：

        residual_n
        =
        |phi_T,n|^2
        +
        |phi_R,n|^2
        -
        a_n^2

    理想情况下：

        residual_n = 0
    """
    _validate_coefficient_pair(
        coefficients,
        "coefficients",
    )

    allocated_energy = (
        np.abs(
            coefficients.transmission
        ) ** 2
        + np.abs(
            coefficients.reflection
        ) ** 2
    )

    available_energy = (
        coefficients.amplitudes**2
    )

    return np.asarray(
        allocated_energy
        - available_energy,
        dtype=np.float64,
    )


def simulate_dual_side_key_generation(
    channel_controller_to_ris: ArrayLike,
    channel_ris_to_transmission_user: ArrayLike,
    channel_ris_to_reflection_user: ArrayLike,
    coefficients: StarCoefficientPair,
    active_mask: ArrayLike,
    *,
    reverse_coefficients: StarCoefficientPair | None = None,
    direct_channel_transmission: ArrayLike = 0.0j,
    direct_channel_reflection: ArrayLike = 0.0j,
    pilot_power_controller: float = 1.0,
    pilot_power_transmission_user: float = 1.0,
    pilot_power_reflection_user: float = 1.0,
    active_noise_variance: float = 0.0,
    receiver_noise_variance_controller: float = 0.0,
    receiver_noise_variance_transmission_user: float = 0.0,
    receiver_noise_variance_reflection_user: float = 0.0,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    rng: np.random.Generator | None = None,
) -> DualSideKeyGenerationResult:
    """
    模拟STAR-RIS透射侧和反射侧的双向密钥生成。

    系统包含两条密钥生成链路：

    透射侧：

        Controller
        <->
        STAR-RIS
        <->
        Transmission user

    反射侧：

        Controller
        <->
        STAR-RIS
        <->
        Reflection user

    参数：
        coefficients:
            正向信道探测所采用的实际STAR-RIS系数。

        reverse_coefficients:
            反向信道探测所采用的实际STAR-RIS系数。

            若为None，则默认：

                reverse_coefficients = coefficients

            这对应理想互易硬件，或者前向和反向具有相同
            静态实现误差的情况。

            若前向和反向硬件响应存在方向相关失配，则分别
            传入不同的coefficients和reverse_coefficients。

        active_mask:
            STAR-RIS有源单元位置。

        transmission_weight:
            透射侧指标权重。

        reflection_weight:
            反射侧指标权重。

    返回：
        两个分支各自的双向观测和密钥指标，以及联合加权指标。
    """
    coefficient_count = _validate_coefficient_pair(
        coefficients,
        "coefficients",
    )

    coefficients_reverse = (
        coefficients
        if reverse_coefficients is None
        else reverse_coefficients
    )

    reverse_coefficient_count = (
        _validate_coefficient_pair(
            coefficients_reverse,
            "reverse_coefficients",
        )
    )

    if (
        coefficient_count
        != reverse_coefficient_count
    ):
        raise ValueError(
            "forward and reverse STAR-RIS "
            "coefficient vectors must have equal lengths"
        )

    if transmission_weight < 0.0:
        raise ValueError(
            "transmission_weight cannot be negative"
        )

    if reflection_weight < 0.0:
        raise ValueError(
            "reflection_weight cannot be negative"
        )

    weight_sum = (
        transmission_weight
        + reflection_weight
    )

    if weight_sum <= 0.0:
        raise ValueError(
            "at least one branch weight must be positive"
        )

    normalized_transmission_weight = (
        transmission_weight
        / weight_sum
    )

    normalized_reflection_weight = (
        reflection_weight
        / weight_sum
    )

    generator = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    # ---------------------------------------------------------
    # 透射侧双向信道探测
    # ---------------------------------------------------------
    transmission_probing = (
        simulate_bidirectional_probing(
            channel_a_to_ris=(
                channel_controller_to_ris
            ),
            channel_ris_to_b=(
                channel_ris_to_transmission_user
            ),
            surface_coefficients=(
                coefficients.transmission
            ),
            active_mask=active_mask,
            surface_coefficients_reverse=(
                coefficients_reverse.transmission
            ),
            direct_channel=(
                direct_channel_transmission
            ),
            pilot_power_a=(
                pilot_power_controller
            ),
            pilot_power_b=(
                pilot_power_transmission_user
            ),
            active_noise_variance=(
                active_noise_variance
            ),
            receiver_noise_variance_a=(
                receiver_noise_variance_controller
            ),
            receiver_noise_variance_b=(
                receiver_noise_variance_transmission_user
            ),
            rng=generator,
        )
    )

    # ---------------------------------------------------------
    # 反射侧双向信道探测
    # ---------------------------------------------------------
    reflection_probing = (
        simulate_bidirectional_probing(
            channel_a_to_ris=(
                channel_controller_to_ris
            ),
            channel_ris_to_b=(
                channel_ris_to_reflection_user
            ),
            surface_coefficients=(
                coefficients.reflection
            ),
            active_mask=active_mask,
            surface_coefficients_reverse=(
                coefficients_reverse.reflection
            ),
            direct_channel=(
                direct_channel_reflection
            ),
            pilot_power_a=(
                pilot_power_controller
            ),
            pilot_power_b=(
                pilot_power_reflection_user
            ),
            active_noise_variance=(
                active_noise_variance
            ),
            receiver_noise_variance_a=(
                receiver_noise_variance_controller
            ),
            receiver_noise_variance_b=(
                receiver_noise_variance_reflection_user
            ),
            rng=generator,
        )
    )

    # ---------------------------------------------------------
    # 透射侧物理层密钥指标
    # ---------------------------------------------------------
    transmission_metrics = (
        evaluate_key_generation(
            transmission_probing.observation_at_a,
            transmission_probing.observation_at_b,
        )
    )

    # ---------------------------------------------------------
    # 反射侧物理层密钥指标
    # ---------------------------------------------------------
    reflection_metrics = (
        evaluate_key_generation(
            reflection_probing.observation_at_a,
            reflection_probing.observation_at_b,
        )
    )

    # ---------------------------------------------------------
    # 联合加权互信息
    # ---------------------------------------------------------
    weighted_mutual_information = (
        normalized_transmission_weight
        * (
            transmission_metrics
            .mutual_information_bits_per_sample
        )
        + normalized_reflection_weight
        * (
            reflection_metrics
            .mutual_information_bits_per_sample
        )
    )

    # ---------------------------------------------------------
    # 联合加权KDR
    # ---------------------------------------------------------
    weighted_kdr = (
        normalized_transmission_weight
        * (
            transmission_metrics
            .key_disagreement_rate
        )
        + normalized_reflection_weight
        * (
            reflection_metrics
            .key_disagreement_rate
        )
    )

    # ---------------------------------------------------------
    # 联合加权观测相关性
    # ---------------------------------------------------------
    weighted_correlation = (
        normalized_transmission_weight
        * (
            transmission_metrics
            .correlation_magnitude
        )
        + normalized_reflection_weight
        * (
            reflection_metrics
            .correlation_magnitude
        )
    )

    return DualSideKeyGenerationResult(
        transmission=(
            BranchKeyGenerationResult(
                probing=transmission_probing,
                metrics=transmission_metrics,
            )
        ),
        reflection=(
            BranchKeyGenerationResult(
                probing=reflection_probing,
                metrics=reflection_metrics,
            )
        ),
        weighted_mutual_information=float(
            weighted_mutual_information
        ),
        weighted_key_disagreement_rate=float(
            weighted_kdr
        ),
        weighted_correlation=float(
            weighted_correlation
        ),
    )