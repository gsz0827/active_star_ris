from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from active_star_ris.secure_key_generation import (
    complex_gaussian_mutual_information,
)

from .channels import complex_normal
from .models import ComplexArray


@dataclass(frozen=True)
class EveBranchObservation:
    observation_forward: ComplexArray
    observation_reverse: ComplexArray

    effective_channel_forward: ComplexArray
    effective_channel_reverse: ComplexArray

    forwarded_active_noise_forward: ComplexArray
    forwarded_active_noise_reverse: ComplexArray


def simulate_eve_branch(
    controller_to_ris_forward: np.ndarray,
    user_to_ris_reverse: np.ndarray,
    ris_to_eve_forward: np.ndarray,
    ris_to_eve_reverse: np.ndarray,
    direct_controller_to_eve: np.ndarray,
    direct_user_to_eve: np.ndarray,
    phi_forward: np.ndarray,
    phi_reverse: np.ndarray,
    active_noise_forward: np.ndarray,
    active_noise_reverse: np.ndarray,
    *,
    pilot_power_forward: float,
    pilot_power_reverse: float,
    pilot_symbols_forward: int,
    pilot_symbols_reverse: int,
    receiver_noise_variance: float,
    tx_coefficient_forward: complex,
    tx_coefficient_reverse: complex,
    rng: np.random.Generator,
) -> EveBranchObservation:
    """模拟Eve对一个STAR-RIS分支的正反向监听。"""

    if pilot_power_forward <= 0.0:
        raise ValueError(
            "pilot_power_forward must be positive"
        )
    if pilot_power_reverse <= 0.0:
        raise ValueError(
            "pilot_power_reverse must be positive"
        )
    if pilot_symbols_forward < 1:
        raise ValueError(
            "pilot_symbols_forward must be positive"
        )
    if pilot_symbols_reverse < 1:
        raise ValueError(
            "pilot_symbols_reverse must be positive"
        )
    if receiver_noise_variance < 0.0:
        raise ValueError(
            "receiver_noise_variance cannot be negative"
        )

    g_forward = np.asarray(
        controller_to_ris_forward,
        dtype=np.complex128,
    )
    h_user_reverse = np.asarray(
        user_to_ris_reverse,
        dtype=np.complex128,
    )
    h_eve_forward = np.asarray(
        ris_to_eve_forward,
        dtype=np.complex128,
    )
    h_eve_reverse = np.asarray(
        ris_to_eve_reverse,
        dtype=np.complex128,
    )

    if g_forward.ndim != 2:
        raise ValueError(
            "controller_to_ris_forward must be two-dimensional"
        )

    expected_shape = g_forward.shape

    for name, values in (
        ("user_to_ris_reverse", h_user_reverse),
        ("ris_to_eve_forward", h_eve_forward),
        ("ris_to_eve_reverse", h_eve_reverse),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: "
                f"expected {expected_shape}, got {values.shape}"
            )

    samples, num_elements = expected_shape

    phi_f = np.asarray(
        phi_forward,
        dtype=np.complex128,
    ).reshape(-1)

    phi_r = np.asarray(
        phi_reverse,
        dtype=np.complex128,
    ).reshape(-1)

    if (
        phi_f.size != num_elements
        or phi_r.size != num_elements
    ):
        raise ValueError(
            "surface coefficient size mismatch"
        )

    noise_f = np.asarray(
        active_noise_forward,
        dtype=np.complex128,
    )
    noise_r = np.asarray(
        active_noise_reverse,
        dtype=np.complex128,
    )

    if (
        noise_f.shape != expected_shape
        or noise_r.shape != expected_shape
    ):
        raise ValueError(
            "active noise matrix shape mismatch"
        )

    direct_f = np.asarray(
        direct_controller_to_eve,
        dtype=np.complex128,
    ).reshape(-1)

    direct_r = np.asarray(
        direct_user_to_eve,
        dtype=np.complex128,
    ).reshape(-1)

    if direct_f.size != samples:
        raise ValueError(
            "direct_controller_to_eve sample count mismatch"
        )

    if direct_r.size != samples:
        raise ValueError(
            "direct_user_to_eve sample count mismatch"
        )

    # Controller -> RIS -> Eve
    effective_forward = (
        direct_f
        + np.sum(
            g_forward
            * phi_f[None, :]
            * h_eve_forward,
            axis=1,
        )
    ) * tx_coefficient_forward

    # User -> RIS -> Eve
    effective_reverse = (
        direct_r
        + np.sum(
            h_user_reverse
            * phi_r[None, :]
            * h_eve_reverse,
            axis=1,
        )
    ) * tx_coefficient_reverse

    # Eve接收到与合法接收端相同的RIS内部噪声实现，
    # 但经过Eve自己的RIS-Eve信道传播。
    forwarded_forward = np.sum(
        h_eve_forward
        * phi_f[None, :]
        * noise_f,
        axis=1,
    )

    forwarded_reverse = np.sum(
        h_eve_reverse
        * phi_r[None, :]
        * noise_r,
        axis=1,
    )

    receiver_noise_forward = complex_normal(
        rng,
        samples,
        variance=receiver_noise_variance,
    )

    receiver_noise_reverse = complex_normal(
        rng,
        samples,
        variance=receiver_noise_variance,
    )

    estimation_scale_forward = np.sqrt(
        max(
            pilot_power_forward
            * pilot_symbols_forward,
            1.0e-12,
        )
    )

    estimation_scale_reverse = np.sqrt(
        max(
            pilot_power_reverse
            * pilot_symbols_reverse,
            1.0e-12,
        )
    )

    observation_forward = (
        effective_forward
        + forwarded_forward
        / estimation_scale_forward
        + receiver_noise_forward
        / estimation_scale_forward
    )

    observation_reverse = (
        effective_reverse
        + forwarded_reverse
        / estimation_scale_reverse
        + receiver_noise_reverse
        / estimation_scale_reverse
    )

    return EveBranchObservation(
        observation_forward=np.asarray(
            observation_forward,
            dtype=np.complex128,
        ),
        observation_reverse=np.asarray(
            observation_reverse,
            dtype=np.complex128,
        ),
        effective_channel_forward=np.asarray(
            effective_forward,
            dtype=np.complex128,
        ),
        effective_channel_reverse=np.asarray(
            effective_reverse,
            dtype=np.complex128,
        ),
        forwarded_active_noise_forward=np.asarray(
            forwarded_forward,
            dtype=np.complex128,
        ),
        forwarded_active_noise_reverse=np.asarray(
            forwarded_reverse,
            dtype=np.complex128,
        ),
    )


def estimate_eve_leakage_bits_per_sample(
    observation_a: np.ndarray,
    observation_b: np.ndarray,
    eve_observation: EveBranchObservation,
) -> float:
    """估计Eve对一个二进制原始密钥样本的泄漏量。

    Eve联合利用正向和反向监听结果：
        Z_E = [Z_forward, Z_reverse]

    返回值限制在[0, 1]，因为当前量化器每个保留
    样本最多生成一个二进制原始密钥比特。
    """
    a = np.asarray(
        observation_a,
        dtype=np.complex128,
    ).reshape(-1)

    b = np.asarray(
        observation_b,
        dtype=np.complex128,
    ).reshape(-1)

    eve_forward = np.asarray(
        eve_observation.observation_forward,
        dtype=np.complex128,
    ).reshape(-1)

    eve_reverse = np.asarray(
        eve_observation.observation_reverse,
        dtype=np.complex128,
    ).reshape(-1)

    if not (
        a.size
        == b.size
        == eve_forward.size
        == eve_reverse.size
    ):
        raise ValueError(
            "legitimate and Eve observations "
            "must have equal sample counts"
        )

    eve_matrix = np.column_stack(
        (
            eve_forward,
            eve_reverse,
        )
    )

    leakage_from_a = (
        complex_gaussian_mutual_information(
            a,
            eve_matrix,
        )
    )

    leakage_from_b = (
        complex_gaussian_mutual_information(
            b,
            eve_matrix,
        )
    )

    leakage = max(
        leakage_from_a,
        leakage_from_b,
    )

    return float(
        np.clip(
            leakage,
            0.0,
            1.0,
        )
    )