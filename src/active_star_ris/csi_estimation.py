from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .channels import complex_normal

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class CSIErrorConfig:
    """CSI估计误差配置。

    nmse_db_min和nmse_db_max用于定义训练阶段随机采样的NMSE范围。
    例如：
        -25 dB表示误差较小；
        -10 dB表示误差相对较大。
    """

    nmse_db_min: float = -25.0
    nmse_db_max: float = -10.0

    def validate(self) -> None:
        if self.nmse_db_min > self.nmse_db_max:
            raise ValueError(
                "nmse_db_min must not exceed nmse_db_max"
            )


@dataclass(frozen=True)
class CSIRealization:
    """一次不完美CSI生成结果。"""

    true_channel: ComplexArray
    estimated_channel: ComplexArray
    estimation_error: ComplexArray
    target_nmse_linear: float


def generate_imperfect_csi(
    true_channel: ArrayLike,
    nmse_db: float,
    rng: np.random.Generator,
) -> CSIRealization:
    """根据真实信道和目标NMSE生成估计CSI。

    模型：
        h_hat = h_true + e

    其中e为零均值复高斯估计误差。

    Parameters
    ----------
    true_channel:
        环境中的真实复信道。
    nmse_db:
        目标归一化均方误差，单位为dB。
    rng:
        NumPy随机数生成器。

    Returns
    -------
    CSIRealization
        包含真实信道、估计信道和估计误差。
    """

    true = np.asarray(
        true_channel,
        dtype=np.complex128,
    )

    if true.size == 0:
        raise ValueError(
            "true_channel must not be empty"
        )

    # 真实信道的平均功率。
    signal_power = max(
        float(np.mean(np.abs(true) ** 2)),
        1.0e-12,
    )

    # 将dB形式的NMSE转换为线性值。
    nmse_linear = float(
        10.0 ** (nmse_db / 10.0)
    )

    # 根据目标NMSE确定复高斯误差方差。
    error_variance = (
        signal_power * nmse_linear
    )

    estimation_error = complex_normal(
        true.shape,
        rng,
        variance=error_variance,
    )

    estimated_channel = (
        true + estimation_error
    )

    return CSIRealization(
        true_channel=true,
        estimated_channel=np.asarray(
            estimated_channel,
            dtype=np.complex128,
        ),
        estimation_error=np.asarray(
            estimation_error,
            dtype=np.complex128,
        ),
        target_nmse_linear=nmse_linear,
    )


def sample_nmse_db(
    config: CSIErrorConfig,
    rng: np.random.Generator,
) -> float:
    """在指定NMSE范围内随机采样一个误差水平。"""

    config.validate()

    return float(
        rng.uniform(
            config.nmse_db_min,
            config.nmse_db_max,
        )
    )


def calculate_empirical_nmse(
    true_channel: ArrayLike,
    estimated_channel: ArrayLike,
) -> float:
    """计算一次实现中的实际NMSE线性值。"""

    true = np.asarray(
        true_channel,
        dtype=np.complex128,
    )
    estimate = np.asarray(
        estimated_channel,
        dtype=np.complex128,
    )

    if true.shape != estimate.shape:
        raise ValueError(
            "true_channel and estimated_channel "
            "must have the same shape"
        )

    denominator = max(
        float(np.mean(np.abs(true) ** 2)),
        1.0e-12,
    )

    numerator = float(
        np.mean(
            np.abs(estimate - true) ** 2
        )
    )

    return numerator / denominator