from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class BidirectionalProbingResult:
    """
    双向信道探测结果。

    forward:
        A -> STAR-RIS -> B

    reverse:
        B -> STAR-RIS -> A
    """

    effective_channel_forward: ComplexArray
    effective_channel_reverse: ComplexArray

    observation_at_a: ComplexArray
    observation_at_b: ComplexArray

    # 每个STAR-RIS单元产生的内部噪声。
    # 形状为(num_samples, num_elements)，且无源单元位置为0。
    active_noise_forward: ComplexArray
    active_noise_reverse: ComplexArray

    forwarded_active_noise_at_a: ComplexArray
    forwarded_active_noise_at_b: ComplexArray

    receiver_noise_at_a: ComplexArray
    receiver_noise_at_b: ComplexArray

    @property
    def effective_channel(
        self,
    ) -> ComplexArray:
        """
        保留旧代码兼容性。
        """
        return self.effective_channel_forward


def _as_channel_matrix(
    values: ArrayLike,
    name: str,
) -> ComplexArray:
    """
    将信道或STAR-RIS系数转换为二维复数矩阵。

    输入形状：
        (num_elements,)
        或
        (num_samples, num_elements)

    输出形状：
        (num_samples, num_elements)
    """
    array = np.asarray(
        values,
        dtype=np.complex128,
    )

    if array.ndim == 1:
        array = array[np.newaxis, :]

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a one-dimensional "
            "or two-dimensional array"
        )

    if array.shape[1] == 0:
        raise ValueError(
            f"{name} cannot contain zero elements"
        )

    if not np.all(
        np.isfinite(array.real)
    ):
        raise ValueError(
            f"{name} contains non-finite real values"
        )

    if not np.all(
        np.isfinite(array.imag)
    ):
        raise ValueError(
            f"{name} contains non-finite imaginary values"
        )

    return array


def _broadcast_channel_rows(
    values: ComplexArray,
    num_samples: int,
    name: str,
) -> ComplexArray:
    """
    将单行信道或系数广播到指定样本数。
    """
    if values.shape[0] == num_samples:
        return values

    if values.shape[0] == 1:
        return np.repeat(
            values,
            num_samples,
            axis=0,
        )

    raise ValueError(
        f"{name} has {values.shape[0]} channel samples, "
        f"but {num_samples} samples are required"
    )


def _broadcast_direct_channel(
    values: ArrayLike,
    num_samples: int,
) -> ComplexArray:
    """
    将直达信道广播为每个样本对应一个复信道值。
    """
    direct = np.asarray(
        values,
        dtype=np.complex128,
    ).reshape(-1)

    if direct.size == 1:
        return np.full(
            num_samples,
            direct.item(),
            dtype=np.complex128,
        )

    if direct.size != num_samples:
        raise ValueError(
            "direct_channel must be a scalar or have "
            "one value per channel sample"
        )

    if not np.all(
        np.isfinite(direct.real)
    ):
        raise ValueError(
            "direct_channel contains non-finite real values"
        )

    if not np.all(
        np.isfinite(direct.imag)
    ):
        raise ValueError(
            "direct_channel contains non-finite imaginary values"
        )

    return direct


def _validate_active_mask(
    active_mask: ArrayLike,
    num_elements: int,
) -> BoolArray:
    """
    检查有源单元掩码。
    """
    mask = np.asarray(
        active_mask,
        dtype=bool,
    ).reshape(-1)

    if mask.size != num_elements:
        raise ValueError(
            "active_mask length must equal the "
            "number of STAR-RIS elements"
        )

    return mask


def _complex_gaussian_noise(
    rng: np.random.Generator,
    variance: float,
    shape: tuple[int, ...],
) -> ComplexArray:
    """
    生成零均值圆对称复高斯噪声。

    满足：

        E[|n|^2] = variance
    """
    if variance < 0.0:
        raise ValueError(
            "noise variance cannot be negative"
        )

    if variance == 0.0:
        return np.zeros(
            shape,
            dtype=np.complex128,
        )

    standard_deviation = np.sqrt(
        variance / 2.0
    )

    noise = standard_deviation * (
        rng.normal(size=shape)
        + 1j * rng.normal(size=shape)
    )

    return np.asarray(
        noise,
        dtype=np.complex128,
    )


def effective_star_channel(
    channel_a_to_ris: ArrayLike,
    channel_ris_to_b: ArrayLike,
    surface_coefficients: ArrayLike,
    direct_channel: ArrayLike = 0.0j,
) -> ComplexArray:
    """
    计算STAR-RIS辅助链路的有效级联信道。

    数学模型：

        h_eff
        =
        h_direct
        +
        sum_n(
            g_n * phi_n * h_n
        )

    参数：
        channel_a_to_ris:
            A到STAR-RIS的信道。

        channel_ris_to_b:
            STAR-RIS到B的信道。

        surface_coefficients:
            STAR-RIS的实际复系数。

        direct_channel:
            A和B之间的直达信道。
    """
    g = _as_channel_matrix(
        channel_a_to_ris,
        "channel_a_to_ris",
    )

    h = _as_channel_matrix(
        channel_ris_to_b,
        "channel_ris_to_b",
    )

    phi = _as_channel_matrix(
        surface_coefficients,
        "surface_coefficients",
    )

    num_samples = max(
        g.shape[0],
        h.shape[0],
        phi.shape[0],
    )

    g = _broadcast_channel_rows(
        g,
        num_samples,
        "channel_a_to_ris",
    )

    h = _broadcast_channel_rows(
        h,
        num_samples,
        "channel_ris_to_b",
    )

    phi = _broadcast_channel_rows(
        phi,
        num_samples,
        "surface_coefficients",
    )

    if not (
        g.shape[1]
        == h.shape[1]
        == phi.shape[1]
    ):
        raise ValueError(
            "all STAR-RIS vectors must have the "
            "same number of elements"
        )

    direct = _broadcast_direct_channel(
        direct_channel,
        num_samples,
    )

    cascaded_channel = np.sum(
        g * phi * h,
        axis=1,
    )

    return np.asarray(
        direct + cascaded_channel,
        dtype=np.complex128,
    )


def simulate_bidirectional_probing(
    channel_a_to_ris: ArrayLike,
    channel_ris_to_b: ArrayLike,
    surface_coefficients: ArrayLike,
    active_mask: ArrayLike,
    *,
    surface_coefficients_reverse: ArrayLike | None = None,
    direct_channel: ArrayLike = 0.0j,
    pilot_power_a: float = 1.0,
    pilot_power_b: float = 1.0,
    active_noise_variance: float = 0.0,
    receiver_noise_variance_a: float = 0.0,
    receiver_noise_variance_b: float = 0.0,
    rng: np.random.Generator | None = None,
) -> BidirectionalProbingResult:
    """
    模拟STAR-RIS辅助的TDD双向信道探测。

    正向探测：

        A -> STAR-RIS -> B

        z_B
        =
        h_eff_forward
        +
        forwarded_active_noise_at_B / sqrt(P_A)
        +
        receiver_noise_B / sqrt(P_A)

    反向探测：

        B -> STAR-RIS -> A

        z_A
        =
        h_eff_reverse
        +
        forwarded_active_noise_at_A / sqrt(P_B)
        +
        receiver_noise_A / sqrt(P_B)

    参数：
        surface_coefficients:
            正向探测中的STAR-RIS实际系数。

        surface_coefficients_reverse:
            反向探测中的STAR-RIS实际系数。

            当该参数为None时，默认反向与正向使用相同系数，
            对应理想互易硬件或仅存在方向无关静态误差的情况。

        active_mask:
            True表示有源单元，False表示无源单元。

        active_noise_variance:
            每个有源单元内部复高斯噪声的方差。

    注意：
        正向和反向内部放大噪声相互独立，不能作为双方共享随机源。
    """
    if pilot_power_a <= 0.0:
        raise ValueError(
            "pilot_power_a must be positive"
        )

    if pilot_power_b <= 0.0:
        raise ValueError(
            "pilot_power_b must be positive"
        )

    if active_noise_variance < 0.0:
        raise ValueError(
            "active_noise_variance cannot be negative"
        )

    if receiver_noise_variance_a < 0.0:
        raise ValueError(
            "receiver_noise_variance_a cannot be negative"
        )

    if receiver_noise_variance_b < 0.0:
        raise ValueError(
            "receiver_noise_variance_b cannot be negative"
        )

    generator = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    g = _as_channel_matrix(
        channel_a_to_ris,
        "channel_a_to_ris",
    )

    h = _as_channel_matrix(
        channel_ris_to_b,
        "channel_ris_to_b",
    )

    phi_forward = _as_channel_matrix(
        surface_coefficients,
        "surface_coefficients",
    )

    if surface_coefficients_reverse is None:
        phi_reverse = phi_forward
    else:
        phi_reverse = _as_channel_matrix(
            surface_coefficients_reverse,
            "surface_coefficients_reverse",
        )

    num_samples = max(
        g.shape[0],
        h.shape[0],
        phi_forward.shape[0],
        phi_reverse.shape[0],
    )

    g = _broadcast_channel_rows(
        g,
        num_samples,
        "channel_a_to_ris",
    )

    h = _broadcast_channel_rows(
        h,
        num_samples,
        "channel_ris_to_b",
    )

    phi_forward = _broadcast_channel_rows(
        phi_forward,
        num_samples,
        "surface_coefficients",
    )

    phi_reverse = _broadcast_channel_rows(
        phi_reverse,
        num_samples,
        "surface_coefficients_reverse",
    )

    if not (
        g.shape[1]
        == h.shape[1]
        == phi_forward.shape[1]
        == phi_reverse.shape[1]
    ):
        raise ValueError(
            "all STAR-RIS vectors must have the "
            "same number of elements"
        )

    num_elements = g.shape[1]

    mask = _validate_active_mask(
        active_mask,
        num_elements,
    )

    effective_channel_forward = (
        effective_star_channel(
            channel_a_to_ris=g,
            channel_ris_to_b=h,
            surface_coefficients=phi_forward,
            direct_channel=direct_channel,
        )
    )

    effective_channel_reverse = (
        effective_star_channel(
            channel_a_to_ris=g,
            channel_ris_to_b=h,
            surface_coefficients=phi_reverse,
            direct_channel=direct_channel,
        )
    )

    # 正向和反向有源内部噪声独立产生。
    active_noise_forward = (
        _complex_gaussian_noise(
            generator,
            active_noise_variance,
            (num_samples, num_elements),
        )
    )

    active_noise_reverse = (
        _complex_gaussian_noise(
            generator,
            active_noise_variance,
            (num_samples, num_elements),
        )
    )

    # 只有active_mask=True的单元产生内部放大噪声。
    active_multiplier = mask.astype(
        np.float64
    )[np.newaxis, :]

    active_noise_forward = (
        active_noise_forward
        * active_multiplier
    )

    active_noise_reverse = (
        active_noise_reverse
        * active_multiplier
    )

    # 正向探测中，有源噪声经过实际正向系数及RIS到B信道。
    forwarded_noise_at_b = np.sum(
        h
        * phi_forward
        * active_noise_forward,
        axis=1,
    )

    # 反向探测中，有源噪声经过实际反向系数及RIS到A信道。
    forwarded_noise_at_a = np.sum(
        g
        * phi_reverse
        * active_noise_reverse,
        axis=1,
    )

    receiver_noise_at_a = (
        _complex_gaussian_noise(
            generator,
            receiver_noise_variance_a,
            (num_samples,),
        )
    )

    receiver_noise_at_b = (
        _complex_gaussian_noise(
            generator,
            receiver_noise_variance_b,
            (num_samples,),
        )
    )

    sqrt_pilot_power_a = np.sqrt(
        pilot_power_a
    )

    sqrt_pilot_power_b = np.sqrt(
        pilot_power_b
    )

    observation_at_b = (
        effective_channel_forward
        + forwarded_noise_at_b
        / sqrt_pilot_power_a
        + receiver_noise_at_b
        / sqrt_pilot_power_a
    )

    observation_at_a = (
        effective_channel_reverse
        + forwarded_noise_at_a
        / sqrt_pilot_power_b
        + receiver_noise_at_a
        / sqrt_pilot_power_b
    )

    return BidirectionalProbingResult(
        effective_channel_forward=np.asarray(
            effective_channel_forward,
            dtype=np.complex128,
        ),
        effective_channel_reverse=np.asarray(
            effective_channel_reverse,
            dtype=np.complex128,
        ),
        observation_at_a=np.asarray(
            observation_at_a,
            dtype=np.complex128,
        ),
        observation_at_b=np.asarray(
            observation_at_b,
            dtype=np.complex128,
        ),
        active_noise_forward=np.asarray(
            active_noise_forward,
            dtype=np.complex128,
        ),
        active_noise_reverse=np.asarray(
            active_noise_reverse,
            dtype=np.complex128,
        ),
        forwarded_active_noise_at_a=np.asarray(
            forwarded_noise_at_a,
            dtype=np.complex128,
        ),
        forwarded_active_noise_at_b=np.asarray(
            forwarded_noise_at_b,
            dtype=np.complex128,
        ),
        receiver_noise_at_a=np.asarray(
            receiver_noise_at_a,
            dtype=np.complex128,
        ),
        receiver_noise_at_b=np.asarray(
            receiver_noise_at_b,
            dtype=np.complex128,
        ),
    )