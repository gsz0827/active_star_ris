from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from active_star_ris.key_generation import (
    evaluate_key_generation,
)
from active_star_ris.star_key_system import (
    DualSideKeyGenerationResult,
    StarCoefficientPair,
    simulate_dual_side_key_generation,
)

ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class EveObservationResult:
    """
    Eve在一次双向探测过程中的观测结果。
    """

    effective_channel_forward: ComplexArray
    effective_channel_reverse: ComplexArray

    observation_forward: ComplexArray
    observation_reverse: ComplexArray

    forwarded_active_noise_forward: ComplexArray
    forwarded_active_noise_reverse: ComplexArray

    receiver_noise_forward: ComplexArray
    receiver_noise_reverse: ComplexArray


@dataclass(frozen=True)
class BranchSecrecyMetrics:
    """
    单个透射或反射分支的保密密钥指标。
    """

    legitimate_mutual_information_bits_per_sample: float

    leakage_from_a_bits_per_sample: float
    leakage_from_b_bits_per_sample: float
    eve_leakage_bits_per_sample: float

    public_leakage_bits_per_sample: float

    raw_secret_key_margin_bits_per_sample: float
    secret_key_rate_bits_per_sample: float


@dataclass(frozen=True)
class DualSideSecureKeyGenerationResult:
    """
    透射侧和反射侧联合保密密钥生成结果。
    """

    legitimate: DualSideKeyGenerationResult

    transmission_eve: EveObservationResult
    reflection_eve: EveObservationResult

    transmission_secrecy: BranchSecrecyMetrics
    reflection_secrecy: BranchSecrecyMetrics

    weighted_eve_leakage_bits_per_sample: float
    weighted_secret_key_rate_bits_per_sample: float


def _as_channel_matrix(
    values: ArrayLike,
    name: str,
) -> ComplexArray:
    array = np.asarray(
        values,
        dtype=np.complex128,
    )

    if array.ndim == 1:
        array = array[np.newaxis, :]

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be one-dimensional "
            "or two-dimensional"
        )

    if array.shape[1] == 0:
        raise ValueError(
            f"{name} cannot contain zero elements"
        )

    if not np.all(np.isfinite(array.real)):
        raise ValueError(
            f"{name} contains non-finite real values"
        )

    if not np.all(np.isfinite(array.imag)):
        raise ValueError(
            f"{name} contains non-finite imaginary values"
        )

    return array


def _broadcast_rows(
    values: ComplexArray,
    num_samples: int,
    name: str,
) -> ComplexArray:
    if values.shape[0] == num_samples:
        return values

    if values.shape[0] == 1:
        return np.repeat(
            values,
            num_samples,
            axis=0,
        )

    raise ValueError(
        f"{name} has {values.shape[0]} samples, "
        f"but {num_samples} samples are required"
    )


def _broadcast_direct_channel(
    values: ArrayLike,
    num_samples: int,
    name: str,
) -> ComplexArray:
    array = np.asarray(
        values,
        dtype=np.complex128,
    ).reshape(-1)

    if array.size == 1:
        result = np.full(
            num_samples,
            array.item(),
            dtype=np.complex128,
        )
    elif array.size == num_samples:
        result = array
    else:
        raise ValueError(
            f"{name} must be a scalar or contain "
            "one value per channel sample"
        )

    if not np.all(np.isfinite(result.real)):
        raise ValueError(
            f"{name} contains non-finite real values"
        )

    if not np.all(np.isfinite(result.imag)):
        raise ValueError(
            f"{name} contains non-finite imaginary values"
        )

    return np.asarray(
        result,
        dtype=np.complex128,
    )


def _validate_active_mask(
    active_mask: ArrayLike,
    num_elements: int,
) -> BoolArray:
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


def complex_gaussian_mutual_information(
    source: ArrayLike,
    observations: ArrayLike,
    *,
    covariance_regularization: float = 1.0e-10,
) -> float:
    """
    估计复高斯标量源与一个或多个复观测之间的互信息。

    对于标量X和观测向量Z，有：

        I(X; Z) = -log2(1 - R^2)

    其中R^2为复数多重相关系数。

    当observations只有一列时，该表达式退化为：

        I(X; Z)
        =
        -log2(1 - |rho_XZ|^2)
    """
    if covariance_regularization < 0.0:
        raise ValueError(
            "covariance_regularization cannot be negative"
        )

    x = np.asarray(
        source,
        dtype=np.complex128,
    ).reshape(-1)

    z = np.asarray(
        observations,
        dtype=np.complex128,
    )

    if z.ndim == 1:
        z = z[:, np.newaxis]

    if z.ndim != 2:
        raise ValueError(
            "observations must be one-dimensional "
            "or two-dimensional"
        )

    if x.size != z.shape[0]:
        raise ValueError(
            "source and observations must contain "
            "the same number of samples"
        )

    if x.size < 3:
        raise ValueError(
            "at least three samples are required"
        )

    if not np.all(np.isfinite(x.real)):
        raise ValueError(
            "source contains non-finite real values"
        )

    if not np.all(np.isfinite(x.imag)):
        raise ValueError(
            "source contains non-finite imaginary values"
        )

    if not np.all(np.isfinite(z.real)):
        raise ValueError(
            "observations contain non-finite real values"
        )

    if not np.all(np.isfinite(z.imag)):
        raise ValueError(
            "observations contain non-finite imaginary values"
        )

    centered_x = x - np.mean(x)

    centered_z = (
        z
        - np.mean(
            z,
            axis=0,
            keepdims=True,
        )
    )

    denominator = x.size - 1

    variance_x = float(
        np.vdot(
            centered_x,
            centered_x,
        ).real
        / denominator
    )

    if variance_x <= np.finfo(np.float64).eps:
        return 0.0

    covariance_z = (
        centered_z.conj().T
        @ centered_z
        / denominator
    )

    covariance_scale = max(
        float(
            np.trace(
                covariance_z
            ).real
            / covariance_z.shape[0]
        ),
        np.finfo(np.float64).eps,
    )

    regularized_covariance_z = (
        covariance_z
        + covariance_regularization
        * covariance_scale
        * np.eye(
            covariance_z.shape[0],
            dtype=np.complex128,
        )
    )

    cross_covariance = (
        centered_z.conj().T
        @ centered_x
        / denominator
    )

    try:
        regression_vector = np.linalg.solve(
            regularized_covariance_z,
            cross_covariance,
        )
    except np.linalg.LinAlgError:
        regression_vector = (
            np.linalg.pinv(
                regularized_covariance_z
            )
            @ cross_covariance
        )

    explained_variance = float(
        np.vdot(
            cross_covariance,
            regression_vector,
        ).real
    )

    multiple_correlation_squared = (
        explained_variance
        / variance_x
    )

    numerical_floor = 1.0e-12

    multiple_correlation_squared = float(
        np.clip(
            multiple_correlation_squared,
            0.0,
            1.0 - numerical_floor,
        )
    )

    mutual_information = -np.log2(
        max(
            1.0
            - multiple_correlation_squared,
            numerical_floor,
        )
    )

    return float(
        mutual_information
    )


def simulate_eve_observations(
    channel_controller_to_ris: ArrayLike,
    channel_ris_to_user: ArrayLike,
    channel_ris_to_eve: ArrayLike,
    surface_coefficients_forward: ArrayLike,
    surface_coefficients_reverse: ArrayLike,
    active_mask: ArrayLike,
    active_noise_forward: ArrayLike,
    active_noise_reverse: ArrayLike,
    *,
    direct_channel_controller_to_eve: ArrayLike = 0.0j,
    direct_channel_user_to_eve: ArrayLike = 0.0j,
    pilot_power_controller: float = 1.0,
    pilot_power_user: float = 1.0,
    receiver_noise_variance_eve: float = 0.0,
    rng: np.random.Generator | None = None,
) -> EveObservationResult:
    """
    模拟Eve对正向和反向探测时隙的监听。

    正向时隙：

        Controller -> STAR-RIS -> Eve

    反向时隙：

        User -> STAR-RIS -> Eve

    Eve使用与合法接收端相同的STAR-RIS内部噪声实现，
    但具有独立的接收机噪声。
    """
    if pilot_power_controller <= 0.0:
        raise ValueError(
            "pilot_power_controller must be positive"
        )

    if pilot_power_user <= 0.0:
        raise ValueError(
            "pilot_power_user must be positive"
        )

    if receiver_noise_variance_eve < 0.0:
        raise ValueError(
            "receiver_noise_variance_eve "
            "cannot be negative"
        )

    generator = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    g = _as_channel_matrix(
        channel_controller_to_ris,
        "channel_controller_to_ris",
    )

    h_user = _as_channel_matrix(
        channel_ris_to_user,
        "channel_ris_to_user",
    )

    h_eve = _as_channel_matrix(
        channel_ris_to_eve,
        "channel_ris_to_eve",
    )

    phi_forward = _as_channel_matrix(
        surface_coefficients_forward,
        "surface_coefficients_forward",
    )

    phi_reverse = _as_channel_matrix(
        surface_coefficients_reverse,
        "surface_coefficients_reverse",
    )

    noise_forward = _as_channel_matrix(
        active_noise_forward,
        "active_noise_forward",
    )

    noise_reverse = _as_channel_matrix(
        active_noise_reverse,
        "active_noise_reverse",
    )

    num_samples = max(
        g.shape[0],
        h_user.shape[0],
        h_eve.shape[0],
        phi_forward.shape[0],
        phi_reverse.shape[0],
        noise_forward.shape[0],
        noise_reverse.shape[0],
    )

    g = _broadcast_rows(
        g,
        num_samples,
        "channel_controller_to_ris",
    )

    h_user = _broadcast_rows(
        h_user,
        num_samples,
        "channel_ris_to_user",
    )

    h_eve = _broadcast_rows(
        h_eve,
        num_samples,
        "channel_ris_to_eve",
    )

    phi_forward = _broadcast_rows(
        phi_forward,
        num_samples,
        "surface_coefficients_forward",
    )

    phi_reverse = _broadcast_rows(
        phi_reverse,
        num_samples,
        "surface_coefficients_reverse",
    )

    noise_forward = _broadcast_rows(
        noise_forward,
        num_samples,
        "active_noise_forward",
    )

    noise_reverse = _broadcast_rows(
        noise_reverse,
        num_samples,
        "active_noise_reverse",
    )

    element_counts = {
        g.shape[1],
        h_user.shape[1],
        h_eve.shape[1],
        phi_forward.shape[1],
        phi_reverse.shape[1],
        noise_forward.shape[1],
        noise_reverse.shape[1],
    }

    if len(element_counts) != 1:
        raise ValueError(
            "all STAR-RIS channel, coefficient, "
            "and noise matrices must have the "
            "same number of elements"
        )

    num_elements = g.shape[1]

    mask = _validate_active_mask(
        active_mask,
        num_elements,
    )

    active_multiplier = mask.astype(
        np.float64
    )[np.newaxis, :]

    noise_forward = (
        noise_forward
        * active_multiplier
    )

    noise_reverse = (
        noise_reverse
        * active_multiplier
    )

    direct_controller_to_eve = (
        _broadcast_direct_channel(
            direct_channel_controller_to_eve,
            num_samples,
            "direct_channel_controller_to_eve",
        )
    )

    direct_user_to_eve = (
        _broadcast_direct_channel(
            direct_channel_user_to_eve,
            num_samples,
            "direct_channel_user_to_eve",
        )
    )

    effective_channel_forward = (
        direct_controller_to_eve
        + np.sum(
            g
            * phi_forward
            * h_eve,
            axis=1,
        )
    )

    effective_channel_reverse = (
        direct_user_to_eve
        + np.sum(
            h_user
            * phi_reverse
            * h_eve,
            axis=1,
        )
    )

    forwarded_noise_forward = np.sum(
        h_eve
        * phi_forward
        * noise_forward,
        axis=1,
    )

    forwarded_noise_reverse = np.sum(
        h_eve
        * phi_reverse
        * noise_reverse,
        axis=1,
    )

    receiver_noise_forward = (
        _complex_gaussian_noise(
            generator,
            receiver_noise_variance_eve,
            (num_samples,),
        )
    )

    receiver_noise_reverse = (
        _complex_gaussian_noise(
            generator,
            receiver_noise_variance_eve,
            (num_samples,),
        )
    )

    observation_forward = (
        effective_channel_forward
        + forwarded_noise_forward
        / np.sqrt(
            pilot_power_controller
        )
        + receiver_noise_forward
        / np.sqrt(
            pilot_power_controller
        )
    )

    observation_reverse = (
        effective_channel_reverse
        + forwarded_noise_reverse
        / np.sqrt(
            pilot_power_user
        )
        + receiver_noise_reverse
        / np.sqrt(
            pilot_power_user
        )
    )

    return EveObservationResult(
        effective_channel_forward=np.asarray(
            effective_channel_forward,
            dtype=np.complex128,
        ),
        effective_channel_reverse=np.asarray(
            effective_channel_reverse,
            dtype=np.complex128,
        ),
        observation_forward=np.asarray(
            observation_forward,
            dtype=np.complex128,
        ),
        observation_reverse=np.asarray(
            observation_reverse,
            dtype=np.complex128,
        ),
        forwarded_active_noise_forward=np.asarray(
            forwarded_noise_forward,
            dtype=np.complex128,
        ),
        forwarded_active_noise_reverse=np.asarray(
            forwarded_noise_reverse,
            dtype=np.complex128,
        ),
        receiver_noise_forward=np.asarray(
            receiver_noise_forward,
            dtype=np.complex128,
        ),
        receiver_noise_reverse=np.asarray(
            receiver_noise_reverse,
            dtype=np.complex128,
        ),
    )


def evaluate_branch_secrecy(
    observation_at_a: ArrayLike,
    observation_at_b: ArrayLike,
    eve_observation_forward: ArrayLike,
    eve_observation_reverse: ArrayLike,
    *,
    public_leakage_bits_per_sample: float = 0.0,
) -> BranchSecrecyMetrics:
    """
    计算单个STAR-RIS分支的保密密钥率代理。

    Eve的总观测向量为：

        Z_E = [Z_E_forward, Z_E_reverse]
    """
    if public_leakage_bits_per_sample < 0.0:
        raise ValueError(
            "public_leakage_bits_per_sample "
            "cannot be negative"
        )

    observation_a = np.asarray(
        observation_at_a,
        dtype=np.complex128,
    ).reshape(-1)

    observation_b = np.asarray(
        observation_at_b,
        dtype=np.complex128,
    ).reshape(-1)

    eve_forward = np.asarray(
        eve_observation_forward,
        dtype=np.complex128,
    ).reshape(-1)

    eve_reverse = np.asarray(
        eve_observation_reverse,
        dtype=np.complex128,
    ).reshape(-1)

    sample_counts = {
        observation_a.size,
        observation_b.size,
        eve_forward.size,
        eve_reverse.size,
    }

    if len(sample_counts) != 1:
        raise ValueError(
            "all legitimate and Eve observations "
            "must contain the same number of samples"
        )

    legitimate_metrics = evaluate_key_generation(
        observation_a,
        observation_b,
    )

    eve_observation_matrix = np.column_stack(
        (
            eve_forward,
            eve_reverse,
        )
    )

    leakage_from_a = (
        complex_gaussian_mutual_information(
            observation_a,
            eve_observation_matrix,
        )
    )

    leakage_from_b = (
        complex_gaussian_mutual_information(
            observation_b,
            eve_observation_matrix,
        )
    )

    eve_leakage = max(
        leakage_from_a,
        leakage_from_b,
    )

    legitimate_mi = (
        legitimate_metrics
        .mutual_information_bits_per_sample
    )

    raw_margin = (
        legitimate_mi
        - eve_leakage
        - public_leakage_bits_per_sample
    )

    secret_key_rate = max(
        0.0,
        raw_margin,
    )

    return BranchSecrecyMetrics(
        legitimate_mutual_information_bits_per_sample=float(
            legitimate_mi
        ),
        leakage_from_a_bits_per_sample=float(
            leakage_from_a
        ),
        leakage_from_b_bits_per_sample=float(
            leakage_from_b
        ),
        eve_leakage_bits_per_sample=float(
            eve_leakage
        ),
        public_leakage_bits_per_sample=float(
            public_leakage_bits_per_sample
        ),
        raw_secret_key_margin_bits_per_sample=float(
            raw_margin
        ),
        secret_key_rate_bits_per_sample=float(
            secret_key_rate
        ),
    )


def simulate_dual_side_secure_key_generation(
    channel_controller_to_ris: ArrayLike,
    channel_ris_to_transmission_user: ArrayLike,
    channel_ris_to_reflection_user: ArrayLike,
    channel_ris_to_eve_transmission: ArrayLike,
    channel_ris_to_eve_reflection: ArrayLike,
    coefficients: StarCoefficientPair,
    active_mask: ArrayLike,
    *,
    reverse_coefficients: StarCoefficientPair | None = None,
    direct_channel_transmission: ArrayLike = 0.0j,
    direct_channel_reflection: ArrayLike = 0.0j,
    direct_channel_controller_to_eve_transmission: ArrayLike = 0.0j,
    direct_channel_transmission_user_to_eve: ArrayLike = 0.0j,
    direct_channel_controller_to_eve_reflection: ArrayLike = 0.0j,
    direct_channel_reflection_user_to_eve: ArrayLike = 0.0j,
    pilot_power_controller: float = 1.0,
    pilot_power_transmission_user: float = 1.0,
    pilot_power_reflection_user: float = 1.0,
    active_noise_variance: float = 0.0,
    receiver_noise_variance_controller: float = 0.0,
    receiver_noise_variance_transmission_user: float = 0.0,
    receiver_noise_variance_reflection_user: float = 0.0,
    receiver_noise_variance_eve_transmission: float = 0.0,
    receiver_noise_variance_eve_reflection: float = 0.0,
    public_leakage_transmission_bits_per_sample: float = 0.0,
    public_leakage_reflection_bits_per_sample: float = 0.0,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
    rng: np.random.Generator | None = None,
) -> DualSideSecureKeyGenerationResult:
    """
    联合模拟合法双方和Eve的双侧密钥生成过程。
    """
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

    coefficients_reverse = (
        coefficients
        if reverse_coefficients is None
        else reverse_coefficients
    )

    legitimate = simulate_dual_side_key_generation(
        channel_controller_to_ris=(
            channel_controller_to_ris
        ),
        channel_ris_to_transmission_user=(
            channel_ris_to_transmission_user
        ),
        channel_ris_to_reflection_user=(
            channel_ris_to_reflection_user
        ),
        coefficients=coefficients,
        reverse_coefficients=(
            coefficients_reverse
        ),
        active_mask=active_mask,
        direct_channel_transmission=(
            direct_channel_transmission
        ),
        direct_channel_reflection=(
            direct_channel_reflection
        ),
        pilot_power_controller=(
            pilot_power_controller
        ),
        pilot_power_transmission_user=(
            pilot_power_transmission_user
        ),
        pilot_power_reflection_user=(
            pilot_power_reflection_user
        ),
        active_noise_variance=(
            active_noise_variance
        ),
        receiver_noise_variance_controller=(
            receiver_noise_variance_controller
        ),
        receiver_noise_variance_transmission_user=(
            receiver_noise_variance_transmission_user
        ),
        receiver_noise_variance_reflection_user=(
            receiver_noise_variance_reflection_user
        ),
        transmission_weight=(
            transmission_weight
        ),
        reflection_weight=(
            reflection_weight
        ),
        rng=generator,
    )

    transmission_eve = simulate_eve_observations(
        channel_controller_to_ris=(
            channel_controller_to_ris
        ),
        channel_ris_to_user=(
            channel_ris_to_transmission_user
        ),
        channel_ris_to_eve=(
            channel_ris_to_eve_transmission
        ),
        surface_coefficients_forward=(
            coefficients.transmission
        ),
        surface_coefficients_reverse=(
            coefficients_reverse.transmission
        ),
        active_mask=active_mask,
        active_noise_forward=(
            legitimate
            .transmission
            .probing
            .active_noise_forward
        ),
        active_noise_reverse=(
            legitimate
            .transmission
            .probing
            .active_noise_reverse
        ),
        direct_channel_controller_to_eve=(
            direct_channel_controller_to_eve_transmission
        ),
        direct_channel_user_to_eve=(
            direct_channel_transmission_user_to_eve
        ),
        pilot_power_controller=(
            pilot_power_controller
        ),
        pilot_power_user=(
            pilot_power_transmission_user
        ),
        receiver_noise_variance_eve=(
            receiver_noise_variance_eve_transmission
        ),
        rng=generator,
    )

    reflection_eve = simulate_eve_observations(
        channel_controller_to_ris=(
            channel_controller_to_ris
        ),
        channel_ris_to_user=(
            channel_ris_to_reflection_user
        ),
        channel_ris_to_eve=(
            channel_ris_to_eve_reflection
        ),
        surface_coefficients_forward=(
            coefficients.reflection
        ),
        surface_coefficients_reverse=(
            coefficients_reverse.reflection
        ),
        active_mask=active_mask,
        active_noise_forward=(
            legitimate
            .reflection
            .probing
            .active_noise_forward
        ),
        active_noise_reverse=(
            legitimate
            .reflection
            .probing
            .active_noise_reverse
        ),
        direct_channel_controller_to_eve=(
            direct_channel_controller_to_eve_reflection
        ),
        direct_channel_user_to_eve=(
            direct_channel_reflection_user_to_eve
        ),
        pilot_power_controller=(
            pilot_power_controller
        ),
        pilot_power_user=(
            pilot_power_reflection_user
        ),
        receiver_noise_variance_eve=(
            receiver_noise_variance_eve_reflection
        ),
        rng=generator,
    )

    transmission_secrecy = evaluate_branch_secrecy(
        observation_at_a=(
            legitimate
            .transmission
            .probing
            .observation_at_a
        ),
        observation_at_b=(
            legitimate
            .transmission
            .probing
            .observation_at_b
        ),
        eve_observation_forward=(
            transmission_eve
            .observation_forward
        ),
        eve_observation_reverse=(
            transmission_eve
            .observation_reverse
        ),
        public_leakage_bits_per_sample=(
            public_leakage_transmission_bits_per_sample
        ),
    )

    reflection_secrecy = evaluate_branch_secrecy(
        observation_at_a=(
            legitimate
            .reflection
            .probing
            .observation_at_a
        ),
        observation_at_b=(
            legitimate
            .reflection
            .probing
            .observation_at_b
        ),
        eve_observation_forward=(
            reflection_eve
            .observation_forward
        ),
        eve_observation_reverse=(
            reflection_eve
            .observation_reverse
        ),
        public_leakage_bits_per_sample=(
            public_leakage_reflection_bits_per_sample
        ),
    )

    weighted_eve_leakage = (
        normalized_transmission_weight
        * transmission_secrecy
        .eve_leakage_bits_per_sample
        + normalized_reflection_weight
        * reflection_secrecy
        .eve_leakage_bits_per_sample
    )

    weighted_secret_key_rate = (
        normalized_transmission_weight
        * transmission_secrecy
        .secret_key_rate_bits_per_sample
        + normalized_reflection_weight
        * reflection_secrecy
        .secret_key_rate_bits_per_sample
    )

    return DualSideSecureKeyGenerationResult(
        legitimate=legitimate,
        transmission_eve=transmission_eve,
        reflection_eve=reflection_eve,
        transmission_secrecy=(
            transmission_secrecy
        ),
        reflection_secrecy=(
            reflection_secrecy
        ),
        weighted_eve_leakage_bits_per_sample=float(
            weighted_eve_leakage
        ),
        weighted_secret_key_rate_bits_per_sample=float(
            weighted_secret_key_rate
        ),
    )