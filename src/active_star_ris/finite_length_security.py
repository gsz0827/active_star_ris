from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np

from active_star_ris.secure_key_generation import (
    BranchSecrecyMetrics,
)


@dataclass(frozen=True)
class FiniteLengthSecurityParameters:
    """
    有限长度安全参数。

    block_length:
        总探测样本数。

    parameter_estimation_samples:
        用于参数估计、不能直接生成密钥的样本数。

    reconciliation_efficiency:
        信息协调效率beta_rec，取值范围为(0, 1]。

    epsilon_smoothing:
        平滑参数对应的失败概率。

    epsilon_parameter_estimation:
        参数估计失败概率。

    epsilon_privacy_amplification:
        隐私放大失败概率。

    aep_coefficient:
        AEP有限长度修正系数。

    parameter_estimation_coefficient:
        参数估计修正系数。

    authentication_leakage_bits:
        认证公开消息消耗的总比特数。

    implementation_margin_bits_per_sample:
        为未建模硬件误差、估计偏差等保留的额外安全裕量。
    """

    block_length: int

    parameter_estimation_samples: int = 0

    reconciliation_efficiency: float = 0.95

    epsilon_smoothing: float = 1.0e-10
    epsilon_parameter_estimation: float = 1.0e-10
    epsilon_privacy_amplification: float = 1.0e-10

    aep_coefficient: float = 1.0
    parameter_estimation_coefficient: float = 1.0

    authentication_leakage_bits: float = 0.0
    implementation_margin_bits_per_sample: float = 0.0


@dataclass(frozen=True)
class FiniteLengthBranchMetrics:
    """
    单个透射或反射分支的有限长度密钥指标。
    """

    total_samples: int
    parameter_estimation_samples: int
    key_generation_samples: int

    reconciliation_efficiency: float

    asymptotic_legitimate_information_bits_per_sample: float
    reconciled_legitimate_information_bits_per_sample: float
    reconciliation_loss_bits_per_sample: float

    eve_leakage_bits_per_sample: float
    public_leakage_bits_per_sample: float

    aep_penalty_bits_per_sample: float
    parameter_estimation_penalty_bits_per_sample: float
    privacy_amplification_penalty_bits_per_sample: float
    authentication_penalty_bits_per_sample: float
    implementation_margin_bits_per_sample: float

    total_finite_length_penalty_bits_per_sample: float

    raw_finite_length_rate_bits_per_sample: float
    finite_length_rate_bits_per_sample: float

    extractable_secret_bits: int

    total_security_failure_probability: float


@dataclass(frozen=True)
class DualSideFiniteLengthResult:
    """
    透射侧和反射侧联合有限长度结果。
    """

    transmission: FiniteLengthBranchMetrics
    reflection: FiniteLengthBranchMetrics

    normalized_transmission_weight: float
    normalized_reflection_weight: float

    weighted_finite_length_rate_bits_per_sample: float

    aggregate_extractable_secret_bits: int


def _validate_probability(
    value: float,
    name: str,
) -> None:
    if not np.isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    if not 0.0 < value < 1.0:
        raise ValueError(
            f"{name} must lie strictly between 0 and 1"
        )


def _validate_parameters(
    parameters: FiniteLengthSecurityParameters,
) -> None:
    if isinstance(
        parameters.block_length,
        bool,
    ):
        raise ValueError(
            "block_length must be an integer"
        )

    if not isinstance(
        parameters.block_length,
        (int, np.integer),
    ):
        raise ValueError(
            "block_length must be an integer"
        )

    if parameters.block_length < 2:
        raise ValueError(
            "block_length must be at least 2"
        )

    if isinstance(
        parameters.parameter_estimation_samples,
        bool,
    ):
        raise ValueError(
            "parameter_estimation_samples "
            "must be an integer"
        )

    if not isinstance(
        parameters.parameter_estimation_samples,
        (int, np.integer),
    ):
        raise ValueError(
            "parameter_estimation_samples "
            "must be an integer"
        )

    if (
        parameters.parameter_estimation_samples
        < 0
    ):
        raise ValueError(
            "parameter_estimation_samples "
            "cannot be negative"
        )

    if (
        parameters.parameter_estimation_samples
        >= parameters.block_length
    ):
        raise ValueError(
            "parameter_estimation_samples must be "
            "smaller than block_length"
        )

    if not np.isfinite(
        parameters.reconciliation_efficiency
    ):
        raise ValueError(
            "reconciliation_efficiency "
            "must be finite"
        )

    if not (
        0.0
        < parameters.reconciliation_efficiency
        <= 1.0
    ):
        raise ValueError(
            "reconciliation_efficiency "
            "must lie within (0, 1]"
        )

    _validate_probability(
        parameters.epsilon_smoothing,
        "epsilon_smoothing",
    )

    _validate_probability(
        parameters
        .epsilon_parameter_estimation,
        "epsilon_parameter_estimation",
    )

    _validate_probability(
        parameters
        .epsilon_privacy_amplification,
        "epsilon_privacy_amplification",
    )

    nonnegative_values = {
        "aep_coefficient": (
            parameters.aep_coefficient
        ),
        "parameter_estimation_coefficient": (
            parameters
            .parameter_estimation_coefficient
        ),
        "authentication_leakage_bits": (
            parameters
            .authentication_leakage_bits
        ),
        "implementation_margin_bits_per_sample": (
            parameters
            .implementation_margin_bits_per_sample
        ),
    }

    for name, value in nonnegative_values.items():
        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative"
            )


def _validate_branch_metrics(
    secrecy_metrics: BranchSecrecyMetrics,
) -> None:
    nonnegative_values = {
        "legitimate_mutual_information": (
            secrecy_metrics
            .legitimate_mutual_information_bits_per_sample
        ),
        "eve_leakage": (
            secrecy_metrics
            .eve_leakage_bits_per_sample
        ),
        "public_leakage": (
            secrecy_metrics
            .public_leakage_bits_per_sample
        ),
    }

    for name, value in nonnegative_values.items():
        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative"
            )


def evaluate_finite_length_branch(
    secrecy_metrics: BranchSecrecyMetrics,
    parameters: FiniteLengthSecurityParameters,
) -> FiniteLengthBranchMetrics:
    """
    将渐近保密指标转换为有限长度密钥率代理。

    计算公式：

        R_finite
        =
        [
            beta_rec * I_AB
            - I_E
            - L_public
            - Delta_AEP
            - Delta_PE
            - Delta_PA
            - Delta_auth
            - Delta_impl
        ]^+

    注意：
        当前I_AB与I_E仍基于复高斯互信息代理。
        因此该模块给出的是有限长度安全代理，
        而不是完整的可组合安全证明。
    """
    _validate_parameters(
        parameters
    )

    _validate_branch_metrics(
        secrecy_metrics
    )

    total_samples = int(
        parameters.block_length
    )

    parameter_estimation_samples = int(
        parameters
        .parameter_estimation_samples
    )

    key_generation_samples = (
        total_samples
        - parameter_estimation_samples
    )

    legitimate_information = float(
        secrecy_metrics
        .legitimate_mutual_information_bits_per_sample
    )

    eve_leakage = float(
        secrecy_metrics
        .eve_leakage_bits_per_sample
    )

    public_leakage = float(
        secrecy_metrics
        .public_leakage_bits_per_sample
    )

    reconciled_information = (
        parameters.reconciliation_efficiency
        * legitimate_information
    )

    reconciliation_loss = (
        legitimate_information
        - reconciled_information
    )

    aep_penalty = (
        parameters.aep_coefficient
        * np.sqrt(
            np.log2(
                2.0
                / parameters.epsilon_smoothing
            )
            / key_generation_samples
        )
    )

    if parameter_estimation_samples == 0:
        parameter_estimation_penalty = 0.0
        parameter_estimation_epsilon = 0.0
    else:
        parameter_estimation_penalty = (
            parameters
            .parameter_estimation_coefficient
            * np.sqrt(
                np.log2(
                    2.0
                    / parameters
                    .epsilon_parameter_estimation
                )
                / parameter_estimation_samples
            )
        )

        parameter_estimation_epsilon = (
            parameters
            .epsilon_parameter_estimation
        )

    privacy_amplification_penalty = (
        2.0
        * np.log2(
            1.0
            / parameters
            .epsilon_privacy_amplification
        )
        / key_generation_samples
    )

    authentication_penalty = (
        parameters.authentication_leakage_bits
        / key_generation_samples
    )

    implementation_margin = (
        parameters
        .implementation_margin_bits_per_sample
    )

    total_finite_length_penalty = (
        aep_penalty
        + parameter_estimation_penalty
        + privacy_amplification_penalty
        + authentication_penalty
        + implementation_margin
    )

    raw_finite_length_rate = (
        reconciled_information
        - eve_leakage
        - public_leakage
        - total_finite_length_penalty
    )

    finite_length_rate = max(
        0.0,
        float(
            raw_finite_length_rate
        ),
    )

    extractable_secret_bits = max(
        0,
        floor(
            key_generation_samples
            * finite_length_rate
        ),
    )

    total_security_failure_probability = (
        parameters.epsilon_smoothing
        + parameter_estimation_epsilon
        + parameters
        .epsilon_privacy_amplification
    )

    return FiniteLengthBranchMetrics(
        total_samples=total_samples,
        parameter_estimation_samples=(
            parameter_estimation_samples
        ),
        key_generation_samples=(
            key_generation_samples
        ),
        reconciliation_efficiency=float(
            parameters
            .reconciliation_efficiency
        ),
        asymptotic_legitimate_information_bits_per_sample=(
            legitimate_information
        ),
        reconciled_legitimate_information_bits_per_sample=float(
            reconciled_information
        ),
        reconciliation_loss_bits_per_sample=float(
            reconciliation_loss
        ),
        eve_leakage_bits_per_sample=(
            eve_leakage
        ),
        public_leakage_bits_per_sample=(
            public_leakage
        ),
        aep_penalty_bits_per_sample=float(
            aep_penalty
        ),
        parameter_estimation_penalty_bits_per_sample=float(
            parameter_estimation_penalty
        ),
        privacy_amplification_penalty_bits_per_sample=float(
            privacy_amplification_penalty
        ),
        authentication_penalty_bits_per_sample=float(
            authentication_penalty
        ),
        implementation_margin_bits_per_sample=float(
            implementation_margin
        ),
        total_finite_length_penalty_bits_per_sample=float(
            total_finite_length_penalty
        ),
        raw_finite_length_rate_bits_per_sample=float(
            raw_finite_length_rate
        ),
        finite_length_rate_bits_per_sample=float(
            finite_length_rate
        ),
        extractable_secret_bits=int(
            extractable_secret_bits
        ),
        total_security_failure_probability=float(
            total_security_failure_probability
        ),
    )


def evaluate_dual_side_finite_length_security(
    transmission_secrecy: BranchSecrecyMetrics,
    reflection_secrecy: BranchSecrecyMetrics,
    transmission_parameters: FiniteLengthSecurityParameters,
    reflection_parameters: (
        FiniteLengthSecurityParameters
        | None
    ) = None,
    *,
    transmission_weight: float = 0.5,
    reflection_weight: float = 0.5,
) -> DualSideFiniteLengthResult:
    """
    计算透射侧与反射侧联合有限长度指标。

    aggregate_extractable_secret_bits表示两条独立分支
    均被用于产密钥时，可提取密钥比特数之和。
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

    effective_reflection_parameters = (
        transmission_parameters
        if reflection_parameters is None
        else reflection_parameters
    )

    transmission_result = (
        evaluate_finite_length_branch(
            transmission_secrecy,
            transmission_parameters,
        )
    )

    reflection_result = (
        evaluate_finite_length_branch(
            reflection_secrecy,
            effective_reflection_parameters,
        )
    )

    weighted_rate = (
        normalized_transmission_weight
        * transmission_result
        .finite_length_rate_bits_per_sample
        + normalized_reflection_weight
        * reflection_result
        .finite_length_rate_bits_per_sample
    )

    aggregate_secret_bits = (
        transmission_result
        .extractable_secret_bits
        + reflection_result
        .extractable_secret_bits
    )

    return DualSideFiniteLengthResult(
        transmission=transmission_result,
        reflection=reflection_result,
        normalized_transmission_weight=float(
            normalized_transmission_weight
        ),
        normalized_reflection_weight=float(
            normalized_reflection_weight
        ),
        weighted_finite_length_rate_bits_per_sample=float(
            weighted_rate
        ),
        aggregate_extractable_secret_bits=int(
            aggregate_secret_bits
        ),
    )