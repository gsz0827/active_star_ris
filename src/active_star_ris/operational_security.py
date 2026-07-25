from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from active_star_ris.finite_length_security import (
    FiniteLengthBranchMetrics,
)
from active_star_ris.secure_key_generation import (
    BranchSecrecyMetrics,
)


@dataclass(frozen=True)
class OperationalPreReconciliationBound:
    """
    实际信息协调前的条件熵下界。

    此结果尚未扣除：
        1. Cascade实际奇偶校验泄漏；
        2. 一致性验证标签；
        3. 隐私放大安全裕量。

    上述三项由practical_key_generation模块
    根据真实协议运行结果继续扣除。
    """

    key_generation_samples: int

    legitimate_information_bits_per_sample: float
    eve_leakage_bits_per_sample: float
    public_leakage_bits_per_sample: float

    aep_penalty_bits_per_sample: float
    parameter_estimation_penalty_bits_per_sample: float
    implementation_margin_bits_per_sample: float

    private_information_rate_before_authentication: float

    gross_private_information_bits: int
    authentication_leakage_bits: int

    pre_reconciliation_entropy_bound_bits: int


def evaluate_operational_pre_reconciliation_bound(
    secrecy_metrics: BranchSecrecyMetrics,
    finite_length_metrics: FiniteLengthBranchMetrics,
    *,
    authentication_leakage_bits: float = 0.0,
) -> OperationalPreReconciliationBound:
    """
    计算实际Cascade运行之前的安全比特上界。

    与有限长度代理的区别：

    代理模式：
        beta_rec * I_AB

    实际协议模式：
        I_AB - 实际Cascade公开泄漏

    因此本函数不使用reconciliation_efficiency，
    也不扣除有限长度模块中的PA惩罚，
    而是由实际Toeplitz隐私放大裕量替代。
    """
    if not np.isfinite(
        authentication_leakage_bits
    ):
        raise ValueError(
            "authentication_leakage_bits must be finite"
        )

    if authentication_leakage_bits < 0.0:
        raise ValueError(
            "authentication_leakage_bits "
            "cannot be negative"
        )

    num_key_samples = int(
        finite_length_metrics
        .key_generation_samples
    )

    if num_key_samples < 1:
        raise ValueError(
            "key_generation_samples must be positive"
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

    aep_penalty = float(
        finite_length_metrics
        .aep_penalty_bits_per_sample
    )

    parameter_estimation_penalty = float(
        finite_length_metrics
        .parameter_estimation_penalty_bits_per_sample
    )

    implementation_margin = float(
        finite_length_metrics
        .implementation_margin_bits_per_sample
    )

    private_rate = (
        legitimate_information
        - eve_leakage
        - public_leakage
        - aep_penalty
        - parameter_estimation_penalty
        - implementation_margin
    )

    gross_private_bits = max(
        0,
        floor(
            num_key_samples
            * max(
                0.0,
                private_rate,
            )
        ),
    )

    authentication_bits = int(
        ceil(
            authentication_leakage_bits
        )
    )

    pre_reconciliation_bound = max(
        0,
        gross_private_bits
        - authentication_bits,
    )

    return OperationalPreReconciliationBound(
        key_generation_samples=(
            num_key_samples
        ),
        legitimate_information_bits_per_sample=(
            legitimate_information
        ),
        eve_leakage_bits_per_sample=(
            eve_leakage
        ),
        public_leakage_bits_per_sample=(
            public_leakage
        ),
        aep_penalty_bits_per_sample=(
            aep_penalty
        ),
        parameter_estimation_penalty_bits_per_sample=(
            parameter_estimation_penalty
        ),
        implementation_margin_bits_per_sample=(
            implementation_margin
        ),
        private_information_rate_before_authentication=float(
            private_rate
        ),
        gross_private_information_bits=int(
            gross_private_bits
        ),
        authentication_leakage_bits=(
            authentication_bits
        ),
        pre_reconciliation_entropy_bound_bits=int(
            pre_reconciliation_bound
        ),
    )