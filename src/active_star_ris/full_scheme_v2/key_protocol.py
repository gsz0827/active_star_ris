from __future__ import annotations

import hashlib
import math

import numpy as np

from .config import KeyGenerationConfig, ObjectiveConfig, ProbingConfig
from .models import BranchKeyMetrics, BranchObservations, JointKeyMetrics


def binary_entropy(probability: float) -> float:
    p = float(np.clip(probability, 0.0, 1.0))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def feature(values: np.ndarray, name: str) -> np.ndarray:
    if name == "real":
        return np.real(values)
    if name == "imag":
        return np.imag(values)
    if name == "magnitude":
        return np.abs(values)
    if name == "phase":
        return np.angle(values)
    raise ValueError(f"unsupported feature: {name}")


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - np.mean(values)
    scale = float(np.std(centered))
    if scale <= 1.0e-12:
        return np.zeros_like(centered)
    return centered / scale


def complex_correlation(a: np.ndarray, b: np.ndarray) -> complex:
    x = np.asarray(a, dtype=np.complex128).reshape(-1)
    y = np.asarray(b, dtype=np.complex128).reshape(-1)
    if x.size != y.size or x.size < 2:
        return 0.0j
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = np.sqrt(np.sum(np.abs(x) ** 2) * np.sum(np.abs(y) ** 2))
    if denominator <= 1.0e-30:
        return 0.0j
    return complex(np.sum(x * np.conj(y)) / denominator)


def gaussian_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    rho_squared = min(abs(complex_correlation(a, b)) ** 2, 1.0 - 1.0e-12)
    return float(max(0.0, -np.log2(1.0 - rho_squared)))


def quantize_guard_band(
    alice_observation: np.ndarray,
    bob_observation: np.ndarray,
    config: KeyGenerationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alice = standardize(feature(alice_observation, config.feature))
    bob = standardize(feature(bob_observation, config.feature))
    alice_keep = np.abs(alice) >= config.guard_band_sigma
    if config.selection_policy == "intersection":
        keep = alice_keep & (np.abs(bob) >= config.guard_band_sigma)
    else:
        keep = alice_keep
    alice_bits = (alice[keep] >= 0.0).astype(np.uint8)
    bob_bits = (bob[keep] >= 0.0).astype(np.uint8)
    return alice_bits, bob_bits, keep


def _parity(bits: np.ndarray, indices: np.ndarray) -> int:
    if indices.size == 0:
        return 0
    return int(np.bitwise_xor.reduce(bits[indices]))


def cascade_reconcile(
    alice_bits: np.ndarray,
    bob_bits: np.ndarray,
    config: KeyGenerationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    alice = np.asarray(alice_bits, dtype=np.uint8).copy()
    bob = np.asarray(bob_bits, dtype=np.uint8).copy()
    if alice.size != bob.size:
        raise ValueError("Alice and Bob bit arrays must have equal length")
    leakage = 0
    if alice.size == 0:
        return bob, leakage
    for pass_index in range(config.reconciliation_passes):
        permutation = rng.permutation(alice.size)
        block_size = min(
            alice.size,
            config.initial_block_size * (2 ** min(pass_index, config.maximum_block_doublings)),
        )
        for start in range(0, alice.size, block_size):
            block = permutation[start : start + block_size]
            leakage += 1
            if _parity(alice, block) == _parity(bob, block):
                continue
            candidate = block.copy()
            while candidate.size > 1:
                midpoint = candidate.size // 2
                left = candidate[:midpoint]
                right = candidate[midpoint:]
                leakage += 1
                if _parity(alice, left) != _parity(bob, left):
                    candidate = left
                else:
                    candidate = right
            if candidate.size == 1:
                bob[candidate[0]] ^= 1
        if np.array_equal(alice, bob):
            break
    return bob, leakage


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    packed = np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big")
    return packed.tobytes()


def verification_passed(alice_bits: np.ndarray, bob_bits: np.ndarray, tag_bits: int) -> bool:
    bytes_needed = math.ceil(tag_bits / 8)
    alice_digest = hashlib.sha256(_bits_to_bytes(alice_bits)).digest()[:bytes_needed]
    bob_digest = hashlib.sha256(_bits_to_bytes(bob_bits)).digest()[:bytes_needed]
    if tag_bits % 8:
        mask = 0xFF << (8 - tag_bits % 8) & 0xFF
        alice_digest = alice_digest[:-1] + bytes([alice_digest[-1] & mask])
        bob_digest = bob_digest[:-1] + bytes([bob_digest[-1] & mask])
    return alice_digest == bob_digest


def toeplitz_hash(bits: np.ndarray, output_bits: int, seed: np.ndarray) -> np.ndarray:
    source = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if output_bits <= 0:
        return np.zeros(0, dtype=np.uint8)
    expected = source.size + output_bits - 1
    seed = np.asarray(seed, dtype=np.uint8).reshape(-1)
    if seed.size != expected:
        raise ValueError("invalid Toeplitz seed length")
    output = np.empty(output_bits, dtype=np.uint8)
    for row in range(output_bits):
        indices = row + np.arange(source.size - 1, -1, -1)
        output[row] = np.bitwise_xor.reduce(source & seed[indices])
    return output


def probing_duration_seconds(probing: ProbingConfig) -> float:
    per_sample_symbols = (
        probing.pilot_symbols_controller
        + probing.pilot_symbols_transmission_user
        + probing.pilot_symbols_reflection_user
    )
    return float(
        probing.samples_per_step * per_sample_symbols * probing.pilot_symbol_duration_seconds
        + 2.0 * probing.samples_per_step * probing.forward_reverse_guard_seconds
        + probing.branch_switch_guard_seconds
    )


def evaluate_branch_key_metrics(
    observations: BranchObservations,
    key_config: KeyGenerationConfig,
    probing: ProbingConfig,
    rng: np.random.Generator,
    *,
    full_protocol: bool,
) -> BranchKeyMetrics:
    alice_bits, bob_bits, keep = quantize_guard_band(
        observations.observation_alice,
        observations.observation_bob,
        key_config,
    )
    retained = int(alice_bits.size)
    raw_kdr = float(np.mean(alice_bits != bob_bits)) if retained else 1.0
    mi_ab = gaussian_mutual_information(
        observations.observation_alice,
        observations.observation_bob,
    )
    eve_information = max(
        gaussian_mutual_information(
            observations.observation_alice,
            observations.observation_eve_forward,
        ),
        gaussian_mutual_information(
            observations.observation_alice,
            observations.observation_eve_reverse,
        ),
        gaussian_mutual_information(
            observations.observation_bob,
            observations.observation_eve_forward,
        ),
        gaussian_mutual_information(
            observations.observation_bob,
            observations.observation_eve_reverse,
        ),
    )
    reciprocity = float(
        np.clip(
            abs(
                complex_correlation(
                    observations.observation_alice,
                    observations.observation_bob,
                )
            ),
            0.0,
            1.0,
        )
    )

    reconciliation_leakage_bound = int(
        math.ceil(
            key_config.reconciliation_efficiency
            * retained
            * binary_entropy(raw_kdr)
        )
    )
    conditional_min_entropy_bound = int(
        math.floor(
            retained
            * max(0.0, 1.0 - min(1.0, eve_information))
        )
    )
    finite_penalty = int(
        math.ceil(
            np.sqrt(
                max(retained, 1)
                * np.log2(2.0 / key_config.epsilon_security)
            )
        )
    )

    # 训练和正式评价共用同一个有限长度密钥余量定义。训练阶段不运行
    # Cascade，而使用协调泄漏上界；正式评价使用实际公开泄漏。
    public_leakage = (
        reconciliation_leakage_bound + key_config.verification_tag_bits
        if retained
        else 0
    )
    key_margin_bits = (
        conditional_min_entropy_bound
        - public_leakage
        - finite_penalty
        - key_config.privacy_margin_bits
    )
    finite_length_secret_bits = min(
        key_config.maximum_final_key_bits,
        max(0, key_margin_bits),
    )

    # 负余量必须保留为训练信号，不能提前截断为零。
    post_kdr = 0.0 if not full_protocol else raw_kdr
    final_key_bits = 0
    final_match = False

    if full_protocol and retained:
        reconciled_bob, parity_leakage = cascade_reconcile(
            alice_bits,
            bob_bits,
            key_config,
            rng,
        )
        post_kdr = float(np.mean(alice_bits != reconciled_bob))
        verified = verification_passed(
            alice_bits,
            reconciled_bob,
            key_config.verification_tag_bits,
        )
        public_leakage = parity_leakage + key_config.verification_tag_bits
        key_margin_bits = (
            conditional_min_entropy_bound
            - public_leakage
            - finite_penalty
            - key_config.privacy_margin_bits
        )
        finite_length_secret_bits = min(
            key_config.maximum_final_key_bits,
            max(0, key_margin_bits),
        )
        final_key_bits = finite_length_secret_bits

        if verified and final_key_bits > 0:
            seed = rng.integers(
                0,
                2,
                size=retained + final_key_bits - 1,
                dtype=np.uint8,
            )
            alice_key = toeplitz_hash(alice_bits, final_key_bits, seed)
            bob_key = toeplitz_hash(reconciled_bob, final_key_bits, seed)
            final_match = bool(np.array_equal(alice_key, bob_key))
            if not final_match:
                final_key_bits = 0
        else:
            final_key_bits = 0

    secret_bits_for_rate = (
        final_key_bits if full_protocol else finite_length_secret_bits
    )
    public_time = public_leakage / key_config.public_channel_rate_bps
    duration = (
        probing_duration_seconds(probing)
        + public_time
        + key_config.fixed_processing_delay_seconds
    )
    secure_rate = secret_bits_for_rate / max(duration, 1.0e-12)

    return BranchKeyMetrics(
        mutual_information_ab=mi_ab,
        eve_information=eve_information,
        reciprocity=reciprocity,
        raw_kdr=raw_kdr,
        retained_bits=retained,
        post_reconciliation_kdr=post_kdr,
        public_leakage_bits=public_leakage,
        finite_length_secret_bits=int(finite_length_secret_bits),
        final_key_bits=int(final_key_bits),
        final_keys_match=final_match,
        secure_key_rate_bps=float(secure_rate),
        finite_penalty_bits=finite_penalty,
        conditional_min_entropy_bits=conditional_min_entropy_bound,
        key_margin_bits=float(key_margin_bits),
    )

def evaluate_joint_key_metrics(
    transmission: BranchObservations,
    reflection: BranchObservations,
    key_config: KeyGenerationConfig,
    probing: ProbingConfig,
    objective: ObjectiveConfig,
    rng: np.random.Generator,
    *,
    full_protocol: bool,
) -> JointKeyMetrics:
    t = evaluate_branch_key_metrics(
        transmission,
        key_config,
        probing,
        rng,
        full_protocol=full_protocol,
    )
    r = evaluate_branch_key_metrics(
        reflection,
        key_config,
        probing,
        rng,
        full_protocol=full_protocol,
    )
    total_weight = objective.transmission_weight + objective.reflection_weight
    wt = objective.transmission_weight / total_weight
    wr = objective.reflection_weight / total_weight
    return JointKeyMetrics(
        transmission=t,
        reflection=r,
        weighted_secure_key_rate_bps=(
            wt * t.secure_key_rate_bps + wr * r.secure_key_rate_bps
        ),
        weighted_raw_kdr=wt * t.raw_kdr + wr * r.raw_kdr,
        weighted_post_reconciliation_kdr=(
            wt * t.post_reconciliation_kdr
            + wr * r.post_reconciliation_kdr
        ),
        weighted_reciprocity=wt * t.reciprocity + wr * r.reciprocity,
        weighted_key_margin_bits=(
            wt * t.key_margin_bits + wr * r.key_margin_bits
        ),
    )
