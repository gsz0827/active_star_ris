from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from active_star_ris.key_generation import (  # noqa: E402
    evaluate_key_generation,
)


def main() -> None:
    rng = np.random.default_rng(
        20260719
    )
    num_samples = 4000

    shared_channel = (
        rng.normal(size=num_samples)
        + 1j * rng.normal(
            size=num_samples
        )
    )

    noise_a = 0.2 * (
        rng.normal(size=num_samples)
        + 1j * rng.normal(
            size=num_samples
        )
    )

    noise_b = 0.2 * (
        rng.normal(size=num_samples)
        + 1j * rng.normal(
            size=num_samples
        )
    )

    metrics = evaluate_key_generation(
        shared_channel + noise_a,
        shared_channel + noise_b,
    )

    print(
        "Step 1 key-generation metric check"
    )
    print(
        f"|rho_AB|       = "
        f"{metrics.correlation_magnitude:.6f}"
    )
    print(
        f"|rho_AB|^2     = "
        f"{metrics.correlation_squared:.6f}"
    )
    print(
        "Gaussian MI    = "
        f"{metrics.mutual_information_bits_per_sample:.6f} "
        "bit/sample"
    )
    print(
        f"Raw KDR        = "
        f"{metrics.key_disagreement_rate:.6f}"
    )
    print(
        f"Raw bit count  = "
        f"{metrics.num_raw_bits}"
    )

    assert (
        metrics.correlation_magnitude
        > 0.9
    )
    assert (
        metrics.key_disagreement_rate
        < 0.2
    )

    print(
        "STEP 1 MANUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()