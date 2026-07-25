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
from active_star_ris.probing import (  # noqa: E402
    simulate_bidirectional_probing,
)


def main() -> None:
    rng = np.random.default_rng(
        20260719
    )

    num_samples = 4000
    num_elements = 16

    channel_a_to_ris = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    channel_ris_to_b = (
        rng.normal(
            size=(num_samples, num_elements)
        )
        + 1j
        * rng.normal(
            size=(num_samples, num_elements)
        )
    ) / np.sqrt(2.0)

    phases = rng.uniform(
        -np.pi,
        np.pi,
        size=num_elements,
    )

    amplitudes = np.ones(
        num_elements
    )

    active_mask = np.zeros(
        num_elements,
        dtype=bool,
    )

    active_mask[:4] = True
    amplitudes[active_mask] = 1.5

    surface_coefficients = (
        amplitudes
        * np.exp(1j * phases)
    )

    result = simulate_bidirectional_probing(
        channel_a_to_ris=(
            channel_a_to_ris
        ),
        channel_ris_to_b=(
            channel_ris_to_b
        ),
        surface_coefficients=(
            surface_coefficients
        ),
        active_mask=active_mask,
        pilot_power_a=1.0,
        pilot_power_b=1.0,
        active_noise_variance=0.002,
        receiver_noise_variance_a=0.01,
        receiver_noise_variance_b=0.01,
        rng=rng,
    )

    metrics = evaluate_key_generation(
        result.observation_at_a,
        result.observation_at_b,
    )

    forward_noise_power = float(
        np.mean(
            np.abs(
                result.forwarded_active_noise_at_b
            ) ** 2
        )
    )

    reverse_noise_power = float(
        np.mean(
            np.abs(
                result.forwarded_active_noise_at_a
            ) ** 2
        )
    )

    observation_difference_power = float(
        np.mean(
            np.abs(
                result.observation_at_a
                - result.observation_at_b
            ) ** 2
        )
    )

    print(
        "Step 2 bidirectional probing check"
    )
    print(
        f"Number of STAR-RIS elements = "
        f"{num_elements}"
    )
    print(
        f"Number of active elements   = "
        f"{np.sum(active_mask)}"
    )
    print(
        f"Forward active-noise power  = "
        f"{forward_noise_power:.6f}"
    )
    print(
        f"Reverse active-noise power  = "
        f"{reverse_noise_power:.6f}"
    )
    print(
        f"Observation difference      = "
        f"{observation_difference_power:.6f}"
    )
    print(
        f"|rho_AB|                    = "
        f"{metrics.correlation_magnitude:.6f}"
    )
    print(
        f"Gaussian MI                 = "
        f"{metrics.mutual_information_bits_per_sample:.6f} "
        "bit/sample"
    )
    print(
        f"Raw KDR                     = "
        f"{metrics.key_disagreement_rate:.6f}"
    )

    assert forward_noise_power > 0.0
    assert reverse_noise_power > 0.0

    assert not np.allclose(
        result.observation_at_a,
        result.observation_at_b,
    )

    assert (
        metrics.correlation_magnitude
        > 0.9
    )

    print(
        "STEP 2 MANUAL CHECK: PASS"
    )


if __name__ == "__main__":
    main()