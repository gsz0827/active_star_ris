from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback only when optional accelerator is missing
    njit = None


def _evolve_gauss_markov_impl(
    initial: np.ndarray,
    standard_normals: np.ndarray,
    correlation: float,
) -> np.ndarray:
    """JIT-friendly implementation of the repository's existing recurrence.

    The innovation variance is recomputed from the previous sample exactly as in
    ``gauss_markov_update``. ``standard_normals`` has shape ``(T-1, 2, N)``.
    """
    num_samples = standard_normals.shape[0] + 1
    num_elements = initial.size
    result = np.empty((num_samples, num_elements), dtype=np.complex128)
    result[0] = initial
    rho = min(max(correlation, 0.0), 1.0)
    dynamic_scale = np.sqrt(max(0.0, 1.0 - rho * rho)) / np.sqrt(2.0)

    for sample_index in range(1, num_samples):
        previous = result[sample_index - 1]
        power_sum = 0.0
        any_nonzero = False
        for element_index in range(num_elements):
            value = previous[element_index]
            power_sum += value.real * value.real + value.imag * value.imag
            if value.real != 0.0 or value.imag != 0.0:
                any_nonzero = True

        if not any_nonzero:
            for element_index in range(num_elements):
                result[sample_index, element_index] = 0.0j
            continue

        power = max(power_sum / num_elements, 1.0e-30)
        innovation_scale = dynamic_scale * np.sqrt(power)
        for element_index in range(num_elements):
            noise = (
                standard_normals[sample_index - 1, 0, element_index]
                + 1j * standard_normals[sample_index - 1, 1, element_index]
            )
            result[sample_index, element_index] = (
                rho * previous[element_index]
                + innovation_scale * noise
            )
    return result


if njit is not None:
    evolve_gauss_markov = njit(cache=True)(_evolve_gauss_markov_impl)
else:
    evolve_gauss_markov = _evolve_gauss_markov_impl
