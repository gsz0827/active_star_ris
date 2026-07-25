from pathlib import Path

import numpy as np

from active_star_ris.imperfect_csi_sweep import (
    run_imperfect_csi_sweep,
    write_imperfect_csi_sweep_csv,
)
from active_star_ris.simulation import (
    load_config,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "default.yaml"
)


def test_imperfect_csi_sweep_returns_all_records():
    config = load_config(
        CONFIG_PATH
    )

    nmse_values = [
        -25.0,
        -5.0,
    ]

    records = run_imperfect_csi_sweep(
        config=config,
        nmse_values_db=nmse_values,
        num_trials=3,
        seed=401,
        num_elements=8,
        num_active_elements=2,
    )

    scenarios = {
        "passive_random",
        "passive_phase_aligned",
        "partial_active_fixed_beta",
        "partial_active_optimized_beta",
        "partial_active_vector_beta",
    }

    assert len(records) == (
        len(nmse_values)
        * len(scenarios)
    )

    assert {
        record.scenario
        for record in records
    } == scenarios

    assert {
        record.nmse_db
        for record in records
    } == set(nmse_values)

    for record in records:
        assert record.num_trials == 3

        assert np.isfinite(
            record.mean_weighted_sum_rate
        )

        assert np.isfinite(
            record.mean_ris_output_power
        )

        assert (
            0.0
            <= record
            .power_violation_probability
            <= 1.0
        )

        assert (
            record.mean_power_violation
            >= 0.0
        )

        assert (
            record.maximum_power_violation
            >= record.mean_power_violation
        )

        assert (
            record.mean_requested_active_elements
            >= 0.0
        )

        assert (
            record.mean_effective_active_elements
            >= 0.0
        )

        assert (
            record.mean_effective_active_elements
            <= record.mean_requested_active_elements
            + 1.0e-12
        )

        assert (
            0.0
            <= record.active_reduction_probability
            <= 1.0
        )

        assert (
            0.0
            <= record.passive_fallback_probability
            <= 1.0
        )

        assert (
            record.mean_active_gain_std
            >= 0.0
        )

        assert (
            record.mean_active_gain_min
            >= 1.0
        )

        assert (
            record.mean_active_gain_max
            >= record.mean_active_gain_min
            - 1.0e-12
        )

        assert (
            record.mean_active_gain_min
            <= record.mean_active_amplitude
            + 1.0e-12
        )

        assert (
            record.mean_active_amplitude
            <= record.mean_active_gain_max
            + 1.0e-12
        )

        assert (
            record.mean_beta_transmission_std
            >= 0.0
        )

        assert (
            0.0
            <= record.mean_beta_transmission_min
            <= 1.0
        )

        assert (
            0.0
            <= record.mean_beta_transmission_max
            <= 1.0
        )

        assert (
            record.mean_beta_transmission_min
            <= record.mean_beta_transmission_max
            + 1.0e-12
        )

        assert (
            record.mean_beta_transmission_min
            <= record.mean_beta_transmission
            + 1.0e-12
        )

        assert (
            record.mean_beta_transmission
            <= record.mean_beta_transmission_max
            + 1.0e-12
        )


def test_random_passive_is_invariant_across_nmse():
    config = load_config(
        CONFIG_PATH
    )

    records = run_imperfect_csi_sweep(
        config=config,
        nmse_values_db=[
            -30.0,
            0.0,
        ],
        num_trials=5,
        seed=409,
        num_elements=8,
        num_active_elements=2,
    )

    random_records = sorted(
        [
            record
            for record in records
            if record.scenario
            == "passive_random"
        ],
        key=lambda item: item.nmse_db,
    )

    assert len(random_records) == 2

    assert np.isclose(
        random_records[0]
        .mean_weighted_sum_rate,
        random_records[1]
        .mean_weighted_sum_rate,
        rtol=0.0,
        atol=1.0e-12,
    )

    assert np.isclose(
        random_records[0]
        .mean_ris_output_power,
        random_records[1]
        .mean_ris_output_power,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_sweep_csv_is_created(
    tmp_path: Path,
):
    config = load_config(
        CONFIG_PATH
    )

    records = run_imperfect_csi_sweep(
        config=config,
        nmse_values_db=[
            -20.0,
        ],
        num_trials=2,
        seed=419,
        num_elements=8,
        num_active_elements=2,
    )

    output_path = (
        tmp_path
        / "imperfect_csi_sweep.csv"
    )

    returned_path = (
        write_imperfect_csi_sweep_csv(
            records=records,
            output_path=output_path,
        )
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    file_text = output_path.read_text(
        encoding="utf-8-sig"
    )

    assert "nmse_db" in file_text
    assert "scenario" in file_text
    assert (
        "power_violation_probability"
        in file_text
    )

    assert (
        "mean_effective_active_elements"
        in file_text
    )

    assert (
        "passive_fallback_probability"
        in file_text
    )

    assert (
        "mean_active_gain_std"
        in file_text
    )

    assert (
        "mean_active_gain_min"
        in file_text
    )

    assert (
        "mean_active_gain_max"
        in file_text
    )

    assert (
        "mean_beta_transmission_std"
        in file_text
    )

    assert (
        "mean_beta_transmission_min"
        in file_text
    )

    assert (
        "mean_beta_transmission_max"
        in file_text
    )

    assert (
        "partial_active_vector_beta"
        in file_text
    )