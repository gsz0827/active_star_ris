from pathlib import Path

import numpy as np
import pytest

from active_star_ris.optimization import (
    design_passive_surface,
)
from active_star_ris.simulation import (
    evaluate_one_realization_imperfect_csi,
    load_config,
    sample_channels_with_imperfect_csi,
)
from active_star_ris.system import (
    evaluate_two_user_system,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "default.yaml"
)


def test_imperfect_csi_evaluation_runs():
    config = load_config(
        CONFIG_PATH
    )

    rng = np.random.default_rng(
        211
    )

    nmse_db, results = (
        evaluate_one_realization_imperfect_csi(
            config=config,
            rng=rng,
            num_elements=16,
            num_active_elements=4,
            nmse_db=-15.0,
        )
    )

    vector_beta_result = results[
        "partial_active_vector_beta"
    ]

    assert np.isfinite(
        vector_beta_result.weighted_sum_rate
    )

    assert (
        vector_beta_result
        .beta_transmission_std
        >= 0.0
    )

    assert (
        0.0
        <= vector_beta_result
        .beta_transmission_min
        <= 1.0
    )

    assert (
        0.0
        <= vector_beta_result
        .beta_transmission_max
        <= 1.0
    )

    assert (
        vector_beta_result
        .beta_transmission_min
        <= vector_beta_result
        .beta_transmission
        + 1.0e-12
    )

    assert (
        vector_beta_result
        .beta_transmission
        <= vector_beta_result
        .beta_transmission_max
        + 1.0e-12
    )

    assert (
        vector_beta_result
        .ris_power_violation
        >= 0.0
    )

    passive_names = (
        "passive_random",
        "passive_phase_aligned",
    )

    for name in passive_names:
        result = results[name]

        assert (
            result.active_gain_std
            == pytest.approx(0.0)
        )

        assert (
            result.active_gain_min
            == pytest.approx(1.0)
        )

        assert (
            result.active_gain_max
            == pytest.approx(1.0)
        )

    active_names = (
        "partial_active_fixed_beta",
        "partial_active_optimized_beta",
    )

    maximum_active_amplitude = float(
        config["system"][
            "maximum_active_amplitude"
        ]
    )

    for name in active_names:
        result = results[name]

        assert result.active_gain_std >= 0.0
        assert result.active_gain_min >= 1.0

        assert (
            result.active_gain_max
            <= maximum_active_amplitude
            + 1.0e-12
        )

        assert (
            result.active_gain_min
            <= result.active_amplitude
            + 1.0e-12
        )

        assert (
            result.active_amplitude
            <= result.active_gain_max
            + 1.0e-12
        )

        if (
            result.effective_active_elements
            <= 1
        ):
            assert (
                result.active_gain_std
                == pytest.approx(0.0)
            )

    assert nmse_db == -15.0

    scenarios = {
        "passive_random",
        "passive_phase_aligned",
        "partial_active_fixed_beta",
        "partial_active_optimized_beta",
        "partial_active_vector_beta",
    }

    for result in results.values():
        assert np.isfinite(
            result.transmission_snr
        )

        assert np.isfinite(
            result.reflection_snr
        )

        assert np.isfinite(
            result.weighted_sum_rate
        )

        assert (
            result.transmission_snr
            >= 0.0
        )

        assert (
            result.reflection_snr
            >= 0.0
        )

        assert (
            result.weighted_sum_rate
            >= 0.0
        )

        assert (
            result.ris_power_violation
            >= 0.0
        )


def test_default_nmse_is_read_from_config():
    config = load_config(
        CONFIG_PATH
    )

    rng = np.random.default_rng(
        223
    )

    nmse_db, _ = (
        evaluate_one_realization_imperfect_csi(
            config=config,
            rng=rng,
            num_elements=8,
            num_active_elements=2,
        )
    )

    assert nmse_db == pytest.approx(
        float(
            config["csi"][
                "default_nmse_db"
            ]
        )
    )


def test_passive_design_uses_estimated_csi_but_true_evaluation():
    config = load_config(
        CONFIG_PATH
    )

    seed = 227
    num_elements = 16
    selected_nmse_db = -8.0

    # -------------------------------------------------
    # 手工执行：
    # 估计CSI负责设计，真实CSI负责评价。
    # -------------------------------------------------
    manual_rng = np.random.default_rng(
        seed
    )

    snapshot = (
        sample_channels_with_imperfect_csi(
            config=config,
            rng=manual_rng,
            num_elements=num_elements,
            nmse_db=selected_nmse_db,
        )
    )

    true_channel = snapshot.true
    estimated_channel = (
        snapshot.estimated
    )

    system_cfg = config["system"]

    beta = float(
        system_cfg["beta_transmission"]
    )

    manually_designed_surface = (
        design_passive_surface(
            estimated_channel.alice_to_ris,
            estimated_channel
            .ris_to_transmission_user,
            estimated_channel
            .ris_to_reflection_user,
            beta,
            estimated_channel
            .direct_transmission,
            estimated_channel
            .direct_reflection,
        )
    )

    manual_true_metrics = (
        evaluate_two_user_system(
            true_channel.alice_to_ris,
            true_channel
            .ris_to_transmission_user,
            true_channel
            .ris_to_reflection_user,
            manually_designed_surface,
            float(
                system_cfg["transmit_power"]
            ),
            float(
                system_cfg[
                    "user_noise_variance"
                ]
            ),
            float(
                system_cfg[
                    "ris_internal_noise_variance"
                ]
            ),
            float(
                system_cfg[
                    "ris_output_power_budget"
                ]
            ),
            float(
                system_cfg[
                    "transmission_weight"
                ]
            ),
            float(
                system_cfg[
                    "reflection_weight"
                ]
            ),
            true_channel.direct_transmission,
            true_channel.direct_reflection,
        )
    )

    # -------------------------------------------------
    # 调用刚刚新增的正式评价函数。
    # 使用相同随机种子可生成完全相同的信道快照。
    # -------------------------------------------------
    function_rng = np.random.default_rng(
        seed
    )

    _, results = (
        evaluate_one_realization_imperfect_csi(
            config=config,
            rng=function_rng,
            num_elements=num_elements,
            num_active_elements=4,
            nmse_db=selected_nmse_db,
        )
    )

    function_result = results[
        "passive_phase_aligned"
    ]

    assert (
        function_result.weighted_sum_rate
        == pytest.approx(
            manual_true_metrics
            .weighted_sum_rate,
            rel=1.0e-12,
            abs=1.0e-12,
        )
    )

    assert (
        function_result.transmission_snr
        == pytest.approx(
            manual_true_metrics
            .transmission.snr_linear,
            rel=1.0e-12,
            abs=1.0e-12,
        )
    )

    assert (
        function_result.reflection_snr
        == pytest.approx(
            manual_true_metrics
            .reflection.snr_linear,
            rel=1.0e-12,
            abs=1.0e-12,
        )
    )