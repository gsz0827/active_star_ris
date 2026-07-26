from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from active_star_ris.experiment_pipeline import (
    build_environment_config,
    run_single_experiment,
)


def load_default_config():
    with Path(
        "configs/default.yaml"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def test_three_structure_action_dimensions():
    config = load_default_config()

    num_elements = int(
        config["system"][
            "num_elements"
        ]
    )

    passive = build_environment_config(
        config,
        num_active_elements=0,
        quick=True,
    )
    partial = build_environment_config(
        config,
        num_active_elements=8,
        quick=True,
    )
    fully_active = build_environment_config(
        config,
        num_active_elements=num_elements,
        quick=True,
        allow_active_bypass_override=False,
    )

    # 动作包括3N个透射/反射相位和能量分配控制，
    # 每个有源单元另增加一个增益控制。
    assert (
        passive.num_active_elements
        == 0
    )
    assert (
        partial.num_active_elements
        == 8
    )
    assert (
        fully_active.num_active_elements
        == num_elements
    )
    assert not fully_active.allow_active_bypass

    partial = build_environment_config(
        config,
        num_active_elements=8,
        quick=True,
        allow_active_bypass_override=True,
    )

    assert partial.allow_active_bypass


def test_ablation_configuration_is_applied():
    config = load_default_config()

    no_noise = build_environment_config(
        config,
        num_active_elements=2,
        ablation="no_internal_noise",
        quick=True,
    )

    assert np.isclose(
        no_noise
        .domain_randomization
        .ris_internal_noise_variance_min,
        0.0,
    )
    assert np.isclose(
        no_noise
        .domain_randomization
        .ris_internal_noise_variance_max,
        0.0,
    )

    no_cvar = build_environment_config(
        config,
        num_active_elements=2,
        ablation="no_cvar",
        quick=True,
    )

    assert np.isclose(
        no_cvar.robust_mean_weight,
        1.0,
    )
    assert np.isclose(
        no_cvar.robust_cvar_weight,
        0.0,
    )


def test_single_experiment_writes_outputs(
    tmp_path,
    monkeypatch,
):
    config = load_default_config()

    from active_star_ris.td3_agent import (
        TD3Agent,
    )

    loaded_checkpoints: list[Path] = []

    original_load_checkpoint = (
        TD3Agent.load_checkpoint
    )

    def tracking_load_checkpoint(
        self,
        path,
        *,
        load_optimizers=True,
    ):
        loaded_checkpoints.append(
            Path(path)
        )

        return original_load_checkpoint(
            self,
            path,
            load_optimizers=(
                load_optimizers
            ),
        )

    monkeypatch.setattr(
        TD3Agent,
        "load_checkpoint",
        tracking_load_checkpoint,
    )

    summary = run_single_experiment(
        config,
        output_directory=(
            tmp_path / "smoke"
        ),
        scenario="smoke",
        num_active_elements=1,
        ablation="full_model",
        seed=7,
        evaluation_seed=900,
        final_evaluation_episodes=1,
        device="cpu",
        quick=True,
        total_steps_override=12,
    )

    episodes_csv = (
        tmp_path
        / "smoke"
        / "csv"
        / "episodes.csv"
    )

    assert episodes_csv.exists()
    assert episodes_csv.stat().st_size > 0

    assert np.isfinite(
        summary[
            "episode_return_mean"
        ]
    )

    expected_files = (
        "checkpoints/best.pt",
        "checkpoints/final.pt",
        "csv/episodes.csv",
        "csv/periodic_evaluations.csv",
        "csv/final_evaluation_steps.csv",
        "csv/final_evaluation_episodes.csv",
        "csv/final_evaluation_summary.csv",
        "figures/episode_return.png",
        "run_metadata.json",
    )

    for filename in expected_files:
        assert (
            tmp_path
            / "smoke"
            / filename
        ).exists()

    # 最终详细评价必须实际加载过检查点。
    assert loaded_checkpoints

    # 最后加载的检查点应当是验证集表现最佳的best.pt。
    assert (
        loaded_checkpoints[-1].name
        == "best.pt"
    )

    expected_best_checkpoint = (
        tmp_path
        / "smoke"
        / "checkpoints"
        / "best.pt"
    )

    assert (
        Path(
            summary["evaluated_checkpoint"]
        )
        == expected_best_checkpoint
    )

    # 验证集种子和独立测试集种子应当分离。
    assert (
        summary["validation_seed"]
        == 900
    )

    assert (
        summary["final_test_seed"]
        == 1_000_900
    )

    # best.pt的附加信息应被读取并写入结果。
    assert isinstance(
        summary[
            "evaluated_checkpoint_extra"
        ],
        dict,
    )
