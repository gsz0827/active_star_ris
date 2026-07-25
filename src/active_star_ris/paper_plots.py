from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use(
    "Agg"
)

import matplotlib.pyplot as plt
import numpy as np

from active_star_ris.sweep_statistics import (
    SweepAggregateResult,
)


def _select_sorted_summaries(
    summaries: Sequence[SweepAggregateResult],
    sweep_name: str,
    x_attribute: str,
) -> list[SweepAggregateResult]:
    selected = [
        item
        for item in summaries
        if item.sweep_name == sweep_name
    ]

    selected.sort(
        key=lambda item: float(
            getattr(
                item,
                x_attribute,
            )
        )
    )

    return selected


def _save_figure(
    figure,
    output_directory: Path,
    base_filename: str,
) -> tuple[
    Path,
    Path,
]:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_directory
        / f"{base_filename}.png"
    )

    pdf_path = (
        output_directory
        / f"{base_filename}.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return (
        png_path,
        pdf_path,
    )


def _errorbar_plot(
    summaries: Sequence[SweepAggregateResult],
    *,
    sweep_name: str,
    x_attribute: str,
    mean_attribute: str,
    ci_attribute: str,
    x_label: str,
    y_label: str,
    output_directory: Path,
    base_filename: str,
) -> tuple[
    Path,
    Path,
] | None:
    selected = _select_sorted_summaries(
        summaries,
        sweep_name,
        x_attribute,
    )

    if len(selected) == 0:
        return None

    x_values = np.asarray(
        [
            float(
                getattr(
                    item,
                    x_attribute,
                )
            )
            for item in selected
        ],
        dtype=np.float64,
    )

    y_values = np.asarray(
        [
            float(
                getattr(
                    item,
                    mean_attribute,
                )
            )
            for item in selected
        ],
        dtype=np.float64,
    )

    y_errors = np.asarray(
        [
            float(
                getattr(
                    item,
                    ci_attribute,
                )
            )
            for item in selected
        ],
        dtype=np.float64,
    )

    figure = plt.figure(
        figsize=(6.4, 4.8)
    )

    axes = figure.add_subplot(
        111
    )

    axes.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        marker="o",
        capsize=4,
        linewidth=1.5,
    )

    axes.set_xlabel(
        x_label
    )

    axes.set_ylabel(
        y_label
    )

    axes.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    return _save_figure(
        figure,
        output_directory,
        base_filename,
    )


def _success_rate_plot(
    summaries: Sequence[SweepAggregateResult],
    *,
    sweep_name: str,
    x_attribute: str,
    x_label: str,
    output_directory: Path,
    base_filename: str,
) -> tuple[
    Path,
    Path,
] | None:
    selected = _select_sorted_summaries(
        summaries,
        sweep_name,
        x_attribute,
    )

    if len(selected) == 0:
        return None

    x_values = np.asarray(
        [
            float(
                getattr(
                    item,
                    x_attribute,
                )
            )
            for item in selected
        ],
        dtype=np.float64,
    )

    success_values = np.asarray(
        [
            item.dual_side_success_rate
            for item in selected
        ],
        dtype=np.float64,
    )

    figure = plt.figure(
        figsize=(6.4, 4.8)
    )

    axes = figure.add_subplot(
        111
    )

    axes.plot(
        x_values,
        success_values,
        marker="o",
        linewidth=1.5,
    )

    axes.set_xlabel(
        x_label
    )

    axes.set_ylabel(
        "Dual-side success probability"
    )

    axes.set_ylim(
        -0.02,
        1.02,
    )

    axes.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    return _save_figure(
        figure,
        output_directory,
        base_filename,
    )


def _two_series_plot(
    summaries: Sequence[SweepAggregateResult],
    *,
    sweep_name: str,
    x_attribute: str,
    first_mean_attribute: str,
    second_mean_attribute: str,
    first_label: str,
    second_label: str,
    x_label: str,
    y_label: str,
    output_directory: Path,
    base_filename: str,
) -> tuple[
    Path,
    Path,
] | None:
    selected = _select_sorted_summaries(
        summaries,
        sweep_name,
        x_attribute,
    )

    if len(selected) == 0:
        return None

    x_values = np.asarray(
        [
            float(
                getattr(
                    item,
                    x_attribute,
                )
            )
            for item in selected
        ],
        dtype=np.float64,
    )

    first_values = np.asarray(
        [
            float(
                getattr(
                    item,
                    first_mean_attribute,
                )
            )
            for item in selected
        ],
        dtype=np.float64,
    )

    second_values = np.asarray(
        [
            float(
                getattr(
                    item,
                    second_mean_attribute,
                )
            )
            for item in selected
        ],
        dtype=np.float64,
    )

    figure = plt.figure(
        figsize=(6.4, 4.8)
    )

    axes = figure.add_subplot(
        111
    )

    axes.plot(
        x_values,
        first_values,
        marker="o",
        linewidth=1.5,
        label=first_label,
    )

    axes.plot(
        x_values,
        second_values,
        marker="s",
        linewidth=1.5,
        label=second_label,
    )

    axes.set_xlabel(
        x_label
    )

    axes.set_ylabel(
        y_label
    )

    axes.grid(
        True,
        alpha=0.3,
    )

    axes.legend()

    figure.tight_layout()

    return _save_figure(
        figure,
        output_directory,
        base_filename,
    )


def write_sweep_split_csvs(
    summaries: Sequence[SweepAggregateResult],
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """
    按照sweep_name分别输出论文绘图CSV。
    """
    if len(summaries) == 0:
        raise ValueError(
            "summaries cannot be empty"
        )

    directory = Path(
        output_directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    grouped: dict[
        str,
        list[SweepAggregateResult],
    ] = {}

    for summary in summaries:
        grouped.setdefault(
            summary.sweep_name,
            [],
        ).append(
            summary
        )

    output_paths: list[Path] = []

    for sweep_name, group in grouped.items():
        rows = [
            asdict(
                item
            )
            for item in group
        ]

        output_path = (
            directory
            / f"{sweep_name}_summary.csv"
        )

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        output_paths.append(
            output_path
        )

    return tuple(
        output_paths
    )


def generate_paper_figures(
    summaries: Sequence[SweepAggregateResult],
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """
    生成PNG和PDF两种格式的论文实验图。

    仅在相应sweep_name存在时生成对应图像。
    """
    if len(summaries) == 0:
        raise ValueError(
            "summaries cannot be empty"
        )

    directory = Path(
        output_directory
    )

    output_paths: list[Path] = []

    figure_results = [
        _errorbar_plot(
            summaries,
            sweep_name="eve_channel_correlation",
            x_attribute="eve_channel_correlation",
            mean_attribute="quantized_eve_mi_mean",
            ci_attribute="quantized_eve_mi_ci95",
            x_label="Eve channel correlation",
            y_label=(
                "Quantized Eve information "
                "(bit/retained bit)"
            ),
            output_directory=directory,
            base_filename=(
                "eve_quantized_information"
            ),
        ),
        _errorbar_plot(
            summaries,
            sweep_name="eve_channel_correlation",
            x_attribute="eve_channel_correlation",
            mean_attribute=(
                "quantized_min_entropy_mean"
            ),
            ci_attribute=(
                "quantized_min_entropy_ci95"
            ),
            x_label="Eve channel correlation",
            y_label=(
                "Conditional min-entropy "
                "(bit/retained bit)"
            ),
            output_directory=directory,
            base_filename=(
                "eve_conditional_min_entropy"
            ),
        ),
        _errorbar_plot(
            summaries,
            sweep_name="eve_channel_correlation",
            x_attribute="eve_channel_correlation",
            mean_attribute=(
                "operational_bound_rate_mean"
            ),
            ci_attribute=(
                "operational_bound_rate_ci95"
            ),
            x_label="Eve channel correlation",
            y_label=(
                "Operational key bound "
                "(bit/channel sample)"
            ),
            output_directory=directory,
            base_filename=(
                "eve_operational_key_rate"
            ),
        ),
        _success_rate_plot(
            summaries,
            sweep_name="eve_channel_correlation",
            x_attribute="eve_channel_correlation",
            x_label="Eve channel correlation",
            output_directory=directory,
            base_filename=(
                "eve_key_generation_success"
            ),
        ),
        _errorbar_plot(
            summaries,
            sweep_name=(
                "directional_phase_error_std_rad"
            ),
            x_attribute=(
                "directional_phase_error_std_rad"
            ),
            mean_attribute="legitimate_mi_mean",
            ci_attribute="legitimate_mi_ci95",
            x_label=(
                "Directional phase-error "
                "standard deviation (rad)"
            ),
            y_label=(
                "Legitimate mutual information "
                "(bit/sample)"
            ),
            output_directory=directory,
            base_filename=(
                "phase_mismatch_legitimate_mi"
            ),
        ),
        _errorbar_plot(
            summaries,
            sweep_name=(
                "directional_phase_error_std_rad"
            ),
            x_attribute=(
                "directional_phase_error_std_rad"
            ),
            mean_attribute=(
                "operational_bound_rate_mean"
            ),
            ci_attribute=(
                "operational_bound_rate_ci95"
            ),
            x_label=(
                "Directional phase-error "
                "standard deviation (rad)"
            ),
            y_label=(
                "Operational key bound "
                "(bit/channel sample)"
            ),
            output_directory=directory,
            base_filename=(
                "phase_mismatch_operational_rate"
            ),
        ),
        _errorbar_plot(
            summaries,
            sweep_name="guard_band_sigma",
            x_attribute="guard_band_sigma",
            mean_attribute=(
                "operational_bound_rate_mean"
            ),
            ci_attribute=(
                "operational_bound_rate_ci95"
            ),
            x_label=(
                "Guard-band width "
                "(standard deviations)"
            ),
            y_label=(
                "Operational key bound "
                "(bit/channel sample)"
            ),
            output_directory=directory,
            base_filename=(
                "guard_band_operational_rate"
            ),
        ),
        _two_series_plot(
            summaries,
            sweep_name="guard_band_sigma",
            x_attribute="guard_band_sigma",
            first_mean_attribute=(
                "transmission_raw_kdr_mean"
            ),
            second_mean_attribute=(
                "reflection_raw_kdr_mean"
            ),
            first_label="Transmission side",
            second_label="Reflection side",
            x_label=(
                "Guard-band width "
                "(standard deviations)"
            ),
            y_label="Raw key disagreement rate",
            output_directory=directory,
            base_filename="guard_band_raw_kdr",
        ),
        _two_series_plot(
            summaries,
            sweep_name="guard_band_sigma",
            x_attribute="guard_band_sigma",
            first_mean_attribute=(
                "transmission_retention_ratio_mean"
            ),
            second_mean_attribute=(
                "reflection_retention_ratio_mean"
            ),
            first_label="Transmission side",
            second_label="Reflection side",
            x_label=(
                "Guard-band width "
                "(standard deviations)"
            ),
            y_label="Retained-sample ratio",
            output_directory=directory,
            base_filename=(
                "guard_band_retention_ratio"
            ),
        ),
    ]

    for result in figure_results:
        if result is not None:
            output_paths.extend(
                result
            )

    return tuple(
        output_paths
    )