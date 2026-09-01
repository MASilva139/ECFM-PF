# Plotting para archivos dvntuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..styles import PlotStyle
from .common import (
    draw_histogram,
    finalize_figure,
    finite_values,
    first_available_column,
    plot_fit_result as _plot_fit_result,
    require_columns,
    style_axis,
)

def plot_b_mass(
    dataframe: pd.DataFrame,
    *,
    mass_column: str | None = None,
    bins: int = 80,
    mass_range: tuple[float, float] = (5100.0, 5500.0),
    split_charge: bool = False,
    id_column: str = "Bplus_ID",
    save: bool = False,
    filename: str = "b2kmm_b_mass",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    if mass_column is None:
        mass_column = first_available_column(
            dataframe, ("Bplus_MM", "Bplus_M"), "plot_b_mass"
        )
    fig, ax = plt.subplots(figsize=(10, 7))
    if split_charge:
        require_columns(dataframe, (id_column,), "plot_b_mass")
        for label, mask, color in (
            (r"$B^+$", dataframe[id_column] > 0, PlotStyle.prompt),
            (r"$B^-$", dataframe[id_column] < 0, PlotStyle.displaced),
        ):
            values = finite_values(dataframe.loc[mask], mass_column)
            if values.size:
                draw_histogram(
                    ax,
                    values,
                    bins=bins,
                    value_range=mass_range,
                    label=label,
                    color=color,
                    filled=False,
                )
        ax.legend()
    else:
        draw_histogram(
            ax,
            finite_values(dataframe, mass_column),
            bins=bins,
            value_range=mass_range,
            color=PlotStyle.signal,
        )
    style_axis(
        fig,
        ax,
        title="Masa reconstruida del candidato B",
        xlabel=r"$m_B$ [MeV/$c^2$]",
        ylabel="Candidatos/bin",
    )
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        data="data",
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_dimuon_mass(
    dataframe: pd.DataFrame,
    *,
    mass_column: str | None = None,
    bins: int = 100,
    mass_range: tuple[float, float] | None = None,
    log_y: bool = False,
    save: bool = False,
    filename: str = "b2kmm_dimuon_mass",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    if mass_column is None:
        mass_column = first_available_column(
            dataframe,
            ("J_psi_1S_MM", "J_psi_1S_M"),
            "plot_dimuon_mass",
        )
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_histogram(
        ax,
        finite_values(dataframe, mass_column),
        bins=bins,
        value_range=mass_range,
        color=PlotStyle.background_model,
    )
    style_axis(
        fig,
        ax,
        title="Espectro de masa dimuónica",
        xlabel=r"$m_{\mu\mu}$ [MeV/$c^2$]",
        ylabel="Candidatos/bin",
    )
    if log_y:
        ax.set_yscale("log")
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        data="data",
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_q2(
    dataframe: pd.DataFrame,
    *,
    mass_column: str | None = None,
    bins: int = 100,
    q2_range: tuple[float, float] | None = None,
    save: bool = False,
    filename: str = "b2kmm_q2",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    """Grafica q²=m²(μ+μ−) en GeV²/c⁴."""

    if mass_column is None:
        mass_column = first_available_column(
            dataframe,
            ("J_psi_1S_MM", "J_psi_1S_M"),
            "plot_q2",
        )
    mass_gev = finite_values(dataframe, mass_column, scale=1e-3)
    q2 = mass_gev**2
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_histogram(
        ax,
        q2,
        bins=bins,
        value_range=q2_range,
        color=PlotStyle.signal,
    )
    style_axis(
        fig,
        ax,
        title=r"Distribución de $q^2$",
        xlabel=r"$q^2=m^2_{\mu\mu}$ [GeV$^2/c^4$]",
        ylabel="Candidatos/bin",
    )
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        data="data",
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_candidate_kinematics(
    dataframe: pd.DataFrame,
    *,
    bins: int = 60,
    momentum_scale: float = 1e-3,
    save: bool = False,
    filename: str = "b2kmm_candidate_kinematics",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    definitions = (
        ("Bplus_PT", r"$B$", PlotStyle.signal),
        ("Kplus_PT", r"$K^+$", PlotStyle.background_model),
        ("muplus_PT", r"$\mu^+$", PlotStyle.prompt),
        ("muminus_PT", r"$\mu^-$", PlotStyle.displaced),
    )
    require_columns(
        dataframe,
        (column for column, _, _ in definitions),
        "plot_candidate_kinematics",
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), squeeze=False)
    for ax, (column, label, color) in zip(axes.flat, definitions):
        draw_histogram(
            ax,
            finite_values(dataframe, column, scale=momentum_scale),
            bins=bins,
            color=color,
        )
        style_axis(
            fig,
            ax,
            title=f"PT de {label}",
            xlabel=r"$p_T$ [GeV/$c$]",
            ylabel="Candidatos/bin",
        )
    fig.suptitle(r"Cinemática de $B^+\to K^+\mu^+\mu^-$", fontweight="bold")
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        data="data",
        output_dir=output_dir,
        show=show,
    )
    return fig, axes

def plot_ipchi2(
    dataframe: pd.DataFrame,
    *,
    bins: int = 80,
    log_x: bool = True,
    log_y: bool = False,
    save: bool = False,
    filename: str = "b2kmm_ipchi2",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    definitions = (
        ("Kplus_IPCHI2_OWNPV", r"$K^+$", PlotStyle.background_model),
        ("muplus_IPCHI2_OWNPV", r"$\mu^+$", PlotStyle.prompt),
        ("muminus_IPCHI2_OWNPV", r"$\mu^-$", PlotStyle.displaced),
    )
    require_columns(dataframe, (item[0] for item in definitions), "plot_ipchi2")
    fig, ax = plt.subplots(figsize=(10, 7))
    for column, label, color in definitions:
        values = finite_values(dataframe, column)
        values = values[values > 0]
        if values.size == 0:
            continue
        if log_x:
            log_edges = np.logspace(
                np.log10(values.min()), np.log10(values.max()), bins + 1
            )
            hist_bins = log_edges
        else:
            hist_bins = bins
        draw_histogram(
            ax,
            values,
            bins=hist_bins,
            label=label,
            color=color,
            filled=False,
        )
    style_axis(
        fig,
        ax,
        title="Parámetro de impacto respecto al PV",
        xlabel=r"$\chi^2_{IP}$",
        ylabel="Candidatos/bin",
    )
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        data="data",
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_muon_pid(
    dataframe: pd.DataFrame,
    *,
    bins: int = 60,
    save: bool = False,
    filename: str = "b2kmm_muon_pid",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    """Compara ProbNNmu de μ+ y μ−."""

    plus_column = first_available_column(
        dataframe,
        ("muplus_ProbNNmu", "muplus_MC15TuneV1_ProbNNmu"),
        "plot_muon_pid",
    )
    minus_column = first_available_column(
        dataframe,
        ("muminus_ProbNNmu", "muminus_MC15TuneV1_ProbNNmu"),
        "plot_muon_pid",
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    for column, label, color in (
        (plus_column, r"$\mu^+$", PlotStyle.prompt),
        (minus_column, r"$\mu^-$", PlotStyle.displaced),
    ):
        draw_histogram(
            ax,
            finite_values(dataframe, column),
            bins=bins,
            value_range=(0.0, 1.0),
            label=label,
            color=color,
            filled=False,
        )
    style_axis(
        fig,
        ax,
        title="Identificación de muones",
        xlabel="ProbNNmu",
        ylabel="Candidatos/bin",
    )
    ax.legend()
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        data="data",
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_mass_correlation(
    dataframe: pd.DataFrame,
    *,
    b_mass_column: str | None = None,
    dimuon_mass_column: str | None = None,
    bins: tuple[int, int] = (100, 100),
    save: bool = False,
    filename: str = "b2kmm_mass_correlation",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    if b_mass_column is None:
        b_mass_column = first_available_column(
            dataframe, ("Bplus_MM", "Bplus_M"), "plot_mass_correlation"
        )
    if dimuon_mass_column is None:
        dimuon_mass_column = first_available_column(
            dataframe,
            ("J_psi_1S_MM", "J_psi_1S_M"),
            "plot_mass_correlation",
        )
    require_columns(
        dataframe,
        (b_mass_column, dimuon_mass_column),
        "plot_mass_correlation",
    )
    b_mass = dataframe[b_mass_column].to_numpy(dtype=np.float64, copy=False)
    dimuon_mass = dataframe[dimuon_mass_column].to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(b_mass) & np.isfinite(dimuon_mass)
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.hist2d(b_mass[valid], dimuon_mass[valid], bins=bins, cmap="viridis")[3]
    style_axis(
        fig,
        ax,
        title="Correlación de masas",
        xlabel=r"$m_B$ [MeV/$c^2$]",
        ylabel=r"$m_{\mu\mu}$ [MeV/$c^2$]",
    )
    PlotStyle.add_dark_colorbar(fig, ax, image, label="Candidatos/bin")
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        data="data",
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_fit_result(
    result: dict,
    *,
    title: str = "Ajuste de masa",
    xlabel: str = r"Masa [MeV/$c^2$]",
    save: bool = False,
    filename: str = "b2kmm_mass_fit",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    return _plot_fit_result(
        result,
        title=title,
        xlabel=xlabel,
        save=save,
        filename=filename,
        data="data",
        output_dir=output_dir,
        show=show,
    )