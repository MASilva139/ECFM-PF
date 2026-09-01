# Plotting para delphes de tabla plana
from collections.abc import Iterable
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..styles import PlotStyle
from .common import (
    draw_histogram,
    finalize_figure,
    finite_values,
    plot_fit_result as _plot_fit_result,
    require_columns,
    style_axis
)

def plot_event_multiplicities(
    events: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    bins: int = 30,
    log_y: bool = True,
    save: bool = False,
    filename: str = "delphes_event_multiplicities",
    output_dir: str | Path | None = None,
    show: bool = True
):
    if columns is None:
        preferred = ("n_Muon", "n_Jet", "n_Electron", "n_Photon", "n_Particle")
        columns = tuple(column for column in preferred if column in events.columns)
    else:
        columns = tuple(columns)
    if not columns:
        raise KeyError("Events no contiene columnas de multiplicidad seleccionables.")
    require_columns(events, columns, "plot_event_multiplicities")
    n_columns = min(3, len(columns))
    n_rows = int(np.ceil(len(columns) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(6 * n_columns, 4.8 * n_rows),
        squeeze=False,
    )
    for ax, column in zip(axes.flat, columns):
        values = finite_values(events, column)
        draw_histogram(ax, values, bins=bins)
        style_axis(
            fig,
            ax,
            title=column,
            xlabel="Objetos por evento",
            ylabel="Eventos/bin",
        )
        if log_y:
            ax.set_yscale("log")
    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)
    fig.suptitle("Multiplicidades de objetos Delphes", fontweight="bold")
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, axes

def plot_muon_kinematics(
    muons: pd.DataFrame,
    *,
    bins: int = 60,
    save: bool = False,
    filename: str = "delphes_muon_kinematics",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    definitions = (
        ("PT", r"$p_T(\mu)$ [GeV/$c$]"),
        ("Eta", r"$\eta(\mu)$"),
        ("Phi", r"$\phi(\mu)$ [rad]"),
    )
    require_columns(muons, (item[0] for item in definitions), "plot_muon_kinematics")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), squeeze=False)
    for ax, (column, xlabel) in zip(axes.flat, definitions):
        draw_histogram(ax, finite_values(muons, column), bins=bins)
        style_axis(
            fig,
            ax,
            title=column,
            xlabel=xlabel,
            ylabel="Muones/bin",
        )
    fig.suptitle("Cinemática de muones reconstruidos", fontweight="bold")
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, axes

def plot_jet_kinematics(
    jets: pd.DataFrame,
    *,
    bins: int = 60,
    save: bool = False,
    filename: str = "delphes_jet_kinematics",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    definitions = [
        ("PT", r"$p_T(j)$ [GeV/$c$]"),
        ("Eta", r"$\eta(j)$"),
        ("Phi", r"$\phi(j)$ [rad]"),
    ]
    if "Mass" in jets.columns:
        definitions.append(("Mass", r"$m_j$ [GeV/$c^2$]"))
    require_columns(jets, (item[0] for item in definitions), "plot_jet_kinematics")
    n_columns = len(definitions)
    fig, axes = plt.subplots(1, n_columns, figsize=(5.6 * n_columns, 5.5), squeeze=False)
    for ax, (column, xlabel) in zip(axes.flat, definitions):
        draw_histogram(ax, finite_values(jets, column), bins=bins)
        style_axis(fig, ax, title=column, xlabel=xlabel, ylabel="Jets/bin")
    fig.suptitle("Cinemática de jets reconstruidos", fontweight="bold")
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, axes

def plot_dimuon_mass(
    dimuons: pd.DataFrame,
    *,
    bins: int = 100,
    mass_range: tuple[float, float] | None = None,
    category_column: str | None = None,
    log_y: bool = False,
    save: bool = False,
    filename: str = "delphes_dimuon_mass",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    require_columns(dimuons, ("dimuon_mass",), "plot_dimuon_mass")
    fig, ax = plt.subplots(figsize=(10, 7))
    if category_column is None:
        draw_histogram(
            ax,
            finite_values(dimuons, "dimuon_mass"),
            bins=bins,
            value_range=mass_range,
            label="Todos",
            color=PlotStyle.signal,
        )
    else:
        require_columns(dimuons, (category_column,), "plot_dimuon_mass")
        colors = {
            "prompt": PlotStyle.prompt,
            "displaced": PlotStyle.displaced,
            "mixed": PlotStyle.background_model,
        }
        for category, group in dimuons.groupby(category_column, sort=False):
            values = finite_values(group, "dimuon_mass")
            if values.size == 0:
                continue
            draw_histogram(
                ax,
                values,
                bins=bins,
                value_range=mass_range,
                label=str(category),
                color=colors.get(str(category)),
                filled=False,
            )
    style_axis(
        fig,
        ax,
        title="Espectro de masa dimuónica",
        xlabel=r"$m_{\mu\mu}$ [GeV/$c^2$]",
        ylabel="Candidatos/bin",
    )
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_prompt_displaced_mass(
    dimuons: pd.DataFrame,
    **kwargs,
):
    kwargs.setdefault("category_column", "displacement_category")
    kwargs.setdefault("filename", "delphes_prompt_displaced_mass")
    return plot_dimuon_mass(dimuons, **kwargs)

def plot_dimuon_delta_r(
    dimuons: pd.DataFrame,
    *,
    bins: int = 80,
    value_range: tuple[float, float] | None = None,
    save: bool = False,
    filename: str = "delphes_dimuon_delta_r",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    values = finite_values(dimuons, "delta_r")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    draw_histogram(
        ax,
        values,
        bins=bins,
        value_range=value_range,
        color=PlotStyle.background_model,
    )
    style_axis(
        fig,
        ax,
        title="Separación angular de los muones",
        xlabel=r"$\Delta R(\mu^+,\mu^-)$",
        ylabel="Candidatos/bin",
    )
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_displacement_significance(
    dimuons: pd.DataFrame,
    *,
    bins: int = 80,
    significance_range: tuple[float, float] = (0.0, 20.0),
    log_y: bool = False,
    save: bool = False,
    filename: str = "delphes_d0_significance",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    columns = ("muplus_d0_significance", "muminus_d0_significance")
    require_columns(dimuons, columns, "plot_displacement_significance")
    fig, ax = plt.subplots(figsize=(10, 7))

    for column, label, color in (
        (columns[0], r"$\mu^+$", PlotStyle.prompt),
        (columns[1], r"$\mu^-$", PlotStyle.displaced),
    ):
        draw_histogram(
            ax,
            finite_values(dimuons, column),
            bins=bins,
            value_range=significance_range,
            label=label,
            color=color,
            filled=False,
        )
    style_axis(
        fig,
        ax,
        title="Significancia del parámetro de impacto",
        xlabel=r"$|D_0/\sigma_{D_0}|$",
        ylabel="Muones/bin",
    )
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_mass_vs_displacement(
    dimuons: pd.DataFrame,
    *,
    bins: tuple[int, int] = (100, 80),
    mass_range: tuple[float, float] | None = None,
    significance_range: tuple[float, float] = (0.0, 20.0),
    save: bool = False,
    filename: str = "delphes_mass_vs_displacement",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    columns = (
        "dimuon_mass",
        "muplus_d0_significance",
        "muminus_d0_significance",
    )
    require_columns(dimuons, columns, "plot_mass_vs_displacement")
    mass = dimuons["dimuon_mass"].to_numpy(dtype=np.float64, copy=False)
    significance = np.maximum(
        dimuons[columns[1]].to_numpy(dtype=np.float64, copy=False),
        dimuons[columns[2]].to_numpy(dtype=np.float64, copy=False),
    )
    valid = np.isfinite(mass) & np.isfinite(significance)
    if not np.any(valid):
        raise ValueError("No hay valores finitos para masa y significancia.")
    if mass_range is None:
        mass_range = (float(np.min(mass[valid])), float(np.max(mass[valid])))
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.hist2d(
        mass[valid],
        significance[valid],
        bins=bins,
        range=(mass_range, significance_range),
        cmap="viridis",
    )[3]
    style_axis(
        fig,
        ax,
        title="Masa frente a desplazamiento",
        xlabel=r"$m_{\mu\mu}$ [GeV/$c^2$]",
        ylabel=r"máx$\{|D_0/\sigma_{D_0}|\}$",
    )
    PlotStyle.add_dark_colorbar(fig, ax, image, label="Candidatos/bin")
    finalize_figure(
        fig,
        save=save,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_truth_pt_response(
    matched_muons: pd.DataFrame,
    *,
    reconstructed_column: str = "PT",
    truth_column: str = "truth_PT",
    bins: int = 100,
    save: bool = False,
    filename: str = "delphes_truth_pt_response",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    require_columns(
        matched_muons,
        (reconstructed_column, truth_column),
        "plot_truth_pt_response",
    )
    reco = matched_muons[reconstructed_column].to_numpy(dtype=np.float64, copy=False)
    truth = matched_muons[truth_column].to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(reco) & np.isfinite(truth) & (truth > 0)
    if not np.any(valid):
        raise ValueError("No hay muones con PT reconstruido y de verdad válidos.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
    scatter_ax, residual_ax = axes[0]
    scatter_ax.scatter(truth[valid], reco[valid], s=5, alpha=0.25)
    limits = [
        min(float(np.min(truth[valid])), float(np.min(reco[valid]))),
        max(float(np.max(truth[valid])), float(np.max(reco[valid]))),
    ]
    scatter_ax.plot(limits, limits, linestyle="--", color=PlotStyle.signal)
    style_axis(
        fig,
        scatter_ax,
        title="Respuesta en PT",
        xlabel=r"$p_T^{truth}$ [GeV/$c$]",
        ylabel=r"$p_T^{reco}$ [GeV/$c$]",
    )
    relative = (reco[valid] - truth[valid]) / truth[valid]
    draw_histogram(
        residual_ax,
        relative,
        bins=bins,
        color=PlotStyle.background_model,
    )
    style_axis(
        fig,
        residual_ax,
        title="Residuo relativo",
        xlabel=r"$(p_T^{reco}-p_T^{truth})/p_T^{truth}$",
        ylabel="Muones/bin",
    )

    finalize_figure(
        fig,
        save=save,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, axes

def plot_fit_result(
    result: dict,
    *,
    title: str = "Ajuste del espectro dimuónico",
    save: bool = False,
    filename: str = "delphes_dimuon_fit",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    return _plot_fit_result(
        result,
        title=title,
        xlabel=r"$m_{\mu\mu}$ [GeV/$c^2$]",
        save=save,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )

def plot_two_muon_mass(
    pairs: pd.DataFrame,
    *,
    bins: int = 100,
    mass_range: tuple[float, float] | None = None,
    log_y: bool = False,
    show_substracted: bool = True,
    save: bool = False,
    filename: str = "delphes_two_muon_mass",
    output_dir: str | Path | None = None,
    show: bool = True
):
    require_columns(pairs, ("dimuon_mass", "is_opposite_sign"), "plot_two_muon_mass")
    os_mass = finite_values(pairs.loc[pairs["is_opposite_sign"]], "dimuon_mass")
    ss_mass = finite_values(pairs.loc[~pairs["is_opposite_sign"]], "dimuon_mass")
    n_panels = 2 if (show_substracted and ss_mass.size) else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(9*n_panels, 7), squeeze=False)
    ax = axes[0, 0]
    if mass_range is None:
        combined = np.concatenate([os_mass, ss_mass]) if ss_mass.size else os_mass
        mass_range = (float(np.min(combined)), float(np.max(combined))) if combined.size else None
    counts_os, edges = np.histogram(os_mass, bins=bins, range=mass_range)
    draw_histogram(ax, os_mass, bins=bins, value_range=mass_range, label="OS (signo opuesto)", color=PlotStyle.signal)
    if ss_mass.size:
        draw_histogram(ax, ss_mass, bins=bins, value_range=mass_range, label="SS (mismo signo)", color=PlotStyle.background_model, filled=False)
        ax.legend()
    style_axis(fig, ax, title="Masa invariante del par de muones", xlabel=r"$m_{\mu\mu}$ [$GeV/c^2$]", ylabel="Eventos/bin")
    if log_y:
        ax.set_yscale("log")
    if n_panels == 2:
        counts_ss, _ = np.histogram(ss_mass, bins=edges)
        centers = 0.5*(edges[:-1] + edges[1:])
        substracted = counts_os - counts_ss
        ax2 = axes[0, 1]
        ax2.step(centers, substracted, where="mid", color=PlotStyle.prompt)
        ax2.axhline(0.0, color=PlotStyle.text, linewidth = 1.0, linestyle="--")
        style_axis(fig, ax2, title="OS - SS (fondo combinatorio sustraído)", xlabel=r"$m_{\mu\mu}$ [$GeV/c^2$]", ylabel="Eventos/bin (sustraídos)")
    finalize_figure(fig, save=save, filename=filename, data='delphes', output_dir=output_dir, show=show)
    return fig, axes