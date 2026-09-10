# Plotting para delphes de tabla plana
from collections.abc import Iterable
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
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
    data: str,
    save: bool = False,
    filename: str = "event_multiplicities",
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
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, axes

def plot_muon_kinematics(
    muons: pd.DataFrame,
    *,
    bins: int = 60,
    log_x: bool = False,
    log_y: bool = False,
    split_charge: bool = False,
    charge_column: str = "Charge",
    save: bool = False,
    data: str,
    filename: str = "muon_kinematics",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    required = ['PT', 'Eta', 'Phi']
    if split_charge:
        required.append(charge_column)
    require_columns(muons, required, "plot_muon_kinematics")
    derived = muons[required].copy()
    derived['PX'] = derived['PT'] * np.cos(derived['Phi'])
    derived['PY'] = derived['PT'] * np.sin(derived['Phi'])
    derived['PZ'] = derived['PT'] * np.sinh(derived['Eta'])
    derived['P'] = np.sqrt(derived['PX']**2 + derived['PY']**2 + derived['PZ']**2)
    fig = plt.figure(figsize=(20, 8.0))
    gs = gridspec.GridSpec(1, 2, width_ratios=(2, 1.45), wspace=0.25)
    left_gs = gs[0].subgridspec(2, 3, wspace=0.35, hspace=0.35)
    right_gs = gs[1].subgridspec(1, 1)
    axes = {
        'PT': fig.add_subplot(left_gs[0, 0]),
        'Eta': fig.add_subplot(left_gs[0, 1]),
        'Phi': fig.add_subplot(left_gs[0, 2]),
        'PX': fig.add_subplot(left_gs[1, 0]),
        'PY': fig.add_subplot(left_gs[1, 1]),
        'PZ': fig.add_subplot(left_gs[1, 2]),
        'P': fig.add_subplot(right_gs[0, 0])
    }
    labels = {
        "PT": r"$p_T(\mu)$ [GeV/$c$]",
        "Eta": r"$\eta(\mu)$",
        "Phi": r"$\phi(\mu)$ [rad]",
        "PX": r"$p_x(\mu)$ [GeV/$c$]",
        "PY": r"$p_y(\mu)$ [GeV/$c$]",
        "PZ": r"$p_z(\mu)$ [GeV/$c$]",
        "P": r"$p(\mu)$ [GeV/$c$]"
    }
    for column, ax in axes.items():
        if split_charge:
            for charge, sign_label, color in ((1, r"$\mu^+$", PlotStyle.prompt), (-1, r"$\mu^-$", PlotStyle.displaced)):
                subset = derived.loc[derived[charge_column] == charge]
                values = subset[column].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size:
                    draw_histogram(ax, values, bins=bins, label=sign_label, color=color, filled=False)
            ax.legend(fontsize=8)
        else:
            values = derived[column].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            draw_histogram(ax, values, bins=bins)
        style_axis(fig, ax, title=column, xlabel=labels[column], ylabel="Muones/bin")
        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ScalarFormatter())
    fig.suptitle("Cinematica de muones reconstruidos", fontweight="bold", fontsize=20)
    finalize_figure(fig, save=save, data=data, filename=filename, output_dir=output_dir, show=show)
    return fig, axes

def plot_jet_kinematics(
    jets: pd.DataFrame,
    *,
    bins: int = 60,
    save: bool = False,
    data: str,
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
        data=data,
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
    log_x: bool = False,
    log_y: bool = False,
    save: bool = False,
    data: str,
    filename: str = "dimuon_mass",
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
            "mixed": PlotStyle.mixed,
            'invalid': PlotStyle.invalid
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
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()
    finalize_figure(
        fig,
        save=save,
        data=data,
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
    kwargs.setdefault("filename", "prompt_displaced_mass")
    return plot_dimuon_mass(dimuons, **kwargs)

def plot_dimuon_delta_r(
    dimuons: pd.DataFrame,
    *,
    bins: int = 80,
    value_range: tuple[float, float] | None = None,
    save: bool = False,
    data: str,
    filename: str = "dimuon_delta_r",
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
        data=data,
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
    log_x: bool = False,
    log_y: bool = False,
    save: bool = False,
    data: str,
    filename: str = "d0_significance",
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
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()
    finalize_figure(
        fig,
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_min_chi2_ip_proxy(
    dimuons: pd.date_range,
    *,
    bins: int = 80,
    value_range: tuple[float, float] | None = (0.0, 20.0),
    category_column: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    save: bool = False,
    data: str, 
    filename: str = 'sqrt_min_chi2_ip_proxy',
    output_dir: str | Path | None = None,
    show: bool = True
):
    context = 'plot_min_chi2_ip_proxy'
    proxy_column = 'sqrt_min_chi2_ip_proxy'
    required = [proxy_column]
    if category_column is not None:
        required.append(category_column)
    require_columns(dimuons, required, context)
    fig, ax = plt.subplots(figsize=(10, 7))
    if category_column is None:
        values = finite_values(dimuons, proxy_column)
        if values.size == 0:
            raise ValueError(f"{context}: sin valores finitos para graficar.")
        draw_histogram(ax, values, bins=bins, value_range=value_range, color=PlotStyle.signal)
    else:
        colors = {
            'prompt': PlotStyle.prompt,
            'displaced': PlotStyle.displaced,
            'mixed': PlotStyle.mixed,
            'invalid': PlotStyle.invalid
        }
        drew_values = False
        for category, group in dimuons.groupby(category_column, sort=False):
            values = finite_values(group, proxy_column)
            if values.size == 0:
                continue
            drew_values = True
            draw_histogram(
                ax,
                values,
                bins=bins,
                value_range=value_range,
                label=str(category),
                color=colors.get(str(category)),
                filled=False
            )
        if not drew_values:
            raise ValueError(f"{context}: sin valores finitos para graficar.")
        ax.legend()
    style_axis(
        fig,
        ax,
        title=r"Proxy de desplazamiento del candidato $\mu^{+}\mu^{-}$",
        xlabel=r"$\sqrt{\min(\chi^{2}_{IP,\,proxy})}$",
        ylabel="Candidatos/bin"
    )
    if log_y:
        ax.set_yscale('log')
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
    finalize_figure(
        fig,
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show
    )
    return fig, ax

def plot_decay_time(
    dimuons: pd.DataFrame,
    *,
    truth_column: str = 'dimuon_decay_time_truth_ps',
    reco_column:str = 'dimuon_decay_time_reco_ps',
    bins: int = 80,
    value_range: tuple[float, float] | None = None,
    density: bool = False,
    log_x: bool = False,
    log_y: bool = False,
    save: bool = False,
    data: str,
    filename: str = 'dimuon_decay_time',
    output_dir: str | Path | None = None,
    show: bool = True
):
    context = 'plot_decay_time'
    available = [
        (truth_column, "Truth", PlotStyle.truth),
        (reco_column, "Reco", PlotStyle.reco)
    ]
    available = [item for item in available if item[0] in dimuons.columns]
    if not available:
        raise KeyError(f"{context}: no existe {truth_column!r} ni {reco_column!r}.")
    fig, ax = plt.subplots(figsize=(10, 7))
    drew_values = False
    for column, label, color in available:
        values = finite_values(dimuons, column)
        if values.size == 0:
            continue
        drew_values = True
        draw_histogram(
            ax,
            values,
            bins=bins,
            value_range=value_range,
            density=density,
            label=label,
            color=color,
            filled=False
        )
    if not drew_values:
        raise ValueError(f"{context}: sin tiempos finitos para graficar.")
    style_axis(
        fig,
        ax,
        title="Tiempo propio del sistema dimuónico",
        xlabel=r"$t(\mu^{+}\mu^{-})\,[ps]$",
        ylabel="Densidad" if density else "Candidatos/bin"
    )
    if log_y:
        ax.set_yscale('log')
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()
    finalize_figure(
        fig,
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show
    )
    return fig, ax

def plot_truth_vs_reco_decay_time(
    dimuons: pd.DataFrame,
    *,
    truth_column: str = 'dimuon_decay_time_truth_ps',
    reco_column: str = 'dimuon_decay_time_reco_ps',
    bins: int | tuple[int, int] = 80,
    time_range: tuple[float, float] | None = None,
    residual_bins: int= 80,
    save: bool = False,
    data: str,
    filename: str = 'truth_vs_reco_decay_time',
    output_dir: str | Path | None = None,
    show: bool = True
):
    context = 'plot_truth_vs_reco_decay_time'
    require_columns(dimuons, (truth_column, reco_column), context)
    truth = dimuons[truth_column].to_numpy(dtype=np.float64)
    reco = dimuons[truth_column].to_numpy(dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(reco)
    if not np.any(valid):
        raise ValueError(f"{context}: no hay pares truth--reco finitos.")
    truth = truth[valid]
    reco = reco[valid]
    if time_range is None:
        lower = float(min(np.min(truth), np.min(reco)))
        upper = float(max(np.max(truth), np.max(reco)))
        if lower == upper:
            padding = max(abs(lower)*0.05, 1.0)
            lower -= padding
            upper += padding
        time_range = (lower, upper)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), squeeze=False)
    correlation_ax, residual_ax = axes[0]
    image = correlation_ax.hist2D(
        truth,
        reco,
        bins=bins,
        range=(time_range, time_range),
        cmap='viridis'
    )[3]
    correlation_ax.plot(
        time_range,
        time_range,
        linestyle='--',
        color=PlotStyle.signal,
        linewidth=1.5
    )
    style_axis(
        fig,
        correlation_ax,
        title="Respuesta del tiempo propio",
        xlabel=r"$t_{truth}\,[ps]$",
        ylabel=r"$t_{reco}\,[ps]$"
    )
    PlotStyle.add_dark_colorbar(fig, correlation_ax, image, label="Candidatos/bin")
    residual = reco - truth
    draw_histogram(
        residual_ax,
        residual,
        bins=residual_bins,
        color=PlotStyle.reco
    )
    residual_ax.axvline(0.0, color=PlotStyle.signal, linestyle='--', linewidth=1.3)
    style_axis(
        fig,
        residual_ax,
        title="Residuo temporal",
        xlabel=r"$t_{reco}-t_{truth}\,[ps]$",
        ylabel="Candidatos/bin"
    )
    finalize_figure(
        fig,
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show
    )
    return fig, axes

def plot_chi2_df_proxy(
    dimuons: pd.DataFrame,
    *,
    bins: int = 80,
    value_range: tuple[float, float] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    save: bool = False,
    data: str,
    filename: str = "chi2_df_proxy",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    context = "plot_chi2_df_proxy"
    values = finite_values(dimuons, "chi2_df_proxy")
    if values.size == 0:
        raise ValueError(f"{context}: no hay valores finitos para graficar.")
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_histogram(
        ax,
        values,
        bins=bins,
        value_range=value_range,
        color=PlotStyle.signal,
    )
    style_axis(
        fig,
        ax,
        title="Proxy de calidad del ajuste de decaimiento",
        xlabel=r"$\chi^2_{DF,\,proxy}$",
        ylabel="Candidatos/bin",
    )
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
    finalize_figure(
        fig,
        save=save,
        data=data,
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
    displacement_column: str = "sqrt_min_chi2_ip_proxy",
    save: bool = False,
    data: str,
    filename: str = "mass_vs_displacement",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    columns = ("dimuon_mass", displacement_column)
    require_columns(dimuons, columns, "plot_mass_vs_displacement")
    mass = dimuons["dimuon_mass"].to_numpy(dtype=np.float64)
    significance = dimuons[displacement_column].to_numpy(dtype=np.float64)
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
        ylabel=r"$\sqrt{\min(\chi^2_{IP,\,proxy})}$",
    )
    PlotStyle.add_dark_colorbar(fig, ax, image, label="Candidatos/bin")
    finalize_figure(
        fig,
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_mass_vs_chi2_ip(
    dimuons: pd.DataFrame,
    **kwargs,
):
    kwargs.setdefault("filename", "mass_vs_sqrt_min_chi2_ip_proxy")
    kwargs.setdefault("displacement_column", "sqrt_min_chi2_ip_proxy")
    return plot_mass_vs_displacement(dimuons, **kwargs)

def plot_mass_vs_decay_time(
    dimuons: pd.DataFrame,
    *,
    time_column: str = "dimuon_decay_time_reco_ps",
    bins: tuple[int, int] = (100, 80),
    mass_range: tuple[float, float] | None = None,
    time_range: tuple[float, float] | None = None,
    save: bool = False,
    data: str,
    filename: str = "mass_vs_decay_time",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    context = "plot_mass_vs_decay_time"
    require_columns(dimuons, ("dimuon_mass", time_column), context)
    mass = dimuons["dimuon_mass"].to_numpy(dtype=np.float64, copy=False)
    time = dimuons[time_column].to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(mass) & np.isfinite(time)
    if not np.any(valid):
        raise ValueError(f"{context}: no hay valores finitos para graficar.")
    mass = mass[valid]
    time = time[valid]
    if mass_range is None:
        mass_range = (float(np.min(mass)), float(np.max(mass)))
    if time_range is None:
        time_range = (float(np.min(time)), float(np.max(time)))
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.hist2d(
        mass,
        time,
        bins=bins,
        range=(mass_range, time_range),
        cmap="viridis",
    )[3]
    label = (
        r"$t_{truth}$ [ps]"
        if "truth" in time_column.lower()
        else r"$t_{reco}$ [ps]"
    )
    style_axis(
        fig,
        ax,
        title="Masa frente al tiempo propio",
        xlabel=r"$m_{\mu\mu}$ [GeV/$c^2$]",
        ylabel=label,
    )
    PlotStyle.add_dark_colorbar(fig, ax, image, label="Candidatos/bin")
    finalize_figure(
        fig,
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_mass_vs_decay_fit_proxy(
    dimuons: pd.DataFrame,
    *,
    bins: tuple[int, int] = (100, 80),
    mass_range: tuple[float, float] | None = None,
    proxy_range: tuple[float, float] | None = None,
    save: bool = False,
    data: str,
    filename: str = "mass_vs_chi2_df_proxy",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    context = "plot_mass_vs_decay_fit_proxy"
    require_columns(dimuons, ("dimuon_mass", "chi2_df_proxy"), context)
    mass = dimuons["dimuon_mass"].to_numpy(dtype=np.float64, copy=False)
    proxy = dimuons["chi2_df_proxy"].to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(mass) & np.isfinite(proxy)
    if not np.any(valid):
        raise ValueError(f"{context}: no hay valores finitos para graficar.")
    mass = mass[valid]
    proxy = proxy[valid]
    if mass_range is None:
        mass_range = (float(np.min(mass)), float(np.max(mass)))
    if proxy_range is None:
        proxy_range = (float(np.min(proxy)), float(np.max(proxy)))
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.hist2d(
        mass,
        proxy,
        bins=bins,
        range=(mass_range, proxy_range),
        cmap="viridis",
    )[3]
    style_axis(
        fig,
        ax,
        title="Masa frente al proxy de calidad del decay fit",
        xlabel=r"$m_{\mu\mu}$ [GeV/$c^2$]",
        ylabel=r"$\chi^2_{DF,\,proxy}$",
    )
    PlotStyle.add_dark_colorbar(fig, ax, image, label="Candidatos/bin")
    finalize_figure(
        fig,
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, ax

def plot_proxy_cut_scan(
    summary: pd.DataFrame,
    *,
    proxy_label: str = "Umbral del proxy",
    save: bool = False,
    data: str,
    filename: str = "proxy_cut_scan",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    context = "plot_proxy_cut_scan"
    columns = ("threshold", "selection_efficiency", "local_asimov")
    require_columns(summary, columns, context)
    threshold = summary["threshold"].to_numpy(dtype=np.float64, copy=False)
    efficiency = summary["selection_efficiency"].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    significance = summary["local_asimov"].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    if not np.any(np.isfinite(threshold)):
        raise ValueError(f"{context}: no hay umbrales finitos para graficar.")
    order = np.argsort(threshold)
    threshold = threshold[order]
    efficiency = efficiency[order]
    significance = significance[order]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), squeeze=False)
    efficiency_ax, significance_ax = axes[0]
    efficiency_ax.plot(
        threshold,
        efficiency,
        marker="o",
        color=PlotStyle.reco,
    )
    style_axis(
        fig,
        efficiency_ax,
        title="Eficiencia de selección",
        xlabel=proxy_label,
        ylabel="Fracción seleccionada",
    )
    efficiency_ax.set_ylim(0.0, 1.05)
    valid_significance = np.isfinite(threshold) & np.isfinite(significance)
    significance_ax.plot(
        threshold[valid_significance],
        significance[valid_significance],
        marker="o",
        color=PlotStyle.signal,
    )
    style_axis(
        fig,
        significance_ax,
        title="Significancia local del ajuste",
        xlabel=proxy_label,
        ylabel=r"$Z_A$",
    )
    finalize_figure(
        fig,
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, axes

def plot_long_lived_truth_validation(
    dimuons: pd.DataFrame,
    *,
    minimum_time_ps: float = 0.0,
    minimum_displacement_mm: float = 0.0,
    bins: int = 80,
    save: bool = False,
    data: str,
    filename: str = "long_lived_truth_validation",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    context = "plot_long_lived_truth_validation"
    columns = (
        "truth_parent_found",
        "truth_decay_length_mm",
        "dimuon_decay_time_truth_ps",
    )
    require_columns(dimuons, columns, context)
    if minimum_time_ps < 0 or minimum_displacement_mm < 0:
        raise ValueError(f"{context}: los umbrales no pueden ser negativos.")
    parent_found = dimuons["truth_parent_found"].fillna(False).to_numpy(dtype=bool)
    displacement = dimuons["truth_decay_length_mm"].to_numpy(dtype=np.float64)
    time = dimuons["dimuon_decay_time_truth_ps"].to_numpy(dtype=np.float64)
    finite_displacement = np.isfinite(displacement)
    finite_time = np.isfinite(time)
    long_lived = (parent_found & finite_displacement & finite_time & (displacement > minimum_displacement_mm) & (time > minimum_time_ps))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), squeeze=False)
    time_ax, displacement_ax, count_ax = axes[0]
    draw_histogram(
        time_ax,
        time[finite_time],
        bins=bins,
        color=PlotStyle.truth,
    )
    time_ax.axvline(
        minimum_time_ps,
        color=PlotStyle.displaced,
        linestyle="--",
    )
    style_axis(
        fig,
        time_ax,
        title="Tiempo propio truth",
        xlabel=r"$t_{truth}$ [ps]",
        ylabel="Candidatos/bin",
    )
    draw_histogram(
        displacement_ax,
        displacement[finite_displacement],
        bins=bins,
        color=PlotStyle.reco,
    )
    displacement_ax.axvline(
        minimum_displacement_mm,
        color=PlotStyle.displaced,
        linestyle="--",
    )
    style_axis(
        fig,
        displacement_ax,
        title="Distancia de decaimiento truth",
        xlabel=r"$|\vec L_{truth}|$ [mm]",
        ylabel="Candidatos/bin",
    )
    labels = ("Total", "Padre", "Tiempo finito", "Long-lived")
    counts = (
        len(dimuons),
        int(np.count_nonzero(parent_found)),
        int(np.count_nonzero(finite_time)),
        int(np.count_nonzero(long_lived)),
    )
    count_ax.bar(
        labels,
        counts,
        color=(
            PlotStyle.invalid,
            PlotStyle.signal,
            PlotStyle.reco,
            PlotStyle.displaced,
        ),
    )
    count_ax.tick_params(axis="x", rotation=20)
    style_axis(
        fig,
        count_ax,
        title="Disponibilidad de truth desplazado",
        xlabel="Estado",
        ylabel="Candidatos",
    )
    finalize_figure(
        fig,
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )
    return fig, axes

def plot_truth_pt_response(
    matched_muons: pd.DataFrame,
    *,
    reconstructed_column: str = "PT",
    truth_column: str = "truth_PT",
    bins: int = 100,
    save: bool = False,
    data: str,
    filename: str = "truth_pt_response",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    require_columns(matched_muons, (reconstructed_column, truth_column), "plot_truth_pt_response")
    reco = matched_muons[reconstructed_column].to_numpy(dtype=np.float64)
    truth = matched_muons[truth_column].to_numpy(dtype=np.float64)
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
        data=data,
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
    data: str,
    filename: str = "dimuon_fit",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    return _plot_fit_result(
        result,
        title=title,
        xlabel=r"$m_{\mu\mu}$ [GeV/$c^2$]",
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show,
    )

def plot_two_muon_mass(
    pairs: pd.DataFrame,
    *,
    bins: int = 100,
    mass_range: tuple[float, float] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    show_substracted: bool = True,
    save: bool = False,
    data: str,
    filename: str = "delphes_two_muon_mass",
    output_dir: str | Path | None = None,
    show: bool = True
):
    require_columns(pairs, ("dimuon_mass", "is_opposite_sign"), "plot_two_muon_mass")
    os_mass = finite_values(pairs.loc[pairs["is_opposite_sign"]], "dimuon_mass")    # signo opuesto
    ss_mass = finite_values(pairs.loc[~pairs["is_opposite_sign"]], "dimuon_mass")   # mismo signo
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
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
    if n_panels == 2:
        counts_ss, _ = np.histogram(ss_mass, bins=edges)
        centers = 0.5*(edges[:-1] + edges[1:])
        substracted = counts_os - counts_ss
        ax2 = axes[0, 1]
        ax2.step(centers, substracted, where="mid", color=PlotStyle.prompt)
        ax2.axhline(0.0, color=PlotStyle.text, linewidth = 1.0, linestyle="--")
        style_axis(fig, ax2, title="OS - SS (fondo combinatorio sustraído)", xlabel=r"$m_{\mu\mu}$ [$GeV/c^2$]", ylabel="Eventos/bin (sustraídos)")
    finalize_figure(
        fig, 
        save=save, 
        filename=filename, 
        data=data, 
        output_dir=output_dir, 
        show=show
    )
    return fig, axes