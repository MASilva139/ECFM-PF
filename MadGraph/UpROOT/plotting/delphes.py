# plotting/delphes
from collections.abc import Iterable
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter, NullFormatter
from ..styles import PlotStyle
from .common import (
    draw_histogram,
    finalize_figure,
    finite_values,
    histogram_bins,
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
        style_axis(fig, ax, title=column, xlabel="Objetos por evento", ylabel="Eventos/bin")
        if log_y:
            ax.set_yscale("log")
    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)
    fig.suptitle("Multiplicidades de objetos Delphes", fontweight="bold")
    finalize_figure(fig, save=save, data=data, filename=filename, output_dir=output_dir, show=show)
    return fig, axes

def _plot_object_kinematics(
    objects: pd.DataFrame,
    *,
    label: str,
    bins: int,
    log_x: bool,
    log_y: bool,
    rank: pd.Series | None,
    rank_labels: tuple[str, str] | None,
    title: str,
    save: bool,
    data: str,
    filename: str,
    output_dir: str | Path | None,
    show: bool
):
    context = '_plot_object_kinematics'
    require_columns(objects, ('PT', 'Eta', 'Phi'), context)
    derived = objects[['PT', 'Eta', 'Phi']].copy()
    derived.loc[derived['Eta'].abs() >= 900.0, 'Eta'] = np.nan
    derived['PX'] = derived['PT'] * np.cos(derived['Phi'])
    derived['PY'] = derived['PT'] * np.sin(derived['Phi'])
    derived['PZ'] = derived['PT'] * np.sinh(derived['Eta'])
    derived['P'] = np.sqrt(derived['PX']**2 + derived['PY']**2 + derived['PZ']**2)
    if rank is not None:
        derived['__rank'] = rank.to_numpy()
    fig = plt.figure(figsize=(20, 8.0))
    grid = gridspec.GridSpec(1, 2, width_ratios=(2, 1.45), wspace=0.25)
    left = grid[0].subgridspec(2, 3, wspace=0.35, hspace=0.35)
    axes = {
        'PT': fig.add_subplot(left[0, 0]),
        'Eta': fig.add_subplot(left[0, 1]),
        'Phi': fig.add_subplot(left[0, 2]),
        'PX': fig.add_subplot(left[1, 0]),
        "PY": fig.add_subplot(left[1, 1]),
        "PZ": fig.add_subplot(left[1, 2]),
        "P": fig.add_subplot(grid[1]),
    }
    labels = {
        "PT": rf"$p_T({label})$ [GeV/$c$]",
        "Eta": rf"$\eta({label})$",
        "Phi": rf"$\phi({label})$ [rad]",
        "PX": rf"$p_x({label})$ [GeV/$c$]",
        "PY": rf"$p_y({label})$ [GeV/$c$]",
        "PZ": rf"$p_z({label})$ [GeV/$c$]",
        "P": rf"$p({label})$ [GeV/$c$]",
    }
    for column, ax in axes.items():
        if rank is None:
            draw_histogram(ax, finite_values(derived, column), bins=bins)
        else:
            assert rank_labels is not None
            for value, curve_label, color in (
                (1, rank_labels[0], PlotStyle.signal),
                (2, rank_labels[1], PlotStyle.background_model),
            ):
                values = finite_values(derived.loc[derived["__rank"].eq(value)], column)
                if values.size:
                    draw_histogram(ax, values, bins=bins, label=curve_label, color=color, filled=False)
            ax.legend(fontsize=8)
        style_axis(fig, ax, title=column, xlabel=labels[column], ylabel="Objetos/bin")
        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
    fig.suptitle(title, fontweight="bold", fontsize=20)
    finalize_figure(fig, save=save, data=data, filename=filename, output_dir=output_dir, show=show)
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
    context = 'plot_muon_kinematics'
    rank = None
    rank_labels = None
    if split_charge:
        require_columns(muons, (charge_column,), context)
        rank = muons[charge_column].map({1: 1, -1: 2})
        rank_labels = (r'$\mu^+$', r'$\mu^-$')
    return _plot_object_kinematics(
        muons,
        label=r'\mu',
        bins=bins,
        log_x=log_x,
        log_y=log_y,
        rank=rank,
        rank_labels=rank_labels,
        title='Cinemática de muones reconstruidos',
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show
    )

def plot_jet_kinematics(
    jets: pd.DataFrame,
    *,
    bins: int = 60,
    log_y: bool = False,
    save: bool = False,
    data: str,
    filename: str = "delphes_jet_kinematics",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    context = "plot_jet_kinematics"
    definitions = [
        ("PT", r"$p_T(j)$ [GeV/$c$]"),
        ("Eta", r"$\eta(j)$"),
        ("Phi", r"$\phi(j)$ [rad]"),
    ]
    if "Mass" in jets.columns:
        definitions.append(("Mass", r"$m_j$ [GeV/$c^2$]"))
    require_columns(jets, (item[0] for item in definitions), context)
    n_columns = len(definitions)
    fig, axes = plt.subplots(1, n_columns, figsize=(5.6 * n_columns, 5.5), squeeze=False)
    for ax, (column, xlabel) in zip(axes.flat, definitions):
        draw_histogram(ax, finite_values(jets, column), bins=bins)
        style_axis(fig, ax, title=column, xlabel=xlabel, ylabel="Jets/bin")
        if log_y:
            ax.set_yscale('log')
    fig.suptitle("Cinemática de jets reconstruidos", fontweight="bold")
    finalize_figure(fig, save=save, data=data, filename=filename, output_dir=output_dir, show=show,)
    return fig, axes

def _plot_mass_spectrum(
    frame: pd.DataFrame,
    *,
    column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    curve_label: str,
    bins: int,
    mass_range: tuple[float, float] | None,
    log_x: bool,
    log_y: bool,
    density: bool,
    save: bool,
    data: str,
    filename: str,
    output_dir: str | Path | None,
    show: bool,
):
    context = "_plot_mass_spectrum"
    require_columns(frame, (column,), context)
    values = finite_values(frame, column)
    if values.size == 0:
        raise ValueError(f"No hay valores finitos en {column!r}.")
    edges = histogram_bins(values=values, bins=bins, value_range=mass_range, logarithmic=log_x)
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_histogram(ax, values, bins=edges, value_range=mass_range, label=curve_label, color=PlotStyle.signal, density=density)
    style_axis(fig, ax, title=title, xlabel=xlabel, ylabel=('Densidad' if density else ylabel))
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend()
    finalize_figure(fig, save=save, data=data, filename=filename, output_dir=output_dir, show=show)
    return fig, ax

def plot_dimuon_mass(
    dimuons: pd.DataFrame,
    *,
    bins: int = 100,
    mass_range: tuple[float, float] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    density: bool = False,
    save: bool = False,
    data: str,
    filename: str = "dimuon_mass",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    return _plot_mass_spectrum(
        dimuons,
        column='dimuon_mass',
        title=r'Espectro de masa dimuónica ($pp\to\mu^+\mu^-$)',
        xlabel=r'$m_{\mu\mu}$ [GeV/$c^2$]',
        ylabel='Candidatos/bin',
        curve_label='',
        bins=bins,
        mass_range=mass_range,
        log_x=log_x,
        log_y=log_y,
        density=density,
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show
    )

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
    draw_histogram(ax, values, bins=bins, value_range=value_range, color=PlotStyle.background_model,)
    style_axis(fig, ax, title="Separación angular de los muones", xlabel=r"$\Delta R(\mu^+,\mu^-)$", ylabel="Candidatos/bin")
    finalize_figure(fig, save=save, data=data, filename=filename, output_dir=output_dir, show=show,)
    return fig, ax

def plot_two_muon_mass(
    pairs: pd.DataFrame,
    *,
    bins: int = 100,
    mass_range: tuple[float, float] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    save: bool = False,
    data: str,
    filename: str = "delphes_two_muon_mass",
    output_dir: str | Path | None = None,
    show: bool = True
):
    context = 'plot_two_muon_mass'
    require_columns(pairs, ("dimuon_mass", "is_opposite_sign"), context)
    # os_mass = finite_values(pairs.loc[pairs["is_opposite_sign"]], "dimuon_mass")    # signo opuesto
    # ss_mass = finite_values(pairs.loc[~pairs["is_opposite_sign"]], "dimuon_mass")   # mismo signo
    # n_panels = 2 if (show_substracted and ss_mass.size) else 1
    # fig, axes = plt.subplots(1, n_panels, figsize=(9*n_panels, 7), squeeze=False)
    # ax = axes[0, 0]
    # if mass_range is None:
    #     combined = np.concatenate([os_mass, ss_mass]) if ss_mass.size else os_mass
    #     mass_range = (float(np.min(combined)), float(np.max(combined))) if combined.size else None
    # counts_os, edges = np.histogram(os_mass, bins=bins, range=mass_range)
    # draw_histogram(ax, os_mass, bins=bins, value_range=mass_range, label="OS (signo opuesto)", color=PlotStyle.signal)
    # if ss_mass.size:
    #     draw_histogram(ax, ss_mass, bins=bins, value_range=mass_range, label="SS (mismo signo)", color=PlotStyle.background_model, filled=False)
    #     ax.legend()
    # style_axis(fig, ax, title="Masa invariante del par de muones", xlabel=r"$m_{\mu\mu}$ [$GeV/c^2$]", ylabel="Eventos/bin")
    # if log_y:
    #     ax.set_yscale("log")
    # if log_x:
    #     ax.set_xscale('log')
    #     ax.xaxis.set_major_formatter(ScalarFormatter())
    # if n_panels == 2:
    #     counts_ss, _ = np.histogram(ss_mass, bins=edges)
    #     centers = 0.5*(edges[:-1] + edges[1:])
    #     substracted = counts_os - counts_ss
    #     ax2 = axes[0, 1]
    #     ax2.step(centers, substracted, where="mid", color=PlotStyle.prompt)
    #     ax2.axhline(0.0, color=PlotStyle.text, linewidth = 1.0, linestyle="--")
    #     style_axis(fig, ax2, title="OS - SS (fondo combinatorio sustraído)", xlabel=r"$m_{\mu\mu}$ [$GeV/c^2$]", ylabel="Eventos/bin (sustraídos)")
    fig, ax = plt.subplots(figsize=(10, 7))
    for opposite, label, color in (
        (True, 'OS (signo opuesto)', PlotStyle.signal),
        (False, 'SS (mismo signo)', PlotStyle.background_model)
    ):
        values = finite_values(pairs.loc[pairs['is_opposite_sign'].eq(opposite)], 'dimuon_mass')
        log_bins = histogram_bins(values=values, bins=bins, value_range=mass_range, logarithmic=log_x)
        if values.size:
            draw_histogram(ax, values, bins=log_bins, value_range=mass_range, label=label, color=color, filled=opposite)
        style_axis(fig, ax, title='Masa invariante del par de muones', xlabel=r'$m_{\mu\mu}$ [GeV/$c^2$]', ylabel='Eventos/bin')
        if log_y:
            ax.set_yscale('log')
        if log_x:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter)
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig, ax

def plot_particle_kinematics(
    particles: pd.DataFrame,
    *,
    label: str = "Z'",
    bins: int = 60,
    log_x: bool = False,
    log_y: bool = False,
    split_rank: bool = False,   # Separación de partícula lider y sublider por pT
    save: bool = False,
    data: str,
    filename: str = "particle_kinematics",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    rank = None
    rank_labels = None
    context = "plot_particle_kinematics"
    if split_rank:
        require_columns(particles, ('event_id',), context)
        keys = [column for column in ('source_file_id', 'event_id') if column in particles.columns]
        rank = particles.groupby(keys, sort=False)['PT'].rank(method='first', ascending=False)
        rank_labels = (f'{label} líder', f'{label} sublider')
    return _plot_object_kinematics(
        particles,
        label=label,
        bins=bins,
        log_x=log_x,
        log_y=log_y,
        rank=rank,
        rank_labels=rank_labels,
        title=rf'Cinemática de ${label}$',
        save=save,
        data=data,
        filename=filename,
        output_dir=output_dir,
        show=show
    )

def plot_mediator_multiplicity(
    particles: pd.DataFrame,
    *,
    label: str = "Z'",
    save: bool = False,
    data: str,
    filename: str = "mediator_multiplicity",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    context = "plot_mediator_multiplicity"
    require_columns(particles, ("event_id",), context)
    keys = [column for column in ("source_file_id", "event_id") if column in particles.columns]
    counts =particles.groupby(keys, sort=False).size()
    distribution = counts.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.bar(
        distribution.index.to_numpy(dtype=np.int64),
        distribution.to_numpy(dtype=np.int64),
        color=PlotStyle.truth,
        alpha=0.8,
    )
    style_axis(fig, ax, title=rf"Multiplicidad de ${label}$", xlabel="Partículas por evento", ylabel="Eventos")
    ax.set_xticks(distribution.index.to_numpy(dtype=np.int64))
    finalize_figure(fig, save=save, data=data, filename=filename, output_dir=output_dir, show=show)
    return fig, ax

def plot_mediator_pair_observables(
    pairs: pd.DataFrame,
    *,
    label: str = "Z'",
    bins: int = 70,
    pair_mass_range: tuple[float, float] | None = None,
    log_y: bool = False,
    save: bool = False,
    data: str,
    filename: str = "truth_mediator_pair_observables",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    context = "plot_mediator_pair_observables"
    required = ("p1_pt", "p2_pt", "pair_mass", "pair_pt", "pair_rapidity", "pair_delta_r", "pair_delta_phi")
    require_columns(pairs, required, context)
    if pairs.empty:
        raise ValueError(f"{context}: no hay pares truth.")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    individual_pt = np.concatenate([finite_values(pairs, "p1_pt"), finite_values(pairs, "p2_pt"),])
    definitions = (
        (axes[0, 0], individual_pt, None, r"$p_T$ individual", r"$p_T(Z')$ [GeV/$c$]"),
        (axes[0, 1], None, "pair_mass", r"Masa del sistema $Z'Z'$", r"$m(Z'Z')$ [GeV/$c^2$]"),
        (axes[0, 2], None, "pair_pt", r"$p_T$ del sistema", r"$p_T(Z'Z')$ [GeV/$c$]"),
        (axes[1, 0], None, "pair_rapidity", "Rapidez del sistema", r"$y(Z'Z')$"),
        (axes[1, 1], None, "pair_delta_r", "Separación entre mediadores", r"$\Delta R(Z'_1,Z'_2)$"),
        (axes[1, 2], None, "pair_delta_phi", "Separación azimutal", r"$|\Delta\phi(Z'_1,Z'_2)|$ [rad]"),
    )
    for ax, direct, column, title, xlabel in definitions:
        # if column is not None:
        #     require_columns(pairs, (column,), context)
        #     values = finite_values(pairs, column)
        # else:
        #     values = direct
        values = direct if column is None else finite_values(pairs, column)
        value_range = pair_mass_range if column == "pair_mass" else None
        draw_histogram(ax, values, bins=bins, value_range=value_range, color=PlotStyle.signal)
        style_axis(fig, ax, title=title, xlabel=xlabel, ylabel="Eventos/bin")
        if log_y:
            ax.set_yscale("log")
    fig.suptitle(rf"Cinemática de $pp\to {label}{label}$", fontweight="bold")
    finalize_figure(fig, save=save, data=data, filename=filename, output_dir=output_dir, show=show)
    return fig, axes

def plot_particle_mass(
    particles: pd.DataFrame,
    *,
    label: str = "Z'",
    column: str = "Mass",
    bins: int = 100,
    mass_range: tuple[float, float] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    density: bool = False,
    save: bool = False,
    data: str,
    filename: str = "particle_mass",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    # Masa de la partícula individual (columna Mass de Particles).
    return _plot_mass_spectrum(
        particles,
        column=column,
        title=rf"Espectro de masa de ${label}$",
        xlabel=rf"$m({label})$ [GeV/$c^2$]",
        ylabel="Partículas/bin",
        curve_label=rf"${label}$",
        bins=bins, mass_range=mass_range, log_x=log_x, log_y=log_y, density=density,
        save=save, data=data, filename=filename, output_dir=output_dir, show=show,
    )

def plot_particle_pair_mass(
    pairs: pd.DataFrame,
    *,
    label: str = "Z'",
    column: str = "pair_mass",
    bins: int = 100,
    mass_range: tuple[float, float] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    density: bool = False,
    save: bool = False,
    data: str,
    filename: str = "particle_pair_mass",
    output_dir: str | Path | None = None,
    show: bool = True,
):
    # Masa invariante del par (análogo de plot_dimuon_mass). 'pairs' viene de ph.build_particle_pairs.
    return _plot_mass_spectrum(
        pairs,
        column=column,
        title=rf"Espectro de masa de ${label}{label}$",
        xlabel=rf"$m({label}{label})$ [GeV/$c^2$]",
        ylabel="Candidatos/bin",
        curve_label="Todos",
        bins=bins, mass_range=mass_range, log_x=log_x, log_y=log_y, density=density,
        save=save, data=data, filename=filename, output_dir=output_dir, show=show,
    )