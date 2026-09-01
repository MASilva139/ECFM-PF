# Plotting para B2HHH
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from ..styles import PlotStyle
from .common import (
    draw_histogram,
    finalize_figure,
    finite_values,
    require_columns,
    style_axis
)

_HADRONS = ("H1", "H2", "H3")
_MOMENTUM_COMPONENTS = ("PX", "PY", "PZ", "P")

def plot_momentum(
        df,
        *,
        bins: int = 20,
        value_range: tuple[float, float] | None = None,
        log_y: bool = True,
        save: bool = False,
        filename: str = "b2hhh_momentum",
        data: str = 'sim' or 'data',
        output_dir = None,
        show: bool = True
):
    columns = [f"{h}_{c}" for h in _HADRONS for c in _MOMENTUM_COMPONENTS]
    require_columns(df, columns, "plot_momentum")
    fig, axes = plt.subplots(3, 4, figsize=(20, 11), squeeze=False)
    for row, hadron in enumerate(_HADRONS):
        for col, comp in enumerate(_MOMENTUM_COMPONENTS):
            ax = axes[row, col]
            column = f'{hadron}_{comp}'
            draw_histogram(ax, finite_values(df, column, scale=1e-3), bins=bins, value_range=value_range)
            style_axis(fig, ax, title=column, xlabel=rf"${column}$ [$GeV/c^{2}$]", ylabel="Eventos")
            if log_y:
                ax.set_yscale("log")
    fig.suptitle("Distribuciones de momentum de los hadrones", fontweight="bold")
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig, axes

def plot_energy(
    df,
    *,
    bins: int = 20,
    log_y: bool = False,
    save: bool = False,
    filename: str = "b2hhh_energy",
    data: str = 'sim' or 'data',
    output_dir = None,
    show: bool = True
):
    columns = [f"{h}_E" for h in _HADRONS]
    require_columns(df, columns, "plot_energy")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), squeeze=False)
    for ax, hadron in zip(axes[0], _HADRONS):
        column = f"{hadron}_E"
        draw_histogram(ax, finite_values(df, column, scale=1e-3), bins=bins)
        style_axis(fig, ax, title=f"Energía candidato {hadron}", xlabel=rf"${column}$ [GeV]", ylabel="Eventos")
        if log_y:
            ax.set_yscale("log")
    fig.suptitle("Distribuciones de energía de los candidatos", fontweight = 'bold')
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig, axes

def plot_mass(
    df,
    *,
    bins: int = 20,
    mass_range: tuple[float, float] | None = None,
    save: bool = False,
    filename: str = "b2hhh_mass",
    data: str = 'sim' or 'data',
    output_dir = None,
    show: bool = True
):
    require_columns(df, ("B_M", "B_Charge"), "plot_mass")
    # B+, B-, B± en gráfica de 1,3
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), squeeze=False)
    panels = ((1, r"$B^{+}$"), (-1, r"$B^{-}$"), (0, r"$B^{\pm}$"))
    for ax, (charge, label) in zip(axes[0], panels):
        subset = df if charge == 0 else df.loc[df["B_Charge"] == charge]
        draw_histogram(ax, finite_values(subset, "B_M", scale=1e-3), bins=bins, value_range=mass_range)
        style_axis(fig, ax, title=f"Masa invariante del B ({label})", xlabel=r"$M_{B}$ [$GeV/c^{2}$]", ylabel="Eventos/bin")
    fig.suptitle("Distribuciones de masa invariante del B", fontweight='bold')
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig, axes

def plot_mass_fit(ax, result: dict, *, label: str, mass_min: float, mass_max: float):
    from ..fitting.b2hhh import evaluate_fit_components
    x_dense = np.linspace(mass_min, mass_max, 400)
    components = evaluate_fit_components(x_dense, result)
    ax.step(result['centers'], result['counts'], where='mid', color=PlotStyle.text, label='Datos', linewidth=1)
    ax.plot(x_dense, components['total'], color=PlotStyle.signal, linewidth='1.5', label='Ajuste total')
    ax.plot(x_dense, components['signal'], color=PlotStyle.prompt, linestyle='--', linewidth=1.2, label='Señal')
    ax.plot(x_dense, components['background'], color=PlotStyle.background_model, linestyle=':', linewidth=1.2, label='Fondo')
    if "mu" in result["parameters"]:
        mu = result['parameters']['mu']
        ax.axvline(mu, color=PlotStyle.text, linestyle='--', linewidth=1, alpha=0.8, label=f"μ = {mu:.1f} [MeV]")
    style_axis(ax.figure, ax, title=label, xlabel=r"$M_{B}$ [$MeV/c^{2}$]", ylabel="Eventos/bin")
    ax.set_xlim(mass_min, mass_max)
    ax.legend(fontsize=8)

def plot_prob_particle(
        df,
        *,
        bins: int = 50,
        save: bool = False,
        filename: str = "b2hhh_prob_particle",
        data: str = 'sim' or 'data',
        output_dir = None,
        show: bool = True
):
    columns = [f"{h}_Prob{p}" for h in _HADRONS for p in ("K", "Pi")]
    require_columns(df, columns, "plot_prob_particle")
    fig, axes = plt.subplots(3, 2, figsize=(10, 14), squeeze=False)
    for row, hadron in enumerate(_HADRONS):
        for col, particle in enumerate(('K', 'Pi')): # (['K', 'Pi'])
            ax = axes[row, col]
            column = f"{hadron}_Prob{particle}"
            draw_histogram(ax, finite_values(df, column), bins=bins, value_range=(0.0, 1.0))
            style_axis(fig, ax, title=f"Probabilidad de ser {particle} - {hadron}", xlabel=f"Probabilidad de ser {particle}", ylabel="Eventos")
    fig.suptitle("Distribuciones de probabilidad de identificación", fontweight = 'bold')
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig, axes

def plot_probability_distributions(
    df, 
    *,
    bins: int = 50,
    gamma: float = 0.4,
    save: bool = False,
    filename: str = "b2hhh_prob2d",
    data: str = 'sim' or 'data',
    output_dir = None,
    show: bool = True
):
    columns = [f"{h}_Prob{p}" for h in _HADRONS for p in ('K', 'Pi')]
    require_columns(df, columns, "plot_probability_distributions")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), squeeze=False)
    for ax, hadron in zip(axes[0], _HADRONS):
        image = ax.hist2d(
            df[f'H{hadron}_ProbK'],
            df[f'H{hadron}_ProbPi'],
            bins = bins,
            range = [[0.0, 1.0], [0.0, 1.0]],
            cmap = 'coolwarm',
            norm = mcolors.PowerNorm(gamma = gamma)
        )
        style_axis(fig, ax, title=f"{hadron}: Probabilidad K vs. π", xlabel="Prob. de ser kaón", ylabel="Prob. de ser pión")
        PlotStyle.add_dark_colorbar(fig, ax, image, label='Frecuencia')
    fig.suptitle('Distribuciones de probabilidad (2D)', fontweight = 'bold')
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig, axes

def plot_square_mass_dist(
    df,
    *,
    bins: int = 40,
    log_y: bool = True,
    save: bool = False,
    filename: str = "b2hhh_msquare",
    data: str = 'sim' or 'data',
    output_dir = None,
    show: bool = True
):
    require_columns(df, ("m2_12", "m2_13"), "plot_square_mass_dist")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), squeeze=False)
    for ax, (column, label) in zip(axes[0], (("m2_12", r"m^{2}_{12}"), ("m2_13", r"m^{2}_{13}"))):
        draw_histogram(ax, finite_values(df, column, scale=1e-6), bins=bins)
        style_axis(fig, ax, title=f"Masa invariante al cuadrado ({label})", xlabel=rf"${label}$ [$GeV^{2}/c^{4}$]", ylabel="Eventos")
        if log_y:
            ax.set_yscale("log")
    fig.suptitle('Distribuciones de masa cuadrática', fontweight = 'bold')
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig, axes

def plot_binned_dalitz(
    df,
    *,
    bins: int = 25,
    range_x: tuple[float, float] = (0.75, 2.0),
    range_y: tuple[float, float] = (0.0, 30.0),
    save: bool = False,
    filename: str = "b2hhh_binned_dalitz",
    data: str = 'sim' or 'data',
    output_dir = None,
    show: bool = True
):
    require_columns(df, ("R0low", "R0high"), "plot_binned_dalitz")
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(PlotStyle.background)
    _, xb, yb, image = ax.hist2d(
        df['R0low']/1e6,
        df['R0high']/1e6,
        bins = bins,
        range = [range_x, range_y],
        cmap = cmap,
        norm = mcolors.PowerNorm(gamma = 0.5)
    )
    style_axis(fig, ax, title="Diagramma de Dalitz agrupado (binned)", xlabel=r"$R^{0}_{\mathrm{low}}$ [$GeV^2/c^4$]", ylabel=r"$R^{0}_{\mathrm{high}}$ [$GeV^2/c^4$]")
    PlotStyle.add_dark_colorbar(fig, ax, image, label='Eventos por bin')
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig, ax

def plot_proyection_mass(
    df,
    *,
    bins: int = 40,
    # value_range: tuple[float, float] = (800, 3500) or 'auto',
    value_range: tuple[float, float] | None = None,
    save: bool = False,
    filename: str = "b2hhh_projection_mass",
    data: str = 'sim' or 'data',
    output_dir = None,
    show: bool = True
):
    require_columns(dr, ("R0low", "R0high"), "plot_proyection_mass")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), squeeze=False)
    for ax, (column, label) in zip(axes[0], (("R0low", r"R^{0}_{low}"), ("R0high", r"R^{0}_{high}"))):
        values = np.sqrt(finite_values(df, column, scale=1e-6))
        draw_histogram(ax, values, bins=bins, value_range=value_range)
        style_axis(fig, ax, title=f"Proyeccion de la masa {label}", xlabel=rf"$\sqrt{{{column}}}$ [$GeV/c^2$]", ylabel="Eventos")
    fig.suptitle('Proyecciones de la masa invariante del B', fontweight = 'bold')
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig, axes

def plot_dalitz_scatter(
    df,
    *,
    s: float = 0.13,
    color: str = 'red',
    alpha: float = 0.45,
    save: bool = False,
    filename: str = "b2hhh_dalitz_scatter",
    data: str = 'sim' or 'data',
    output_dir = None,
    show: bool = True
):
    require_columns(df, ("m2_12", "m2_13", "R0low", "R0high"), "plot_dalitz_scatter")
    fig, axes = plt.subplots(1, 2, figsize = (14, 7), squeeze=False)
    panels = (
        (axes[0, 0], "m2_12", "m2_13", "Diagrama de Dalitz", r"$m^2_{12}$", r"$m^2_{13}$"),
        (axes[0, 1], "R0low", "R0high", "Diagrama de Dalitz ordenado", r"$R^{0}_{\mathrm{low}}$", r"$R^{0}_{\mathrm{high}}$")
    )
    for ax, xcol, ycol, title, xlabel, ylabel in panels:
        ax.scatter(df[xcol]/1e6, df[ycol]/1e6, s=s, color=color, alpha=alpha, rasterized=True)
        style_axis(fig, ax, title=title, xlabel=rf"{xlabel} [$GeV^2/c^4$]", ylabel=rf"{ylabel} [$GeV^2/c^4$]")
    fig.suptitle(f'Diagramas de Dalitz (Scatter)', fontweight='bold')
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig, axes

def plot_dalitz_sumary(
    dalitz: dict,
    *,
    save: bool = False,
    filename: str = "b2hhh_dalitz_summary",
    data: str = 'sim' or 'data',
    output_dir = None,
    show: bool = True
):
    hBp, hBm = dalitz["hBp"], dalitz["hBm"]
    A_map, sA_map, S_map = dalitz["A_map"], dalitz["sA_map"], dalitz["S_map"]
    xb, yb = dalitz["xb"], dalitz["yb"]
    fig = plt.figure(figsize=(25,8))
    gs = gridspec.GridSpec(1, 2, figure = fig, width_ratios = [1.4, 1.45], wspace = 0.3)
    left_gs = gs[0].subgridspec(2, 2, wspace = 0.45, hspace = 0.35)
    right_gs = gs[1].subgridspec(1, 1)
    ax_bp, ax_bm = fig.add_subplot(left_gs[0, 0]), fig.add_subplot(left_gs[0, 1])
    ax_A, ax_sA = fig.add_subplot(left_gs[1, 0]), fig.add_subplot(left_gs[1, 1])
    ax_S = fig.add_subplot(right_gs[0, 0])
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(PlotStyle.background)
    extend = [xb[0], xb[-1], yb[0], yb[-1]]
    panels = (
        (ax_bp, np.ma.masked_where(hBp.T == 0, hBp.T), r'Dalitz $B^{+}$', 'Eventos por bin', None, None),
        (ax_bm, np.ma.masked_where(hBm.T == 0, hBm.T), r'Dalitz $B^{-}$', 'Eventos por bin', None, None),
        (ax_A, np.ma.masked_invalid(A_map.T), 'Asimetría local', r'$A_{CP}^{local}$', -1, 1),
        (ax_bm, np.ma.masked_invalid(sA_map.T), 'Incertidumbre local', r'$\sigma(A_{CP}^{local})$', None, None),
        (ax_bm, np.ma.masked_where(hBm.T == 0, hBm.T), 'Significancia local', r'$A/\sigma_{A}$', -5, 5),
    )
    for ax, values, title, label, vmin, vmax in panels:
        image = ax.imshow(values, extent=extend, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        style_axis(fig, ax, title=title, xlabel=r"$m^2(KK)_{\mathrm{low}}$ [$GeV^2/c^4$]", ylabel=r"$m^2(KK)_{\mathrm{high}}$ [$GeV^2/c^4$]")
        PlotStyle.add_dark_colorbar(fig, ax, image, label=label)
    fig.suptitle(f"Dalitz ordenado (charm_veto = {dalitz.get('charm_veto')})", fontweight="bold")
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig

def plot_large_CP(
    df,
    *,
    bins: tuple[int, int] = (50, 10),
    save: bool = False,
    filename: str = "b2hhh_large_cp",
    data: str = 'sim' or 'data',
    outuput_dir = None,
    show: bool = True
):
    require_columns(df, ("B_M", "B_Charge"), "plot_large_CP")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), squeeze=False)
    ax_all, ax_split = axes[0]
    draw_histogram(ax_all, finite_values(df, "B_M"), bins=bins[0])
    style_axis(fig, ax_all, title="Masa Invariante del B", xlabel=r"$M_{B}$ [$MeV/c^2$]", ylabel="Eventos")
    ax_split.hist(
        [df.loc[df["B_Charge"] == 1, "B_M"].dropna(), df.loc[df["B_Charge"] == -1, "B_M"].dropna()],
        bins=bins[1],
        histtype="step",
        stacked=True,
        fill=False,
        linewidth=1.3,
        label=[r"$B^{+}$", r"$B^{-}$"]
    )
    ax_split.legend()
    style_axis(fig, ax_split, title="Comparación $B^{+}$ vs $B^{-}$", xlabel=r"$M_{B}$ [$MeV/c^{2}$]", ylabel="Eventos")
    fig.suptitle("Violación CP en la región de gran asimetría del Dalitz", fotweight="bold")
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=outuput_dir, show=show)
    return fig, axes