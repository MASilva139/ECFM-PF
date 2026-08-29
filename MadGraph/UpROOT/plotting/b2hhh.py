from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from ..config import (
    DARK_BACKGROUND,
    LIGHT_TEXT,
    MASS_MIN,
    MASS_MAX
)
from ..fitting.b2hhh import evaluate_fit_components
from ..styles import PlotStyle

def plot_momentum(
        df,
        channel_name: str,
        safe_name: str | None = None,
        bins: int = 20,
        m_range: tuple = (float, float),
        save: bool = True,
        data: str = 'sim' or 'data'
):
    momentum_comps = ['PX', 'PY', 'PZ', 'P']
    fig, axes = plt.subplots(3, 4, figsize=(20, 11), squeeze=False)
    fig.suptitle(
        f'{channel_name} - Distribuciones de momentum',
        fontsize = 17,
        fontweight = 'bold',
        color = LIGHT_TEXT
    )
    fig.patch.set_facecolor(DARK_BACKGROUND)
    for row_idx, hadron_id in enumerate([1, 2, 3]):
        h = f'H{hadron_id}'
        for col_idx, comp in enumerate(momentum_comps):
            ax = axes[row_idx, col_idx]
            column_name = f'{h}_{comp}'
            if column_name not in df.columns:
                raise KeyError(f'No existe la columna {column_name} en el DataFrame')
            values = df[column_name].dropna()
            ax.hist(
                values/1000,
                bins = bins,
                range = m_range,
                histtype = 'stepfilled',
                alpha = 0.75
            )
            PlotStyle.apply_dark_axes_style(
                fig, 
                ax, 
                title = column_name,
                xlabel = rf'${column_name}$ [GeV/$c^{2}$]',
                ylabel = 'Eventos'
            )
            ax.set_yscale('log')
            ax.grid(alpha = 0.18, linestyle = '--')
    fig.tight_layout()
    if save:
        if safe_name is None:
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
            if data == 'sim':
                PlotStyle.save_fig_sim(fig, f'{safe_name}_momentum')
            elif data == 'data':
                PlotStyle.save_fig(fig, f'{safe_name}_momentum')
            else:
                raise ValueError(f"Valor de 'sim' no válido: {data}. Debe ser 'sim' o 'data'.")
    plt.show()
    return fig, axes

def plot_energy(
    df,
    channel_name: str,
    safe_name: str | None = None,
    bins: int = 20,
    save: bool = True,
    data: str = 'sim' or 'data'
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), squeeze=False)
    fig.suptitle(
        f'{channel_name} - Distribuciones de energia de los candidatos a kaon',
        fontsize = 17,
        fontweight = 'bold',
        color = LIGHT_TEXT
    )
    fig.patch.set_facecolor(DARK_BACKGROUND)
    hadrons = ['H1', 'H2', 'H3']
    for i, hadron in enumerate(hadrons):
        ax = axes[0, i]
        h = f'{hadron}_E'
        values = df[h].dropna()/1000
        ax.hist(
            values,
            bins = bins,
            range = (values.min(), values.max()),
            histtype = 'stepfilled',
            alpha = 0.75
        )
        PlotStyle.apply_dark_axes_style(
            fig, ax,
            title = f'Energia del candidato {h}',
            xlabel = rf'${h}$ [GeV]',
            ylabel= 'Eventos'
        )
        ax.set_yscale('log')
        ax.grid(alpha = 0.18, linestyle = '--')
    fig.tight_layout()
    if save:
        if safe_name is None:
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
            if data == 'sim':
                PlotStyle.save_fig_sim(fig, f'{safe_name}_energy')
            elif data == 'data':
                PlotStyle.save_fig(fig, f'{safe_name}_energy')
            else:
                raise ValueError(f"Valor de 'sim' no válido: {data}. Debe ser 'sim' o 'data'.")
    plt.show()
    return fig, axes

def plot_mass(
    df,
    channel_name: str,
    safe_name: str | None = None,
    bins: int = 20,
    g_range: tuple | None = None,
    save: bool = True,
    data: str = 'sim' or 'data'
):
    # B+, B-, B± en gráfica de 1,3
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), squeeze=False)
    fig.suptitle(
        f'{channel_name} - Distribuciones de masa invariante del B',
        fontsize = 17,
        fontweight = 'bold',
        color = LIGHT_TEXT
    )
    fig.patch.set_facecolor(DARK_BACKGROUND)
    for i, (charge, label) in enumerate([(1, r'$B^{+}$'), (-1, r'$B^{-}$'), (0, r'$B^{\pm}$')]):
        ax = axes[0, i]
        if charge == 0:
            masses_arr = df['B_M'].dropna()
        else:
            masses_arr = df.loc[df['B_Charge'] == charge, 'B_M'].dropna()
        if g_range is None:
            ax.hist(
                masses_arr/1000,
                bins = bins,
                range = (MASS_MIN/1000, masses_arr.max()/1000),
                histtype = 'stepfilled',
                alpha = 0.75
            )
        else:
            ax.hist(
                masses_arr/1000,
                bins = bins,
                range = g_range,
                histtype = 'stepfilled',
                alpha = 0.75
            )
        PlotStyle.apply_dark_axes_style(
            fig,
            ax,
            title = f'Masa invariante del B ({label})',
            xlabel = r'$M_{B}$ [GeV/$c^{2}$]',
            ylabel= 'Eventos/bin'
        )
        # ax.set_yscale('log')
        ax.grid(alpha = 0.18, linestyle = '--')
    fig.tight_layout()
    if save:
        if safe_name is None:
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
            if data == 'sim':
                PlotStyle.save_fig_sim(fig, f'{safe_name}_mass_invariant')
            elif data == 'data':
                PlotStyle.save_fig(fig, f'{safe_name}_mass_invariant')
            else:
                raise ValueError(f"Valor de 'sim' no válido: {data}. Debe ser 'sim' o 'data'.")
    plt.show()
    return fig, axes

def plot_prob_particle(
        df,
        channel_name: str,
        safe_name: str | None = None,
        bins: int = 50,
        save: bool = True,
        data: str = 'sim' or 'data'
):
    fig, axes = plt.subplots(3, 2, figsize=(10, 14), squeeze=False)
    fig.suptitle(
        f'{channel_name} - Distribuciones de probabilidad de ser kaon o pion',
        fontsize = 17,
        fontweight = 'bold',
        color = LIGHT_TEXT
    )
    fig.patch.set_facecolor(DARK_BACKGROUND)
    for row_idx, hadron_id in enumerate([1, 2, 3]):
        h = f'H{hadron_id}'
        for col_idx, particle in enumerate(['K', 'Pi']):
            ax = axes[row_idx, col_idx]
            column_name = f'{h}_Prob{particle}'
            values = df[column_name].dropna()
            ax.hist(
                values,
                bins = bins,
                range = (0, 1),
                histtype = 'stepfilled',
                alpha = 0.75
            )
            PlotStyle.apply_dark_axes_style(
                fig,
                ax,
                title=f'Probabilidad de ser {particle} - candidato: {h}',
                xlabel=rf'Probabilidad de ser {particle}',
                ylabel='Eventos'
            )
            ax.grid(alpha=0.18, linestyle='--')
    fig.tight_layout()
    if save:
        if safe_name is None:
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
            if data == 'sim':
                PlotStyle.save_fig_sim(fig, f'{safe_name}_prob_particle')
            elif data == 'data':
                PlotStyle.save_fig(fig, f'{safe_name}_prob_particle')
            else:
                raise ValueError(f"Valor de 'sim' no válido: {data}. Debe ser 'sim' o 'data'.")
    plt.show()
    return fig, axes

def plot_mass_fit(ax, result:dict, label:str):
    x_dense = np.linspace(MASS_MIN, MASS_MAX)
    components = evaluate_fit_components(x_dense, result)
    ax.step(result['centers'], result['counts'], where='mid', color='steelblue', label='Datos', linewidth=1)
    ax.plot(x_dense, components['total'], color='red', linewidth='1.5', label='Ajuste total')
    ax.plot(x_dense, components['signal'], color='lime', linestyle='--', linewidth=1.2, label='Señal')
    ax.plot(x_dense, components['background'], color='orange', linestyle=':', linewidth=1.2, label='Fondo')
    ax.axvline(result['parameters']['mu'], color='white', linestyle='--', linewidth=1, alpha=0.8, label=f"μ = {result['parameters']['mu']:.1f} [MeV]")
    PlotStyle.apply_dark_axes_style(
        ax.figure,
        ax, 
        label,
        r'$M_{B}$ [MeV/$c^{2}$]',
        'Eventos/bin'
    )
    ax.set_xlim(MASS_MIN, MASS_MAX)
    ax.legend(fontsize=8, facecolor=DARK_BACKGROUND, edgecolor=LIGHT_TEXT, labelcolor=LIGHT_TEXT)

def plot_probability_distributions(
    df, 
    channel_name: str,
    safe_name: str | None = None,
    bins: int = 50,
    gamma: float = 0.4,
    save: bool = True,
    data: str = 'sim' or 'data'
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
    fig.patch.set_facecolor(DARK_BACKGROUND)
    for hadron_id, ax in zip([1, 2, 3], axes):
        ax.set_facecolor(DARK_BACKGROUND)
        _, _, _, image = ax.hist2d(
            df[f'H{hadron_id}_ProbK'],
            df[f'H{hadron_id}_ProbPi'],
            bins = bins,
            range = [[0.5, 1.0], [0.0, 0.5]],
            cmap = 'coolwarm',
            norm = mcolors.PowerNorm(gamma = gamma)
        )
        ax.grid(color=LIGHT_TEXT, linestyle='--', alpha=0.12)
        PlotStyle.apply_dark_axes_style(
            fig, ax,
            rf'H{hadron_id}: Probabilidad $K$ vs. $\pi$',
            "Probabilidad de ser kaón",
            "Probabilidad de ser pión"
        )
        PlotStyle.add_dark_colorbar(fig, ax, image, label='Frecuencia')
    fig.suptitle(
        'Distribución de probabilidades',
        color = LIGHT_TEXT,
        fontsize = 18,
        fontweight = 'bold'
    )
    fig.tight_layout()
    if save:
        if safe_name is None:
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
            if data == 'sim':
                PlotStyle.save_fig_sim(fig, f'{safe_name}_dist_prob')
            elif data == 'data':
                PlotStyle.save_fig(fig, f'{safe_name}_dist_prob')
            else:
                raise ValueError(f"Valor de 'sim' no válido: {data}. Debe ser 'sim' o 'data'.")
    plt.show()

def plot_square_mass_dist(
    df,
    channel_name: str,
    safe_name: str | None = None,
    bins: int = 40,
    save: bool = True,
    data: str = 'sim' or 'data'
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), squeeze=False)
    fig.suptitle(
        f'{channel_name} - Distribuciones de masas cuadradas',
        fontsize = 17,
        fontweight = 'bold',
        color = LIGHT_TEXT
    )
    fig.patch.set_facecolor(DARK_BACKGROUND)
    for i, (col, label) in enumerate([('m2_12', rf'm^2_{12}'), ('m2_13', r'm^{2}_{13}')]):
        ax = axes[0, i]
        val = df[col].dropna()/1e6
        ax.hist(
            val,
            bins = bins,
            range = (val.min(), val.max()),
            histtype = 'stepfilled',
            alpha = 0.77
        )
        PlotStyle.apply_dark_axes_style(
            fig,
            ax,
            title = rf'Masa invariante cuadratica {label}',
            xlabel = rf'${label}$ [GeV$^2/c^4$]',
            ylabel = rf'Eventos'
        )
        ax.set_yscale('log')
        ax.grid(alpha=0.25, linestyle='--')
    fig.tight_layout()
    if save:
        if safe_name is None:
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
            if data == 'sim':
                PlotStyle.save_fig_sim(fig, f'{safe_name}_msquare')
            elif data == 'data':
                PlotStyle.save_fig(fig, f'{safe_name}_msquare')
            else:
                raise ValueError(f"Valor de 'sim' no válido: {data}. Debe ser 'sim' o 'data'.")
    plt.show()
    return fig, axes

def plot_binned_dalitz(
    df,
    channel_name: str,
    safe_name: str | None = None,
    bins: int = 25,
    g_range: tuple = [[0.75, 2.0], [0.0, 30.0]],
    save: bool = True,
    data: str = 'sim' or 'data'
):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(DARK_BACKGROUND)
    dalitz_cmap = plt.get_cmap('RdBu_r').copy()
    dalitz_cmap.set_bad(DARK_BACKGROUND)
    histogram_2d, xb, yb, image = ax.hist2d(
        df['R0low']/1e6,
        df['R0high']/1e6,
        bins = bins,
        range = g_range,
        cmap = dalitz_cmap,
        norm = mcolors.PowerNorm(gamma = 0.5)
    )
    PlotStyle.apply_dark_axes_style(
        fig, ax,
        title=f'[{channel_name}] - Diagrama de Dalitz agrupado (binned)',
        xlabel=r'$R^{0}_{\mathrm{Low}}\,[GeV^{2}/c^{4}]$',
        ylabel=r'$R^{0}_{\mathrm{High}}\,[GeV^{2}/c^{4}]$'
    )
    PlotStyle.add_dark_colorbar(fig, ax, image, label='Eventos por bin')
    ax.grid(alpha=0.25, linestyle='--')
    fig.tight_layout()
    if save:
        if safe_name is None:
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
            if data == 'sim':
                PlotStyle.save_fig_sim(fig, f'{safe_name}_binned_dalitz')
            elif data == 'data':
                PlotStyle.save_fig(fig, f'{safe_name}_binned_dalitz')
            else:
                raise ValueError(f"Valor de 'sim' no válido: {data}. Debe ser 'sim' o 'data'.")
    plt.show()

def plot_proyection_mass(
    df, 
    channel_name: str,
    safe_name: str | None = None,
    bins: int = 40,
    g_range: tuple[float, float] = (800, 3500) or 'auto',
    save: bool = True,
    data: str = 'sim' or 'data'
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), squeeze=False)
    fig.patch.set_facecolor(DARK_BACKGROUND)
    fig.suptitle(
        f'[{channel_name}] - Proyecciones de la masa invariante del B',
        fontsize = 17,
        fontweight = 'bold',
        color = LIGHT_TEXT
    )
    low = ('R0low', r'menor')
    high = ('R0high', r'mayor')
    for i, (col, label) in enumerate([low, high]):
        ax = axes[0, i]
        val = df[col].dropna()/1e6
        if g_range == 'auto':
            ax.hist(
                np.sqrt(val),
                bins = bins,
                range = (np.sqrt(val).min(), np.sqrt(val).max()),
                histtype = 'stepfilled',
                alpha = 0.77
            )
        else:
            ax.hist(
                np.sqrt(val),
                bins = bins,
                range = g_range,
                histtype = 'stepfilled',
                alpha = 0.77
            )
        PlotStyle.apply_dark_axes_style(
            fig, ax,
            title = rf'Proyeccion de la masa {label}',
            xlabel = rf'${label}$ [GeV/$c^{2}$]',
            ylabel = rf'Eventos'
        )
        ax.grid(alpha=0.25, linestyle='--')
    fig.tight_layout()
    if save:
        if safe_name is None:
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
            if data == 'sim':
                PlotStyle.save_fig_sim(fig, f'{safe_name}_proyection_mass')
            elif data == 'data':
                PlotStyle.save_fig(fig, f'{safe_name}_proyection_mass')
            else:
                raise ValueError(f"Valor de 'sim' no válido: {data}. Debe ser 'sim' o 'data'.")
    plt.show()

def plot_dalitz_sumary(
    hBp, hBm, A_map, sA_map, S_map, xb, yb, 
    channel_name: str, 
    charm_veto, 
    safe_name: str | None = None,
    save: bool = True,
    data: str = 'sim' or 'data'
):
    fig = plt.figure(figsize=(25,8))
    fig.patch.set_facecolor(DARK_BACKGROUND)
    gs = gridspec.GridSpec(
        1, 2,
        figure = fig,
        width_ratios = [1.4, 1.45],
        wspace = 0.3
    )
    left_gs = gs[0].subgridspec(2, 2, wspace = 0.45, hspace = 0.35)
    right_gs = gs[1].subgridspec(1, 1)
    ax_bp = fig.add_subplot(left_gs[0, 0])
    ax_bm = fig.add_subplot(left_gs[0, 1])
    ax_A = fig.add_subplot(left_gs[1, 0])
    ax_sA = fig.add_subplot(left_gs[1, 1])
    ax_S = fig.add_subplot(right_gs[0, 0])
    fig.suptitle(
        f'{channel_name} - Dalitz ordenado (charm veto = {charm_veto})',
        fontsize = 17,
        fontweight = 'bold',
        color = LIGHT_TEXT
    )
    dalitz_cmap = plt.get_cmap('RdBu_r').copy()
    dalitz_cmap.set_bad(DARK_BACKGROUND)
    asym_cmap = plt.get_cmap('RdBu_r').copy()
    asym_cmap.set_bad(DARK_BACKGROUND)
    # unc_cmap = plt.get_cmap('viridis').copy()
    unc_cmap = plt.get_cmap('RdBu_r').copy()
    unc_cmap.set_bad(DARK_BACKGROUND)
    sig_cmap = plt.get_cmap('RdBu_r').copy()
    sig_cmap.set_bad(DARK_BACKGROUND)
    ext = [xb[0], xb[-1], yb[0], yb[-1]]
    panels = [
        {
            'ax': ax_bp,
            'values': np.ma.masked_where(hBp.T == 0, hBp.T),
            'title': r'Dalitz $B^{+}$',
            'cmap': dalitz_cmap,
            'label': 'Eventos por bin',
            'vmin': None,
            'vmax': None
        },
        {
            'ax': ax_bm,
            'values': np.ma.masked_where(hBm.T == 0, hBm.T),
            'title': r'Dalitz $B^{-}$',
            'cmap': dalitz_cmap,
            'label': 'Eventos por bin',
            'vmin': None,
            'vmax': None
        },
        {
            'ax': ax_A,
            'values': np.ma.masked_invalid(A_map.T),
            'title': r'Asimetría local',
            'cmap': asym_cmap,
            'label': r'$A_{CP}^{local}$',
            'vmin': -1,
            'vmax': 1
        },
        {
            'ax': ax_sA,
            'values': np.ma.masked_invalid(sA_map.T),
            'title': r'Incertidumbre local',
            'cmap': unc_cmap,
            'label': r'$\sigma(A_{CP}^{local})$',
            'vmin': None,
            'vmax': None
        },
        {
            'ax': ax_S,
            'values': np.ma.masked_invalid(S_map.T),
            'title': r'Significancia de la asimetría local',
            'cmap': sig_cmap,
            'label': r'$A/\sigma_{A}$',
            'vmin': -5,
            'vmax': 5
        }
    ]
    for panel in panels:
        ax = panel['ax']
        image = ax.imshow(
            panel['values'],
            extent = ext,
            origin = 'lower',
            aspect = 'auto',
            cmap = panel['cmap'],
            vmin = panel['vmin'],
            vmax = panel['vmax'],
            interpolation = 'nearest'
        )
        PlotStyle.apply_dark_axes_style(
            fig, ax, panel['title'],
            r'$m^{2}(KK)_{\mathrm{Low}}\,[GeV^{2}/c^{4}]$',
            r'$m^{2}(KK)_{\mathrm{High}}\,[GeV^{2}/c^{4}]$'
        )
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
        PlotStyle.add_dark_colorbar(fig, ax, image, label = panel['label'])
    fig.tight_layout()
    if save:
        if safe_name is None:
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
            if data == 'sim':
                PlotStyle.save_fig_sim(fig, f'{safe_name}_dalitz_completo')
            elif data == 'data':
                PlotStyle.save_fig(fig, f'{safe_name}_dalitz_completo')
            else:
                raise ValueError(f"Valor de 'sim' no válido: {data}. Debe ser 'sim' o 'data'.")
    plt.show()

def plot_dalitz_scatter(
    df, 
    channel_name: str,
    safe_name: str | None = None,
    s: float = 0.13,
    color: str = 'red',
    alpha: float = 0.45,
    rasterized: bool = True,
    save: bool = True,
    data: str = 'sim' or 'data'
):
    fig, axes = plt.subplots(1, 2, figsize = (14, 7))
    fig.suptitle(f'[{channel_name}] -- Diagramas de Dalitz (Scatter)', fontsize=17, fontweight='bold')
    panels = [
        {
            'ax': axes[0],
            'title': 'Diagrama de Dalitz',
            'xlabel': r'$m_{12}^{2}$ [GeV$^2/c^4$]',
            'ylabel': r'$m_{13}^{2}$ [GeV$^2/c^4$]',
            'x': df['m2_12']/1e6,
            'y': df['m2_13']/1e6,
        }, 
        {
            'ax': axes[1],
            'title': 'Diagrama de Dalitz ordenado',
            'xlabel': r'$R_{\mathrm{Low}}^{0}$ [GeV$^2/c^4$]',
            'ylabel': r'$R_{\mathrm{High}}^{0}$ [GeV$^2/c^4$]',
            'x': df['R0low']/1e6,
            'y': df['R0high']/1e6
        }
    ]
    for panel in panels:
        ax = panel['ax']
        ax.scatter(
            panel['x'],
            panel['y'],
            s = s,
            color = color,
            alpha = alpha,
            rasterized = rasterized
        )
        ax.set_title(panel['title'])
        ax.set_xlabel(panel['xlabel'])
        ax.set_ylabel(panel['ylabel'])
        ax.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    if save:
        if safe_name is None:       
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
            if data == 'sim':
                PlotStyle.save_fig_sim(fig, f'{safe_name}_dalitz_scatter')
            elif data == 'data':
                PlotStyle.save_fig(fig, f'{safe_name}_dalitz_scatter')
            else:
                raise ValueError(f"Valor de 'sim' no válido: {data}. Debe ser 'sim' o 'data'.")
    plt.show()
    return fig, axes

def plot_large_CP(
    df,
    channel_name: str,
    safe_name: str | None = None,
    bins: tuple[int, int] = (50, 10),
    save: bool = True,
    data: str = 'sim' or 'data'
):
    f1_data = df['B_M']
    f2_data = [
        df.query('B_Charge == 1')['B_M'],
        df.query('B_Charge == -1')['B_M']
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), squeeze=False)
    fig.suptitle(
        f'[{channel_name}] - Violacion CP en una region del Dalitz con asimetria considerable',
        fontsize = 17,
        fontweight = 'bold',
        color = LIGHT_TEXT
    )
    fig.patch.set_facecolor(DARK_BACKGROUND)
    panels = [
        {
            'ax': axes[0,0],
            'title': 'Masa invariante del B',
            'values': f1_data,
            'bins': bins[0],
            'xlabel': r'$M_{B}$ [MeV/$c^{2}$]',
            'ylabel': 'Eventos',
            'histtype': 'stepfilled',
            'stacked': None,
            'fill': None,
            'linewidth': 1.2,
            'label': None
        }, 
        {
            'ax': axes[0,1],
            'title': r'Comparacion de masa: $B^{+}$ vs $B^{-}$',
            'values': f2_data,
            'bins': bins[1],
            'xlabel': r'$M_{B}$ [MeV/$c^{2}$]',
            'ylabel': 'Eventos',
            'histtype': 'step',
            'stacked': True,
            'fill': False,
            'linewidth': 1.3,
            'label': [r'$B^{+}$', r'$B^{-}$']
        }
    ]
    for panel in panels:
        ax = panel['ax']
        if panel['stacked'] is not None:
            ax.hist(
                panel['values'],
                bins = panel['bins'],
                histtype = panel['histtype'],
                stacked = panel['stacked'],
                fill = panel['fill'],
                linewidth = panel['linewidth'],
                label = panel['label']
            )
            ax.legend(fontsize=18)
        else:
            ax.hist(
                panel['values'],
                bins = panel['bins'],
                histtype = panel['histtype'],
                linewidth = panel['linewidth']
            )
        PlotStyle.apply_dark_axes_style(
            fig, ax, panel['title'], panel['xlabel'], panel['ylabel']
        )
        ax.grid(alpha=0.25, linestyle='--')
    fig.tight_layout()
    if save:
        if safe_name is None:
            safe_name = (
                channel_name
                .replace('→', 'to')     # U+2192: →
                .replace(' ', '_')
                .replace('±', "pm")        # U+00B1: ±
                .replace('+', 'plus')
                .replace('-', 'minus')  
            )
            if data == 'sim':
                PlotStyle.save_fig_sim(fig, f'{safe_name}_large_CP')
            elif data == 'data':
                PlotStyle.save_fig(fig, f'{safe_name}_large_CP')
            else:
                raise ValueError(f"Valor de 'sim' no válido: {data}. Debe ser 'sim' o 'data'.")
    plt.show()