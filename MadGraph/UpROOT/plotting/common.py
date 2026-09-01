from collections.abc import Iterable
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..styles import PlotStyle

def require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    context: str
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise KeyError(f"{context}: faltan columnas: {', '.join(missing)}.")

def first_available_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    context: str
) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    formatted = ', '.join(candidates)
    raise KeyError(f"{context}: ninguna columna disponible entre: {formatted}.")

def finite_values(
    df: pd.DataFrame,
    column: str, 
    *,
    scale: float = 1.0
) -> np.ndarray:
    require_columns(df, (column,), "finite_values")
    values = df[column].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return values*float(scale)

def positive_values(
    df: pd.DataFrame,
    column: str,
    *,
    scale: float = 1.0
) -> np.ndarray:
    values = finite_values(df, column, scale=scale)
    return values[values > 0]

def draw_histogram(
    ax,
    values,
    *,
    bins: int | np.ndarray = 60,
    value_range: tuple[float, float] | None = None,
    label: str | None = None,
    color: str | None = None,
    density: bool = False,
    filled: bool = True,
    alpha: float = 0.72,
    linewidth: float = 1.7
):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return ax.hist(
        values,
        bins = bins,
        range = value_range,
        density = density,
        histtype = "stepfilled" if filled else "step",
        alpha = alpha if filled else 1.0,
        linewidth = linewidth,
        label = label,
        color = color
    )

def style_axis(
    fig,
    ax,
    *,
    title: str,
    xlabel: str, 
    ylabel: str, 
    grid: bool = True
) -> None:
    PlotStyle.apply_dark_axes_style(fig, ax, title=title, xlabel=xlabel, ylabel=ylabel)
    if grid:
        ax.grid(alpha=0.18, linestyle="--")

def finalize_figure(
    fig,
    *,
    save: bool = False,
    filename: str | None = None,
    data: str = 'delphes',
    output_dir: str | Path | None = None,
    dpi: int = 500,
    show: bool = True
) -> Path | None:
    return PlotStyle.finalize(
        fig, 
        save=save,
        filename=filename,
        data=data,
        output_dir=output_dir,
        dpi=dpi,
        show=show
    )

def plot_fit_result(
    result: dict, 
    *,
    title: str, 
    xlabel: str,
    ylabel: str = "Candidatos/bin",
    save: bool = False,
    filename: str | None = None,
    data: str = 'dephes',
    output_dir: str | Path | None = None,
    show: bool = True
):
    required = {"counts", "centers", "uncertainties", "expected"}
    missing = sorted(required.difference(result))
    if missing:
        raise KeyError(f"Resultado de ajuste incompleto: {', '.join(missing)}.")
    counts = np.asarray(result["counts"], dtype=np.float64)
    centers = np.asarray(result["centers"], dtype=np.float64)
    errors = np.asarray(result["uncertainties"], dtype=np.float64)
    expected = np.asarray(result["expected"], dtype=np.float64)
    fig, (ax, pull_ax) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    ax.errorbar(centers, counts, yerr=errors, fmt="o", markersize=3.5, color=PlotStyle.text, label="Datos")
    ax.plot(centers, expected, color=PlotStyle.signal, linewidth=2.0, label="Ajuste total")
    components = result.get("components", {})
    for name, values in components.items():
        color = (PlotStyle.background_model if name.lower().startswith("background") else PlotStyle.prompt)
        ax.plot(centers, np.asarray(values, dtype=np.float64), linestyle="--", linewidth=1.6, color=color, label=name)
    style_axis(fig, ax, title=title, xlabel="", ylabel=ylabel)
    ax.legend()
    pulls = np.divide(counts-expected, errors, out=np.zeros_like(counts), where=errors >0)
    pull_ax.axhline(0.0, color=PlotStyle.text, linewidth=1.0)
    pull_ax.axhline(3.0, color=PlotStyle.displaced, linewidth=0.8, linestyle=":")
    pull_ax.axhline(-3.0, color=PlotStyle.displaced, linewidth=0.8, linestyle=":")
    pull_ax.scatter(centers, pulls, s=14, color=PlotStyle.background_model)
    style_axis(fig, pull_ax, title="", xlabel=xlabel, ylabel="Pull")
    finalize_figure(fig, save=save, filename=filename, data=data, output_dir=output_dir, show=show)
    return fig, (ax, pull_ax)