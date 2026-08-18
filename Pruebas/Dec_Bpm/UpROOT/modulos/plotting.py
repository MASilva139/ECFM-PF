from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .config import FIG_DIR


plt.style.use("seaborn-v0_8-whitegrid")


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def save_figure(
    fig,
    filename: str,
    output_dir: str | Path = FIG_DIR,
    show: bool = True,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_hist1d(
    values,
    *,
    bins: int,
    value_range: tuple[float, float],
    title: str,
    xlabel: str,
    filename: str,
    ylabel: str = "Candidatos",
    color: str = "tab:blue",
    label: str | None = None,
    output_dir: str | Path = FIG_DIR,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        _finite(values),
        bins=bins,
        range=value_range,
        histtype="step",
        linewidth=1.4,
        color=color,
        label=label,
    )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if label:
        ax.legend()
    path = save_figure(fig, filename, output_dir=output_dir)
    return fig, ax, path


def plot_hist_comparison(
    series: Mapping[str, np.ndarray],
    *,
    bins: int,
    value_range: tuple[float, float],
    title: str,
    xlabel: str,
    filename: str,
    ylabel: str = "Candidatos",
    colors: tuple[str, ...] = ("tab:red", "tab:blue", "tab:green", "tab:orange"),
    output_dir: str | Path = FIG_DIR,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    for (label, values), color in zip(series.items(), colors, strict=False):
        ax.hist(
            _finite(values),
            bins=bins,
            range=value_range,
            histtype="step",
            linewidth=1.4,
            color=color,
            label=label,
        )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    path = save_figure(fig, filename, output_dir=output_dir)
    return fig, ax, path


def plot_hist2d(
    x_values,
    y_values,
    *,
    bins: tuple[int, int],
    value_range: tuple[tuple[float, float], tuple[float, float]],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    log_z: bool = False,
    output_dir: str | Path = FIG_DIR,
):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.hist2d(
        x_values[finite_mask],
        y_values[finite_mask],
        bins=bins,
        range=value_range,
        cmin=1,
        norm=LogNorm() if log_z else None,
        cmap="viridis",
    )
    fig.colorbar(image[3], ax=ax, label="Candidatos")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    path = save_figure(fig, filename, output_dir=output_dir)
    return fig, ax, path

