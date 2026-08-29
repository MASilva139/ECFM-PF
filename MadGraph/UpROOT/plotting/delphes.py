from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..styles import PlotStyle

def _required(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    context: str
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{context}: faltan columnas: {', '.join(missing)}")

def _finite_values(df: pd.DataFrame, column: str) -> np.ndarray:
    _required(df, (column), '_finite_values')
    values = df[column].to_numpy(dtype=np.float64, copy=False)
    return values[np.isfinite(values)]

def plot_dimuons_mass(
    dimuons: pd.DataFrame,
    *,
    bins: int = 100,
    mass_range: tuple[float, float] | None = None,
    category_column: str  | None = None,
    title: str = "Espectro de masa dimuónica",
    save_path: str | Path | None = None,
    show: bool = True
):
    _required(dimuons, ('dimuon_mass'), 'plot_dimuon_mass')
    fig, ax = plt.subplots(figsize=(10, 7))
    if category_column is None:
        masses = _finite_values(dimuons, 'dimuons_mass')
        ax.hist(
            masses,
            bins=bins,
            range=mass_range,
            histtype='stepfilled',
            alpha=0.75,
            label='Todos'
        )
    else:
        _required(dimuons, (category_column), 'plot_dimuon_mass')
        colors = {
            'prompt': '#55D98B',
            'displaced': '#B80A0A',
            'mixed': "#62D9FF"
        }
        for category, group in dimuons.groupby(category_column, sort=False):
            values = _finite_values(group, 'dimuon_mass')
            ax.hist(
                values, 
                bins=bins,
                range=mass_range,
                histtype='step',
                linewidth=1.8,
                label=str(category),
                color=colors.get(str(category))
            )
    PlotStyle.apply_dark_axes_style(fig, ax, )