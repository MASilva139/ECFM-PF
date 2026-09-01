import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..config import MASS_MIN, MASS_MAX
from ..plotting.common import finalize_figure

def normalize_fit_models(fit_model: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(fit_model, str):
        return [fit_model]
    return list(fit_model)

def plot_mass_by_change(
    df: pd.DataFrame,
    channel_name: str,
    *,
    fit_model: str | list[str] | tuple[str, ...] = "gauss_exp",
    mass_column: str = "B_M",
    charge_column: str = "B_charge",
    mass_min: float | None = None,
    mass_max: float | None = None,
    save: bool = False,
    filename: str | None = None,
    data: str = 'data',
    output_dir = None,
    show: bool = True
) -> dict[str, dict[int, dict]]:
    from ..fitting.b2hhh import fit_mass
    from ..plotting.b2hhh import plot_mass_fit
    mass_min = MASS_MIN if mass_min is None else mass_min
    mass_max = MASS_MAX if mass_max is None else mass_max
    fit_models = normalize_fit_models(fit_model)
    fig, axes = plt.subplots(len(fit_models), 2, figsize={14, 5*len(fit_models)}, squeeze=False)
    fig.suptitle(f"{channel_name} - Masa Invariante del B", fontweight="bold")
    results: dict[str, dict[int, dict]] = {}
    for row, model in enumerate(fit_models):
        results[model] = {}
        for col, (charge, label) in enumerate(((1, r"$B^{+}$"), (-1, r"$B^{-}$"))):
            ax = axes[row, col]
            masses_arr = df.loc[df[charge_column] == charge, mass_column].to_numpy()
            print(rf"{label} ({model}): {len(masses_arr):,} candidatos en [{mass_min}, {mass_max}] $MeV/c^2$")
            result = fit_mass(masses_arr, model=model, verbose=True)
            results[model][charge] = result
            plot_mass_fit(ax, result, label=f"{label} - {model}", mass_min=mass_min, mass_max=mass_max)
            finalize_figure(
                fig,
                save=save,
                filename=filename or f"b2hhh_{channel_name}_mass_fit",
                data=data,
                output_dir=output_dir,
                show=show
            )
            return results