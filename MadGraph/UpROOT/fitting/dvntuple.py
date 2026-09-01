# Fitting para archivos dvntuple
from collections.abc import Iterable
import numpy as np
import pandas as pd
from .common import require_columns, sideband_background_estimate
from .delphes import compare_peak_models, fit_dimuon_peak

def first_available_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
    context: str,
) -> str:
    for column in candidates:
        if column in dataframe.columns:
            return column
    raise KeyError(f"{context}: no existe ninguna de {', '.join(candidates)}.")

def fit_b_mass(
    dataframe: pd.DataFrame,
    *,
    mass_column: str | None = None,
    target_mass: float = 5279.34,
    mass_window: tuple[float, float] = (5100.0, 5500.0),
    bins: int = 80,
    model: str = "crystalball_exp",
    initial_sigma: float = 18.0,
    verbose: bool = True,
) -> dict:
    if mass_column is None:
        mass_column = first_available_column(
            dataframe,
            ("Bplus_MM", "Bplus_M"),
            "fit_b_mass",
        )
    result = fit_dimuon_peak(
        dataframe,
        target_mass=target_mass,
        mass_window=mass_window,
        mass_column=mass_column,
        bins=bins,
        model=model,
        initial_sigma=initial_sigma,
        verbose=verbose,
    )
    result["analysis"] = "B2KMuMu"
    result["observable"] = "B_mass"
    result["unit"] = "MeV/c^2"
    return result

def fit_dimuon_resonance(
    dataframe: pd.DataFrame,
    *,
    target_mass: float,
    mass_window: tuple[float, float],
    mass_column: str | None = None,
    bins: int = 80,
    model: str = "gauss_exp",
    initial_sigma: float | None = None,
    verbose: bool = True,
) -> dict:
    if mass_column is None:
        mass_column = first_available_column(
            dataframe,
            ("J_psi_1S_MM", "J_psi_1S_M"),
            "fit_dimuon_resonance",
        )
    result = fit_dimuon_peak(
        dataframe,
        target_mass=target_mass,
        mass_window=mass_window,
        mass_column=mass_column,
        bins=bins,
        model=model,
        initial_sigma=initial_sigma,
        verbose=verbose,
    )
    result["analysis"] = "B2KMuMu"
    result["observable"] = "dimuon_mass"
    result["unit"] = "MeV/c^2"
    return result

def fit_jpsi_peak(
    dataframe: pd.DataFrame,
    *,
    mass_column: str | None = None,
    mass_window: tuple[float, float] = (3000.0, 3200.0),
    bins: int = 80,
    model: str = "double_gauss_exp",
    verbose: bool = True,
) -> dict:
    return fit_dimuon_resonance(
        dataframe,
        target_mass=3096.9,
        mass_window=mass_window,
        mass_column=mass_column,
        bins=bins,
        model=model,
        initial_sigma=12.0,
        verbose=verbose,
    )

def compare_b_mass_models(
    dataframe: pd.DataFrame,
    *,
    mass_column: str | None = None,
    target_mass: float = 5279.34,
    mass_window: tuple[float, float] = (5100.0, 5500.0),
    bins: int = 80,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    if mass_column is None:
        mass_column = first_available_column(
            dataframe,
            ("Bplus_MM", "Bplus_M"),
            "compare_b_mass_models",
        )
    return compare_peak_models(
        dataframe,
        target_mass=target_mass,
        mass_window=mass_window,
        mass_column=mass_column,
        bins=bins,
    )

def fit_b_charge_samples(
    dataframe: pd.DataFrame,
    *,
    id_column: str = "Bplus_ID",
    mass_column: str | None = None,
    **fit_options,
) -> dict[str, dict]:
    require_columns(dataframe, (id_column,), "fit_b_charge_samples")
    particle_id = dataframe[id_column].to_numpy(dtype=np.float64, copy=False)
    selections = {
        "Bplus": dataframe.loc[particle_id > 0],
        "Bminus": dataframe.loc[particle_id < 0],
    }
    results = {}
    for label, sample in selections.items():
        if sample.empty:
            continue
        results[label] = fit_b_mass(
            sample,
            mass_column=mass_column,
            verbose=False,
            **fit_options,
        )
    return results

def estimate_b_sidebands(
    dataframe: pd.DataFrame,
    *,
    mass_column: str | None = None,
    signal_window: tuple[float, float] = (5228.0, 5330.0),
    left_band: tuple[float, float] = (5100.0, 5200.0),
    right_band: tuple[float, float] = (5400.0, 5500.0),
) -> dict[str, float | int]:
    if mass_column is None:
        mass_column = first_available_column(
            dataframe,
            ("Bplus_MM", "Bplus_M"),
            "estimate_b_sidebands",
        )
    require_columns(dataframe, (mass_column,), "estimate_b_sidebands")
    return sideband_background_estimate(
        dataframe[mass_column],
        signal_window=signal_window,
        left_band=left_band,
        right_band=right_band,
    )

def summarize_charge_fits(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for charge, result in results.items():
        rows.append(
            {
                "sample": charge,
                "mean": result["parameters"].get("mean", np.nan),
                "signal_yield": result.get("signal_yield", np.nan),
                "background_yield": result.get("background_yield", np.nan),
                "chi2_ndf": result.get("reduced_chi2", np.nan),
                "converged": result.get("converged", False),
            }
        )
    return pd.DataFrame(rows)