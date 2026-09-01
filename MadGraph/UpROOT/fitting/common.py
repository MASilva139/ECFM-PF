from collections.abc import Iterable, Sequence
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    context: str
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise KeyError(f"{context}: faltan columnas: {', '.join(missing)}.")

def finite_array(values) -> np.array:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]

def values_from_dataframe(
    dataframe: pd.DataFrame,
    column: str,
    *,
    value_range: tuple[float, float] | None = None
) -> np.ndarray:
    require_columns(dataframe, (column), "values_from_dataframe")
    values = finite_array(dataframe[column])
    if value_range is not None:
        lower, upper = value_range
        if lower > upper:
            raise ValueError("value_range: lower < upper.")
        values = values[(values >= lower) & (values <= upper)]
    return values

def prepare_histogram(
    values,
    *, 
    bins: int = 80,
    value_range: tuple[float, float]
) -> dict:
    lower, upper = value_range
    array = finite_array(values)
    array = array[(array >= lower) & (array <= upper)]
    if array.size == 0:
        raise ValueError("array no tiene valores en el intervalo.")
    counts, edges = np.histogram(array, bins=bins, range=value_range)
    centers = 0.5 * (edges[:1] + edges[1:])
    uncertainties = np.sqrt(np.maximum(counts, 1.0))
    return {
        "values": array,
        "counts": counts.astype(np.float64),
        "edges": edges,
        "centers": centers,
        "uncertainties": uncertainties,
        "bin_width": float(edges[1] - edges[0]),
        "value_range": (float[lower], float(upper)),
        "n_entries": int(array.size)
    }

def fit_histogram(
    histogram: dict,
    *,
    model_fn,
    initial_parameters: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bouds: Sequence[float],
    parameter_names: Sequence[str],
    model_name: str,
    maxfev: int = 300_000,
) -> dict:
    if len(initial_parameters) != len(parameter_names):
        raise ValueError("initial_parameters y parameter_names no coinciden.")
    centers = np.asarray(histogram["centers"], dtype=np.float64)
    counts = np.asarray(histogram["counts"], dtype=np.float64)
    uncertainties = np.asarray(histogram["uncertainties"], dtype=np.float64)
    try:
        popt, pcov = curve_fit(
            model_fn,
            centers,
            counts,
            p0=np.asarray(initial_parameters, dtype=np.float64),
            sigma=uncertainties,
            absolute_sigma=True,
            bounds=(lower_bounds, upper_bouds),
            maxfev=maxfev
        )
        converged = True
        warning = None
    except (RuntimeError, ValueError, FloatingPointError) as e:
        popt = np.asarray(initial_parameters, dtype=np.float64)
        pcov = np.full((len(popt), len(popt)), np.nan)
        converged = False
        warning = str(e)
    expected = np.asarray(model_fn(centers, *popt), dtype=np.float64)
    residuals = counts - expected
    pulls = np.divide(residuals, uncertainties, out=np.zeros_like(residuals), where=uncertainties > 0)
    chi2 = float(np.sum(pulls**2))
    ndf = int(len(counts) - len(popt))
    n_parameters = len(popt)
    n_points = len(counts)
    # Aproximaciones para comparar modelos
    aic = chi2 +2.0*n_parameters
    bic = chi2 + n_parameters*np.log(max(n_points, 1))
    errors = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
    parameters = dict(zip(parameter_names, popt))
    parameter_errors = dict(zip(parameter_names, errors))
    return {
        **histogram,
        "model_name": model_name,
        "model_fn": model_fn,
        "parameter_names": list(parameter_names),
        "parameters": parameters,
        "parameter_error": parameter_errors,
        "popt": popt,
        "pcov": pcov,
        "expected": expected,
        "residuals": residuals,
        "pulls": pulls,
        "chi2": chi2,
        "ndf": ndf,
        "reduced_chi2": chi2/ndf if ndf > 0 else np.nan,
        "aic": float(aic),
        "bic": float(bic),
        "converged": converged,
        "warning": warning
    }

def counting_significance(signal: float, background: float) -> dict[str, float]:
    signal = float(signal)
    background = float(background)
    sample = signal/np.sqrt(background) if background > 0 else np.nan
    if background > 0:
        term = (signal + background)*np.log1p(signal/background) - signal
        asimov = np.sqrt(max(2.0*term, 0.0))
    else:
        asimov = np.nan
    return {
        "signal": signal,
        "background": background,
        "s_over_sqrt_b": sample,
        "asimov_significance": float(asimov)
    }

def sideband_background_estimate(
    values,
    *,
    signal_window: tuple[float, float],
    left_band: tuple[float, float],
    right_band: tuple[float, float]
) -> dict[str, float | int]:
    array = finite_array(values)
    s0, s1 = signal_window
    l0, l1 = left_band
    r0, r1 = right_band
    n_signal_region = int(np.count_nonzero((array  >= s0) & (array <= s1)))
    n_left = int(np.count_nonzero((array >= l0) & (array <= l1)))
    n_right = int(np.count_nonzero((array >= r0) & (array <= r1)))
    signal_width = s1 - s0
    sideband_width = (l1-l0)+(r1-r0)
    scale = signal_width / sideband_width
    background = (n_left + n_right) * scale
    signal = n_signal_region-background
    significance = counting_significance(max(signal, 0.0), background)
    return {
        "n_signal_region": n_signal_region,
        "n_left": n_left,
        "n_right": n_right,
        "background_estimate": float(background),
        "signal_estimate": float(signal),
        **significance
    }