# Fitting para delphes de tabla plana
from collections.abc import Callable, Iterable
import numpy as np
import pandas as pd
from .common import (
    counting_significance,
    fit_histogram,
    prepare_histogram,
    require_columns,
    sideband_background_estimate,
    values_from_dataframe
)
from .models import (
    crystal_ball,
    double_gaussian,
    exponential_background,
    gaussian
)

VALID_MODELS = (
    'gauss_exp',
    'double_gauss_exp',
    'crystalball_exp'
)

def _model_configuration(
    model: str, 
    *,
    histogram: dict,
    target_mass: float,
    initial_sigma: float | None
) -> dict:
    lower, upper = histogram['value_range']
    span = upper - lower
    bin_width = histogram['bin_width']
    counts = histogram['counts']
    pivot = 0.5*(lower+upper)
    maximum = max(float(np.max(counts)), 1.0)
    median = max(float(np.median(counts)), 1.0)
    sigma0 = (float(initial_sigma) if initial_sigma is not None else max(1.5*bin_width, 0.005*max(abs(target_mass), span)))
    sigma0 = min(max(sigma0, 0.30*bin_width), 0.20*span)
    min_sigma = max(0.15*bin_width, np.finfo(float).eps)
    max_sigma = 0.35*span
    slope_limit = 30.0/span
    if model == 'gauss_exp':
        def total(x, amplitude, mean, sigma, bg_amplitude, slope):
            return (gaussian(x, amplitude, mean, sigma) + exponential_background(x, bg_amplitude, slope, pivot))
        def components(x, parameters):
            amplitude, mean, sigma, bg_amplitude, slope = parameters
            return {
                "signal": gaussian(x, amplitude, mean, sigma),
                "background": exponential_background(x, bg_amplitude, slope, pivot)
            }
        return {
            "function": total,
            "components": components,
            "parameter_names": (
                "amplitude",
                "mean",
                "sigma",
                "background_amplitude",
                "slope"
            ),
            "p0": (maximum, target_mass, sigma0, median, 0.0),
            "lower": (0.0, lower, min_sigma, 0.0, -slope_limit),
            "upper": (np.inf, upper, max_sigma, np.inf, slope_limit)
        }
    if model == "double_gauss_exp":
        def total(x, amplitude, mean, sigma_core, core_fraction, sigma_tail, bg_amplitude, slope):
            return (double_gaussian(x, amplitude, mean, sigma_core, core_fraction, sigma_tail) + exponential_background(x, bg_amplitude, slope, pivot))
        def components(x, parameters):
            (amplitude, mean, sigma_core, core_fraction, sigma_tail, bg_amplitude, slope) = parameters
            return {
                "signal": double_gaussian(x, amplitude, mean, sigma_core, core_fraction, sigma_tail),
                "background": exponential_background(x, bg_amplitude, slope, pivot)
            }
        return {
            "function": total,
            "components": components,
            "parameter_names": (
                "amplitude",
                "mean",
                "sigma_core",
                "core_fraction",
                "sigma_tail",
                "background_amplitude",
                "slope"
            ),
            "p0": (maximum, target_mass, 0.7*sigma0, 0.75, 1.0*sigma0, median, 0.0),
            "lower": (0.0, lower, min_sigma, 0.0, min_sigma, 0.0, -slope_limit),
            "upper": (np.inf, upper, max_sigma, 1.0, max_sigma, np.inf, slope_limit)
        }
    if model == "crystalball_exp":
        def total(x, amplitude, beta, m, mean, sigma, bg_amplitude, slope):
            return (crystal_ball(x, amplitude, beta, m, mean, sigma) + exponential_background(x, bg_amplitude, slope, pivot))
        def components(x, parameters):
            amplitude, beta, m, mean, sigma, bg_amplitude, slope = parameters
            return {
                "signal": crystal_ball(x, amplitude, beta, m, mean, sigma),
                "background": exponential_background(x, bg_amplitude, slope, pivot)
            }
        return {
            "function": total,
            "components": components,
            "parameter_names": (
                "amplitude",
                "beta",
                "m", 
                "mean",
                "sigma",
                "background_amplitude",
                "slope"
            ),
            "p0": (maximum, 2.0, 3.0, target_mass, sigma0, median, 0.0),
            "lower": (0.0, 0.2, 1.01, lower, min_sigma, 0.0, -slope_limit),
            "upper": (np.inf, 20.0, 50.0, upper, max_sigma, np.inf, slope_limit)
        }
    raise ValueError(f"Modelo desconocido {model!r}.\nOpciones: {', '.join(VALID_MODELS)}.")

def fit_dimuon_peak(
    dimuons: pd.DataFrame,
    *, 
    target_mass: float,
    mass_window: tuple[float, float],
    mass_column: str = "dimuon_mass",
    bins: int = 80,
    model: str = "gauss_exp",
    initial_sigma: float | None = None,
    verbose: bool = True
) -> dict:
    lower, upper = mass_window
    masses = values_from_dataframe(dimuons, mass_column, value_range=mass_window)
    histogram = prepare_histogram(masses, bins=bins, value_range=mass_window)
    configuration = _model_configuration(
        model,
        histogram=histogram,
        target_mass=target_mass,
        initial_sigma=initial_sigma
    )
    result = fit_histogram(
        histogram,
        model_fn=configuration["function"],
        initial_parameters=configuration["p0"],
        lower_bounds=configuration["lower"],
        upper_bouds=configuration["upper"],
        parameter_names=configuration["parameter_names"],
        model_name=model
    )
    components = configuration["components"](result["centers"], result["popt"])
    result["components"] = components
    result["target_mass"] = float(target_mass)
    result["mass_column"] = mass_column
    result["signal_yield"] = float(np.sum(components["signal"]))
    result["background_yield"] = float(np.sum(components["background"]))
    result["significance"] = counting_significance(result["signal_yield"], result["background_yield"])
    if verbose:
        parameters = result["parameters"]
        print(f"Modelo      = {model}")
        print(f"Media       = {parameters['mean']:.6g}")
        if "sigma" in parameters:
            print(f"Sigma       = {parameters['sigma_core']:.6g}")
        print(f"Señal aprox = {result['signal_yield']:.1f}")
        print(f"χ²/ndf      = {result['chi2']:.1f}/{result['ndf']}")
        if not result['converged']:
            print(f"[WARNING] - {result['warning']}")
    return result

def compare_peak_models(
    dimuons: pd.DataFrame,
    *,
    target_mass: float,
    mass_window: tuple[float, float],
    mass_column: str = 'dimuon_mass',
    bins: int = 80,
    models: Iterable[str] = VALID_MODELS
) -> tuple[pd.DataFrame, dict[str, dict]]:
    results: dict[str, dict] = {}
    rows: list[dict] = []
    for model in models:
        result = fit_dimuon_peak(
            dimuons,
            target_mass=target_mass,
            mass_window=mass_window,
            mass_column=mass_column,
            bins=bins,
            model=model,
            verbose=False
        )
        results[model] = result
        rows.append({
            "model": model,
            "converged": result["converged"],
            "chi2": result["chi2"],
            "ndf": result["ndf"],
            "reduced_chi2": result["reduced_chi2"],
            "aic": result["aic"],
            "bic": result["bic"],
            "signal_yield": result["signal_yield"],
            "background_yield": result["background_yield"]
        })
    sumary = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    return sumary, results

def fit_by_displacement_category(
    dimuons: pd.DataFrame,
    *,
    target_mass: float,
    mass_window: tuple[float, float],
    category_column: str = "displacement_category",
    categories: Iterable[str] = ("prompt", "displaced"),
    mass_column: str = "dimuon_mass",
    bins: int = 80,
    model: str = "gauss_exp"
) -> dict[str, dict]:
    require_columns(dimuons, (category_column), "fit_by_displacement_category")
    results = {}
    for category in categories:
        selected = dimuons.loc[dimuons[category_column].eq(category)]
        if selected.empty:
            continue
        results[str(category)] = fit_dimuon_peak(
            selected,
            target_mass=target_mass,
            mass_window=mass_window,
            mass_column=mass_column,
            bins=bins,
            model=model,
            verbose=False
        )
    return results

def estimate_dimuon_sideband(
    dimuons: pd.DataFrame,
    *,
    signal_window: tuple[float, float],
    left_band: tuple[float, float],
    right_band: tuple[float, float],
    mass_column: str = "dimuon_mass"
) -> dict[str, float | int]:
    values = values_from_dataframe(dimuons, mass_column)
    return sideband_background_estimate(
        values,
        signal_window=signal_window,
        left_band=left_band,
        right_band=right_band
    )

def scan_mass_hypotheses(
    dimuons: pd.DataFrame,
    hypotheses: Iterable[float],
    *,
    half_window: float | Callable[[float], float],
    mass_column: str = "dimuon_mass",
    bins: int = 80,
    model: str = "gauss_exp",
    initial_sigma: float | Callable[[float], float] | None = None
) -> tuple[pd.DataFrame, dict[float, float]]:
    results: dict[float, dict] = {}
    rows: list[dict] = []
    for hypothesis in hypotheses:
        mass = float(hypothesis)
        width = float(half_window(mass) if callable(half_window) else half_window)
        sigma = (float(initial_sigma(mass)) if callable(initial_sigma) else initial_sigma)
        try:
            result = fit_dimuon_peak(
                dimuons,
                target_mass=mass,
                mass_window=(mass - width, mass + width),
                mass_column=mass_column,
                bins=bins,
                initial_sigma=sigma,
                verbose=False
            )
        except ValueError as e:
            rows.append({
                "mass_hypothesis": mass,
                "converged": False,
                "warning": str(e)
            })
            continue
        results[mass] = result
        significance = result["significance"]
        rows.append({
            "mass_hypothesis": mass,
            "fitted_mass": result["parameters"].get("mean", np.nan),
            "signal_yield": result["signal_yield"],
            "background_yield": result["background_yield"],
            "local_asimov": significance["asimov_significance"],
            "chi2_ndf": result["reduced_chi2"],
            "converged": result["converged"],
            "warning": result["warning"]
        })
    return pd.DataFrame(rows), results

def fit_resolution_models(masses, resolutions, *, degree: int = 2) -> dict:
    masses = np.asarray(masses, dtype=np.float64)
    resolutions = np.asarray(resolutions, dtype=np.float64)
    valid = np.isfinite(masses) & np.isfinite(resolutions) & (resolutions > 0)
    masses = masses[valid]
    resolutions = resolutions[valid]
    coefficients = np.polyfit(masses, resolutions, degree)
    predicted = np.polyval(coefficients, masses)
    residuals = resolutions - predicted
    return {
        "degree": degree,
        "coefficients": coefficients,
        "masses": masses,
        "resolutions": resolutions,
        "predicted": predicted,
        "residuals": residuals,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "function": np.poly1d(coefficients)
    }