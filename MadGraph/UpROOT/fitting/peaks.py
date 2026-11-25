# fitting/peaks (script para picos)
from collections.abc import Iterable
import numpy as np
import pandas as pd
from .common import fit_histogram, prepare_histogram, values_from_dataframe
from .models import crystal_ball, double_gaussian, exponential_background, gaussian

VALID_MODELS = ('ge', 'dge', 'cbe')

def _configuration(model: str, *, histogram: dict, target_mass: float, sigma: float | None)-> dict:
    lower, upper = histogram['value_range']
    span = upper - lower
    bin_width = histogram['bin_width']
    counts = histogram['counts']
    pivot = 0.5 * (lower + upper)
    maximum = max(float(np.max(counts)), 1.0)
    median = max(float(np.median(counts)), 1.0)
    sigma0 = (float(sigma) if sigma is not None else max(1.5 * bin_width, 0.005 * max(abs(target_mass), span)))
    sigma0 = min(max(sigma0, 0.30 * bin_width), 0.20 * span)
    min_sigma = max(0.15 * bin_width, np.finfo(float).eps)
    max_sigma = 0.35 * span
    slope_limit = 30.0 / span
    if model == "gauss_exp":
        def total(x, amplitude, mean, sigma, bg_amplitude, slope):
            return gaussian(x, amplitude, mean, sigma) + exponential_background(x, bg_amplitude, slope, pivot)
        def components(x, parameters):
            amplitude, mean, sigma, bg_amplitude, slope = parameters
            return {
                "signal": gaussian(x, amplitude, mean, sigma),
                "background": exponential_background(x, bg_amplitude, slope, pivot),
            }
        return {
            "function": total,
            "components": components,
            "parameter_names": ("amplitude", "mean", "sigma", "background_amplitude", "slope"),
            "p0": (maximum, target_mass, sigma0, median, 0.0),
            "lower": (0.0, lower, min_sigma, 0.0, -slope_limit),
            "upper": (np.inf, upper, max_sigma, np.inf, slope_limit),
        }
    if model == "double_gauss_exp":
        def total(x, amplitude, mean, sigma_core, core_fraction, sigma_tail, bg_amplitude, slope):
            return double_gaussian(x, amplitude, mean, sigma_core, core_fraction, sigma_tail) + exponential_background(x, bg_amplitude, slope, pivot)
        def components(x, parameters):
            (amplitude, mean, sigma_core, core_fraction, sigma_tail, bg_amplitude, slope) = parameters
            return {
                "signal": double_gaussian(x, amplitude, mean, sigma_core, core_fraction, sigma_tail),
                "background": exponential_background(x, bg_amplitude, slope, pivot),
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
                "slope",
            ),
            "p0": (maximum, target_mass, 0.7 * sigma0, 0.75, sigma0, median, 0.0),
            "lower": (0.0, lower, min_sigma, 0.0, min_sigma, 0.0, -slope_limit),
            "upper": (np.inf, upper, max_sigma, 1.0, max_sigma, np.inf, slope_limit),
        }
    if model == "crystalball_exp":
        def total(x, amplitude, beta, m, mean, sigma, bg_amplitude, slope):
            return crystal_ball(x, amplitude, beta, m, mean, sigma) + exponential_background(x, bg_amplitude, slope, pivot)
        def components(x, parameters):
            amplitude, beta, m, mean, sigma, bg_amplitude, slope = parameters
            return {
                "signal": crystal_ball(x, amplitude, beta, m, mean, sigma),
                "background": exponential_background(x, bg_amplitude, slope, pivot),
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
                "slope",
            ),
            "p0": (maximum, 2.0, 3.0, target_mass, sigma0, median, 0.0),
            "lower": (0.0, 0.2, 1.01, lower, min_sigma, 0.0, -slope_limit),
            "upper": (np.inf, 20.0, 50.0, upper, max_sigma, np.inf, slope_limit),
        }
    raise ValueError(f"Modelo desconocido {model!r}: {', '.join(VALID_MODELS)}.")

# Ajuste de pico con fondo exponencial
def fit_peak(
    dataframe: pd.DataFrame,
    *,
    target_mass: float,
    mass_window: tuple[float, float],
    mass_column: str,
    bins: int = 80,
    model: str = "gauss_exp",
    sigma: float | None = None,
    verbose: bool = True,
) -> dict:
    masses = values_from_dataframe(dataframe, mass_column, value_range=mass_window)
    histogram = prepare_histogram(masses, bins=bins, value_range=mass_window)
    configuration = _configuration(
        model,
        histogram=histogram,
        target_mass=target_mass,
        sigma=sigma,
    )
    result = fit_histogram(
        histogram,
        model_fn=configuration["function"],
        initial_parameters=configuration["p0"],
        lower_bounds=configuration["lower"],
        upper_bouds=configuration["upper"],
        parameter_names=configuration["parameter_names"],
        model_name=model,
    )
    components = configuration["components"](result["centers"], result["popt"])
    result["components"] = components
    result["target_mass"] = float(target_mass)
    result["mass_column"] = mass_column
    result["signal_yield"] = float(np.sum(components["signal"]))
    result["background_yield"] = float(np.sum(components["background"]))
    if verbose:
        print(f"Modelo = {model}")
        print(f"Media = {result['parameters']['mean']:.6g}")
        print(f"χ²/ndf = {result['chi2']:.1f}/{result['ndf']}")
        if not result["converged"]:
            print(f"[WARNING] - {result['warning']}")
    return result

def compare_peak_models(
    dataframe: pd.DataFrame,
    *,
    target_mass: float,
    mass_window: tuple[float, float],
    mass_column: str,
    bins: int = 80,
    models: Iterable[str] = VALID_MODELS,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Compara modelos genéricos mediante χ², AIC y BIC."""

    results: dict[str, dict] = {}
    rows = []
    for model in models:
        result = fit_peak(
            dataframe,
            target_mass=target_mass,
            mass_window=mass_window,
            mass_column=mass_column,
            bins=bins,
            model=model,
            verbose=False,
        )
        results[model] = result
        rows.append(
            {
                "model": model,
                "converged": result["converged"],
                "chi2": result["chi2"],
                "ndf": result["ndf"],
                "reduced_chi2": result["reduced_chi2"],
                "aic": result["aic"],
                "bic": result["bic"],
                "signal_yield": result["signal_yield"],
                "background_yield": result["background_yield"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    return summary, results