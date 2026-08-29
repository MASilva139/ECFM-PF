from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from .config import FIG_DIR
from .plotting import save_figure

Model = Callable[..., np.ndarray]

@dataclass(slots=True)
class FitResult:
    model_name: str
    model: Model
    parameter_names: tuple[str, ...]
    parameters: np.ndarray
    covariance: np.ndarray
    counts: np.ndarray
    edges: np.ndarray
    centers: np.ndarray
    fit_range: tuple[float, float]

    @property
    def errors(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.covariance), 0, None))
    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {"value": float(value), "error": float(error)}
            for name, value, error in zip(
                self.parameter_names,
                self.parameters,
                self.errors,
                strict=True,
            )
        }

def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def polynomial_1(x, c0, c1):
    dx = x - 5280.0
    return c0 + c1 * dx

def polynomial_2(x, c0, c1, c2):
    dx = x - 5280.0
    return c0 + c1 * dx + c2 * dx**2

def polynomial_3(x, c0, c1, c2, c3):
    dx = x - 5280.0
    return c0 + c1 * dx + c2 * dx**2 + c3 * dx**3

def exponential_background(x, amplitude, slope):
    return amplitude * np.exp(slope * (x - 5280.0))

def landau_like(x, amplitude, most_probable_value, width):
    z = np.clip((x - most_probable_value) / width, -100, 100)
    return amplitude * np.exp(-0.5 * (z + np.exp(-z)))

def gaussian_plus_polynomial_1(x, a, mean, sigma, c0, c1):
    return gaussian(x, a, mean, sigma) + polynomial_1(x, c0, c1)

def gaussian_plus_exponential(x, a, mean, sigma, b, slope):
    return gaussian(x, a, mean, sigma) + exponential_background(x, b, slope)

def double_gaussian(x, a1, mean1, sigma1, a2, mean2, sigma2):
    return gaussian(x, a1, mean1, sigma1) + gaussian(x, a2, mean2, sigma2)

def double_gaussian_plus_exponential(
    x,
    a1,
    mean1,
    sigma1,
    a2,
    mean2,
    sigma2,
    background,
    slope,
):
    return double_gaussian(x, a1, mean1, sigma1, a2, mean2, sigma2) + (
        exponential_background(x, background, slope)
    )

def breit_wigner(x, amplitude, mean, gamma):
    half_gamma_squared = (0.5 * gamma) ** 2
    return amplitude * half_gamma_squared / (
        (x - mean) ** 2 + half_gamma_squared
    )

def fit_histogram(
    values,
    *,
    model_name: str,
    model: Model,
    initial_parameters: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]],
    parameter_names: Sequence[str],
    bins: int,
    histogram_range: tuple[float, float],
    fit_range: tuple[float, float],
) -> FitResult:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins, range=histogram_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= fit_range[0]) & (centers <= fit_range[1])
    uncertainties = np.sqrt(np.maximum(counts[mask], 1))
    parameters, covariance = curve_fit(
        model,
        centers[mask],
        counts[mask],
        p0=initial_parameters,
        sigma=uncertainties,
        absolute_sigma=True,
        bounds=bounds,
        maxfev=200_000,
    )
    return FitResult(
        model_name=model_name,
        model=model,
        parameter_names=tuple(parameter_names),
        parameters=parameters,
        covariance=covariance,
        counts=counts,
        edges=edges,
        centers=centers,
        fit_range=fit_range,
    )

def plot_fit_result(
    result: FitResult,
    *,
    title: str,
    xlabel: str,
    filename: str,
    output_dir: str | Path = FIG_DIR,
):
    x_fit = np.linspace(*result.fit_range, 1000)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.stairs(result.counts, result.edges, color="black", label="Datos")
    ax.plot(
        x_fit,
        result.model(x_fit, *result.parameters),
        color="tab:red",
        label="Ajuste",
    )
    ax.set(title=title, xlabel=xlabel, ylabel="Candidatos")
    ax.legend()
    path = save_figure(fig, filename, output_dir=output_dir)
    return fig, ax, path

def print_fit_parameters(result: FitResult) -> None:
    for name, estimate in result.as_dict().items():
        print(f"{name}: {estimate['value']:.6g} ± {estimate['error']:.3g}")

def fit_and_plot(
    values,
    *,
    model_name: str,
    model: Model,
    initial_parameters: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]],
    parameter_names: Sequence[str],
    bins: int,
    histogram_range: tuple[float, float],
    fit_range: tuple[float, float],
    title: str,
    xlabel: str,
    filename: str,
    output_dir: str | Path = FIG_DIR,
) -> FitResult:
    result = fit_histogram(
        values,
        model_name=model_name,
        model=model,
        initial_parameters=initial_parameters,
        bounds=bounds,
        parameter_names=parameter_names,
        bins=bins,
        histogram_range=histogram_range,
        fit_range=fit_range,
    )
    plot_fit_result(
        result,
        title=title,
        xlabel=xlabel,
        filename=filename,
        output_dir=output_dir,
    )
    print_fit_parameters(result)
    return result