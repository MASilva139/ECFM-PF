from collections.abc import Callable, Iterable
import matplotlib.pyplot as plt
import pandas as pd
from ..styles import PlotStyle
from ..plotting.common import finalize_figure, style_axis

def fit_mediator_mass(
    pairs: pd.DataFrame,
    *,
    target_mass: float | None = None,
    mass_window: tuple[float, float] | None = None,
    hypotheses: Iterable[float] | None = None,
    half_window: float | Callable[[float], float] = 5.0,
    mass_column: str = "dimuon_mass",
    model: str = "gauss_exp",
    bins: int = 80,
    plot: bool = True,
    save: bool = False,
    filename: str | None = None,
    output_dir = None,
    show: bool = True
) -> dict:
    from ..fitting import delphes as fit_delphes
    from ..plotting import delphes as plot_delphes
    has_target = target_mass is not None
    has_scan = hypotheses is not None
    if has_target == has_scan:
        raise ValueError("Debe escogerse entre:\n- 'target_mass': ajuste de masa fija.\n- 'hypotheses': escaneo de masas desconocidas.")
    if has_target:
        window = mass_window or (0.85*target_mass, 1.15*target_mass)
        result = fit_delphes.fit_dimuon_peak(
            pairs,
            target_mass=target_mass,
            mass_window=window,
            mass_column=mass_column,
            bins=bins,
            model=model,
            verbose=True
        )
        if plot:
            plot_delphes.plot_fit_result(
                result,
                title=f"Ajuste del mediador (masa objetivo = {target_mass:.2f} [GeV])",
                save=save,
                filename=filename or "delphes_mediator_fit",
                output_dir=output_dir,
                show=show
            )
        return {'mode': "fixed_target", **result}
    scan_summary, scan_results = fit_delphes.scan_mass_hypotheses(
        pairs,
        hypotheses,
        half_window=half_window,
        mass_column=mass_column,
        bins=bins,
        model=model
    )
    if plot:
        fig, ax = plot.subplots(figsize=(10, 6.5))
        ax.plot(
            scan_summary["mass_hypothesis"],
            scan_summary["local_asimov"],
            marker="o",
            markersize=4,
            color=PlotStyle.signal
        )
        ax.axhline(0.0, color=PlotStyle.text, linewidth=1.0, linestyle="--")
        converged = scan_summary.loc[scan_summary["converged"].fillna(False)]
        if not converged.empty:
            best = converged.sort_values("local_asimov", ascending=False).iloc[0]
            ax.axvline(best["mass_hypothesis"], color=PlotStyle.displaced, linestyle=":", label=f"Mejor candidato = {best['mass_hypothesis']:.1f} [GeV]")
            ax.legend()
        style_axis(
            fig,
            ax, 
            title="Escaneo de hipótesis de masa del mediador",
            xlabel=r"Masa hipotética [$GeV/c^2$]",
            ylabel="Significancia local (Asimov)"
        )
        finalize_figure(
            fig,
            save=save,
            filename=filename or "delphes_mediator_scan",
            output_dir=output_dir,
            show=show
        )
    return {'mode': "mass_scan", 'summary': scan_summary, "results": scan_results}