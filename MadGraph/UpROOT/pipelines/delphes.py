# pipelines/delphes
from collections.abc import Callable, Iterable
import matplotlib.pyplot as plt
import pandas as pd
from ..styles import PlotStyle
from ..plotting.common import finalize_figure, style_axis

def prepare_truth_mediator_pair_analysis(
    particles: pd.DataFrame,
    *,
    parent_pid: int,
    status: int | Iterable[int] | None = 23,
    events: pd.DataFrame | None = None,
    jets: pd.DataFrame | None = None,
    missing_et: pd.DataFrame | None = None,
    require_two: bool = True,
) -> dict[str, object]:
    from ..physics import delphes as physics_delphes
    truth_mediators = physics_delphes.select_particles(
        particles,
        parent_pid,
        status=status,
    )
    truth_pairs = physics_delphes.build_particle_pairs(
        particles,
        pid=parent_pid,
        status=status,
        require_two=require_two,
    )
    event_observables = physics_delphes.attach_pair_recoil(
        truth_pairs,
        missing_et=missing_et,
        jets=jets,
        events=events,
    ) if not truth_pairs.empty else truth_pairs.copy()
    summary = physics_delphes.summarize_mediator(particles, pid=parent_pid, status=status,)
    if events is not None and "event_id" in events.columns:
        event_keys = [column for column in ("source_file_id", "event_id") if column in events.columns]
        total_events = int(events[event_keys].drop_duplicates().shape[0])
        summary["n_total_events"] = total_events
        summary["fraction_total_events_with_mediator"] = (
            float(summary["n_events_with_mediator"]/total_events)
            if total_events
            else float("nan")
        )
    summary["n_truth_pairs"] = int(len(truth_pairs))
    summary["n_events_with_missing_et"] = int(
        event_observables["missing_et"].notna().sum()
        if "missing_et" in event_observables.columns
        else 0
    )
    summary["n_events_with_leading_jet"] = int(
        event_observables["leading_jet_pt"].notna().sum()
        if "leading_jet_pt" in event_observables.columns
        else 0
    )
    return {
        "truth_mediators": truth_mediators,
        "truth_pairs": truth_pairs,
        "event_observables": event_observables,
        "summary": summary,
    }

def prepare_particle_analysis(
    particles: pd.DataFrame,
    *,
    pid: int,
    status: int | Iterable[int] | None = None,
    **kwargs,
) -> dict[str, object]:
    return prepare_truth_mediator_pair_analysis(particles, parent_pid=pid, status=status, **kwargs,)

def prepare_truth_zp_pair_analysis(
    particles: pd.DataFrame,
    *,
    zp_pid: int = 1023,
    status: int | Iterable[int] | None = 23,
    **kwargs,
) -> dict[str, object]:
    return prepare_truth_mediator_pair_analysis(particles, parent_pid=zp_pid, status=status, **kwargs,)

def prepare_mediator_pair_analysis(
    muons: pd.DataFrame,
    *,
    parent_pid: int,
    particles: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    target_mass: float | None = None,
    pairing: str = "auto",
    truth_status: int | Iterable[int] | None = None,
    truth_require_two: bool = False,
    min_muon_pt: float = 0.0,
    max_abs_muon_eta: float | None = None,
    max_isolation: float | None = None,
    mass_window: tuple[float, float] | None = None,
    max_delta_r: float | None = None,
) -> dict[str, pd.DataFrame]:
    from ..physics import delphes as physics_delphes
    allowed_pairings = {"auto", "truth", "reco"}
    if pairing not in allowed_pairings:
        raise ValueError(f"pairing debe ser una de {sorted(allowed_pairings)}.")
    if pairing == "truth" and particles is None:
        raise ValueError("pairing='truth' requiere el DataFrame Particles.")
    selected_muons = physics_delphes.select_muons(
        muons,
        min_pt=min_muon_pt,
        max_abs_eta=max_abs_muon_eta,
        max_isolation=max_isolation,
    )
    dimuons = physics_delphes.build_dimuons(selected_muons, events=events)
    dimuons = physics_delphes.calc_dimuon_observables(dimuons)
    dimuons = physics_delphes.select_dimuons(
        dimuons,
        mass_window=mass_window,
        min_muon_pt=min_muon_pt,
        max_abs_muon_eta=max_abs_muon_eta,
        max_delta_r=max_delta_r,
        max_isolation=max_isolation,
    )
    matched_dimuons = pd.DataFrame()
    truth_mediators = pd.DataFrame()
    truth_pairs = pd.DataFrame()
    effective_pairing = "truth" if pairing == "auto" and particles is not None else pairing
    if effective_pairing == "auto":
        effective_pairing = "reco"
    if particles is not None:
        matched_dimuons = physics_delphes.match_dimuon_truth_parents(
            dimuons,
            particles,
            parent_pid=parent_pid,
        )
        truth_mediators = physics_delphes.select_particles(
            particles,
            parent_pid,
            status=truth_status,
        )
        truth_pairs = physics_delphes.build_particle_pairs(
            particles,
            pid=parent_pid,
            status=truth_status,
            require_two=truth_require_two,
        )
    if effective_pairing == "truth":
        pair_candidates = physics_delphes.select_truth_matched_dimuons(matched_dimuons, parent_pid=parent_pid, target_mass=target_mass,)
        mediator_pairs = physics_delphes.build_mediator_pairs(pair_candidates, target_mass=target_mass, require_distinct_truth_parents=True,)
    else:
        pair_candidates = dimuons
        mediator_pairs = physics_delphes.build_mediator_pairs(pair_candidates, target_mass=target_mass,)
    return {
        "selected_muons": selected_muons,
        "dimuons": dimuons,
        "matched_dimuons": matched_dimuons,
        "pair_candidates": pair_candidates,
        "mediator_pairs": mediator_pairs,
        "truth_mediators": truth_mediators,
        "truth_pairs": truth_pairs,
    }

def prepare_zp_pair_analysis(
    muons: pd.DataFrame,
    *,
    particles: pd.DataFrame | None = None,
    zp_pid: int = 1023,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    # Canal opcional zp zp -> 4 muones
    return prepare_mediator_pair_analysis(muons, parent_pid=zp_pid, particles=particles, **kwargs,)

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