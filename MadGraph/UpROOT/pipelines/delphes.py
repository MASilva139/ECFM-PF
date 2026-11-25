# pipelines/delphes
from collections.abc import Callable, Iterable
import matplotlib.pyplot as plt
import pandas as pd

def prepare_dimuon_analysis(
    muons: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    min_muon_pt: float = 0.0,
    max_abs_muon_eta: float | None = None,
    max_isolation: float | None = None,
    mass_window: tuple[float, float] | None = None,
    max_delta_r: float | None = None,
) -> pd.DataFrame:
    from ..physics import delphes as phy
    selected_muons = phy.select_muons(
        muons,
        min_pt=min_muon_pt,
        max_abs_eta=max_abs_muon_eta,
        max_isolation=max_isolation
    )
    dimuons = phy.build_dimuons(selected_muons, events=events)
    dimuons = phy.calc_dimuon_observables(dimuons)
    dimuons = phy.select_dimuons(
        dimuons,
        mass_window=mass_window,
        min_muon_pt=min_muon_pt,
        max_abs_muon_eta=max_abs_muon_eta,
        max_delta_r=max_delta_r,
        max_isolation=max_isolation
    )
    return dimuons

def prepare_particle_analysis(
    particles: pd.DataFrame,
    *,
    pid: int,
    status: int | Iterable[int] | None = None,
    events: pd.DataFrame | None = None,
    require_two: bool = True,
) -> dict[str, object]:
    from ..physics import delphes as phy
    selected_particles = phy.select_particles(particles, pid, status=status)
    particle_pairs = phy.build_particle_pairs(particles, pid=pid, status=status, require_two=require_two)
    summary = phy.summarize_mediator(particles, pid=pid, status=status)
    if events is not None:
        keys = [column for column in ("source_file_id", "event_id") if column in events.columns]
        if "event_id" not in keys:
            raise KeyError("prepare_particle_analysis: Events requiere event_id.")
        total_events = int(events[keys].drop_duplicates().shape[0])
        summary["n_total_events"] = total_events
        summary["fraction_total_events_with_particle"] = (summary["n_events_with_particle"] / total_events if total_events else float("nan"))
    summary["n_particle_pairs"] = int(len(particle_pairs))
    return {
        "particles": selected_particles,
        "particle_pairs": particle_pairs,
        "summary": summary
    }

def prepare_truth_mediator_pair_analysis(
    particles: pd.DataFrame,
    *,
    parent_pid: int,
    status: int | Iterable[int] | None = 23,
    **kwargs,
) -> dict[str, object]:
    return prepare_particle_analysis(particles, pid=parent_pid, status=status, **kwargs)

def prepare_zp_pair_analysis(
    particles: pd.DataFrame,
    *,
    zp_pid: int = 1023,
    status: int | Iterable[int] | None = 23,
    **kwargs,
) -> dict[str, object]:
    return prepare_particle_analysis(particles, pid=zp_pid, status=status, **kwargs)