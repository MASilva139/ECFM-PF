# physics/delphes
from collections.abc import Iterable
from itertools import combinations
import numpy as np
import pandas as pd
import re
from .common import (delta_r, delta_phi, invariant_mass)

MUON_MASS_GEV = 0.1056583755
ELECTRON_MASS_GEV = 0.00051099895
MOMENTUM_UNIT = 'GeV'
POSITION_UNIT = 'mm'
C_MM_PS = 0.299792458

_COLUMN_NAMES = {
    "object_index": "index",
    "PT": "pt",
    "Eta": "eta",
    "Phi": "phi",
    "Charge": "charge",
    "PX": "px",
    "PY": "py",
    "PZ": "pz",
    "P": "p",
    "E": "e",
}

def _required(df: pd.DataFrame, columns: tuple[str, ...], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        formatted = ", ".join(missing)
        raise KeyError(f"{context}: columnas faltantes: {formatted}")

def _snake_case(name: str) -> str:
    if name in _COLUMN_NAMES:
        return _COLUMN_NAMES[name]
    converted = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return converted.lower()

def _float_column(df: pd.DataFrame, column: str, context: str) -> np.ndarray:
    _required(df, (column,), context)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)

def _normalise_integers(values: int | Iterable[int], *, name:str) -> set[int]:
    if isinstance(values, (int, np.integer)):
        return {int(values)}
    if isinstance(values, (str, bytes)):
        raise TypeError(f'{name} debe ser un entero o un iterable de enteros.')
    result = {int(value) for value in values}
    if not result:
        raise ValueError(f'{name}: vacío')
    return values

def _event_keys(df: pd.DataFrame) -> list[str]:
    _required(df, ('event_id',), "_event_keys")
    return [column for column in ('source_file_id', 'event_id') if column in df.columns]

def _four_vector(objects: pd.DataFrame, *, context: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # cartesian = (
    #     ('E', 'Px', 'Py', 'Pz'),
    #     ('E', 'PX', 'PY', 'PZ'),
    #     ('energy', 'px', 'py', 'pz'),
    # )
    # for energy_column, px_column, py_column, pz_column in cartesian:
    #     if {energy_column, px_column, py_column, pz_column}.issubset(objects.columns):
    #         return tuple(_float_column(objects, column, context) for column in (energy_column, px_column, py_column, pz_column))
    for columns in (
        ('E', 'Px', 'Py', 'Pz'), 
        ('E', 'PX', 'PY', 'PZ'),
        ('e', 'px', 'py', 'pz')
    ):
        if set(columns).issubset(objects.columns):
            return tuple(_float_column(objects, column, context) for column in columns)
    _required(objects, ('PT', 'Eta', 'Phi'), context)
    pt = _float_column(objects, 'PT', context)
    eta = _float_column(objects, 'Eta', context)
    phi = _float_column(objects, 'Phi', context)
    mass = (_float_column(objects, 'Mass', context) if 'Mass' in objects.columns else np.zeros(len(objects), dtype=np.float64))
    # if 'Mass' in objects.columns:
    #     mass = _float_column(objects, 'Mass', context)
    # else:
    #     mass = np.zeros(len(objects), dtype=np.float64)
    px = pt*np.cos(phi)
    py = pt*np.sin(phi)
    pz = pt*np.sinh(eta)
    e = np.sqrt(np.maximum(px**2 + py**2 + pz**2 + mass**2, 0.0))
    return e, px, py, pz

def select_particles(
    particles: pd.DataFrame,
    pid: int | Iterable[str],
    *,
    match_pid: bool = True,
    status: int | Iterable[int] | None = None
) -> pd.DataFrame:
    context = 'select_particles'
    _required(particles, ('event_id', 'PID'), context)
    selected_pids = _normalise_integers(pid, name='pid')
    # assert selected_pids is not None
    pid_values = pd.to_numeric(particles['PID'], errors='coerce')
    mask = (pid_values.abs().isin({abs(value) for value in selected_pids}) if match_pid else pid_values.isin(selected_pids))
    # if match_pid:
    #     mask = pid_values.abs().isin({abs(value) for value in selected_pids})
    # else:
    #     mask = pid_values.isin(selected_pids)
    if status is not None:
        _required(particles, ("Status",), context)
        statuses = _normalise_integers(status, name='status')
        mask &= pd.to_numeric(particles["Status"], errors='coerce').isin(statuses)
    return particles.loc[mask].copy()

def build_particle_pairs(
    particles: pd.DataFrame,
    *,
    pid: int | Iterable[int],
    match_pid: bool = True,
    status: int | Iterable[int] | None = None,
    require_two: bool = False
) -> pd.DataFrame:
    context = "build_particle_pairs"
    selected = select_particles(particles, pid, match_pid=match_pid, status=status)
    _required(selected, ("object_index",), context)
    if selected.empty:
        return pd.DataFrame()
    keys = _event_keys(selected)
    e, px, py, pz = _four_vector(selected, context=context)
    selected = selected.copy()
    selected["__e"] = e
    selected["__px"] = px
    selected["__py"] = py
    selected["__pz"] = pz
    selected["__pt"] = np.hypot(px, py)
    selected["__phi"] = np.arctan2(py, px)
    selected["__eta"] = np.arcsinh(np.divide(pz, selected["__pt"].to_numpy(), out=np.full(len(selected), np.nan), where=selected["__pt"].to_numpy() > 0,))
    counts = selected.groupby(keys, sort=False).size().rename("n_particles")
    selected = selected.merge(counts, on=keys, how="left", validate="many_to_one")
    mask = (selected["n_particles"].eq(2) if require_two else selected["n_particles"].ge(2))
    selected = selected.loc[mask].copy()
    if selected.empty:
        return pd.DataFrame()
    selected = selected.sort_values([*keys, "__pt", "object_index"], ascending=[*[True] * len(keys), False, True], kind="stable")
    selected["__pair_rank"] = selected.groupby(keys, sort=False).cumcount()
    selected = selected.loc[selected["__pair_rank"].lt(2)].copy()
    canonical = {
        "object_index": "index",
        "PID": "pid",
        "Status": "status",
        "Mass": "mass",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "T": "t",
        "root_uid": "root_uid",
        "__e": "e",
        "__px": "px",
        "__py": "py",
        "__pz": "pz",
        "__pt": "pt",
        "__eta": "eta",
        "__phi": "phi",
    }
    value_columns = [column for column in canonical if column in selected.columns]
    def ranked(rank: int, prefix: str) -> pd.DataFrame:
        frame = selected.loc[selected["__pair_rank"].eq(rank), [*keys, "n_particles", *value_columns],].copy()
        return frame.rename(columns={column: f"{prefix}_{canonical[column]}" for column in value_columns})
    first = ranked(0, "p1")
    second = ranked(1, "p2").drop(columns="n_particles")
    result = first.merge(second, on=keys, how="inner", validate="one_to_one", sort=False)
    e = result["p1_e"].to_numpy() + result["p2_e"].to_numpy()
    px = result["p1_px"].to_numpy() + result["p2_px"].to_numpy()
    py = result["p1_py"].to_numpy() + result["p2_py"].to_numpy()
    pz = result["p1_pz"].to_numpy() + result["p2_pz"].to_numpy()
    pt = np.hypot(px, py)
    result["pair_e"] = e
    result["pair_px"] = px
    result["pair_py"] = py
    result["pair_pz"] = pz
    result["pair_mass"] = invariant_mass(e, px, py, pz)
    result["pair_pt"] = pt
    result["pair_eta"] = np.arcsinh(np.divide(pz, pt, out=np.full(len(result), np.nan), where=pt > 0))
    result["pair_phi"] = np.arctan2(py, px)
    numerator = e + pz
    denominator = e - pz
    result["pair_rapidity"] = np.where((numerator > 0) & (denominator > 0), 0.5 * np.log(numerator / denominator), np.nan,)
    result["pair_delta_phi"] = delta_phi(result["p1_phi"], result["p2_phi"])
    result["pair_delta_r"] = delta_r(result["p1_eta"], result["p1_phi"], result["p2_eta"], result["p2_phi"])
    return result

def summarize_mediator(
    particles: pd.DataFrame,
    *,
    pid: int | Iterable[int],
    status: int | Iterable[int] | None = None,
    match_pid: bool = True
) -> dict[str, object]:
    all_selected = select_particles(particles, pid, match_pid=match_pid)
    selected = select_particles(particles, pid, match_pid=match_pid, status=status)
    keys = _event_keys(particles)
    total_events = int(particles[keys].drop_duplicates().shape[0])
    counts = (selected.groupby(keys, sort=False).size() if not selected.empty else pd.Series(dtype=np.int64))
    masses = (pd.to_numeric(selected["Mass"], errors="coerce").dropna().to_numpy() if "Mass" in selected.columns else np.asarray([], dtype=np.float64))
    status_counts = (
        pd.to_numeric(all_selected["Status"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
        if "Status" in all_selected.columns
        else {}
    )
    n_events = int(len(counts))
    return {
        "pid": sorted(_normalise_integers(pid, name="pid")),
        "status_filter": status,
        "n_total_events": total_events,
        "n_particle_rows_all_statuses": int(len(all_selected)),
        "n_particle_rows_selected": int(len(selected)),
        "n_events_with_particle": n_events,
        "n_events_with_exactly_two": int(counts.eq(2).sum()),
        "fraction_total_events_with_particle": (n_events / total_events if total_events else np.nan),
        "multiplicity_counts": counts.value_counts().sort_index().astype(int).to_dict(),
        "status_counts": status_counts,
        "mass_mean_gev": float(np.mean(masses)) if masses.size else np.nan,
        "mass_std_gev": float(np.std(masses)) if masses.size else np.nan,
        "mass_median_gev": float(np.median(masses)) if masses.size else np.nan,
    }

def calc_four_momentum(df: pd.DataFrame, *, mass: float | str = MUON_MASS_GEV) -> pd.DataFrame:
    context = "calc_four_momentum"
    _required(df, ("PT", "Eta", "Phi"), context)
    result = df.copy()
    pt = result["PT"].to_numpy(dtype=np.float64)
    eta = result["Eta"].to_numpy(dtype=np.float64)
    phi = result["Phi"].to_numpy(dtype=np.float64)
    if isinstance(mass, str):
        _required(result, (mass,), context)
        particle_mass = result[mass].to_numpy(dtype=np.float64)
    else:
        particle_mass = float(mass)
        if particle_mass < 0:
            raise ValueError("La masa no puede ser negativa.")
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    momentum_squared = (px**2 + py**2 + pz**2)
    result["PX"] = px
    result["PY"] = py
    result["PZ"] = pz
    result["P"] = np.sqrt(momentum_squared)
    result["E"] = np.sqrt(momentum_squared + particle_mass**2)
    return result

def select_muons(
    muons: pd.DataFrame,
    *,
    min_pt: float = 0.0,
    max_abs_eta: float | None = None,
    max_isolation: float | None = None,
    charge: int | None = None
) -> pd.DataFrame:
    if min_pt < 0:
        raise ValueError("'min_pt' no puede ser negativo.")
    if charge not in (None, -1, 1):
        raise ValueError("'charge'= None | -1 | 1")
    _required(muons, ("event_id", "PT"), "select_muons")
    mask = muons["PT"].ge(min_pt)
    if max_abs_eta is not None:
        _required(muons, ("Eta",), "select_muons")
        mask &= muons["Eta"].abs().le(max_abs_eta)
    if max_isolation is not None:
        _required(muons, ("IsolationVar",), "select_muons")
        mask &= muons["IsolationVar"].le(max_isolation)
    if charge is not None:
        _required(muons, ("Charge",), "select_muons")
        mask &= muons["Charge"].eq(charge)
    return muons.loc[mask].copy()

def select_jets(
    jets: pd.DataFrame,
    *,
    min_pt: float = 0.0,
    max_abs_eta: float | None = None,
    require_btag: bool = False,
) -> pd.DataFrame:
    _required(jets, ("event_id", "PT"), "select_jets")
    mask = jets["PT"].ge(min_pt)
    if max_abs_eta is not None:
        _required(jets, ("Eta",), "select_jets")
        mask &= jets["Eta"].abs().le(max_abs_eta)
    if require_btag:
        _required(jets, ("BTag",), "select_jets")
        mask &= jets["BTag"].ne(0)
    return jets.loc[mask].copy()

def add_selected_counts(
    events: pd.DataFrame,
    *,
    muons: pd.DataFrame | None = None,
    jets: pd.DataFrame | None = None,
    electrons: pd.DataFrame | None = None,
    photons: pd.DataFrame | None = None,
) -> pd.DataFrame:
    context = "add_selected_counts"
    _required(events, ("event_id",), context)
    result = events.copy()
    # collections = (
    #     ("Muon", muons),
    #     ("Jet", jets),
    #     ("Electron", electrons),
    #     ("Photons", photons),
    # )
    # for name, objects in collections:
    #     if objects is None:
    #         continue
    #     _required(objects, ("event_id",), context)
    #     counts = objects.groupby("event_id", sort=False).size()
    #     result[f"n_{name}_selected"] = (result["event_id"].map(counts).fillna(0).astype(np.int64))
    for name, objects in (('Muon', muons), ('Jet', jets), ('Electron', electrons), ('Photon', photons)):
        if objects is None:
            continue
        _required(objects, ('event_id',), context)
        keys = [column for column in _event_keys(result) if column in objects]
        counts = objects.groupby(keys, sort=False).size().rename(f'n_{name}_selected')
        result = result.merge(counts, on=keys, how='left', validate='one_to_one')
        result[f'n_{name}_selected'] = (result[f'n_{name}_selected'].fillna(0).astype(np.int64))
    return result

def filter_events(
    events: pd.DataFrame,
    *,
    min_muons: int = 1,
    min_jets: int = 0,
    max_electrons: int | None = None,
    use_selected: bool = False,
) -> pd.DataFrame:
    context = "filter_events"
    _required(events, ("event_id",), context)
    mask = pd.Series(True, index=events.index)
    def count(collection: str) -> pd.Series:
        selected = (f"n_{collection}_selected")
        original = (f"n_{collection}")
        column = selected if use_selected and selected in events.columns else original
        _required(events, (column,), context)
        return events[column]
    if min_muons:
        mask &= count("Muon").ge(min_muons)
    if min_jets:
        mask &= count("Jet").ge(min_jets)
    if max_electrons is not None:
        mask &= count("Electron").le(max_electrons)
    return events.loc[mask].copy()

def leading_object(objects: pd.DataFrame) -> pd.DataFrame:
    _required(objects, ("event_id", "PT"), "leading_object")
    keys = _event_keys(objects)
    columns = ["event_id", "PT"]
    ascending = [*[True] * len(keys), False]
    if "object_index" in objects.columns:
        columns.append("object_index")
        ascending.append(True)
    return (
        objects
        .sort_values(columns, ascending=ascending, kind="stable")
        .drop_duplicates(keys)
        .reset_index(drop=True)
    )

def calc_dimuon_variables(dimuons: pd.DataFrame, *, muon_mass: float = MUON_MASS_GEV,) -> pd.DataFrame:
    context = "calc_dimuon_variables"
    required = tuple(f"{side}_{attribute}" for side in ("muplus", "muminus") for attribute in ("pt", "eta", "phi"))
    _required(dimuons, required, context)
    result = dimuons.copy()
    for side in ("muplus", "muminus"):
        # pt = result[f"{side}_pt"].to_numpy(dtype=np.float64)
        # eta = result[f"{side}_eta"].to_numpy(dtype=np.float64)
        # phi = result[f"{side}_phi"].to_numpy(dtype=np.float64)
        pt = _float_column(result, f'{side}_pt', context)
        eta = _float_column(result, f'{side}_eta', context)
        phi = _float_column(result, f'{side}_phi', context)
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)
        pz = pt*np.sinh(eta)
        momentum_squared = (px**2 + py**2 + pz**2)
        e = np.sqrt(momentum_squared + muon_mass**2)
        result[f"{side}_px"] = px
        result[f"{side}_py"] = py
        result[f"{side}_pz"] = pz
        result[f"{side}_p"] = np.sqrt(momentum_squared)
        result[f"{side}_e"] = e
    px = (result["muplus_px"].to_numpy() + result["muminus_px"].to_numpy())
    py = (result["muplus_py"].to_numpy() + result["muminus_py"].to_numpy())
    pz = (result["muplus_pz"].to_numpy() + result["muminus_pz"].to_numpy())
    e = (result["muplus_e"].to_numpy() + result["muminus_e"].to_numpy())
    pt = np.hypot(px, py)
    delta_eta = (result["muplus_eta"].to_numpy() - result["muminus_eta"].to_numpy())
    result["dimuon_px"] = px
    result["dimuon_py"] = py
    result["dimuon_pz"] = pz
    result["dimuon_e"] = e
    result["dimuon_pt"] = pt
    result["dimuon_phi"] = np.arctan2(py, px)
    result["dimuon_eta"] = np.arcsinh(np.divide(pz, pt, out=np.full(len(result), np.nan), where=pt > 0))
    result["dimuon_mass"] = invariant_mass(e, px, py, pz)
    result["delta_eta"] = delta_eta
    result["delta_phi"] = delta_phi(result['muplus_phi'], result['muminus_phi'])
    result["delta_r"] = delta_r(result['muplus_eta'], result['muplus_phi'], result['muminus_eta'], result['muminus_phi'])
    charge_columns = {"muplus_charge", "muminus_charge"}
    if charge_columns.issubset(result.columns):
        result["dimuon_charge"] = (result["muplus_charge"] + result["muminus_charge"])
    return result

def build_dimuons(muons: pd.DataFrame, *, events: pd.DataFrame | None = None, muon_mass: float = MUON_MASS_GEV) -> pd.DataFrame:
    context = "build_dimuons"
    _required(muons, ("event_id", "object_index", "PT", "Eta", "Phi", "Charge"), context)
    keys = _event_keys(muons)
    # excluded = {"event_id", "source_file_id", "source_entry",}
    excluded = {*keys, 'source_entry'}
    attributes = [column for column in muons.columns if column not in excluded]
    rc = {column: _snake_case(column) for column in attributes}
    if len(set(rc.values())) != len(rc):
        raise ValueError(f"{context}: Conversión de nombres genera nombres duplicados.")
    def signed(charge: int, prefix: str) -> pd.DataFrame:
        # if charge > 0:
        #     mask = muons["Charge"].gt(0)
        # else:
        #     mask = muons["Charge"].lt(0)
        mask = muons['Charge'].gt(0) if charge > 0 else muons['Charge'].lt(0)
        frame = muons.loc[mask, ["event_id", *attributes]].copy()
        renamed = {column: (f"{prefix}_{rc[column]}") for column in attributes}
        return frame.rename(columns=renamed)
    positive = signed(1, "muplus")
    negative = signed(-1, "muminus")
    candidates = positive.merge(negative, on="event_id", how="inner", validate="many_to_many", sort=False,)
    candidates["candidate_index"] = (candidates.groupby("event_id", sort=False).cumcount().astype(np.int32))
    if events is not None:
        merge_keys = [column for column in keys if column in events.columns]
        _required(events, ("event_id",), context)
        # if events['event_id'].duplicated().any():
        if events.duplicated(merge_keys).any():
            raise ValueError(f"{context}: Events contiene claves duplicadas.")
        # candidates = candidates.merge(events, on="event_id", how="left", validate="many_to_one", sort=False,)
        candidates = candidates.merge(events, on=merge_keys, how="left", validate="many_to_one", sort=False)
    # else:
    elif 'source_entry' in muons.columns:
        # identity_columns = [column for column in ("source_file_id", "source_entry") if column in muons.columns]
        # if identity_columns:
        #     identity = (muons[["event_id", *identity_columns]].drop_duplicates("event_id"))
        #     candidates = candidates.merge(identity, on="event_id", how="left", validate="many_to_one", sort=False,)
        identity = muons[[*keys, 'source_entry']].drop_duplicates(keys)
        candidates = candidates.merge(identity, on=keys, how='left', validate='many_to_one')
    return calc_dimuon_variables(candidates, muon_mass=muon_mass)

def select_dimuons(
    dimuons: pd.DataFrame,
    *,
    mass_window: tuple[float, float] | None = None,
    min_muon_pt: float | None = None,
    max_abs_muon_eta: float | None = None,
    max_delta_r: float | None = None,
    max_isolation: float | None = None
) -> pd.DataFrame:
    context = "select_dimuons"
    mask = pd.Series(True, index=dimuons.index)
    if mass_window is not None:
        _required(dimuons, ("dimuon_mass",), context)
        minimum, maximum = mass_window
        if minimum > maximum:
            raise ValueError("mass_window: el mínimo supera al máximo.")
        mask &= dimuons["dimuon_mass"].between(minimum, maximum)
    if min_muon_pt is not None:
        _required(dimuons, ("muplus_pt", "muminus_pt"), context)
        mask &= (dimuons["muplus_pt"].ge(min_muon_pt) & dimuons["muminus_pt"].ge(min_muon_pt))
    if max_abs_muon_eta is not None:
        _required(dimuons, ("muplus_eta", "muminus_eta"), context)
        mask &= (dimuons["muplus_eta"].abs().le(max_abs_muon_eta) & dimuons["muminus_eta"].abs().le(max_abs_muon_eta))
    if max_delta_r is not None:
        _required(dimuons, ("delta_r",), context)
        mask &= dimuons["delta_r"].le(max_delta_r)
    if max_isolation is not None:
        _required(dimuons, ("muplus_isolation", "muminus_isolation"), context)
        mask &= (dimuons["muplus_isolation"].le(max_isolation) & dimuons["muminus_isolation"].le(max_isolation))
    return dimuons.loc[mask].copy()

def calc_dimuon_observables(dimuons: pd.DataFrame, *, muon_mass: float = MUON_MASS_GEV) -> pd.DataFrame:
    result = calc_dimuon_variables(dimuons, muon_mass=muon_mass)
    result['dimuon_mass2'] = (result["dimuon_mass"]**2)
    result['muon_sum_pt'] = (result['muplus_pt'] + result['muminus_pt'])
    pt_sum = result['muon_sum_pt'].to_numpy(dtype=np.float64)
    pt_difference = (result["muplus_pt"] - result['muminus_pt']).abs().to_numpy(dtype=np.float64)
    result['pt_balance'] = np.divide(pt_difference, pt_sum, out=np.full(len(result), np.nan), where=pt_sum != 0)
    result['acoplanarity'] = (np.pi - result['delta_phi'].abs())
    e = result['dimuon_e'].to_numpy(dtype=np.float64)
    pz = result['dimuon_pz'].to_numpy(dtype=np.float64)
    numerator = e + pz
    denominator = e - pz
    # valid_rapidity = ((numerator > 0) & (denominator > 0))
    # result['dimuon_rapidity'] = np.divide(numerator, denominator, out=np.full(len(result), np.nan), where=valid_rapidity)
    # result['dimuon_rapidity'] = (0.5*np.log(result['dimuon_rapidity']))
    result['dimuon_rapidity'] = np.where((numerator > 0) & (denominator > 0), 0.5 * np.log(numerator/denominator), np.nan)
    result['cos_theta_star'] = np.tanh((result['muminus_eta'] - result['muplus_eta'])/2.0)
    return result

def compute_counting_asymmetry(n_positive: int, n_negative: int) -> dict[str, float | int]:
    total = (n_positive + n_negative)
    if total == 0:
        return {
            "A": np.nan,
            "sigma": np.nan,
            "significance": np.nan,
            "N_positive": 0,
            "N_negative": 0,
        }
    asymmetry = (n_negative - n_positive) / total
    sigma = np.sqrt((1 - asymmetry**2) / total)
    significance = (asymmetry / sigma if sigma > 0 else np.nan)
    return {
        "A": asymmetry,
        "sigma": sigma,
        "significance": significance,
        "N_positive": n_positive,
        "N_negative": n_negative,
    }

def compute_charge_asymmetry(objects: pd.DataFrame, *, charge_column: str = 'Charge') -> dict[str, float | int]:
    _required(objects, (charge_column,), 'compute_charge_asymmetry')
    n_positive = int(objects[charge_column].gt(0).sum())
    n_negative = int(objects[charge_column].lt(0).sum())
    return compute_counting_asymmetry(n_positive, n_negative)

def print_asymmetry(label: str, result: dict[str, float | int]) -> None:
    print(f"\n======= {label} =======")
    print(f" N+ = {result['Np']:,}       |       N- = {result['Nm']:,}")
    print(f" A = {result['A']:+.4f} ± {result['sigma']:.4f}")
    print(f" Significancia = {result['significance']:+.2f} σ")

def build_two_muon_system(muons: pd.DataFrame, *, muon_mass: float = MUON_MASS_GEV, require_two: bool = True) -> pd.DataFrame:
    context = "build_two_muon_system"
    _required(muons, ("event_id", "object_index", "PT", "Eta", "Phi", "Charge"), context)
    keys = _event_keys(muons)
    counts = muons.groupby(keys, sort=False).size()
    valid_events = counts.index[counts.eq(2) if require_two else counts.ge(2)]
    if len(valid_events) == 0:
        raise ValueError("No hay eventos con el número de muones requerido.")
    selected = muons.set_index(keys).loc[valid_events].reset_index()
    selected = selected.sort_values([*keys, "PT", "object_index"], ascending=[*[True] * len(keys), False, True], kind="stable",)
    selected = selected.loc[selected.groupby(keys, sort=False).cumcount().lt(2)].copy()
    selected["__rank"] = selected.groupby(keys, sort=False).cumcount()
    def side(rank: int, prefix: str) -> pd.DataFrame:
        columns = ["object_index", "PT", "Eta", "Phi", "Charge"]
        frame = selected.loc[selected["__rank"].eq(rank), [*keys, *columns]].copy()
        return frame.rename(columns={column: f"{prefix}_{_snake_case(column)}" for column in columns})
    result = side(0, "mu1").merge(side(1, "mu2"), on=keys, how="inner", validate="one_to_one")
    for prefix in ("mu1", "mu2"):
        pt = result[f"{prefix}_pt"].to_numpy(dtype=np.float64)
        eta = result[f"{prefix}_eta"].to_numpy(dtype=np.float64)
        phi = result[f"{prefix}_phi"].to_numpy(dtype=np.float64)
        result[f"{prefix}_px"] = pt * np.cos(phi)
        result[f"{prefix}_py"] = pt * np.sin(phi)
        result[f"{prefix}_pz"] = pt * np.sinh(eta)
        result[f"{prefix}_energy"] = np.sqrt((pt * np.cosh(eta)) ** 2 + muon_mass**2)
    px = result["mu1_px"] + result["mu2_px"]
    py = result["mu1_py"] + result["mu2_py"]
    pz = result["mu1_pz"] + result["mu2_pz"]
    energy = result["mu1_energy"] + result["mu2_energy"]
    result["dimuon_mass"] = invariant_mass(energy, px, py, pz)
    result["dimuon_pt"] = np.hypot(px, py)
    result["delta_r"] = delta_r(result["mu1_eta"], result["mu1_phi"], result["mu2_eta"], result["mu2_phi"])
    result["charge_product"] = result["mu1_charge"] * result["mu2_charge"]
    result["is_opposite_sign"] = result["charge_product"].lt(0)
    return result