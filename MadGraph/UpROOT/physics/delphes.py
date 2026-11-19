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
    "D0": "d0",
    "DZ": "dz",
    "ErrorD0": "error_d0",
    "ErrorDZ": "error_dz",
    "IsolationVar": "isolation",
    "IsolationVarRhoCorr": "isolation_rho_corr",
    "Particle_ref": "particle_ref",
    "PX": "px",
    "PY": "py",
    "PZ": "pz",
    "P": "p",
    "E": "energy",
}

def _required(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    context: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        formatted = ", ".join(missing)
        raise KeyError(f"{context}: columnas faltantes: {formatted}")

def _snake_case(name: str) -> str:
    if name in _COLUMN_NAMES:
        return _COLUMN_NAMES[name]
    converted = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return converted.lower()

def _float_column(
    df: pd.DataFrame,
    column: str,
    context: str
) -> np.ndarray:
    _required(df, (column,), context)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)

def _point_components(
    df: pd.DataFrame,
    columns: tuple[str, str, str],
    *,
    constant: tuple[float, float, float] | None,
    context: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if constant is not None:
        if len(constant) != 3:
            raise ValueError(f"{context}: el punto constante debe tener tres coordenadas.")
        values = np.asarray(constant, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{context}: coordenadas constantes deben ser finitas.")
        return tuple(np.full(len(df), value, dtype=np.float64) for value in values)
    _required(df, columns, context)
    return tuple(_float_column(df, column, context) for column in columns)

def _resolution_values(
    df: pd.DataFrame,
    resolution: float | str,
    *,
    context: str
) -> np.ndarray:
    if isinstance(resolution, str):
        return _float_column(df, resolution, context)
    value = float(resolution)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{context}: resolución constante debe ser positiva y finita.")
    return np.full(len(df), value, dtype=np.float64)

def _divide_valid(
    numerator: np.ndarray,
    denominator: np.ndarray
) -> np.ndarray:
    valid = (np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0))
    return np.divide(numerator, denominator, out=np.full(len(numerator), np.nan, dtype=np.float64), where=valid)

def _normalise_parent_pids(parent_pid: int | Iterable[int] | None) -> set[int] | None:
    if parent_pid is None:
        return None
    if isinstance(parent_pid, (int, np.integer)):
        return {int(parent_pid)}
    if isinstance(parent_pid, (str, bytes)):
        raise TypeError('parent_pid debe ser un entero o un iterable de enteros.')
    values = {int(value) for value in parent_pid}
    if not values:
        raise ValueError('parent_pid: vacío')
    return values

def _event_key_columns(df: pd.DataFrame) -> list[str]:
    _required(df, ('event_id',), "_event_key_columns")
    return [column for column in ('source_file_id', 'event_id') if column in df.columns]

def _four_vector_from_objects(
    objects: pd.DataFrame,
    *,
    context: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cartesian = (
        ('E', 'Px', 'Py', 'Pz'),
        ('E', 'PX', 'PY', 'PZ'),
        ('energy', 'px', 'py', 'pz'),
    )
    for energy_column, px_column, py_column, pz_column in cartesian:
        if {energy_column, px_column, py_column, pz_column}.issubset(objects.columns):
            return tuple(_float_column(objects, column, context) for column in (energy_column, px_column, py_column, pz_column))
    _required(objects, ('PT', 'Eta', 'Phi'), context)
    pt = _float_column(objects, 'PT', context)
    eta = _float_column(objects, 'Eta', context)
    phi = _float_column(objects, 'Phi', context)
    if 'Mass' in objects.columns:
        mass = _float_column(objects, 'Mass', context)
    else:
        mass = np.zeros(len(objects), dtype=np.float64)
    px = pt*np.cos(phi)
    py = pt*np.sin(phi)
    pz = pt*np.sinh(eta)
    energy = np.sqrt(np.maximum(px**2 + py**2 + pz**2 + mass**2, 0.0))
    return energy, px, py, pz

def select_particles(
    particles: pd.DataFrame,
    pid: int | Iterable[str],
    *,
    match_absolute_pid: bool = True,
    status: int | Iterable[int] | None = None
) -> pd.DataFrame:
    context = 'select_particles'
    _required(particles, ('event_id', 'PID'), context)
    selected_pids = _normalise_parent_pids(pid)
    assert selected_pids is not None
    pid_values = pd.to_numeric(particles['PID'], errors='coerce')
    if match_absolute_pid:
        mask = pid_values.abs().isin({abs(value) for value in selected_pids})
    else:
        mask = pid_values.isin(selected_pids)
    if status is not None:
        _required(particles, ("Status",), context)
        if isinstance(status, (int, np.integer)):
            statuses = {int(status)}
        elif isinstance(status, (str, bytes)):
            raise TypeError("status debe ser un entero o un iterable de enteros.")
        else:
            statuses = {int(value) for value in status}
        if not statuses:
            raise ValueError('status no puede ser un iterable vacio.')
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
    context = 'particle_pairs'
    selected = select_particles(
        particles,
        pid,
        match_absolute_pid=match_pid,
        status=status
    )
    _required(selected, ('object_index',), context)
    if selected.empty:
        return pd.DataFrame()
    event_keys = _event_key_columns(selected)
    energy, px, py, pz = _four_vector_from_objects(selected, context=context)
    selected = selected.copy()
    selected['__energy'] = energy
    selected['__px'] = px
    selected['__py'] = py
    selected['__pz'] = pz
    selected['__pt'] = np.hypot(px, py)
    selected['__phi'] = np.arctan2(py, px)
    selected['__eta'] = np.arcsinh(np.divide(pz, selected['__pt'].to_numpy(dtype=np.float64), out=np.full(len(selected), np.nan), where=selected['__pt'].to_numpy(dtype=np.float64) > 0,))
    counts = selected.groupby(event_keys, sort=False).size().rename('n_truth_particles')
    selected = selected.merge(counts, on=event_keys, how='left', validate='many_to_one')
    if require_two:
        selected = selected.loc[selected['n_truth_particles'].eq(2)].copy()
    else:
        selected = selected.loc[selected["n_truth_particles"].ge(2)].copy()
    if selected.empty:
        return pd.DataFrame()
    selected = selected.sort_values(
        [*event_keys, '__pt', 'object_index'],
        ascending=[*[True]*len(event_keys), False, True],
        kind='stable'
    )
    selected['__pair_rank'] = selected.groupby(event_keys, sort=False).cumcount()
    selected = selected.loc[selected['__pair_rank'].lt(2)].copy()
    canonical = {
        'object_index': 'index',
        'PID': 'pid',
        'Status': 'status',
        'Mass': 'mass',
        'X': 'x',
        'Y': 'y',
        'Z': 'z',
        'T': 't',
        'root_uid': 'root_uid',
        '__energy': 'energy',
        '__px': 'px',
        '__py': 'py',
        '__pz': 'pz',
        '__pt': 'pt',
        '__eta': 'eta',
        '__phi': 'phi'
    }
    value_columns = [column for column in canonical if column in selected.columns]
    def ranked(rank: int, prefix: str) -> pd.DataFrame:
        frame = selected.loc[selected['__pair_rank'].eq(rank), [*event_keys, 'n_truth_particles', *value_columns]].copy()
        return frame.rename(columns={column: f'{prefix}_{canonical[column]}' for column in value_columns})
    first = ranked(0, 'truth1')
    second = ranked(1, 'truth2').drop(columns='n_truth_particles')
    result = first.merge(second, on=event_keys, how='inner', validate='one_to_one', sort=False)
    e = result['truth1_energy'].to_numpy(dtype=np.float64) + result['truth2_energy'].to_numpy(dtype=np.float64)
    pair_px = result['truth1_px'].to_numpy(dtype=np.float64) + result['truth2_px'].to_numpy(dtype=np.float64)
    pair_py = result['truth1_py'].to_numpy(dtype=np.float64) + result['truth2_py'].to_numpy(dtype=np.float64)
    pair_pz = result['truth1_pz'].to_numpy(dtype=np.float64) + result['truth2_pz'].to_numpy(dtype=np.float64)
    pair_pt = np.hypot(pair_px, pair_py)
    result['truth_pair_energy'] = e
    result['truth_pair_px'] = pair_px
    result['truth_pair_py'] = pair_py
    result['truth_pair_pz'] = pair_pz
    result['truth_pair_mass'] = invariant_mass(e, pair_px, pair_py, pair_pz)
    result['truth_pair_pt'] = pair_pt
    result['truth_pair_eta'] = np.arcsinh(np.divide(pair_pz, pair_pt, out=np.full(len(result), np.nan), where=pair_pt > 0))
    result['truth_pair_phi'] = np.arctan2(pair_py, pair_px)
    numerator = e + pair_pz
    denominator = e - pair_pz
    result['truth_pair_rapidity'] = np.where((numerator > 0) & (denominator > 0), 0.5*np.log(numerator/denominator), np.nan)
    result['truth_pair_delta_phi'] = delta_phi(result['truth1_phi'], result['truth2_phi'])
    result['truth_pair_delta_r'] = delta_r(result['truth1_eta'], result['truth1_phi'], result['truth2_eta'], result['truth2_phi'])
    return result

def summarize_mediator(
    particles: pd.DataFrame,
    *,
    pid: int | Iterable[int],
    status: int | Iterable[int] | None = None,
    match_pid: bool = True
) -> dict[str, object]:
    context = 'summarize_mediator'
    _required(particles, ('event_id', 'PID'), context)
    all_mediators = select_particles(particles, pid, match_absolute_pid=match_pid)
    selected = select_particles(particles, pid, match_absolute_pid=match_pid, status=status)
    event_keys = _event_key_columns(particles)
    total_events = int(particles[event_keys].drop_duplicates().shape[0])
    selected_counts = selected.groupby(event_keys, sort=False).size() if not selected.empty else pd.Series(dtype=np.int64)
    events_with_mediator = int(len(selected_counts))
    exact_two = int(selected_counts.eq(2).sum())
    status_counts = (
        pd.to_numeric(all_mediators["Status"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
        if "Status" in all_mediators.columns
        else {}
    )
    multiplicity_counts = (selected_counts.value_counts().sort_index().astype(int).to_dict() if not selected_counts.empty else {})
    masses = (pd.to_numeric(selected["Mass"], errors="coerce").dropna().to_numpy(dtype=np.float64) if "Mass" in selected.columns else np.asarray([], dtype=np.float64))
    if {"D1", "D2"}.issubset(selected.columns):
        daughter1 = pd.to_numeric(selected["D1"], errors="coerce").to_numpy(dtype=np.float64)
        daughter2 = pd.to_numeric(selected["D2"], errors="coerce").to_numpy(dtype=np.float64)
        has_daughter_indices = (
            np.isfinite(daughter1)
            & np.isfinite(daughter2)
            & (daughter1 >= 0)
            & (daughter2 >= daughter1)
        )
    else:
        has_daughter_indices = np.zeros(len(selected), dtype=bool)
    return {
        "pid": sorted(_normalise_parent_pids(pid) or []),
        "status_filter": status,
        "n_total_events": total_events,
        "n_mediator_rows_all_statuses": int(len(all_mediators)),
        "n_mediator_rows_selected": int(len(selected)),
        "n_events_with_mediator": events_with_mediator,
        "n_events_with_exactly_two": exact_two,
        "fraction_total_events_with_mediator": (
            float(events_with_mediator/total_events) if total_events else np.nan
        ),
        "fraction_selected_events_with_exactly_two": (
            float(exact_two/events_with_mediator) if events_with_mediator else np.nan
        ),
        "status_counts": status_counts,
        "multiplicity_counts": multiplicity_counts,
        "mass_mean_gev": float(np.mean(masses)) if masses.size else np.nan,
        "mass_std_gev": float(np.std(masses)) if masses.size else np.nan,
        "mass_median_gev": float(np.median(masses)) if masses.size else np.nan,
        "n_mediators_with_daughter_indices": int(np.count_nonzero(has_daughter_indices)),
        "fraction_mediators_with_daughter_indices": (
            float(np.mean(has_daughter_indices)) if len(selected) else np.nan
        ),
    }
    
def attach_pair_recoil(
    pairs: pd.DataFrame,
    *,
    missing_et: pd.DataFrame | None = None,
    jets: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    context = "attach_pair_recoil"
    _required(
        pairs,
        ("event_id", "truth_pair_pt", "truth_pair_phi"),
        context,
    )
    result = pairs.copy()
    event_keys = _event_key_columns(result)
    if events is not None:
        event_merge_keys = [column for column in event_keys if column in events.columns]
        if "event_id" not in event_merge_keys:
            raise KeyError(f"{context}: Events requiere event_id.")
        if events.duplicated(event_merge_keys).any():
            raise ValueError(f"{context}: Events contiene eventos duplicados.")
        event_columns = [column for column in events.columns if column in event_merge_keys or column not in result.columns]
        result = result.merge(events[event_columns], on=event_merge_keys, how="left", validate="one_to_one", sort=False,)
    if missing_et is not None:
        met_keys = [column for column in event_keys if column in missing_et.columns]
        if "event_id" not in met_keys:
            raise KeyError(f"{context}: MissingET requiere event_id.")
        met_column = next((column for column in ("MET", "PT") if column in missing_et.columns), None)
        if met_column is None:
            raise KeyError(f"{context}: MissingET requiere MET o PT.")
        met_fields = [column for column in (met_column, "Phi", "Eta") if column in missing_et.columns]
        met_frame = (
            missing_et[[*met_keys, *met_fields]]
            .sort_values([*met_keys, met_column], ascending=[*[True]*len(met_keys), False], kind="stable")
            .drop_duplicates(met_keys, keep="first")
            .rename(columns={
                met_column: "missing_et",
                "Phi": "missing_et_phi",
                "Eta": "missing_et_eta",
            })
        )
        result = result.merge(met_frame, on=met_keys, how="left", validate="one_to_one", sort=False,)
    if jets is not None:
        jet_keys = [column for column in event_keys if column in jets.columns]
        if "event_id" not in jet_keys:
            raise KeyError(f"{context}: Jets requiere event_id.")
        _required(jets, ("PT",), context)
        jet_fields = [column for column in ("object_index", "PT", "Eta", "Phi", "Mass", "BTag") if column in jets.columns]
        leading_jet = (
            jets[[*jet_keys, *jet_fields]]
            .sort_values([*jet_keys, "PT"], ascending=[*[True]*len(jet_keys), False], kind="stable")
            .drop_duplicates(jet_keys, keep="first")
            .rename(columns={column: f"leading_jet_{_snake_case(column)}" for column in jet_fields})
        )
        result = result.merge(leading_jet, on=jet_keys, how="left", validate="one_to_one", sort=False,)
    pair_pt = result["truth_pair_pt"].to_numpy(dtype=np.float64)
    if {"missing_et", "missing_et_phi"}.issubset(result.columns):
        met = result["missing_et"].to_numpy(dtype=np.float64)
        result["delta_phi_pair_met"] = np.abs(delta_phi(result["truth_pair_phi"], result["missing_et_phi"]))
        result["met_minus_truth_pair_pt"] = met - pair_pt
        result["met_response"] = np.divide(met, pair_pt, out=np.full(len(result), np.nan), where=np.isfinite(pair_pt) & (pair_pt > 0),)
        result["pair_met_balance"] = np.divide(np.abs(met-pair_pt), met+pair_pt, out=np.full(len(result), np.nan), where=np.isfinite(met+pair_pt) & ((met+pair_pt) > 0),)
    if {"leading_jet_pt", "leading_jet_phi"}.issubset(result.columns):
        jet_pt = result["leading_jet_pt"].to_numpy(dtype=np.float64)
        result["delta_phi_pair_leading_jet"] = np.abs(delta_phi(result["truth_pair_phi"], result["leading_jet_phi"]))
        result["pair_leading_jet_balance"] = np.divide(np.abs(pair_pt-jet_pt), pair_pt+jet_pt, out=np.full(len(result), np.nan), where=np.isfinite(pair_pt+jet_pt) & ((pair_pt+jet_pt) > 0),)
    return result

def calc_four_momentum(
    df: pd.DataFrame,
    *,
    mass: float | str = MUON_MASS_GEV,
) -> pd.DataFrame:
    _required(df, ("PT", "Eta", "Phi"), "calc_four_momentum")
    result = df.copy()
    pt = result["PT"].to_numpy(dtype=np.float64)
    eta = result["Eta"].to_numpy(dtype=np.float64)
    phi = result["Phi"].to_numpy(dtype=np.float64)
    if isinstance(mass, str):
        _required(result, (mass,), "calc_four_momentum")
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
    max_abs_d0: float | None = None,
    max_abs_dz: float | None = None,
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
    if max_abs_d0 is not None:
        _required(muons, ("D0",), "select_muons")
        mask &= muons["D0"].abs().le(max_abs_d0)
    if max_abs_dz is not None:
        _required(muons, ("DZ",), "select_muons")
        mask &= muons["DZ"].abs().le(max_abs_dz)
    if charge is not None:
        _required(muons, ("Charge",), "select_muons")
        mask &= muons["Charge"].eq(charge)
    return muons.loc[mask].copy()

def impact_parameter_significance(muons: pd.DataFrame) -> pd.DataFrame:
    _required(muons, ('D0', 'DZ', 'ErrorD0', 'ErrorDZ'), 'impact_parameter_significance')
    result = muons.copy()
    d0 = result['D0'].to_numpy(dtype=np.float64)
    dz = result['DZ'].to_numpy(dtype=np.float64)
    error_d0 = result['ErrorD0'].to_numpy(dtype=np.float64)
    error_dz = result['ErrorDZ'].to_numpy(dtype=np.float64)
    result['D0_significance'] = _divide_valid(d0, error_d0)
    result['DZ_significance'] = _divide_valid(dz, error_dz)
    return result

def add_ip_proxies(
    dimuons: pd.DataFrame,
    *,
    require_valid: bool = False
) -> pd.DataFrame:
    context = 'add_ip_proxies'
    required = ('muplus_d0', 'muminus_d0', 'muplus_error_d0', 'muminus_error_d0')
    _required(dimuons, required, context)
    result = dimuons.copy()
    plus_d0 = _float_column(result, 'muplus_d0', context)
    minus_d0 = _float_column(result, 'muminus_d0', context)
    plus_error = _float_column(result, 'muplus_error_d0', context)
    minus_error = _float_column(result, 'muminus_error_d0', context)
    plus_significance = np.abs(_divide_valid(plus_d0, plus_error))
    minus_significance = np.abs(_divide_valid(minus_d0, minus_error))
    valid = np.isfinite(plus_significance) & np.isfinite(minus_significance)
    result['muplus_d0_significance'] = plus_significance
    result['muminus_d0_significance'] = minus_significance
    result['muplus_chi2_ip_proxy'] = plus_significance**2
    result['muminus_chi2_ip_proxy'] = minus_significance**2
    result['sqrt_min_chi2_ip_proxy'] = np.minimum(plus_significance, minus_significance)
    result['min_chi2_ip_proxy'] = result['sqrt_min_chi2_ip_proxy']**2
    result['ip_proxy_valid'] = valid
    if require_valid and not bool(np.any(valid)):
        raise ValueError('add_ip_proxies: sin candidatos con ErrorD0.')
    return result

def attach_dimuon_track_vertices(
    dimuons: pd.DataFrame,
    tracks: pd.DataFrame,
    *,
    track_reference_column: str | None = None,
    muplus_reference_column: str = 'muplus_particle_ref',
    muminus_reference_column: str = 'muminus_particle_ref'
) -> pd.DataFrame:
    context = 'attach_dimuon_track_vertices'
    if track_reference_column is None:
        track_reference_column = next((column for column in ("Particle_ref", 'particle_ref') if column in tracks.columns), None,)
    if track_reference_column is None:
        raise KeyError(f"{context}: Tracks requiere 'Particle_ref' o 'particle_ref'.")
    _required(dimuons, ("event_id", muplus_reference_column, muminus_reference_column), context)
    _required(tracks, ("event_id", track_reference_column, "X", "Y", "Z"), context)
    optional_fields = [column for column in (
        "object_index",
        "Xd",
        "Yd",
        "Zd",
        "XFirstHit",
        "YFirstHit",
        "ZFirstHit",
        "D0",
        "DZ",
        "ErrorD0",
        "ErrorDZ",
        "ErrorD0DZ"
    ) if column in tracks.columns]
    lookup = tracks.loc[tracks[track_reference_column].notna() & tracks[track_reference_column].ne(0), [
        "event_id",
        track_reference_column,
        "X",
        "Y",
        "Z",
        *optional_fields
    ],].copy()
    keys = ["event_id", track_reference_column]
    if lookup.duplicated(keys).any():
        raise ValueError(f"{context}: existen varias trazas para la misma referencia de partícula dentro de un evento.")
    result = dimuons.copy()
    existing_track_cols = [c for side in ('muplus', 'muminus') for c in result.columns if c.startswith(f'{side}_track_')]
    if existing_track_cols:
        result = result.drop(columns=existing_track_cols)
    result["__row_position"] = np.arange(len(result), dtype=np.int64)
    field_names = {
        "object_index": "index",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "Xd": "xd",
        "Yd": "yd",
        "Zd": "zd",
        "XFirstHit": "x_first_hit",
        "YFirstHit": "y_first_hit",
        "ZFirstHit": "z_first_hit",
        "D0": "d0",
        "DZ": "dz",
        "ErrorD0": "error_d0",
        "ErrorDZ": "error_dz",
        "ErrorD0DZ": "error_d0_dz"
    }
    for side, reference_column in (("muplus", muplus_reference_column), ("muminus", muminus_reference_column)):
        renamed = lookup.rename(columns={
            track_reference_column: reference_column, **{
                field: f"{side}_track_{field_names[field]}" for field in lookup.columns if field in field_names
            },
        })
        result = result.merge(renamed, on=["event_id", reference_column], how="left", validate="many_to_one", sort=False)
        result[f"{side}_track_matched"] = result[f"{side}_track_x"].notna()
    result["dimuon_tracks_matched"] = (result["muplus_track_matched"] & result["muminus_track_matched"])
    return (
        result
        .sort_values("__row_position", kind="stable")
        .drop(columns="__row_position")
        .set_axis(dimuons.index)
    )

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
    _required(events, ("event_id",), "add_selected_counts")
    result = events.copy()
    collections = (
        ("Muon", muons),
        ("Jet", jets),
        ("Electron", electrons),
        ("Photons", photons),
    )
    for name, objects in collections:
        if objects is None:
            continue
        _required(objects, ("event_id",), "add_selected_counts")
        counts = objects.groupby("event_id", sort=False).size()
        result[f"n_{name}_selected"] = (result["event_id"].map(counts).fillna(0).astype(np.int64))
    return result

def filter_events(
    events: pd.DataFrame,
    *,
    min_muons: int = 1,
    min_jets: int = 0,
    max_electrons: int | None = None,
    use_selected: bool = False,
) -> pd.DataFrame:
    _required(events, ("event_id",), "filter_events")
    mask = pd.Series(True, index=events.index)
    def count(collection: str) -> pd.Series:
        selected_column = (f"n_{collection}_selected")
        original_column = (f"n_{collection}")
        if (use_selected and selected_column in events.columns):
            column = selected_column
        else:
            column = original_column
        _required(events, (column,), "filter_events")
        return events[column]
    if min_muons > 0:
        mask &= count("Muon").ge(min_muons)
    if min_jets > 0:
        mask &= count("Jet").ge(min_jets)
    if max_electrons is not None:
        mask &= count("Electron").le(max_electrons)
    return events.loc[mask].copy()

def leading_object(objects: pd.DataFrame) -> pd.DataFrame:
    _required(objects, ("event_id", "PT"), "leading_object")
    order_columns = ["event_id", "PT"]
    ascending = [True, False]
    if "object_index" in objects.columns:
        order_columns.append("object_index")
        ascending.append(True)
    return (
        objects
        .sort_values(order_columns, ascending=ascending, kind="stable")
        .drop_duplicates("event_id", keep="first")
        .reset_index(drop=True)
    )

def calc_dimuon_variables(
    dimuons: pd.DataFrame,
    *,
    muon_mass: float = MUON_MASS_GEV,
) -> pd.DataFrame:
    required = tuple(f"{side}_{attribute}" for side in ("muplus", "muminus") for attribute in ("pt", "eta", "phi"))
    _required(dimuons, required, "calc_dimuon_variables")
    result = dimuons.copy()
    for side in ("muplus", "muminus"):
        pt = result[f"{side}_pt"].to_numpy(dtype=np.float64)
        eta = result[f"{side}_eta"].to_numpy(dtype=np.float64)
        phi = result[f"{side}_phi"].to_numpy(dtype=np.float64)
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)
        pz = pt*np.sinh(eta)
        momentum_squared = (px**2 + py**2 + pz**2)
        energy = np.sqrt(momentum_squared + muon_mass**2)
        result[f"{side}_px"] = px
        result[f"{side}_py"] = py
        result[f"{side}_pz"] = pz
        result[f"{side}_p"] = np.sqrt(momentum_squared)
        result[f"{side}_energy"] = energy
    px = (result["muplus_px"].to_numpy() + result["muminus_px"].to_numpy())
    py = (result["muplus_py"].to_numpy() + result["muminus_py"].to_numpy())
    pz = (result["muplus_pz"].to_numpy() + result["muminus_pz"].to_numpy())
    energy = (result["muplus_energy"].to_numpy() + result["muminus_energy"].to_numpy())
    pt = np.hypot(px, py)
    delta_eta = (result["muplus_eta"].to_numpy() - result["muminus_eta"].to_numpy())
    result["dimuon_px"] = px
    result["dimuon_py"] = py
    result["dimuon_pz"] = pz
    result["dimuon_energy"] = energy
    result["dimuon_pt"] = pt
    result["dimuon_phi"] = np.arctan2(py, px)
    result["dimuon_eta"] = np.arcsinh(np.divide(pz, pt, out=np.full(len(result), np.nan), where=pt > 0))
    result["dimuon_mass"] = invariant_mass(energy, px, py, pz)
    result["delta_eta"] = delta_eta
    result["delta_phi"] = delta_phi(result['muplus_phi'], result['muminus_phi'])
    result["delta_r"] = delta_r(result['muplus_eta'], result['muplus_phi'], result['muminus_eta'], result['muminus_phi'])
    charge_columns = {"muplus_charge", "muminus_charge"}
    if charge_columns.issubset(result.columns):
        result["dimuon_charge"] = (result["muplus_charge"] + result["muminus_charge"])
    return result

def build_dimuons(
    muons: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    muon_mass: float = MUON_MASS_GEV,
) -> pd.DataFrame:
    _required(muons, ("event_id", "object_index", "PT", "Eta", "Phi", "Charge"), "build_dimuons")
    excluded = {"event_id", "source_file_id", "source_entry",}
    attributes = [column for column in muons.columns if column not in excluded]
    rc = {column: _snake_case(column) for column in attributes}
    if len(set(rc.values())) != len(rc):
        raise ValueError("Conversión de nombres genera nombres duplicados.")
    def signed_muons(charge: int, prefix: str) -> pd.DataFrame:
        if charge > 0:
            mask = muons["Charge"].gt(0)
        else:
            mask = muons["Charge"].lt(0)
        selected = muons.loc[mask, ["event_id", *attributes]].copy()
        renamed = {column: (f"{prefix}_{rc[column]}") for column in attributes}
        return selected.rename(columns=renamed)
    positive = signed_muons(1, "muplus")
    negative = signed_muons(-1, "muminus")
    candidates = positive.merge(negative, on="event_id", how="inner", validate="many_to_many", sort=False,)
    candidates["candidate_index"] = (candidates.groupby("event_id", sort=False).cumcount().astype(np.int32))
    if events is not None:
        _required(events, ("event_id",), "build_dimuons")
        if events['event_id'].duplicated().any():
            raise ValueError("DataFrame Events contiene 'event_id' duplicados.")
        candidates = candidates.merge(events, on="event_id", how="left", validate="many_to_one", sort=False,)
    else:
        identity_columns = [column for column in ("source_file_id", "source_entry") if column in muons.columns]
        if identity_columns:
            identity = (muons[["event_id", *identity_columns]].drop_duplicates("event_id"))
            candidates = candidates.merge(identity, on="event_id", how="left", validate="many_to_one", sort=False,)
    return calc_dimuon_variables(candidates, muon_mass=muon_mass)

def match_dimuon_truth_parents(
    dimuons: pd.DataFrame,
    particles: pd.DataFrame,
    *,
    parent_pid: int | Iterable[int] | None = None,
    match_absolute_pid: bool = True,
    muplus_reference_column: str = "muplus_particle_ref",
    muminus_reference_column: str = "muminus_particle_ref",
) -> pd.DataFrame:
    context = "match_dimuon_truth_parents"
    _required(dimuons, ("event_id", muplus_reference_column, muminus_reference_column), context,)
    _required(particles, ("event_id", "object_index", "root_uid", "PID", "M1", "M2"), context,)
    result = dimuons.copy()
    event_keys = [column for column in ("source_file_id", "event_id") if column in result.columns and column in particles.columns]
    if "event_id" not in event_keys:
        event_keys.append("event_id")
    row_column = "__dimuon_row"
    base = result.reset_index(drop=True).copy()
    base[row_column] = np.arange(len(base), dtype=np.int64)
    daughter_columns = [
        *event_keys,
        "root_uid",
        "object_index",
        "PID",
        "M1",
        "M2",
    ]
    daughters = particles.loc[particles["root_uid"].notna() & particles["root_uid"].ne(0), daughter_columns,].copy()
    if daughters.duplicated([*event_keys, "root_uid"]).any():
        raise ValueError(f"{context}: root_uid duplicado dentro de un evento.")
    def match_side(side: str, reference_column: str) -> pd.DataFrame:
        references = base[[row_column, *event_keys, reference_column]].rename(columns={reference_column: "__particle_reference"})
        renamed = daughters.rename(columns={
            "root_uid": "__particle_reference",
            "object_index": f"truth_{side}_index",
            "PID": f"truth_{side}_pid",
            "M1": f"__{side}_m1",
            "M2": f"__{side}_m2",
        })
        return references.merge(renamed, on=[*event_keys, "__particle_reference"], how="left", validate="many_to_one", sort=False,).sort_values(row_column, kind="stable")
    plus = match_side("muplus", muplus_reference_column)
    minus = match_side("muminus", muminus_reference_column)
    allowed_pids = _normalise_parent_pids(parent_pid)
    parent_fields = [
        column
        for column in (
            "root_uid", "PID", "Status", "Mass", "E", "Px", "Py", "Pz",
            "PX", "PY", "PZ", "PT", "Eta", "Phi", "Rapidity", "X", "Y", "Z", "T",
        )
        if column in particles.columns
    ]
    parents = particles[[*event_keys, "object_index", *parent_fields]].copy()
    if parents.duplicated([*event_keys, "object_index"]).any():
        raise ValueError(f"{context}: object_index duplicado dentro de un evento.")
    parent_field_names = {
        "root_uid": "root_uid",
        "PID": "pid",
        "Status": "status",
        "Mass": "mass",
        "E": "energy",
        "Px": "px",
        "Py": "py",
        "Pz": "pz",
        "PX": "px",
        "PY": "py",
        "PZ": "pz",
        "PT": "pt",
        "Eta": "eta",
        "Phi": "phi",
        "Rapidity": "rapidity",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "T": "t",
    }
    parent_rename = {"object_index": "truth_parent_index", **{column: f"truth_parent_{parent_field_names[column]}" for column in parent_fields},}
    parents = parents.rename(columns=parent_rename)
    candidates: list[pd.DataFrame] = []
    priority = 0
    for plus_mother in ("__muplus_m1", "__muplus_m2"):
        plus_values = pd.to_numeric(plus[plus_mother], errors="coerce").to_numpy(dtype=np.float64)
        for minus_mother in ("__muminus_m1", "__muminus_m2"):
            minus_values = pd.to_numeric(minus[minus_mother], errors="coerce").to_numpy(dtype=np.float64)
            common = (np.isfinite(plus_values) & np.isfinite(minus_values) & (plus_values >= 0) & (plus_values == minus_values))
            if np.any(common):
                candidate = plus.loc[common, [row_column, *event_keys]].copy()
                candidate["truth_parent_index"] = plus_values[common].astype(np.int64)
                candidate["__parent_priority"] = priority
                candidates.append(candidate)
            priority += 1
    if candidates:
        parent_candidates = pd.concat(candidates, ignore_index=True).merge(parents, on=[*event_keys, "truth_parent_index"], how="left", validate="many_to_one", sort=False,)
        if allowed_pids is not None:
            parent_pid_values = pd.to_numeric(parent_candidates["truth_parent_pid"], errors="coerce")
            if match_absolute_pid:
                allowed = parent_pid_values.abs().isin({abs(value) for value in allowed_pids})
            else:
                allowed = parent_pid_values.isin(allowed_pids)
            parent_candidates = parent_candidates.loc[allowed]
        chosen = (
            parent_candidates
            .sort_values([row_column, "__parent_priority"], kind="stable")
            .drop_duplicates(row_column, keep="first")
        )
    else:
        chosen = pd.DataFrame(columns=[row_column, *parent_rename.values()])
    side_columns = pd.DataFrame({
        row_column: base[row_column],
        "truth_muplus_index": pd.to_numeric(plus["truth_muplus_index"], errors="coerce"),
        "truth_muplus_pid": pd.to_numeric(plus["truth_muplus_pid"], errors="coerce"),
        "truth_muminus_index": pd.to_numeric(minus["truth_muminus_index"], errors="coerce"),
        "truth_muminus_pid": pd.to_numeric(minus["truth_muminus_pid"], errors="coerce"),
    })
    generated_columns = [column for column in base.columns if column.startswith("truth_parent_") or column.startswith("truth_muplus_") or column.startswith("truth_muminus_")]
    base = base.drop(columns=generated_columns, errors="ignore")
    chosen_columns = [column for column in chosen.columns if column not in {*event_keys, "__parent_priority"}]
    base = base.merge(side_columns, on=row_column, how="left", validate="one_to_one", sort=False)
    base = base.merge(chosen[chosen_columns], on=row_column, how="left", validate="one_to_one", sort=False,)
    base["truth_daughters_matched"] = (base["truth_muplus_index"].notna() & base["truth_muminus_index"].notna())
    base["truth_parent_found"] = base.get("truth_parent_index", pd.Series(np.nan, index=base.index)).notna()
    for column in ("truth_muplus_index", "truth_muplus_pid", "truth_muminus_index", "truth_muminus_pid", "truth_parent_index", "truth_parent_pid",):
        if column in base.columns:
            base[column] = pd.array(base[column], dtype="Int64")
    base = base.sort_values(row_column, kind="stable").drop(columns=row_column)
    base.index = dimuons.index
    return base

def select_truth_matched_dimuons(
    dimuons: pd.DataFrame,
    *,
    parent_pid: int | Iterable[int] | None = None,
    match_absolute_pid: bool = True,
    unique_parent: bool = True,
    target_mass: float | None = None,
) -> pd.DataFrame:
    context = "select_truth_matched_dimuons"
    _required(dimuons, ("truth_parent_found", "truth_parent_index", "truth_parent_pid"), context)
    mask = dimuons["truth_parent_found"].fillna(False).astype(bool)
    allowed_pids = _normalise_parent_pids(parent_pid)
    if allowed_pids is not None:
        pid_values = pd.to_numeric(dimuons["truth_parent_pid"], errors="coerce")
        if match_absolute_pid:
            mask &= pid_values.abs().isin({abs(value) for value in allowed_pids})
        else:
            mask &= pid_values.isin(allowed_pids)
    result = dimuons.loc[mask].copy()
    if result.empty or not unique_parent:
        return result
    event_keys = _event_key_columns(result)
    ranking_column = "__truth_candidate_rank"
    if target_mass is not None:
        _required(result, ("dimuon_mass",), context)
        result[ranking_column] = (result["dimuon_mass"] - float(target_mass)).abs()
        ascending = [*[True]*len(event_keys), True, True]
    else:
        if "dimuon_pt" in result.columns:
            result[ranking_column] = -pd.to_numeric(result["dimuon_pt"], errors="coerce").fillna(0.0)
        else:
            result[ranking_column] = 0.0
        ascending = [*[True]*len(event_keys), True, True]
    result = (
        result
        .sort_values([*event_keys, "truth_parent_index", ranking_column], ascending=ascending, kind="stable")
        .drop_duplicates([*event_keys, "truth_parent_index"], keep="first")
        .drop(columns=ranking_column)
    )
    return result

def build_mediator_pairs(
    dimuons: pd.DataFrame,
    *,
    target_mass: float | None = None,
    require_distinct_muons: bool = True,
    require_distinct_truth_parents: bool = False,
) -> pd.DataFrame:
    context = "build_mediator_pairs"
    required = (
        "event_id", "muplus_index", "muminus_index", "dimuon_mass",
        "dimuon_pt", "dimuon_px", "dimuon_py", "dimuon_pz", "dimuon_energy",
    )
    working = dimuons
    if not {"dimuon_px", "dimuon_py", "dimuon_pz", "dimuon_energy"}.issubset(working.columns):
        working = calc_dimuon_variables(working)
    _required(working, required, context)
    if target_mass is not None and (not np.isfinite(target_mass) or target_mass <= 0):
        raise ValueError(f"{context}: target_mass debe ser positivo y finito.")
    if require_distinct_truth_parents:
        _required(working, ("truth_parent_index", "truth_parent_found"), context)
    if working.empty:
        return pd.DataFrame()
    event_keys = _event_key_columns(working)
    candidates = working.reset_index(drop=True).copy()
    if "candidate_index" not in candidates.columns:
        candidates["candidate_index"] = candidates.groupby(event_keys, sort=False).cumcount().astype(np.int32)
    rows: list[dict[str, object]] = []
    for event_key, group in candidates.groupby(event_keys, sort=False, dropna=False):
        best: tuple[float, float, pd.Series, pd.Series] | None = None
        records = [row for _, row in group.iterrows()]
        for first, second in combinations(records, 2):
            first_muons = {int(first["muplus_index"]), int(first["muminus_index"])}
            second_muons = {int(second["muplus_index"]), int(second["muminus_index"])}
            if require_distinct_muons and first_muons.intersection(second_muons):
                continue
            if require_distinct_truth_parents:
                if (pd.isna(first["truth_parent_found"]) or pd.isna(second["truth_parent_found"]) or not bool(first["truth_parent_found"]) or not bool(second["truth_parent_found"])):
                    continue
                if int(first["truth_parent_index"]) == int(second["truth_parent_index"]):
                    continue
            mass1 = float(first["dimuon_mass"])
            mass2 = float(second["dimuon_mass"])
            if not np.isfinite(mass1) or not np.isfinite(mass2):
                continue
            if target_mass is None:
                denominator = mass1 + mass2
                score = ((mass1 - mass2)/denominator)**2 if denominator > 0 else np.inf
            else:
                score = ((mass1 - target_mass)/target_mass)**2 + ((mass2 - target_mass)/target_mass)**2
            pt_sum = float(first["dimuon_pt"]) + float(second["dimuon_pt"])
            ranking = (float(score), -pt_sum)
            if best is None or ranking < best[:2]:
                best = (ranking[0], ranking[1], first, second)
        if best is None:
            continue
        score, _, first, second = best
        if float(second["dimuon_pt"]) > float(first["dimuon_pt"]):
            first, second = second, first
        event_values = event_key if isinstance(event_key, tuple) else (event_key,)
        row: dict[str, object] = dict(zip(event_keys, event_values))
        row["n_dimuon_candidates"] = int(len(group))
        row["n_unique_candidate_muons"] = int(len({
            *group["muplus_index"].astype(int).tolist(),
            *group["muminus_index"].astype(int).tolist(),
        }))
        for number, candidate in ((1, first), (2, second)):
            row.update({
                f"mediator{number}_candidate_index": int(candidate["candidate_index"]),
                f"mediator{number}_muplus_index": int(candidate["muplus_index"]),
                f"mediator{number}_muminus_index": int(candidate["muminus_index"]),
                f"mediator{number}_mass": float(candidate["dimuon_mass"]),
                f"mediator{number}_pt": float(candidate["dimuon_pt"]),
                f"mediator{number}_px": float(candidate["dimuon_px"]),
                f"mediator{number}_py": float(candidate["dimuon_py"]),
                f"mediator{number}_pz": float(candidate["dimuon_pz"]),
                f"mediator{number}_energy": float(candidate["dimuon_energy"]),
            })
            if "dimuon_eta" in candidate:
                row[f"mediator{number}_eta"] = float(candidate["dimuon_eta"])
            if "dimuon_phi" in candidate:
                row[f"mediator{number}_phi"] = float(candidate["dimuon_phi"])
            if "truth_parent_index" in candidate and pd.notna(candidate["truth_parent_index"]):
                row[f"mediator{number}_truth_parent_index"] = int(candidate["truth_parent_index"])
            if "truth_parent_pid" in candidate and pd.notna(candidate["truth_parent_pid"]):
                row[f"mediator{number}_truth_parent_pid"] = int(candidate["truth_parent_pid"])
        row["pairing_score"] = score
        row["pairing_method"] = ("truth_parent" if require_distinct_truth_parents else ("target_mass" if target_mass is not None else "mass_balance"))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    px = result["mediator1_px"].to_numpy() + result["mediator2_px"].to_numpy()
    py = result["mediator1_py"].to_numpy() + result["mediator2_py"].to_numpy()
    pz = result["mediator1_pz"].to_numpy() + result["mediator2_pz"].to_numpy()
    energy = result["mediator1_energy"].to_numpy() + result["mediator2_energy"].to_numpy()
    pt = np.hypot(px, py)
    result["mediator_mass_average"] = 0.5*(result["mediator1_mass"] + result["mediator2_mass"])
    result["mediator_mass_difference"] = (result["mediator1_mass"] - result["mediator2_mass"]).abs()
    result["mediator_mass_asymmetry"] = np.divide(result["mediator_mass_difference"], result["mediator1_mass"] + result["mediator2_mass"], out=np.full(len(result), np.nan), where=(result["mediator1_mass"] + result["mediator2_mass"]).to_numpy() > 0,)
    result["four_muon_energy"] = energy
    result["four_muon_px"] = px
    result["four_muon_py"] = py
    result["four_muon_pz"] = pz
    result["four_muon_mass"] = invariant_mass(energy, px, py, pz)
    result["four_muon_pt"] = pt
    result["four_muon_eta"] = np.arcsinh(np.divide(pz, pt, out=np.full(len(result), np.nan), where=pt > 0))
    result["four_muon_phi"] = np.arctan2(py, px)
    if {"mediator1_eta", "mediator1_phi", "mediator2_eta", "mediator2_phi"}.issubset(result.columns):
        result["mediator_delta_phi"] = delta_phi(result["mediator1_phi"], result["mediator2_phi"])
        result["mediator_delta_r"] = delta_r(result["mediator1_eta"], result["mediator1_phi"], result["mediator2_eta"], result["mediator2_phi"],)
    return result

def select_dimuons(
    dimuons: pd.DataFrame,
    *,
    mass_window: tuple[float, float] | None = None,
    min_muon_pt: float | None = None,
    max_abs_muon_eta: float | None = None,
    max_delta_r: float | None = None,
    max_isolation: float | None = None
) -> pd.DataFrame:
    mask = pd.Series(True, index=dimuons.index)
    if mass_window is not None:
        _required(dimuons, ("dimuon_mass",), "select_dimuons")
        minimum, maximum = mass_window
        if minimum > maximum:
            raise ValueError("mass_window: el mínimo supera al máximo.")
        mask &= dimuons["dimuon_mass"].between(minimum, maximum)
    if min_muon_pt is not None:
        _required(dimuons, ("muplus_pt", "muminus_pt"), "select_dimuons")
        mask &= (dimuons["muplus_pt"].ge(min_muon_pt) & dimuons["muminus_pt"].ge(min_muon_pt))
    if max_abs_muon_eta is not None:
        _required(dimuons, ("muplus_eta", "muminus_eta"), "select_dimuons")
        mask &= (dimuons["muplus_eta"].abs().le(max_abs_muon_eta) & dimuons["muminus_eta"].abs().le(max_abs_muon_eta))
    if max_delta_r is not None:
        _required(dimuons, ("delta_r",), "select_dimuons")
        mask &= dimuons["delta_r"].le(max_delta_r)
    if max_isolation is not None:
        _required(dimuons, ("muplus_isolation", "muminus_isolation"), "select_dimuons")
        mask &= (dimuons["muplus_isolation"].le(max_isolation) & dimuons["muminus_isolation"].le(max_isolation))
    return dimuons.loc[mask].copy()

def match_muon_truth(
    muons: pd.DataFrame,
    particles: pd.DataFrame,
) -> pd.DataFrame:
    reference = next((column for column in ("Particle_ref", "particle_ref") if column in muons.columns), None)
    if reference is None:
        raise KeyError("match_muon_truth requiere la columna 'Particle_ref' o 'particle_ref'.")
    _required(muons, ("event_id", reference), "match_muon_truth")
    _required(particles, ("event_id", "object_index", "root_uid"), "match_muon_truth")
    optional_fields = (
        "PID",
        "Status",
        "PT",
        "Eta",
        "Phi",
        "Mass",
        "Charge",
        "M1",
        "M2",
        "D1",
        "D2",
    )
    available_fields = [field for field in optional_fields if field in particles.columns]
    valid_identifiers = (particles["root_uid"].notna() & particles["root_uid"].ne(0))
    columns = [
        "event_id",
        "object_index",
        "root_uid",
        *available_fields,
    ]
    truth = particles.loc[valid_identifiers, columns].copy()
    truth = truth.rename(columns={column: f"truth_{column}" for column in ("object_index", "root_uid", *available_fields)})
    result = muons.merge(truth, left_on=["event_id", reference], right_on=["event_id", "truth_root_uid"], how="left", validate="many_to_one", sort=False,)
    result["truth_matched"] = result["truth_root_uid"].notna()
    if "truth_PID" in result.columns:
        result["truth_is_muon"] = (result["truth_PID"].abs().eq(13).fillna(False))
        if "Charge" in result.columns:
            result["truth_charge_consistent"] = (result["Charge"].eq(-np.sign(result["truth_PID"])).fillna(False))
    if ("PT" in result.columns and "truth_PT" in result.columns):
        result["pt_difference"] = (result["PT"] - result["truth_PT"])
    return result

def attach_mother(
    muons: pd.DataFrame,
    particles: pd.DataFrame,
    *,
    mother: int = 1,
) -> pd.DataFrame:
    if mother not in (1, 2):
        raise ValueError("mother debe ser 1 o 2.")
    mother_column = (f"truth_M{mother}")
    _required(muons, ("event_id", mother_column), "attach_mother")
    _required(particles, ("event_id", "object_index"), "attach_mother")
    optional_fields = (
        "PID",
        "Status",
        "PT",
        "Eta",
        "Phi",
        "Mass",
        "M1",
        "M2",
        "D1",
        "D2",
        "root_uid",
    )
    fields = [field for field in optional_fields if field in particles.columns]
    prefix = (f"mother{mother}_")
    lookup = particles[["event_id", "object_index", *fields]].rename(columns={"object_index": f"{prefix}index", **{field: f"{prefix}{field}" for field in fields}})
    return muons.merge(
        lookup,
        left_on=["event_id", mother_column],
        right_on=["event_id", f"{prefix}index"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

def prepare_training_features(
    events: pd.DataFrame,
    muons: pd.DataFrame,
    *,
    jets: pd.DataFrame | None = None,
    electrons: pd.DataFrame | None = None,
    min_jets: int = 3,
    max_electrons: int | None = None,
    min_muon_pt: float = 0.0,
    max_abs_muon_eta: float | None = None,
) -> pd.DataFrame:
    selected_muons = select_muons(
        muons,
        min_pt=min_muon_pt,
        max_abs_eta=max_abs_muon_eta,
    )
    counted_events = add_selected_counts(
        events,
        muons=selected_muons,
        jets=jets,
        electrons=electrons
    )
    accepted_events = filter_events(
        counted_events,
        min_muons=1,
        min_jets=min_jets,
        max_electrons=max_electrons,
        use_selected=True,
    )
    leading_muons = leading_object(selected_muons)
    leading_muons = calc_four_momentum(leading_muons, mass=MUON_MASS_GEV)
    columns = [
        "event_id",
        "PT",
        "Eta",
        "Phi",
        "PX",
        "PY",
        "PZ",
        "P",
        "E",
    ]
    for optional in ("Charge", "D0", "DZ", "IsolationVar"):
        if optional in leading_muons.columns:
            columns.append(optional)
    renamed = leading_muons[columns].rename(
        columns={
            "PT": "Muon_PT",
            "Eta": "Muon_Eta",
            "Phi": "Muon_Phi",
            "PX": "Muon_px",
            "PY": "Muon_py",
            "PZ": "Muon_pz",
            "P": "Muon_P",
            "E": "Muon_E",
            "Charge": "Muon_Charge",
            "D0": "Muon_D0",
            "DZ": "Muon_DZ",
            "IsolationVar": "Muon_IsolationVar"
        }
    )
    return accepted_events.merge(
        renamed,
        on="event_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

def split_train_test(
    df: pd.DataFrame,
    *,
    event_column: str = "event_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _required(df, (event_column,), "split_train_test")
    training_mask = (df[event_column].astype(np.int64).mod(2).eq(0))
    train = df.loc[training_mask].copy()
    test = df.loc[~training_mask].copy()
    return train, test

def calc_dimuon_observables(
    dimuons: pd.DataFrame,
    *,
    muon_mass: float = MUON_MASS_GEV
) -> pd.DataFrame:
    result = calc_dimuon_variables(dimuons, muon_mass=muon_mass)
    result['dimuon_mass2'] = (result["dimuon_mass"]**2)
    result['muon_sum_pt'] = (result['muplus_pt'] + result['muminus_pt'])
    pt_sum = result['muon_sum_pt'].to_numpy(dtype=np.float64)
    pt_difference = (result["muplus_pt"] - result['muminus_pt']).abs().to_numpy(dtype=np.float64)
    result['pt_balance'] = np.divide(pt_difference, pt_sum, out=np.full(len(result), np.nan), where=pt_sum != 0)
    result['acoplanarity'] = (np.pi - result['delta_phi'].abs())
    energy = result['dimuon_energy'].to_numpy(dtype=np.float64)
    pz = result['dimuon_pz'].to_numpy(dtype=np.float64)
    numerator = energy + pz
    denominator = energy - pz
    valid_rapidity = ((numerator > 0) & (denominator > 0))
    result['dimuon_rapidity'] = np.divide(numerator, denominator, out=np.full(len(result), np.nan), where=valid_rapidity)
    result['dimuon_rapidity'] = (0.5*np.log(result['dimuon_rapidity']))
    result['cos_theta_star'] = np.tanh((result['muminus_eta'] - result['muplus_eta'])/2.0)
    return result

def calc_truth_decay_time(
    dimuons: pd.DataFrame,
    particles: pd.DataFrame,
    *,
    parent_pid: int | Iterable[int] | None = None,
    match_absolute_pid: bool = True,
    muplus_reference_column: str = 'muplus_particle_ref',
    muminus_reference_column: str = 'muminus_particle_ref',
    c_mm_ps: float = C_MM_PS
) -> pd.DataFrame:
    context = 'calc_truth_decay_time'
    if not np.isfinite(c_mm_ps) or c_mm_ps <= 0:
        raise ValueError(f"{context}: error con el valor de c.")
    _required(dimuons, ("event_id", muplus_reference_column, muminus_reference_column), context)
    particle_fields = {
        "event_id",
        "object_index",
        "root_uid",
        "PID",
        "M1",
        "M2",
        "Mass",
        "X",
        "Y",
        "Z"
    }
    _required(particles, tuple(sorted(particle_fields)), context)
    cartesian_momentum = {"PX", "PY", "PZ"}
    cylindrical_momentum = {"PT", "Eta", "Phi"}
    if not cartesian_momentum.issubset(particles.columns):
        _required(particles, tuple(sorted(cylindrical_momentum)), context)
    result = dimuons.copy()
    positions = np.arange(len(result), dtype=np.int64)
    event_ids = result["event_id"].to_numpy()
    truth = particles.loc[:, [
        "event_id",
        "object_index",
        "root_uid",
        "PID",
        "M1",
        "M2",
        "Mass",
        "X",
        "Y",
        "Z",
        *(["PX", "PY", "PZ"] if cartesian_momentum.issubset(particles.columns) else ["PT", "Eta", "Phi"]),
    ]].copy()
    valid_references = truth["root_uid"].notna() & truth["root_uid"].ne(0)
    daughter_lookup = truth.loc[
        valid_references,
        ["event_id", "root_uid", "object_index", "PID", "M1", "M2", "X", "Y", "Z"],
    ]
    if daughter_lookup.duplicated(["event_id", "root_uid"]).any():
        raise ValueError(f"{context}: Particles contiene root_uid duplicados dentro de un evento.")
    parent_lookup = truth.drop(columns=["M1", "M2"])
    if parent_lookup.duplicated(["event_id", "object_index"]).any():
        raise ValueError(f"{context}: Particles contiene object_index duplicados dentro de un evento.")
    def match_daughter(side: str, reference_column: str) -> pd.DataFrame:
        reference_frame = pd.DataFrame({
            "__row_position": positions,
            "event_id": event_ids,
            "__particle_reference": result[reference_column].to_numpy(),
        })
        renamed = daughter_lookup.rename(columns={
            "root_uid": "__particle_reference",
            **{column: f"__{side}_{column.lower()}" for column in ("object_index", "PID", "M1", "M2", "X", "Y", "Z")},
        })
        matched = reference_frame.merge(
            renamed,
            on=["event_id", "__particle_reference"],
            how="left",
            validate="many_to_one",
            sort=False
        )
        return matched.sort_values("__row_position", kind="stable").reset_index(drop=True)
    plus = match_daughter("muplus", muplus_reference_column)
    minus = match_daughter('muminus', muminus_reference_column)
    parent_pids = _normalise_parent_pids(parent_pid)
    potential_parents: list[pd.DataFrame] = []
    priority = 0
    for plus_mother in ('__muplus_m1', '__muplus_m2'):
        plus_values = pd.to_numeric(plus[plus_mother], errors='coerce').to_numpy(dtype=np.float64)
        for minus_mother in ('__muminus_m1', '__muminus_m2'):
            minus_values = pd.to_numeric(minus[minus_mother], errors='coerce').to_numpy(dtype=np.float64)
            common = (np.isfinite(plus_values) & np.isfinite(minus_values) & (plus_values >= 0) & (plus_values == minus_values))
            if np.any(common):
                potential_parents.append(pd.DataFrame({
                    "__row_position": positions[common],
                    "event_id": event_ids[common],
                    "truth_parent_index": plus_values[common].astype(np.int64),
                    "__parent_priority": priority
                }))
            priority += 1
    parent_columns = {
        "object_index": 'truth_parent_index',
        'root_uid': 'truth_parent_root_uid',
        'PID': 'truth_parent_pid',
        'Mass': 'truth_parent_mass',
        'X': 'truth_parent_x',
        'Y': 'truth_parent_y',
        'Z': 'truth_parent_z',
        'PX': 'truth_parent_px',
        'PY': 'truth_parent_py',
        'PZ': 'truth_parent_pz',
        'PT': 'truth_parent_pt',
        'Eta': 'truth_parent_eta',
        'Phi': 'truth_parent_phi',
    }
    available_parent_columns = {source: target for source, target in parent_columns.items() if source in parent_lookup.columns}
    renamed_parent_lookup = parent_lookup.rename(columns=available_parent_columns)
    if potential_parents:
        candidates = pd.concat(potential_parents, ignore_index=True)
        candidates = candidates.merge(
            renamed_parent_lookup,
            on=['event_id', 'truth_parent_index'],
            how='left',
            validate='many_to_one',
            sort=False
        )
        if parent_pids is not None:
            pid_values = pd.to_numeric(candidates['truth_parent_pid'], errors='coerce')
            allowed = (pid_values.abs().isin({abs(value) for value in parent_pids}) if match_absolute_pid else pid_values.isin(parent_pids))
            candidates = candidates.loc[allowed]
        chosen = (
            candidates
            .sort_values(['__row_position', '__parent_priority'], kind='stable')
            .drop_duplicates('__row_position', keep='first')
        )
    else:
        empty_columns = dict.fromkeys((
            '__row_position',
            'event_id',
            'truth_parent_index',
            '__parent_priority',
            *available_parent_columns.values()
        ))
        chosen = pd.DataFrame(columns=list(empty_columns))
    scaffold = pd.DataFrame({'__row_position': positions})
    chosen = scaffold.merge(
        chosen.drop(columns='event_id', errors='ignore'),
        on='__row_position',
        how='left',
        validate='one_to_one',
        sort=False
    ).sort_values('__row_position', kind='stable')
    def daughter_values(frame: pd.DataFrame, side: str, field: str) -> np.ndarray:
        return pd.to_numeric(frame[f"__{side}_{field}"], errors='coerce').to_numpy(dtype=np.float64)
    plus_x = daughter_values(plus, 'muplus', 'x')
    plus_y = daughter_values(plus, 'muplus', 'y')
    plus_z = daughter_values(plus, 'muplus', 'z')
    minus_x = daughter_values(minus, 'muminus', 'x')
    minus_y = daughter_values(minus, 'muminus', 'y')
    minus_z = daughter_values(minus, 'muminus', 'z')
    daughters_finite = (np.isfinite(plus_x) & np.isfinite(plus_y) & np.isfinite(plus_z) & np.isfinite(minus_x) & np.isfinite(minus_y) & np.isfinite(minus_z))
    decay_x = np.where(daughters_finite, 0.5*(plus_x + minus_x), np.nan)
    decay_y = np.where(daughters_finite, 0.5*(plus_y + minus_y), np.nan)
    decay_z = np.where(daughters_finite, 0.5*(plus_z + minus_z), np.nan)
    daughter_separation = np.where(daughters_finite, np.sqrt((plus_x - minus_x)**2 + (plus_y - minus_y)**2 + (plus_z - minus_z)**2), np.nan,)
    def parent_values(column: str) -> np.ndarray:
        if column not in chosen.columns:
            return np.full(len(result), np.nan, dtype=np.float64)
        return pd.to_numeric(chosen[column], errors='coerce').to_numpy(dtype=np.float64)
    production_x = parent_values('truth_parent_x')
    production_y = parent_values('truth_parent_y')
    production_z = parent_values('truth_parent_z')
    parent_mass = parent_values('truth_parent_mass')
    if cartesian_momentum.issubset(particles.columns):
        parent_px = parent_values('truth_parent_px')
        parent_py = parent_values('truth_parent_py')
        parent_pz = parent_values('truth_parent_pz')
    else:
        parent_pt = parent_values('truth_parent_pt')
        parent_eta = parent_values('truth_parent_eta')
        parent_phi = parent_values('truth_parent_phi')
        parent_px = parent_pt*np.cos(parent_phi)
        parent_py = parent_pt*np.sin(parent_phi)
        parent_pz = parent_pt*np.sinh(parent_eta)
    dx = decay_x - production_x
    dy = decay_y - production_y
    dz = decay_z - production_z
    momentum_squared = parent_px**2 + parent_py**2 + parent_pz**2
    momentum = np.sqrt(momentum_squared)
    transverse_momentum_squared = parent_px**2 + parent_py**2
    transverse_momentum = np.sqrt(transverse_momentum_squared)
    displacement_squared = dx**2 + dy**2 + dz**2
    displacement = np.sqrt(displacement_squared)
    projection_numerator = dx*parent_px + dy*parent_py + dz*parent_pz
    transverse_projection = dx*parent_px + dy*parent_py
    valid_time = (
        daughters_finite & np.isfinite(production_x) & np.isfinite(production_y) & np.isfinite(production_z)
        & np.isfinite(parent_mass) & (parent_mass > 0) & np.isfinite(momentum_squared) & (momentum_squared > 0)
    )
    projected_length = np.divide(
        projection_numerator, 
        momentum,
        out=np.full(len(result), np.nan, dtype=np.float64),
        where=valid_time
    )
    lxy = np.divide(
        transverse_projection,
        transverse_momentum,
        out=np.full(len(result), np.nan, dtype=np.float64),
        where=(valid_time & (transverse_momentum_squared > 0))
    )
    ctau_mm = np.divide(
        parent_mass*projection_numerator,
        momentum_squared,
        out=np.full(len(result), np.nan, dtype=np.float64),
        where=valid_time
    )
    parent_index = parent_values('truth_parent_index')
    parent_pid_values = parent_values('truth_parent_pid')
    parent_found = np.isfinite(parent_index) & np.isfinite(parent_pid_values)
    result['truth_parent_index'] = pd.array(parent_index, dtype='Int64')
    result['truth_parent_pid'] = pd.array(parent_pid_values, dtype='Int64')
    result['truth_parent_found'] = parent_found
    result['truth_decay_vertex_x'] = decay_x
    result['truth_decay_vertex_y'] = decay_y
    result['truth_decay_vertex_z'] = decay_z
    result['truth_decay_vertex_separation_mm'] = daughter_separation
    result['truth_decay_length_mm'] = np.where(valid_time, displacement, np.nan)
    result['truth_projected_decay_length_mm'] = projected_length
    result['truth_lxy_mm'] = lxy
    result['truth_ctau_mm'] = ctau_mm
    result['dimuon_decay_time_truth_ps'] = ctau_mm/c_mm_ps
    result['truth_decay_time_valid'] = valid_time
    return result

def check_long_lived_truth(
    dimuons: pd.DataFrame,
    *,
    time_column: str = 'dimuon_decay_time_truth_ps',
    displacement_column: str = 'truth_decay_length_mm',
    minimum_time_ps: float = 0.00,
    minimum_displacement_mm: float = 0.00,
    raise_on_failure: bool = False
) -> dict[str, int | float | bool]:
    context = 'check_long_lived_truth'
    if minimum_time_ps < 0 or minimum_displacement_mm < 0:
        raise
    _required(dimuons, (time_column, displacement_column), context)
    time = _float_column(dimuons, time_column, context)
    displacement = _float_column(dimuons, displacement_column, context)
    finite_time = np.isfinite(time)
    finite_displacement = np.isfinite(displacement)
    long_lived = (finite_time & finite_displacement & (time > minimum_time_ps) & (displacement > minimum_displacement_mm))
    selected_times = time[long_lived]
    parent_found = (dimuons['truth_parent_found'].fillna(False).to_numpy(dtype=bool) if 'truth_parent_found' in dimuons.columns else finite_time)
    total = len(dimuons)
    summary: dict[str, int | float | bool] = {
        "n_candidates": total,
        'n_parent_found': int(np.count_nonzero(parent_found)),
        'n_finite_time': int(np.count_nonzero(finite_time)),
        'n_finite_displacement': int(np.count_nonzero(finite_displacement)),
        'n_long_lived_truth': int(np.count_nonzero(long_lived)),
        'fraction_long_lived_truth': (float(np.count_nonzero(long_lived)/total) if total else np.nan),
        'median_decay_time_truth_ps': (float(np.median(selected_times)) if len(selected_times) else np.nan),
        'maximum_decay_time_truth_ps': (float(np.max(selected_times)) if len(selected_times) else np.nan),
        'has_long_lived_truth': bool(np.any(long_lived))
    }
    if raise_on_failure and not summary['has_long_lived_truth']:
        raise ValueError(f"{context}: no hay candidatos por encima de los umbrales truth indicados.")
    return summary

def calc_reco_decay_time(
    dimuons: pd.DataFrame,
    *,
    primary_vertex_columns: tuple[str, str, str] = ('pv_x', 'pv_y', 'pv_z'),
    secondary_vertex_columns: tuple[str, str, str] = ('sv_x', 'sv_y', 'sv_z'),
    primary_vertex: tuple[float, float, float] | None = None,
    momentum_columns: tuple[str, str, str] = ('dimuon_px', 'dimuon_py', 'dimuon_pz'),
    mass_column: str = 'dimuon_mass',
    dimension: str = '3d',
    c_mm_ps: float = C_MM_PS
) -> pd.DataFrame:
    context = 'calc_reco_decay_time'
    if dimension not in {'3d', 'transverse'}:
        raise ValueError(f"{context}: el parámetro 'dimension' debe ser '3d' o 'transverse'.")
    _required(dimuons, (*momentum_columns, mass_column), context)
    result = dimuons.copy()
    pv_x, pv_y, pv_z = _point_components(result, primary_vertex_columns, constant=primary_vertex, context=context)
    sv_x, sv_y, sv_z = _point_components(result, secondary_vertex_columns, constant=None, context=context)
    px, py, pz = (_float_column(result, column, context) for column in momentum_columns)
    mass = _float_column(result, mass_column, context)
    dx = sv_x - pv_x
    dy = sv_y - pv_y
    dz = sv_z - pv_z
    transverse_momentum_squared = px**2 + py**2
    transverse_momentum = np.sqrt(transverse_momentum_squared)
    transverse_projection = dx*px + dy*py
    lxy = np.divide(
        transverse_projection,
        transverse_momentum,
        out=np.full(len(result), np.nan, dtype=np.float64),
        where=(np.isfinite(transverse_projection) & np.isfinite(transverse_momentum) & (transverse_momentum > 0))
    )
    if dimension == '3d':
        momentum_squared = transverse_momentum_squared + pz**2
        momentum = np.sqrt(momentum_squared)
        projection_numerator = transverse_projection + dz*pz
    else:
        momentum_squared = transverse_momentum_squared
        momentum = transverse_momentum
        projection_numerator = transverse_projection
    valid = (np.isfinite(mass) & (mass > 0) & np.isfinite(projection_numerator) & np.isfinite(momentum_squared) & (momentum_squared > 0))
    projected_length = np.divide(
        projection_numerator,
        momentum,
        out=np.full(len(result), np.nan, dtype=np.float64),
        where=valid
    )
    ctau_mm = np.divide(
        mass*projection_numerator,
        momentum_squared,
        out=np.full(len(result), np.nan, dtype=np.float64),
        where=valid
    )
    result['dimuon_lxy_reco_mm'] = lxy
    result['dimuon_projected_decay_length_reco_mm'] = projected_length
    result['dimuon_ctau_reco_mm'] = ctau_mm
    result['dimuon_decay_time_reco_ps'] = ctau_mm/c_mm_ps
    result['reco_decay_time_valid'] = valid
    return result

def calc_decay_fit_proxy(
    dimuons: pd.DataFrame,
    *,
    vertex_xy_resolution_mm: float | str,
    vertex_z_resolution_mm: float | str,
    pointing_resolution_mm: float | str,
    muplus_vertex_columns: tuple[str, str, str] = ('muplus_track_x', 'muplus_track_y', 'muplus_track_z'),
    muminus_vertex_columns: tuple[str, str, str] = ('muminus_track_x', 'muminus_track_y', 'muminus_track_z'),
    primary_vertex_columns: tuple[str, str, str] = ('pv_x', 'pv_y', 'pv_z'),
    secondary_vertex_columns: tuple[str, str, str] | None = None,
    primary_vertex: tuple[float, float, float] | None = None,
    momentum_columns: tuple[str, str, str] = ('dimuon_px', 'dimuon_py', 'dimuon_pz')
) -> pd.DataFrame:
    context = 'calc_decay_fit_proxy'
    _required(dimuons, momentum_columns, context)
    result = dimuons.copy()
    plus_x, plus_y, plus_z = _point_components(
        result,
        muplus_vertex_columns,
        constant=None,
        context=context
    )
    minus_x, minus_y, minus_z = _point_components(
        result,
        muminus_vertex_columns,
        constant=None,
        context=context
    )
    pv_x, pv_y, pv_z = _point_components(
        result,
        primary_vertex_columns,
        constant=primary_vertex,
        context=context
    )
    if secondary_vertex_columns is None:
        sv_x = 0.5*(plus_x + minus_x)
        sv_y = 0.5*(plus_y + minus_y)
        sv_z = 0.5*(plus_z + minus_z)
    else:
        sv_x, sv_y, sv_z = _point_components(
            result,
            secondary_vertex_columns,
            constant=None,
            context=context
        )
    px, py, pz = (_float_column(result, column, context) for column in momentum_columns)
    sigma_vertex_xy = _resolution_values(result, vertex_xy_resolution_mm, context=context)
    sigma_vertex_z = _resolution_values(result, vertex_z_resolution_mm, context=context)
    sigma_pointing = _resolution_values(result, pointing_resolution_mm, context=context)
    separation_xy = np.hypot(plus_x - minus_x, plus_y - minus_y)
    separation_z = np.abs(plus_z - minus_z)
    vertex_xy_pull = _divide_valid(separation_xy, sigma_vertex_xy)
    vertex_z_pull = _divide_valid(separation_z, sigma_vertex_z)
    flight_x = sv_x - pv_x
    flight_y = sv_y - pv_y
    flight_z = sv_z - pv_z
    flight = np.column_stack((flight_x, flight_y, flight_z))
    momentum = np.column_stack((px, py, pz))
    flight_norm = np.linalg.norm(flight, axis=1)
    momentum_norm = np.linalg.norm(momentum, axis=1)
    cross_norm = np.linalg.norm(np.cross(flight, momentum), axis=1)
    pointing_miss = np.divide(
        cross_norm,
        momentum_norm,
        out=np.full(len(result), np.nan, dtype=np.float64),
        where=(np.isfinite(cross_norm) & np.isfinite(momentum_norm) & np.isfinite(flight_norm) & (momentum_norm > 0) & (flight_norm > 0))
    )
    pointing_pull = _divide_valid(pointing_miss, sigma_pointing)
    cosine = np.divide(
        np.sum(flight*momentum, axis=1),
        flight_norm*momentum_norm,
        out=np.full(len(result), np.nan, dtype=np.float64),
        where=(np.isfinite(flight_norm) & np.isfinite(momentum_norm) & (flight_norm > 0) & (momentum_norm > 0)),
    )
    pointing_angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    components = np.column_stack((vertex_xy_pull, vertex_z_pull, pointing_pull))
    valid = np.all(np.isfinite(components), axis=1)
    chi2_proxy = np.where(valid, np.sum(components**2, axis=1), np.nan)
    result['dimuon_sv_proxy_x'] = sv_x
    result['dimuon_sv_proxy_y'] = sv_y
    result['dimuon_sv_proxy_z'] = sv_z
    result['dimuon_vertex_separation_xy_mm'] = separation_xy
    result['dimuon_vertex_separation_z_mm'] = separation_z
    result['dimuon_pointing_miss_mm'] = pointing_miss
    result['dimuon_pointing_angle_rad'] = pointing_angle
    result['chi2_df_vertex_xy_component'] = vertex_xy_pull**2
    result['chi2_df_vertex_z_component'] = vertex_z_pull**2
    result['chi2_df_pointing_component'] = pointing_pull**2
    result['chi2_df_proxy'] = chi2_proxy
    result['chi2_df_proxy_valid'] = valid
    return result

def classify_dimuon_displacement(
    dimuons: pd.DataFrame,
    *,
    prompt_max_significance: float = 3.0,
    displaced_min_significance: float = 5.0
) -> pd.DataFrame:
    _required(dimuons, ('muplus_d0', 'muminus_d0', 'muplus_error_d0', 'muminus_error_d0'), 'Classify_dimuon_displacement')
    if (prompt_max_significance >= displaced_min_significance):
        raise ValueError("'prompt_max_significance' debe ser menor que 'displaced_min_significance'.")
    result = add_ip_proxies(dimuons)
    valid = result['ip_proxy_valid']
    prompt = (valid & result['muplus_d0_significance'].le(prompt_max_significance) & result['muminus_d0_significance'].le(prompt_max_significance))
    displaced = (valid & result['muplus_d0_significance'].ge(displaced_min_significance) & result['muminus_d0_significance'].ge(displaced_min_significance))
    result['displacement_category'] = np.select(
        [~valid, prompt, displaced],
        ['invalid', 'prompt', 'displaced'],
        default='mixed'
    )
    return result

def select_displacement_category(dimuons: pd.DataFrame, category: str) -> pd.DataFrame:
    available = {'prompt', 'displaced', 'mixed', 'invalid'}
    if category not in available:
        formatted = ', '.join(sorted(available))
        raise ValueError(f'Categoría desconocida: {category!r}.\nOpciones: {formatted}.')
    _required(dimuons, ('displacement_category',), 'select_displacement_category')
    return dimuons.loc[dimuons['displacement_category'].eq(category)].copy()

def prepare_dimuon_analysis(
    muons: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    min_muon_pt: float = 0.0,
    max_abs_muon_eta: float | None = None,
    max_isolation: float | None = None,
    mass_window: tuple[float, float] | None = None,
    max_delta_r: float | None = None,
    classify_displacement: bool = True,
    prompt_max_significance: float = 3.0,
    displaced_min_significance: float = 5.0
) -> pd.DataFrame:
    selected_muons = select_muons(
        muons,
        min_pt=min_muon_pt,
        max_abs_eta=max_abs_muon_eta,
        max_isolation=max_isolation
    )
    dimuons = build_dimuons(selected_muons, events=events)
    dimuons = calc_dimuon_observables(dimuons)
    dimuons = select_dimuons(
        dimuons,
        mass_window=mass_window,
        min_muon_pt=min_muon_pt,
        max_abs_muon_eta=max_abs_muon_eta,
        max_delta_r=max_delta_r,
        max_isolation=max_isolation
    )
    displacement_columns = {
        'muplus_d0',
        'muminus_d0',
        'muplus_error_d0',
        'muminus_error_d0'
    }
    if (classify_displacement and displacement_columns.issubset(dimuons.columns)):
        dimuons = classify_dimuon_displacement(
            dimuons,
            prompt_max_significance=(prompt_max_significance),
            displaced_min_significance=(displaced_min_significance)
        )
    return dimuons

def compute_counting_asymmetry(
    n_positive: int,
    n_negative: int,
) -> dict[str, float | int]:
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

def compute_charge_asymmetry(
    objects: pd.DataFrame,
    *,
    charge_column: str = 'Charge'
) -> dict[str, float | int]:
    _required(objects, (charge_column,), 'compute_charge_asymmetry')
    n_positive = int(objects[charge_column].gt(0).sum())
    n_negative = int(objects[charge_column].lt(0).sum())
    return compute_counting_asymmetry(n_positive, n_negative)

def print_asymmetry(label: str, result: dict[str, float | int]) -> None:
    print(f"\n======= {label} =======")
    print(f" N+ = {result['Np']:,}       |       N- = {result['Nm']:,}")
    print(f" A = {result['A']:+.4f} ± {result['sigma']:.4f}")
    print(f" Significancia = {result['significance']:+.2f} σ")

def build_two_muon_system(
    muons: pd.DataFrame,
    *,
    muon_mass: float = MUON_MASS_GEV,
    require_two: bool = True
) -> pd.DataFrame:
    required = {"source_file_id", "event_id", "object_index", "PT", "Eta", "Phi", "Charge"}
    missing = required.difference(muons.columns)
    if missing:
        raise KeyError(f"build_two_muon_system: faltan columnas {sorted(missing)}.")
    group_keys = ["source_file_id", "event_id"]
    counts = muons.groupby(group_keys).size()
    valid_events = counts[counts == 2].index if require_two else counts[counts >= 2].index
    if len(valid_events) == 0:
        raise ValueError("No hay eventos con el número de muones requerido.")
    selected = muons.set_index(group_keys).loc[valid_events].reset_index()
    selected = selected.sort_values(group_keys + ["PT"], ascending=[True, True, False])
    grouped = selected.groupby(group_keys, sort=False)
    first = grouped.nth(0).reset_index(drop=True)
    second = grouped.nth(1).reset_index(drop=True)
    px = first["PT"]*np.cos(first["Phi"]) + second["PT"]*np.cos(second["Phi"])
    py = first["PT"]*np.sin(first["Phi"]) + second["PT"]*np.sin(second["Phi"])
    pz = first["PT"]*np.sinh(first["Eta"]) + second["PT"]*np.sinh(second["Eta"])
    e1 = np.sqrt((first["PT"]*np.cosh(first["Eta"]))**2 + muon_mass**2)
    e2 = np.sqrt((second["PT"]*np.cosh(second["Eta"]))**2 + muon_mass**2)
    energy =  e1 + e2
    mass = np.sqrt(np.maximum(energy**2 - px**2 -py**2 - pz**2, 0.0))
    pt = np.sqrt(px**2 + py**2)
    delta_eta  = first["Eta"] - second["Eta"]
    raw_delta_phi = first["Phi"] - second["Phi"]
    delta_phi = np.arctan2(np.sin(raw_delta_phi), np.cos(raw_delta_phi))
    charge_product = first["Charge"] * second["Charge"]
    result = pd.DataFrame({
        "source_file_id": first["source_file_id"],
        "event_id": first["event_id"],
        "dimuon_mass": mass,
        "dimuon_pt": pt,
        "delta_r": np.sqrt(delta_eta**2 + delta_phi**2),
        "charge_product": charge_product,
        "is_opposite_sign": charge_product < 0,
        "mu1_pt": first["PT"],
        "mu1_eta": first["Eta"],
        "mu1_phi": first["Phi"],
        "mu1_charge": first["Charge"],
        "mu2_pt": second["PT"],
        "mu2_eta": second["Eta"],
        "mu2_phi": second["Phi"],
        "mu2_charge": second["Charge"]
    })
    n_os = int(result["is_opposite_sign"].sum())
    n_ss = len(result) - n_os
    print(f"Eventos con 2 muones: {len(result):,} (OS: {n_os:,} | SS: {n_ss:,})")
    return result