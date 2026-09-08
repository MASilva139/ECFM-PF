# Físicas para delphes (tabla plana)
from collections.abc import Iterable
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
    _float_column(df, (column,), context)
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
    for side, reference_column in (("muplus", muplus_reference_column), ('muminus', muminus_reference_column)):
        renamed = lookup.rename(columns={
            track_reference_column: reference_column, **{
                field: f"{side}_track_{field_names[field]}" for field in lookup.columns if field in field_names
            },
        })
        result = result.merge(
            renamed, 
            on=["event_id", reference_column], 
            how="left", 
            validate="many_to_one", 
            sort=False
        )
        result[f"{side}_track_matched"] = result[f"{side}_track_x"].notna()
    result["dimuon_tracks_matched"] = (result['muplus_track_matched'] & result["muminus_track_matched"])
    return (result.sort_values("__row_position", kind="stable").drop(columns="__row_position").set_axis(dimuons.index))

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
    candidates = positive.merge(
        negative,
        on="event_id",
        how="inner",
        validate="many_to_many",
        sort=False,
    )
    candidates["candidate_index"] = (
        candidates
        .groupby("event_id", sort=False)
        .cumcount()
        .astype(np.int32)
    )
    if events is not None:
        _required(events, ("event_id",), "build_dimuons")
        if events['event_id'].duplicated().any():
            raise ValueError("DataFrame Events contiene 'event_id' duplicados.")
        candidates = candidates.merge(
            events,
            on="event_id",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    else:
        identity_columns = [column for column in ("source_file_id", "source_entry") if column in muons.columns]
        if identity_columns:
            identity = (muons[["event_id", *identity_columns]].drop_duplicates("event_id"))
            candidates = candidates.merge(
                identity,
                on="event_id",
                how="left",
                validate="many_to_one",
                sort=False,
            )
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
    result = muons.merge(
        truth,
        left_on=["event_id", reference],
        right_on=["event_id", "truth_root_uid"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
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
    result['pt_balance'] = np.divide(
        pt_difference, 
        pt_sum, 
        out=np.full(len(result), np.nan),
        where=pt_sum != 0
    )
    result['acoplanarity'] = (np.pi - result['delta_phi'].abs())
    energy = result['dimuon_energy'].to_numpy(dtype=np.float64)
    pz = result['dimuon_pz'].to_numpy(dtype=np.float64)
    numerator = energy + pz
    denominator = energy - pz
    valid_rapidity = ((numerator > 0) & (denominator > 0))
    result['dimuon_rapidity'] = np.divide(
        numerator,
        denominator,
        out=np.full(len(result), np.nan),
        where=valid_rapidity
    )
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
    displacement_column: str = 'truth_decay_legth_mm',
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
        where=(np.isfinite(cross_norm) & np.isfinite(momentum_norm) & np.isfinite(flight_norm) (momentum_norm > 0) & (flight_norm > 0),)
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
    available = {'prompt', 'displaced', 'mixed'}
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