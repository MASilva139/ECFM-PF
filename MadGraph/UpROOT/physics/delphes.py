from __future__ import annotations
import numpy as np
import pandas as pd

MUON_MASS_GEV = 0.1056583755
ELECTRON_MASS_GEV = 0.00051099895

def _required(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    context: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        formatted = ", ".join(missing)
        raise KeyError(f"{context}: columnas faltantes: {formatted}")

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
) -> pd.DataFrame:
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
) -> pd.DataFrame:
    _required(events, ("event_id",), "add_selected_counts")
    result = events.copy()
    collections = (
        ("Muon", muons),
        ("Jet", jets),
        ("Electron", electrons),
    )
    for name, objects in collections:
        if objects is None:
            continue
        _required(objects, ("event_id",), "add_selected_counts")
        counts = objects.groupby("event_id", sort=False).size()
        result[f"n_{name}_selected"] = (
            result["event_id"]
            .map(counts)
            .fillna(0)
            .astype(np.int64)
        )
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
    if min_muons:
        mask &= count("Muon").ge(min_muons)
    if min_jets:
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
    "E": "energy",
    "P": "p",
}

def calc_dimuon_variables(
    dimuons: pd.DataFrame,
    *,
    muon_mass: float = MUON_MASS_GEV,
) -> pd.DataFrame:
    required = tuple(f"{side}_{attribute}" for side in ("muplus", "muminus") for attribute in ("pt", "eta", "phi"))
    _required(dimuons, required, "calc_dimuon_variables")
    result = dimuons.copy()
    for side in ("muplus", "muminus"):
        temporary = pd.DataFrame(
            {
                "PT": result[f"{side}_pt"].to_numpy(),
                "Eta": result[f"{side}_eta"].to_numpy(),
                "Phi": result[f"{side}_phi"].to_numpy(),
            },
            index=result.index,
        )
        momentum = calc_four_momentum(temporary, mass=muon_mass)
        mapping = (("PX", "px"), ("PY", "py"), ("PZ", "pz"), ("E", "energy"), ("P", "p"))
        for original, output in mapping:
            result[f"{side}_{output}"] = momentum[original].to_numpy()
    px = (result["muplus_px"].to_numpy() + result["muminus_px"].to_numpy())
    py = (result["muplus_py"].to_numpy() + result["muminus_py"].to_numpy())
    pz = (result["muplus_pz"].to_numpy() + result["muminus_pz"].to_numpy())
    energy = (result["muplus_energy"].to_numpy() + result["muminus_energy"].to_numpy())
    pt = np.hypot(px, py)
    delta_eta = (result["muplus_eta"].to_numpy() - result["muminus_eta"].to_numpy())
    raw_delta_phi = (result["muplus_phi"].to_numpy() - result["muminus_phi"].to_numpy())
    delta_phi = np.arctan2(np.sin(raw_delta_phi), np.cos(raw_delta_phi))
    result["dimuon_px"] = px
    result["dimuon_py"] = py
    result["dimuon_pz"] = pz
    result["dimuon_energy"] = energy
    result["dimuon_pt"] = pt
    result["dimuon_phi"] = np.arctan2(py, px)
    result["dimuon_eta"] = np.arcsinh(np.divide(pz, pt, out=np.full(len(result), np.nan), where=pt > 0))
    result["dimuon_mass"] = np.sqrt(np.maximum(energy**2 - px**2 - py**2 - pz**2, 0.0))
    result["delta_eta"] = delta_eta
    result["delta_phi"] = delta_phi
    result["delta_r"] = np.hypot(delta_eta, delta_phi)
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
    _required(muons, ("event_id", "object_index", "PT", "Eta", "Phi", "Charge", ), "build_dimuons",)
    excluded = {
        "event_id",
        "source_file_id",
        "source_entry",
    }
    attributes = [column for column in muons.columns if column not in excluded]
    def signed_muons(charge: int, prefix: str) -> pd.DataFrame:
        if charge > 0:
            mask = muons["Charge"].gt(0)
        else:
            mask = muons["Charge"].lt(0)
        selected = muons.loc[mask, ["event_id", *attributes]].copy()
        renamed = {column: (f"{prefix}_" f"{_COLUMN_NAMES.get(column, column.lower())}") for column in attributes}
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
    truth = truth.rename(
        columns={column: f"truth_{column}" for column in ("object_index", "root_uid", *available_fields)})
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
        "E",
    ]
    if "Charge" in leading_muons.columns:
        columns.append("Charge")
    renamed = leading_muons[columns].rename(
        columns={
            "PT": "Muon_PT",
            "Eta": "Muon_Eta",
            "Phi": "Muon_Phi",
            "PX": "Muon_px",
            "PY": "Muon_py",
            "PZ": "Muon_pz",
            "E": "Muon_E",
            "Charge": "Muon_Charge",
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

def compute_charge_asymmetry(
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