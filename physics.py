"""Física vectorizada para árboles Delphes leídos con Uproot y Awkward.

Este módulo está pensado para guardarse en ``MadGraph/UpROOT/physics.py``.
No abre archivos ROOT: recibe el ``ak.Array`` producido por ``io.load_awkward``.

Ejemplo mínimo
--------------
>>> import UpROOT.io as io
>>> import UpROOT.physics as physics
>>> events = io.load_awkward(
...     branches=physics.delphes_branches(), dataset="mpmm_10k_1"
... )
>>> muons = physics.select_muons(physics.build_muons(events), min_pt=1.0)
>>> candidates = physics.build_dimuons(muons, events=events)
>>> masses = physics.histogram_values(candidates, "mass")
>>> table = physics.candidates_to_dataframe(candidates)

Los momentos y masas se expresan en GeV; D0, DZ y sus errores, en mm.
La estructura ``evento -> objetos/candidatos`` se conserva hasta convertir
explícitamente el resultado a NumPy o a un DataFrame.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal

import awkward as ak
import numpy as np
import pandas as pd
import vector


vector.register_awkward()

MUON_MASS_GEV = 0.1056583755
ELECTRON_MASS_GEV = 0.00051099895

ChargeSelection = Literal["opposite", "same", "all"]
DisplacementCategory = Literal["prompt", "displaced", "mixed"]
DatasetSchema = Literal["delphes", "dvntuple", "b2hhh", "unknown"]


def _root_branch(collection: str, attribute: str) -> str:
    """Construye el nombre completo que espera el ``io.py`` del repositorio."""
    return f"{collection}/{collection}.{attribute}"


def delphes_branches(
    *,
    impact_parameters: bool = True,
    isolation: bool = False,
    event_information: bool = True,
    missing_energy: bool = False,
    jets: bool = False,
) -> list[str]:
    """Devuelve ramas ligeras para reconstruir dimuones; excluye Particle.*.

    El resultado puede pasarse directamente a ``io.load_awkward``. Las ramas
    ``Particle.*`` se solicitan aparte mediante :func:`truth_branches`, ya que
    pueden consumir órdenes de magnitud más memoria que las ramas de muones.
    """
    branches = [
        _root_branch("Muon", "PT"),
        _root_branch("Muon", "Eta"),
        _root_branch("Muon", "Phi"),
        _root_branch("Muon", "Charge"),
    ]

    if impact_parameters:
        branches.extend(
            _root_branch("Muon", attribute)
            for attribute in ("D0", "DZ", "ErrorD0", "ErrorDZ")
        )

    if isolation:
        branches.extend(
            _root_branch("Muon", attribute)
            for attribute in ("IsolationVar", "IsolationVarRhoCorr")
        )

    if event_information:
        branches.extend(
            _root_branch("Event", attribute)
            for attribute in ("Number", "Weight")
        )

    if missing_energy:
        branches.extend(
            _root_branch("MissingET", attribute)
            for attribute in ("MET", "Eta", "Phi")
        )
        branches.append(_root_branch("ScalarHT", "HT"))

    if jets:
        branches.extend(
            _root_branch("Jet", attribute)
            for attribute in ("PT", "Eta", "Phi", "Mass", "BTag")
        )

    return list(dict.fromkeys(branches))


def truth_branches(
    *,
    status: bool = False,
    vertices: bool = False,
    genealogy: bool = False,
) -> list[str]:
    """Devuelve ramas opcionales de partículas generadas; pueden ser pesadas."""
    attributes = ["PID", "PT", "Eta", "Phi", "Mass", "Charge"]

    if status:
        attributes.append("Status")
    if vertices:
        attributes.extend(("X", "Y", "Z", "T"))
    if genealogy:
        attributes.extend(("M1", "M2", "D1", "D2"))

    return [_root_branch("Particle", attribute) for attribute in attributes]


def detect_schema(events: ak.Array) -> DatasetSchema:
    """Distingue las tres estructuras presentes en el repositorio ECFM-PF."""
    fields = {str(field).rsplit("/", 1)[-1] for field in ak.fields(events)}

    if {"Muon.PT", "Muon.Eta", "Muon.Phi"}.issubset(fields):
        return "delphes"
    if "Muon" in fields and _has_nested_field(events, "Muon", "PT"):
        return "delphes"
    if {"Bplus_M", "J_psi_1S_M"}.issubset(fields):
        return "dvntuple"
    if {"H1_PX", "H2_PX", "H3_PX"}.issubset(fields):
        return "b2hhh"

    return "unknown"


def _has_nested_field(events: ak.Array, collection: str, attribute: str) -> bool:
    if collection not in ak.fields(events):
        return False

    nested_fields = set(ak.fields(events[collection]))
    return attribute in nested_fields or f"{collection}.{attribute}" in nested_fields


def get_field(
    events: ak.Array,
    name: str,
    *,
    required: bool = True,
    default: Any = None,
) -> Any:
    """Acepta ``Muon/Muon.PT``, ``Muon.PT`` y colecciones anidadas.

    También admite nombres ``Muon_PT`` usados por algunas conversiones.
    """
    available_fields = set(ak.fields(events))
    clean_name = str(name).rsplit("/", 1)[-1]

    candidates = [str(name), clean_name]
    if "." in clean_name:
        collection, attribute = clean_name.split(".", 1)
        candidates.extend(
            (
                _root_branch(collection, attribute),
                f"{collection}_{attribute}",
            )
        )

        if collection in available_fields:
            nested = events[collection]
            nested_fields = set(ak.fields(nested))
            if attribute in nested_fields:
                return nested[attribute]
            if clean_name in nested_fields:
                return nested[clean_name]

    for candidate in dict.fromkeys(candidates):
        if candidate in available_fields:
            return events[candidate]

    if not required:
        return default

    schema = detect_schema(events)
    extra = ""
    if schema in {"dvntuple", "b2hhh"}:
        extra = (
            f" El conjunto corresponde al esquema {schema!r}, no a un árbol "
            "Delphes; sus variables necesitan un módulo de física diferente."
        )

    available_preview = ", ".join(sorted(available_fields)[:12])
    raise KeyError(
        f"No se encontró la rama {name!r}. Campos disponibles: "
        f"{available_preview or '(ninguno)'}.{extra}"
    )


def _validate_same_multiplicity(
    reference: ak.Array,
    values: ak.Array,
    field_name: str,
) -> None:
    """Impide combinar atributos que no representen los mismos objetos."""
    try:
        reference_counts = ak.num(reference, axis=1)
        value_counts = ak.num(values, axis=1)
    except (ValueError, np.exceptions.AxisError) as exc:
        raise ValueError(
            f"La rama {field_name!r} no tiene estructura evento -> objetos."
        ) from exc

    if not bool(ak.all(reference_counts == value_counts)):
        raise ValueError(
            f"La multiplicidad por evento de {field_name!r} no coincide "
            "con la de la colección principal."
        )


def build_objects(
    events: ak.Array,
    collection: str,
    *,
    mass: float | None = None,
    charge_required: bool = False,
    sort_by_pt: bool = False,
) -> ak.Array:
    """Construye cuatrimomentos irregulares para Muon, Electron, Jet o Particle."""
    prefix = str(collection).strip()
    if not prefix:
        raise ValueError("Debes indicar el nombre de una colección Delphes.")

    pt = get_field(events, f"{prefix}.PT")
    eta = get_field(events, f"{prefix}.Eta")
    phi = get_field(events, f"{prefix}.Phi")

    for field_name, values in (("Eta", eta), ("Phi", phi)):
        _validate_same_multiplicity(pt, values, f"{prefix}.{field_name}")

    if mass is None:
        defaults = {"Muon": MUON_MASS_GEV, "Electron": ELECTRON_MASS_GEV, "Photon": 0.0}
        mass_values = get_field(events, f"{prefix}.Mass", required=False)
        if mass_values is None:
            if prefix not in defaults:
                raise KeyError(
                    f"La colección {prefix!r} no contiene {prefix}.Mass. "
                    "Indica mass explícitamente si conoces su valor."
                )
            mass_values = ak.full_like(pt, defaults[prefix], dtype=np.float64)
        else:
            _validate_same_multiplicity(pt, mass_values, f"{prefix}.Mass")
    else:
        if mass < 0:
            raise ValueError("La masa de los objetos no puede ser negativa.")
        mass_values = ak.full_like(pt, mass, dtype=np.float64)

    payload: dict[str, Any] = {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "mass": mass_values,
    }

    optional_fields = {
        "Charge": "charge",
        "D0": "d0",
        "DZ": "dz",
        "ErrorD0": "error_d0",
        "ErrorDZ": "error_dz",
        "IsolationVar": "isolation",
        "IsolationVarRhoCorr": "isolation_rho",
        "SumPt": "sum_pt",
        "SumPtCharged": "sum_pt_charged",
        "SumPtNeutral": "sum_pt_neutral",
        "SumPtChargedPU": "sum_pt_charged_pu",
        "BTag": "btag",
        "PID": "pid",
        "Status": "status",
        "X": "x_vertex",
        "Y": "y_vertex",
        "Z": "z_vertex",
        "T": "time",
        "M1": "mother_1",
        "M2": "mother_2",
        "D1": "daughter_1",
        "D2": "daughter_2",
    }

    for original_name, output_name in optional_fields.items():
        values = get_field(events, f"{prefix}.{original_name}", required=False)
        if values is not None:
            _validate_same_multiplicity(pt, values, f"{prefix}.{original_name}")
            payload[output_name] = values

    if charge_required and "charge" not in payload:
        raise KeyError(f"La colección {prefix!r} requiere la rama {prefix}.Charge.")

    objects = ak.zip(payload, with_name="Momentum4D")

    if sort_by_pt:
        ordering = ak.argsort(objects.pt, axis=1, ascending=False, stable=True)
        objects = objects[ordering]

    return objects


def build_muons(
    events: ak.Array,
    *,
    mass: float = MUON_MASS_GEV,
    sort_by_pt: bool = True,
) -> ak.Array:
    """Reconstruye muones; Delphes no guarda normalmente una rama Muon.Mass."""
    return build_objects(
        events,
        "Muon",
        mass=mass,
        charge_required=True,
        sort_by_pt=sort_by_pt,
    )


def build_electrons(
    events: ak.Array,
    *,
    mass: float = ELECTRON_MASS_GEV,
    sort_by_pt: bool = True,
) -> ak.Array:
    """Reconstruye electrones con masa fija en GeV."""
    return build_objects(
        events,
        "Electron",
        mass=mass,
        charge_required=True,
        sort_by_pt=sort_by_pt,
    )


def build_jets(events: ak.Array, *, sort_by_pt: bool = True) -> ak.Array:
    """Reconstruye jets utilizando la masa guardada en Jet.Mass."""
    return build_objects(events, "Jet", sort_by_pt=sort_by_pt)


def build_generated_particles(
    events: ak.Array,
    *,
    sort_by_pt: bool = False,
) -> ak.Array:
    """Reconstruye Particle.* únicamente cuando fue cargado de forma explícita."""
    particles = build_objects(events, "Particle", sort_by_pt=sort_by_pt)
    if "pid" not in ak.fields(particles):
        raise KeyError("Para identificar partículas generadas necesitas Particle.PID.")
    return particles


def select_generated_particles(
    particles: ak.Array,
    *,
    pdg_id: int,
    include_antiparticle: bool = True,
    status: int | None = None,
) -> ak.Array:
    """Selecciona una especie generada, por ejemplo ``pdg_id=13`` para muones."""
    if "pid" not in ak.fields(particles):
        raise KeyError("El arreglo no contiene el campo 'pid'.")

    if include_antiparticle:
        mask = np.abs(particles.pid) == abs(pdg_id)
    else:
        mask = particles.pid == pdg_id

    if status is not None:
        if "status" not in ak.fields(particles):
            raise KeyError("Carga Particle.Status para seleccionar por estado.")
        mask = mask & (particles.status == status)

    return particles[mask]


def impact_parameter_significance(
    objects: ak.Array,
    *,
    component: Literal["d0", "dz"] = "d0",
) -> ak.Array:
    """Calcula |D0|/ErrorD0 o |DZ|/ErrorDZ sin divisiones por cero."""
    if component not in {"d0", "dz"}:
        raise ValueError("component debe ser 'd0' o 'dz'.")

    error_name = f"error_{component}"
    available = set(ak.fields(objects))
    missing = {component, error_name} - available
    if missing:
        names = ", ".join(sorted(missing))
        raise KeyError(f"Faltan campos para calcular la significancia: {names}.")

    displacement = objects[component]
    uncertainty = objects[error_name]
    valid = uncertainty > 0
    safe_uncertainty = ak.where(valid, uncertainty, 1.0)
    significance = np.abs(displacement) / safe_uncertainty
    return ak.where(valid, significance, np.nan)


def select_muons(
    muons: ak.Array,
    *,
    min_pt: float | None = None,
    max_abs_eta: float | None = None,
    min_eta: float | None = None,
    max_eta: float | None = None,
    max_isolation: float | None = None,
    min_abs_d0: float | None = None,
    max_abs_d0: float | None = None,
    min_d0_significance: float | None = None,
    max_d0_significance: float | None = None,
) -> ak.Array:
    """Aplica cortes sin aplanar ni eliminar eventos sin muones seleccionados."""
    if min_pt is not None and min_pt < 0:
        raise ValueError("min_pt no puede ser negativo.")
    if max_abs_eta is not None and max_abs_eta < 0:
        raise ValueError("max_abs_eta no puede ser negativo.")
    if min_eta is not None and max_eta is not None and min_eta > max_eta:
        raise ValueError("min_eta no puede ser mayor que max_eta.")

    mask = ak.ones_like(muons.pt, dtype=np.bool_)

    if min_pt is not None:
        mask = mask & (muons.pt >= min_pt)
    if max_abs_eta is not None:
        mask = mask & (np.abs(muons.eta) <= max_abs_eta)
    if min_eta is not None:
        mask = mask & (muons.eta >= min_eta)
    if max_eta is not None:
        mask = mask & (muons.eta <= max_eta)

    if max_isolation is not None:
        if "isolation" not in ak.fields(muons):
            raise KeyError("Carga Muon.IsolationVar para aplicar max_isolation.")
        mask = mask & (muons.isolation <= max_isolation)

    if min_abs_d0 is not None or max_abs_d0 is not None:
        if "d0" not in ak.fields(muons):
            raise KeyError("Carga Muon.D0 para aplicar cortes sobre D0.")
        abs_d0 = np.abs(muons.d0)
        if min_abs_d0 is not None:
            mask = mask & (abs_d0 >= min_abs_d0)
        if max_abs_d0 is not None:
            mask = mask & (abs_d0 <= max_abs_d0)

    if min_d0_significance is not None or max_d0_significance is not None:
        significance = impact_parameter_significance(muons, component="d0")
        if min_d0_significance is not None:
            mask = mask & (significance >= min_d0_significance)
        if max_d0_significance is not None:
            mask = mask & (significance <= max_d0_significance)

    return muons[mask]


def delta_phi(first: ak.Array, second: ak.Array) -> ak.Array:
    """Devuelve Δφ restringido al intervalo [-π, π]."""
    difference = first.phi - second.phi
    return np.arctan2(np.sin(difference), np.cos(difference))


def delta_r(first: ak.Array, second: ak.Array) -> ak.Array:
    """Calcula ΔR = sqrt((Δη)² + (Δφ)²)."""
    return np.sqrt((first.eta - second.eta) ** 2 + delta_phi(first, second) ** 2)


def invariant_mass(*objects: ak.Array) -> ak.Array:
    """Suma dos o más cuatrimomentos y devuelve su masa invariante en GeV."""
    if len(objects) < 2:
        raise ValueError("Se requieren al menos dos cuatrimomentos.")

    total = objects[0]
    for current in objects[1:]:
        total = total + current

    return total.mass


def _event_scalar(
    events: ak.Array,
    field_name: str,
    *,
    default: float | int | ak.Array,
) -> ak.Array:
    """Convierte ramas Event/MissingET de cero o un objeto en valores por evento."""
    values = get_field(events, field_name, required=False)
    if values is None:
        return ak.Array(default) if isinstance(default, np.ndarray) else default

    try:
        counts = ak.num(values, axis=1)
    except (ValueError, np.exceptions.AxisError):
        return values

    if bool(ak.any(counts > 1)):
        raise ValueError(
            f"La rama {field_name!r} contiene más de un valor por evento; "
            "no es posible convertirla a un escalar sin perder información."
        )

    first = ak.firsts(values, axis=1)
    if isinstance(default, ak.Array):
        valid = ~ak.is_none(first)
        return ak.where(valid, ak.fill_none(first, 0), default)

    return ak.fill_none(first, default)


def build_dimuons(
    muons: ak.Array,
    *,
    events: ak.Array | None = None,
    charge: ChargeSelection = "opposite",
    min_mass: float | None = None,
    max_mass: float | None = None,
) -> ak.Array:
    """Forma todos los pares por evento y calcula observables de dimuones.

    Los eventos con menos de dos muones se conservan como listas vacías.
    ``charge='opposite'`` es apropiado para resonancias μ⁺μ⁻; ``'same'``
    proporciona una muestra de control y ``'all'`` conserva ambas opciones.
    """
    if charge not in {"opposite", "same", "all"}:
        raise ValueError("charge debe ser 'opposite', 'same' o 'all'.")
    if min_mass is not None and max_mass is not None and min_mass > max_mass:
        raise ValueError("min_mass no puede ser mayor que max_mass.")
    if "charge" not in ak.fields(muons):
        raise KeyError("Los muones necesitan el campo 'charge'.")
    if events is not None and len(events) != len(muons):
        raise ValueError("events y muons deben contener la misma cantidad de eventos.")

    pairs = ak.combinations(muons, 2, axis=1, fields=("first", "second"))
    first = pairs.first
    second = pairs.second

    if charge == "opposite":
        pair_mask = first.charge * second.charge < 0
    elif charge == "same":
        pair_mask = first.charge * second.charge > 0
    else:
        pair_mask = ak.ones_like(first.charge, dtype=np.bool_)

    first = first[pair_mask]
    second = second[pair_mask]
    system = first + second

    event_index = ak.local_index(muons, axis=0)
    event_index, candidate_mass = ak.broadcast_arrays(event_index, system.mass)
    muon_count, _ = ak.broadcast_arrays(ak.num(muons, axis=1), candidate_mass)

    payload: dict[str, Any] = {
        "event_index": event_index,
        "n_muons": muon_count,
        "mass": candidate_mass,
        "pt": system.pt,
        "eta": system.eta,
        "phi": system.phi,
        "rapidity": system.rapidity,
        "charge": first.charge + second.charge,
        "delta_eta": first.eta - second.eta,
        "delta_phi": delta_phi(first, second),
        "delta_r": delta_r(first, second),
        "mu1_pt": first.pt,
        "mu1_eta": first.eta,
        "mu1_phi": first.phi,
        "mu1_charge": first.charge,
        "mu2_pt": second.pt,
        "mu2_eta": second.eta,
        "mu2_phi": second.phi,
        "mu2_charge": second.charge,
    }

    optional_muon_fields = (
        "d0",
        "dz",
        "error_d0",
        "error_dz",
        "isolation",
        "isolation_rho",
    )
    available_muon_fields = set(ak.fields(muons))
    for field_name in optional_muon_fields:
        if field_name in available_muon_fields:
            payload[f"mu1_{field_name}"] = first[field_name]
            payload[f"mu2_{field_name}"] = second[field_name]

    for component in ("d0", "dz"):
        required = {component, f"error_{component}"}
        if required.issubset(available_muon_fields):
            first_significance = impact_parameter_significance(first, component=component)
            second_significance = impact_parameter_significance(second, component=component)
            payload[f"mu1_{component}_significance"] = first_significance
            payload[f"mu2_{component}_significance"] = second_significance
            payload[f"min_{component}_significance"] = np.minimum(
                first_significance, second_significance
            )
            payload[f"max_{component}_significance"] = np.maximum(
                first_significance, second_significance
            )

    if events is not None:
        source_index = ak.local_index(events, axis=0)
        event_number = _event_scalar(events, "Event.Number", default=source_index)
        event_weight = _event_scalar(events, "Event.Weight", default=1.0)
        payload["event_number"] = ak.broadcast_arrays(event_number, candidate_mass)[0]
        payload["event_weight"] = ak.broadcast_arrays(event_weight, candidate_mass)[0]

        for input_name, output_name in (
            ("MissingET.MET", "met"),
            ("MissingET.Phi", "met_phi"),
            ("ScalarHT.HT", "scalar_ht"),
        ):
            if get_field(events, input_name, required=False) is not None:
                values = _event_scalar(events, input_name, default=np.nan)
                payload[output_name] = ak.broadcast_arrays(values, candidate_mass)[0]

    candidates = ak.zip(payload)

    if min_mass is not None:
        candidates = candidates[candidates.mass >= min_mass]
    if max_mass is not None:
        candidates = candidates[candidates.mass <= max_mass]

    return candidates


def select_dimuons(
    candidates: ak.Array,
    *,
    min_mass: float | None = None,
    max_mass: float | None = None,
    min_pt: float | None = None,
    max_delta_r: float | None = None,
    category: DisplacementCategory | None = None,
    prompt_max_d0_significance: float = 3.0,
    displaced_min_d0_significance: float = 5.0,
) -> ak.Array:
    """Selecciona candidatos y clasifica pares prompt/desplazados opcionalmente.

    Los umbrales 3 y 5 son configurables y constituyen cortes de análisis, no
    una reconstrucción de vértice secundario ni una identificación universal.
    """
    if category is not None and category not in {"prompt", "displaced", "mixed"}:
        raise ValueError("category debe ser 'prompt', 'displaced', 'mixed' o None.")
    if min_mass is not None and max_mass is not None and min_mass > max_mass:
        raise ValueError("min_mass no puede ser mayor que max_mass.")

    mask = ak.ones_like(candidates.mass, dtype=np.bool_)

    if min_mass is not None:
        mask = mask & (candidates.mass >= min_mass)
    if max_mass is not None:
        mask = mask & (candidates.mass <= max_mass)
    if min_pt is not None:
        mask = mask & (candidates.pt >= min_pt)
    if max_delta_r is not None:
        mask = mask & (candidates.delta_r <= max_delta_r)

    if category is not None:
        available_fields = set(ak.fields(candidates))
        required = {"min_d0_significance", "max_d0_significance"}
        if not required.issubset(available_fields):
            raise KeyError(
                "Para clasificar candidatos necesitas Muon.D0 y Muon.ErrorD0."
            )

        is_prompt = candidates.max_d0_significance <= prompt_max_d0_significance
        is_displaced = candidates.min_d0_significance >= displaced_min_d0_significance

        if category == "prompt":
            mask = mask & is_prompt
        elif category == "displaced":
            mask = mask & is_displaced
        else:
            mask = mask & ~is_prompt & ~is_displaced

    return candidates[mask]


def best_mass_candidate(candidates: ak.Array, target_mass: float) -> ak.Array:
    """Elige por evento el candidato más cercano a una masa objetivo en GeV.

    Los eventos sin candidatos se devuelven como ``None``; no se pierden.
    """
    if target_mass < 0:
        raise ValueError("target_mass no puede ser negativo.")

    ordering = ak.argsort(np.abs(candidates.mass - target_mass), axis=1)
    return ak.firsts(candidates[ordering], axis=1)


def leading_pt_candidate(candidates: ak.Array) -> ak.Array:
    """Devuelve el candidato de mayor pT por evento, o None si no existe."""
    ordering = ak.argsort(candidates.pt, axis=1, ascending=False, stable=True)
    return ak.firsts(candidates[ordering], axis=1)


def histogram_values(
    candidates: ak.Array,
    field: str = "mass",
    *,
    drop_nonfinite: bool = True,
) -> np.ndarray:
    """Aplana únicamente la variable solicitada para usarla en histogramas."""
    if field not in ak.fields(candidates):
        raise KeyError(f"El candidato no contiene el campo {field!r}.")

    values = ak.flatten(candidates[field], axis=1)
    values = ak.drop_none(values)
    result = np.asarray(ak.to_numpy(values))

    if drop_nonfinite:
        result = result[np.isfinite(result)]

    return result


def weighted_histogram(
    candidates: ak.Array,
    *,
    field: str = "mass",
    bins: int | Sequence[float] = 80,
    value_range: tuple[float, float] | None = None,
    weight_field: str = "event_weight",
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula un histograma y usa pesos de evento cuando están disponibles."""
    if field not in ak.fields(candidates):
        raise KeyError(f"El candidato no contiene el campo {field!r}.")

    values = np.asarray(ak.to_numpy(ak.flatten(candidates[field], axis=1)))
    weights: np.ndarray | None = None

    if weight_field in ak.fields(candidates):
        weights = np.asarray(
            ak.to_numpy(ak.flatten(candidates[weight_field], axis=1))
        )

    valid = np.isfinite(values)
    if weights is not None:
        valid = valid & np.isfinite(weights)
        weights = weights[valid]

    return np.histogram(values[valid], bins=bins, range=value_range, weights=weights)


def candidates_to_dataframe(
    candidates: ak.Array,
    *,
    fields: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Crea una tabla compacta: una fila por candidato, no por Particle.*."""
    available_fields = list(ak.fields(candidates))
    selected_fields = (
        available_fields if fields is None else list(dict.fromkeys(str(name) for name in fields))
    )

    missing = [name for name in selected_fields if name not in available_fields]
    if missing:
        raise KeyError(f"Campos de candidatos no disponibles: {', '.join(missing)}.")

    flattened = ak.flatten(candidates, axis=1)
    columns: dict[str, np.ndarray] = {}

    for field_name in selected_fields:
        values = flattened[field_name]
        if ak.fields(values):
            raise ValueError(f"El campo {field_name!r} contiene registros anidados.")

        try:
            array = ak.to_numpy(values, allow_missing=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"El campo {field_name!r} no es escalar por candidato."
            ) from exc

        if isinstance(array, np.ma.MaskedArray):
            if np.issubdtype(array.dtype, np.number):
                array = array.astype(np.float64).filled(np.nan)
            else:
                array = array.astype(object).filled(None)

        columns[field_name] = np.asarray(array)

    return pd.DataFrame(columns)


def summarize_candidates(candidates: ak.Array) -> dict[str, int]:
    """Resume eventos, candidatos y eventos con al menos un candidato."""
    multiplicities = ak.num(candidates, axis=1)
    return {
        "events": len(candidates),
        "events_with_candidates": int(ak.sum(multiplicities > 0)),
        "candidates": int(ak.sum(multiplicities)),
        "maximum_candidates_per_event": (
            int(ak.max(multiplicities)) if len(multiplicities) else 0
        ),
    }


__all__ = [
    "ELECTRON_MASS_GEV",
    "MUON_MASS_GEV",
    "best_mass_candidate",
    "build_dimuons",
    "build_electrons",
    "build_generated_particles",
    "build_jets",
    "build_muons",
    "build_objects",
    "candidates_to_dataframe",
    "delphes_branches",
    "delta_phi",
    "delta_r",
    "detect_schema",
    "get_field",
    "histogram_values",
    "impact_parameter_significance",
    "invariant_mass",
    "leading_pt_candidate",
    "select_dimuons",
    "select_generated_particles",
    "select_muons",
    "summarize_candidates",
    "truth_branches",
    "weighted_histogram",
]
