from collections.abc import Iterable, Sequence
from typing import Any, Literal
import awkward as ak
import numpy as np
import pandas as pd
import vector

vector.register_awkward()
MUON_M_GEV = 0.1056583745
ELECTRON_M_GEV = 0.0005109989461
Z_M_GEV = 91.1876

ChargeSel = Literal['op', 'same', 'all']
Displacement = Literal['prompt', 'displaced', 'mixed']
DatasetSchema = Literal["delphes", 'dvntuple', 'b2hhh', 'other']

def _root_branch(collection: str, atribute: str) -> str:
    return f"{collection}/{collection}.{atribute}"

def delphes_branches(
    *,
    impact_parameters: bool = True,
    isolation: bool = False, 
    event_inf: bool = False, 
    miss_energy: bool = False, 
    jets: bool = False
) -> list[str]:
    branches = [
        _root_branch('Muon', 'PT'),
        _root_branch('Muon', 'Eta'),
        _root_branch('Muon', 'Phi'),
        _root_branch('Muon', 'Charge')
    ]
    if impact_parameters:
        branches.extend(_root_branch('Muon', atribute) for atribute in ('D0', 'DZ', 'ErrorD0', 'ErrorDZ'))
    if isolation:
        branches.extend(_root_branch('Muon', atribute) for atribute in ('IsolationVar', 'IsolationVarRhoCorr'))
    if event_inf:
        branches.extend(_root_branch('Event', atribute) for atribute in ('Number', 'Weight'))
    if miss_energy:
        branches.extend(_root_branch('MissingET', atribute) for atribute in ('MET', 'Eta', 'Phi'))
        branches
    if miss_energy:
        branches.extend(_root_branch('MissingET', atribute) for atribute in ('MET', 'Eta', 'Phi'))
        branches.append(_root_branch('ScalarHT', 'HT'))
    if jets:
        branches.extend(_root_branch('Jet', atribute) for atribute in ('PT', 'Eta', 'Phi', 'Mass', 'BTag'))
    return list(dict.fromkeys(branches))

def _nested_field(events: ak.Array, collection: str, atribute: str) -> bool:
    if collection not in ak.fields(events):
        return False
    n_fields = set(ak.fields(events[collection]))
    return atribute in n_fields or f'{collection}.{atribute}' in n_fields

def _validate_same_mult(
    reference: ak.Array,
    values: ak.Array,
    f_name: str
) -> None:
    try:
        reference_counts = ak.num(reference, axis=1)
        value_counts = ak.num(values, axis=1)
    except (ValueError, np.exceptions.AxisError) as e:
        raise ValueError(f'Rama {f_name!r}, sin estructura evento -> objeto.')
    if not bool(ak.all(reference_counts == value_counts)):
        raise ValueError(f'Multiplicidad por evento: {f_name!r}.')

def truth_branches(     # Ramas opcionales
    *, 
    status: bool = False,
    vertices: bool = False,
    genealogy: bool = False
) -> list[str]:
    atributes = ['PID', 'PT', 'Eta', 'Phi', 'Mass', 'Charge']
    if status:
        atributes.append('Status')
    if vertices:
        atributes.extend(('X', 'Y', 'Z', 'T'))
    if genealogy:
        atributes.extend(('M1', 'M2', 'D1', 'D2'))
    return [_root_branch('Particle', atribute) for atribute in atributes]

def detect_schema(events: ak.array) -> DatasetSchema:
    fields = {str(field).rsplit('/', 1)[-1] for field in ak.fields(events)}
    if {'Muon.PT', 'Muon.Eta', 'Muon.Phi'}.issubset(fields):
        return 'delphes'
    if 'Muon' in fields and _nested_field(events, 'Muon', 'PT'):
        return 'delphes'
    if {'Bplus_M', 'J_psi_1S_M'}.issubset(fields):
        return 'dvntuple'
    if {'H1_PX', 'H2_PX', 'H3_PX'}.issubset(fields):
        return 'b2hhh'
    return 'other'

def get_field(
    events: ak.Array,
    name: str,
    *,
    required: bool = True,
    default: Any = None
) -> Any:
    av_fields = set(ak.fields(events))
    c_name = str(name).rsplit('/', 1)[-1]
    candidates = [str(name), c_name]
    if '.' in c_name:
        collection, atribute = c_name.split('.', 1)
        candidates.extend((_root_branch(collection, atribute), f'{collection}_{atribute}',))
        if collection in av_fields:
            nested = events[collection]
            nested_fields = set(ak.fields(nested))
            if atribute in nested_fields:
                return nested[atribute]
            if c_name in nested_fields:
                return nested[c_name]
    for candidate in dict.fromkeys(candidates):
        if candidate in av_fields:
            return events[candidate]
    if not required:
        return default
    schema = detect_schema(events)
    extra = ''
    if schema in {'dvntuple', 'b2hhh'}:
        extra = (f"Esquema: {schema!r}, no pertenece a un árbol Delphes.")
    av_preview = ', '.join(sorted(av_fields)[:12])
    raise KeyError(f'No se encontró la rama {name!r}. Campos disponibles: {av_fields or '(0)'}.{extra}')

def objects(
    events: ak.Array,
    collection: str,
    *,
    mass: float | None = None,
    charge: bool = False,
    sort_pt: bool = False
) -> ak.Array:
    prefix = str(collection).strip()
    if not prefix:
        raise ValueError('Indique nombre de colección Delphes.')
    pt = get_field(events, f'{prefix}.PT')
    eta = get_field(events, f'{prefix}.Eta')
    phi = get_field(events, f'{prefix}.Phi')
    for f_name, values in (('Eta', eta), ('Phi', phi)):
        _validate_same_mult(pt, values, f'{prefix}.{f_name}')
    if mass is None:
        defaults = {'Muon': MUON_M_GEV, 'Electron': ELECTRON_M_GEV, 'Photon': 0.0, 'Z': Z_M_GEV}
        mass_val = get_field(events, f'{prefix}.Mass', required=False)
        if mass_val is None:
            if prefix not in defaults:
                raise KeyError(f'Colección {prefix!r} no contiene {prefix}.Mass.')
            mass_val = ak.full_like(pt, defaults[prefix], dtype=np.float64)
        else:
            _validate_same_mult(pt, mass_val, f'{prefix}.Mass')
    else:
        if mass < 0:
            raise ValueError('Masa negativa')
        mass_val = ak.full_like(pt, mass, dtype=np.float64)
    payload: dict[str, Any] = {
        'pt': pt,
        'eta': eta,
        'phi': phi,
        'mass': mass_val
    }
    opt_fields = {
        'Charge': 'charge',
        'D0': 'd0',
        'DZ': 'dz',
        'ErrorD0': 'error_d0',
        'ErrorDZ': 'error_dz',
        'IsolationVar': 'isolation',
        'IsolationVarRhoCorr': 'isolation_rho',
        'SumPt': 'sum_pt',
        'SumPtCharged': 'sum_pt_charged',
        'SumPtNeutral': 'sum_pt_neutral',
        'BTag': 'btag',
        'PID': 'pid',
        'Status': 'status',
        'X': 'x_vertex',
        'Y': 'y_vertex',
        'Z': 'z_vertex',
        'T': 'time',
        'M1': 'mother_1',
        'M2': 'mother_2',
        'D1': 'daughter_1',
        'D2': 'daughter2'
    }
    for or_name, out_name in opt_fields.items():
        values = get_field(events, f'{prefix}.{or_name}', required=False)
        if values is not None:
            _validate_same_mult(pt, values, f'{prefix}.{or_name}')
            payload[out_name] = values
    if charge and 'charge' not in payload:
        raise KeyError(f'La colección {prefix!r} requiere la rama {prefix}.Charge')
    obj = ak.zip(payload, with_name='Momentum4D')
    if sort_pt:
        ord = ak.argsort(obj.pt, axis=1, ascending=False, stable=True)
        obj = obj[ord]
    return obj

def muons(
    events: ak.Array,
    *,
    mass: float = MUON_M_GEV,
    sort_pt: bool = True
) -> ak.Array:
    return objects(events, 'Muon', mass=mass, charge=True, sort_pt=sort_pt)

def electrons(
    events: ak.Array,
    *,
    mass: float = ELECTRON_M_GEV,
    sort_pt: bool = True
) -> ak.Array:
    return objects(events, 'Electron', mass=mass, charge=True, sort_pt=sort_pt)

def jets(
    events: ak.Array,
    *,
    sort_pt: bool = True
) -> ak.Array:
    return objects(events, 'Jet', charge=True, sort_pt=sort_pt)

def generated_particles(
    events: ak.Array,
    *,
    sort_pt: bool = True
) -> ak.Array:
    particles =  objects(events, 'Particle', charge=True, sort_pt=sort_pt)
    if 'pid' not in ak.fields(particles):
        raise KeyError('Se necesita Particle.PID para identificar partículas')
    return particles

def sel_gen_particles(
    particles: ak.Array,
    *,
    pdg_id: int,
    antiparticle: bool = True,
    status: int | None = None
) -> ak.Array:
    if 'pid' not in ak.fields(particles):
        raise KeyError("El arreglo no contiene el campo 'pid'.")
    if antiparticle:
        mask = np.abs(particles.pid) == abs(pdg_id)
    else:
        mask = particles.pid == pdg_id
    if status is not None:
        if 'status' not in ak.fields(particles):
            raise KeyError('Carga Particle.Status para seleccionar por estado')
        mask = mask & (particles.status == status)
    return particles[mask]

def impact_parameter_sig(
    objects: ak.Array,
    *,
    component: Literal['d0', 'dz'] = 'd0'
) -> ak.Array:
    if component not in {'d0', 'dz'}:
        raise ValueError("campo component debe ser 'd0' o 'dz'.")
    e_name = f'error_{component}'
    available = set(ak.fields(objects))
    miss = {component, e_name} - available
    if miss:
        names = ", ".join(sorted(miss))
        raise KeyError(f'La significancia necesita: {names}.')
    displacement = objects[component]
    uncertainty = objects[e_name]
    valid = uncertainty > 0
    safe_unc = ak.where(valid, uncertainty, 1.0)
    significance = np.abs(displacement)/safe_unc
    return ak.where(valid, significance, np.nan)

def sel_muons(
    muons: ak.Array,
    *,
    min_pt: float | None = None,
    max_abs_eta: float | None = None,
    min_eta: float | None = None,
    max_eta: float | None = None,
    max_isolation: float | None = None,
    min_abs_d0: float | None = None,
    max_abs_d0: float | None = None,
    min_d0_sig: float | None = None,
    max_d0_sig: float | None = None
) -> ak.Array:
    if min_pt is not None and min_pt < 0:
        raise('min_pt no puede ser negativo.')
    if max_abs_eta is not None and max_abs_eta < 0:
        raise('max_abs_eta no puede ser negativo.')
    if min_eta is not None and min_eta < 0:
        raise('min_eta no puede ser negativo.')
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
        if 'isolation' not in ak.fields(muons):
            raise KeyError('Cargar Muon.IsolationVar para aplicar max_isolation.')
        mask = mask & (muons.isolation <= max_isolation)
    if min_d0_sig is not None or max_d0_sig is not None:
        sig = impact_parameter_sig(muons, component='d0')
        if min_d0_sig is not None:
            mask = mask & (sig >= min_d0_sig)
        if max_d0_sig is not None:
            mask = mask & (sig <= max_d0_sig)
    return muons[mask]

def delta_phi(
    first: ak.Array,
    second: ak.Array
) -> ak.Array:
    dif = first.phi - second.phi
    return np.arctan2(np.sin(dif), np.cos(dif))

def delta_r(first: ak.Array, second: ak.Array) -> ak.Array:
    return np.sqrt(
        (first.eta - second.eta)**2 + delta_phi(first, second)**2
    )

def mass_inv(*objects: ak.Array) -> ak.Array:
    if len(objects) < 2:
        raise ValueError('Se requieren por lo menos 2 cuatrimomentos.')
    total = objects[0]
    for current in objects[1:]:
        total = total + current
    return total.mass

def _scalar_ev(
    events: ak.Array,
    field: str,
    *,
    default: float | int | ak.Array
) -> ak.Array:
    val = get_field(events, field, required=False)
    if val is None:
        return ak.Array(default) if isinstance(default, np.ndarray) else default
    try:
        counts = ak.num(val, axis=1)
    except (ValueError, np.exceptions.AxisError):
        return val
    if bool(ak.any(counts > 1)):
        raise ValueError(f'Rama {field!r} contiene más de un valor por evento')
    first = ak.firsts(val, axis=1)
    if isinstance(default, ak.Array):
        valid = ~ak.is_none(first)
        return ak.where(valid, ak.fill_none(first, 0), default)
    return ak.fill_none(first, default)

def dimuons(
    muons: ak.Array,
    *,
    events: ak.Array | None = None,
    charge: ChargeSel = 'op',
    min_mass: float | None = None,
    max_mass: float | None = None
) -> ak.Array:
    if charge not in {'op', 'same', 'all'}:
        raise ValueError("charge = 'op' | 'same' | 'all'.")
    if min_mass is not None and max_mass is not None and min_mass > max_mass:
        raise ValueError("'min_mass' no puede ser mayor que 'max_mass'.")
    if 'charge' not in ak.fields(muons):
        raise KeyError("Los muones necesitan el campo 'charge'.")
    if events is not None and len(events) != len(muons):
        raise ValueError("'events' y 'muons' deben contener la misma cantidad de eventos.")
    pairs = ak.combinations(muons, 2, axis=1, fields=("first", "second"))
    first = pairs.first
    second = pairs.second
    if charge == 'op':
        p_mask = first.charge*second.charge < 0
    elif charge == 'same':
        p_mask = first.charge*second.charge > 0
    else:
        p_mask = ak.ones_like(first.charge, dtype=np.bool_)
    first = first[p_mask]
    second = second[p_mask]
    system = first + second
    event_index = ak.local_index(muons, axis=0)
    event_index, candidate_mass = ak.broadcast_arrays(event_index, system.mass)
    muons_count, _ = ak.broadcast_arrays(ak.num(muons, axis=1), candidate_mass)
    payload: dict[str, Any] = {
        'event_index': event_index,
        'n_muons': muons_count,
        'mass': candidate_mass,
        'pt': system.pt,
        'eta': system.eta,
        'phi': system.phi,
        'rapidity': system.rapidity,
        'charge': first.charge + second.charge,
        'delta_eta': first.eta - second.eta,
        'delta_phi': delta_phi(first, second),
        'delta_r': delta_r(first, second),
        'mu1_pt': first.pt,
        'mu1_eta': first.eta,
        'mu1_phi': first.phi,
        'mu1_charge': first.charge,
        'mu2_pt': second.pt,
        'mu2_eta': second.eta,
        'mu2_phi': second.phi,
        'mu2_charge': second.charge
    }
    optional_fields = ('d0', 'dz', 'error_d0', 'error_dz', 'isolation', 'isolation_rho')
    av_muon_fields = set(ak.fields(muons))
    for field in optional_fields:
        if field in av_muon_fields:
            payload[f'mu1_{field}'] = first[field]
            payload[f'mu2_{field}'] = second[field]
    for comp in ('d0', 'dz'):
        req = {comp, f'error_{comp}'}
        if req.issubset(av_muon_fields):
            first_sig = impact_parameter_sig(first, component=comp)
            second_sig = impact_parameter_sig(second, component=comp)
            payload[f'mu1_{comp}_sig'] = first_sig
            payload[f'mu2_{comp}_sig'] = second_sig
            payload[f'min_{comp}_sig'] = np.minimum(first_sig, second_sig)
            payload[f'max_{comp}_sig'] = np.maximum(first_sig, second_sig)
    if events is not None:
        source_i = ak.local_index(events, axis=0)
        event_n = _scalar_ev(events, 'Event.Number', default=source_i)
        event_w = _scalar_ev(events, 'Event.Weight', default=1.0)
        payload['event_number'] = ak.broadcast_arrays(event_n, candidate_mass)[0]
        payload['event_weight'] = ak.broadcast_arrays(event_w, candidate_mass)[0]
        for i_name, out_name in (
            ('MissingET.MET', 'met'),
            ('MissingET.Phi', 'met_phi'),
            ('ScalarHT.HT', 'scalar_ht')
        ):
            if get_field(events, i_name, out_name, required=False) is not None:
                val = _scalar_ev(events, i_name, default=np.nan)
                payload[out_name] = ak.broadcast_arrays(val, candidate_mass)[0]
    candidates = ak.zip(payload)
    if min_mass is not None:
        candidates = candidates[candidates.mass >= min_mass]
    if max_mass is not None:
        candidates = candidates[candidates.mass <= max_mass]
    return candidates

def sel_dimuons(
    candidates: ak.Array,
    *,
    min_mass: float | None = None,
    max_mass: float | None = None,
    min_pt: float | None = None,
    max_delta_r: float | None = None,
    category: Displacement | None = None,
    prompt_max_d0: float = 3.0,     # Corte de análisis 1
    displaced_min_d0: float = 5.0   # Corte de análisis 2
) -> ak.Array:
    if category is not None and category not in {'prompt', 'displaced', 'mixed'}:
        raise ValueError("category = 'prompt' | 'displaced' | 'mixed' | None.")
    if min_mass is not None and max_mass is not None and min_mass > max_mass:
        raise ValueError('max_mass > min_mass')
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
        av_fields = set(ak.fields(candidates))
        req = {'min_d0_sig', 'max_d0_sig'}
        if not req.issubset(av_fields):
            raise KeyError('Faltan Muon.D0 y Muon.ErrorD0.')
        prompt = candidates.max_d0_sig <= prompt_max_d0
        displaced = candidates.min_d0_sig >= displaced_min_d0
        if category == 'prompt':
            mask = mask & prompt
        elif category == 'displaced':
            mask = mask & displaced
        else:
            mask = mask & ~prompt & ~displaced
    return candidates[mask]

def mass_candidate(candidates: ak.Array, target_mass: float) -> ak.Array:
    if target_mass < 0:
        raise ValueError("'target_mass' no puede ser negativo.")
    ord = ak.argsort(np.abs(candidates.mass - target_mass), axis=1)
    return ak.firsts(candidates[ord], axis=1)

def pt_candidates(candidates: ak.Array) -> ak.Array:
    ord = ak.argsort(candidates.pt, axis=1, ascending=False, stable=True)
    return ak.firsts(candidates[ord], axis=1)
