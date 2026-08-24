import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import awkward as ak
import numpy as np
import pandas as pd
import uproot

MUON_M_GEV = 0.1056583745
DEF_COLLECTIONS: tuple[str, ...] = (
    'Muon',
    'Jet',
    'Electron',
    'Photon',
    'MissingET',
    'ScalarHT'
)
TREE_NAMES: dict[str, str] = {
    'Event': 'Events',
    'Muon': 'Muons',
    'Electron': 'Electrons',
    'Photon': 'Photons',
    'Jet': 'Jets',
    'FatJet': 'FatJets',
    'GenJet': 'GenJets',
    'Particle': 'Particles',
    'Track': 'Tracks',
    'Tower': 'Towers',
    'Weight': 'Weights',
    'EFlowTrack': 'EFlowTracks',
    'EFlowPhoton': 'EFlowPhotons',
    'EFlowNeutralHadron': 'EFlowNeutralHadrons',
    'MissingET': 'MissingET',
    'GenMissingET': 'GenMissingET',
    'ScalarHT': 'ScalarHT'
}

class ConversionError(RuntimeError):
    """Error de conversión"""

@dataclass(frozen=True, slots=True)
class InputFile:
    path: Path
    file_id: int
    event_offset: int
    entries: int

def _clean_name(value: str) -> str:
    result = re.sub(r'^0-9A-Za-z_', '_', str(value))
    result = re.sub(r'_+', '_', result).strip('_')
    return result or 'value'

def _tree_name(collection: str) -> str:
    return TREE_NAMES.get(collection, _clean_name(collection))

def _parse_branch_name(branch_name: str) -> tuple[str, str] | None:
    if '/' not in branch_name:
        return None
    collection, leaf_name = branch_name.split('/', 1)
    prefix = f'{collection}.'
    if not leaf_name.startswith(prefix):
        return None
    attribute = leaf_name[len(prefix) :]
    if not attribute or '/' in attribute:
        return None
    return collection, attribute

def _available_branches(tree: Any) -> list[str]:
    try:
        names = tree.keys(recursive=True)
    except TypeError:
        names = tree.keys()
    return list(dict.fromkeys(str(name) for name in names))

def discover_collections(tree: Any) -> dict[str, dict[str, str]]:
    collections: dict[str, dict[str, str]] = defaultdict(dict)
    for branch_name in _available_branches(tree):
        parsed = _parse_branch_name(branch_name)
        if parsed is None:
            continue
        collection, attribute = parsed
        collections[collection][attribute] = branch_name
    return {name: dict(fields) for name, fields in collections.items()}

def _as_np(values: Any, *, label: str) -> np.ndarray:
    try:
        result = ak.to_numpy(values, allow_missing=True)
    except (TypeError, ValueError) as e:
        raise ConversionError(f'La columna {label!r} no puede represaentarse como valores escalares.') from e
    if isinstance(result, np.ma.MaskedArray):
        if np.issubdtype(result.dtype, np.floating):
            result = result.filled(np.nan)
        elif np.issubdtype(result.dtype, np.integer):
            result = result.filled(0)
        elif np.issubdtype(result.dtype, np.bool_):
            result = result.filled(False)
        else:
            result = result.astype(object).filled("")
    array = np.asarray(result)
    if array.ndim != 1:
        raise ConversionError(f"La columna {label!r} tiene {array.ndim} dimensiones; se alacenará como árbol de valores relacionados.")
    return array

def _is_str_array(values: np.ndarray) -> bool:
    if values.dtype.kind in {'U', 'S'}:
        return True
    if values.dtype.kind != 'O':
        return False
    return all(isinstance(value, (str, bytes)) for value in values)

def _column_type(values: np.ndarray) -> Any:
    if _is_str_array(values):
        return 'string'
    if values.dtype.kind not in {'b', 'i', 'u', 'f'}:
        raise ConversionError(f"Tipo no compatible con TTree: {values.dtype!s}.")
    return values.dtype

class RootTreeWriter:
    def __init__(self, writable_file: Any) -> None:
        self.file = writable_file
        self.schemas: dict[str, tuple[str, ...]] = {}
        self.rows: dict[str, int] = defaultdict(int)
    def append(self, tree_name: str, columns: Mapping[str, Any]) -> None:
        if not columns:
            raise ConversionError(f"Árbol {tree_name!r} no tiene columnas.")
        normalized: dict[str, np.ndarray] = {}
        for name, values in columns.items():
            if isinstance(values, np.ndarray):
                array = values
            else:
                array = _as_np(values, label=f'{tree_name}.{name}')
            normalized[str(name)] = array
        sizes = {len(values) for values in normalized.values()}
        if len(sizes) != 1:
            detail = ", ".join(f"{name}={len(values)}" for name, values in normalized.items())
            raise ConversionError(f"Columnas del árbol {tree_name!r} tienen tamaños distintos: {detail}.")
        column_names = tuple(normalized)
        if tree_name not in self.schemas:
            schema = {name: _column_type(values) for name, values in normalized.items()}
            self.file.mktree(tree_name, schema)
            self.schemas[tree_name] = column_names
        elif column_names != self.schemas[tree_name]:
            raise ConversionError(f"Esquema del árbol {tree_name!r} cambió entre bloques. Esperando: {self.schemas[tree_name]}; recibido: {column_names}.")
        size = sizes.pop()
        if size == 0:
            return
        writable = {name: values.tolist() if _is_str_array(values) else values for name, values in normalized.items()}
        self.file[tree_name].extend(writable)
        self.rows[tree_name] += size

def _write_utf8_column(
    *,
    writer: RootTreeWriter,
    tree_name: str,
    column_name: str,
    values: np.ndarray,
    parent_ids: Mapping[str, np.ndarray],
    target_columns: dict[str, np.ndarray]
) -> None:
    encoded = [value if isinstance(values, bytes) else str(value).encode("utf-8") for value in values]
    lengths = np.fromiter((len(value) for value in encoded), dtype=np.int32)
    target_columns[f"{column_name}_nbytes"] = lengths
    if not bool(np.any(lengths)):
        return
    columns = {name: np.repeat(parent, lengths) for name, parent in parent_ids.items()}
    columns["byte_index"] = np.concatenate([np.arange(length, dtype=np.int32) for length in lengths])
    columns["utf/_byte"] = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    writer.append(f"{tree_name}__{_clean_name(column_name)}Utf8", columns)

def _list_depth(values: ak.Array) -> int:
    layout = ak.to_layout(values)
    minimum, maximum = layout.minmax_depth
    if minimum != maximum:
        raise ConversionError(f"Estructura con profundidad variable: {minimum}...{maximum}.")
    return maximum

def _optional_mask(values: ak.Array) -> np.ndarray | None:
    layout = ak.to_layout(values)
    if not getattr(layout, 'is_option', False):
        return None
    return np.asarray(ak.to_numpy(~ak.is_none(values, axis=0)), dtype=np.bool_)

def _write_nested_values(
    *,
    writer: RootTreeWriter,
    collection_tree: str,
    attribute_path: str,
    values: ak.Array,
    parent_ids: Mapping[str, np.ndarray],
    level: int = 0
) -> None:
    fields = list(ak.fields(values))
    if fields:
        for field in fields:
            _write_nested_values(
                writer=writer,
                collection_tree=collection_tree,
                attribute_path=f"{attribute_path}_{field}",
                values=values[field],
                parent_ids=parent_ids,
                level=level
            )
        return
    depth = _list_depth(values)
    if depth <= 1:
        clean_path = _clean_name(attribute_path)
        is_reference = clean_path.lower().endswith("refs")
        table_suffix = clean_path[:-5].rstrip("_") if is_reference else clean_path
        table_name = (f"{collection_tree}__{table_suffix}Links" if is_reference else f"{collection_tree}__{table_suffix}Values")
        value_name = "reference_uid" if is_reference else "value"
        columns = dict(parent_ids)
        scalar_values = _as_np(values, label=f"{table_name}.{value_name}")
        if _is_str_array(scalar_values):
            _write_utf8_column(
                writer=writer,
                tree_name=table_name,
                column_name=value_name,
                values=scalar_values,
                parent_ids=parent_ids,
                target_columns=columns
            )
        else:
            columns[value_name] = scalar_values
        valid_mask = _optional_mask(values)
        if valid_mask is not None:
            columns["value_is_valid"] = valid_mask
        writer.append(table_name, columns)
        return
    indices = ak.local_index(values, axis=1)
    expanded_ids: dict[str, np.ndarray] = {}
    for name, parent in parent_ids.items():
        repeated = ak.broadcast_arrays(ak.Array(parent), indices)[0]
        expanded_ids[name] = _as_np(
            ak.flatten(repeated, axis=1),
            label=f"{collection_tree}.{name}"
        )
    index_name = "value_index" if level == 0 else f"value_index_{level}"
    expanded_ids[index_name] = np.asarray(
        ak.to_numpy(ak.flatten(indices, axis=1)),
        dtype=np.int32
    )
    flattened = ak.flatten(values, axis=1)
    _write_nested_values(
        writer=writer,
        collection_tree=collection_tree,
        attribute_path=attribute_path,
        values=flattened,
        parent_ids=expanded_ids,
        level=level + 1
    )

def _add_object_field(
    *,
    writer: RootTreeWriter,
    collection_tree: str,
    attribute_path: str,
    values: ak.Array,
    base_columns: dict[str, np.ndarray],
    parent_ids: Mapping[str, np.ndarray]
) -> None:
    fields = list(ak.fields(values))
    if fields:
        for field in fields:
            _add_object_field(
                writer=writer,
                collection_tree=collection_tree,
                attribute_path=f"{attribute_path}_{field}",
                values=values[field],
                base_columns=base_columns,
                parent_ids=parent_ids
            )
        return
    depth = _list_depth(values)
    if depth > 1:
        _write_nested_values(
            writer=writer,
            collection_tree=collection_tree,
            attribute_path=attribute_path,
            values=values,
            parent_ids=parent_ids
        )
        return
    column_name = _clean_name(attribute_path)
    if column_name == "fUniqueID":
        column_name = "root_uid"
    if column_name in base_columns:
        raise ConversionError(f"La normalización genera una columna duplicada: {collection_tree}.{column_name}.")
    scalar_values = _as_np(values, label=f"{collection_tree}.{column_name}")
    if _is_str_array(scalar_values):
        _write_utf8_column(
            writer=writer,
            tree_name=collection_tree,
            column_name=column_name,
            values=scalar_values,
            parent_ids=parent_ids,
            target_columns=base_columns
        )
    else:
        base_columns[column_name] = scalar_values
    valid_mask = _optional_mask(values)
    if valid_mask is not None:
        base_columns[f"{column_name}_is_valid"] = valid_mask

def _iter_batches(
    tree: Any,
    branches: Sequence[str],
    step_size: str | int
) -> Iterable[dict[str, ak.Array]]:
    try:
        yield from tree.iterate(
            expressions=list(branches),
            step_size=step_size,
            library="ak",
            how=dict
        )
    except Exception as e:
        raise ConversionError(f"No fue posible leer un bloque de ramas. Ramas solicitadas:\n{', '.join(branches)}.\n\nError original: {e}") from e

def _scalar_event_column(values: ak.Array, *, label: str) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        counts = ak.num(values, axis=1)
    except (ValueError, np.exceptions.AxisError):
        return _as_np(values, label=label), _optional_mask(values)
    if bool(ak.any(counts > 1)):
        raise ConversionError(f"Rama de evento {label!r} contiene varios valores; debe tratarse como una colección independiente.")
    if bool(ak.all(counts == 1)):
        return _as_np(ak.flatten(values, axis=1), label=label), None
    first = ak.firsts(values, axis=1)
    valid = np.asarray(ak.to_numpy(~ak.is_none(first)), dtype=np.bool_)
    return _as_np(first, label=label), valid

def _event_columns(
    batch: Mapping[str, ak.Array],
    *,
    writer: RootTreeWriter,
    source: InputFile,
    cursor: int,
    event_fields: Mapping[str, str],
    count_fields: Mapping[str, str]
) -> dict[str, np.ndarray]:
    first = next(iter(batch.values()))
    size = len(first)
    entries = np.arange(cursor, cursor + size, dtype=np.int64)
    columns: dict[str, np.ndarray] = {
        "source_file_id": np.full(size, source.file_id, dtype=np.int32),
        "event_id": source.event_offset + entries,
        "source_entry": entries
    }
    for attribute, branch_name in event_fields.items():
        output_name = {
            "Number": "event_number",
            "Weight": "event_weight",
            "fUniqueID": "root_uid"
        }.get(attribute, _clean_name(attribute))
        values, valid = _scalar_event_column(batch[branch_name], label=f"Events.{output_name}")
        if _is_str_array(values):
            parent_ids = {key: columns[key] for key in ("source_file_id", "event_id", "source_entry")}
            _write_utf8_column(
                writer=writer,
                tree_name="Events",
                column_name=output_name,
                values=values,
                parent_ids=parent_ids,
                target_columns=columns
            )
        else:
            columns[output_name] = values
        if valid is not None:
            columns[f"{output_name}_is_valid"] = valid
    for collection, branch_name in count_fields.items():
        columns[f"n_{_clean_name(collection)}"] = _as_np(batch[branch_name], label=f"Events.n_{collection}")
    return columns

def _write_events(
    *,
    writer: RootTreeWriter,
    tree: Any,
    source: InputFile,
    discovered: Mapping[str, Mapping[str, str]],
    selected_collections: Sequence[str],
    excluded: set[str],
    step_size: str | int,
) -> None:
    available = set(_available_branches(tree))
    event_fields = {attribute: branch for attribute, branch in discovered.get("Event", {}).items() if branch not in excluded}
    count_fields = {collection: f"{collection}_size" for collection in selected_collections if f"{collection}_size" in available and f"{collection}_size" not in excluded}
    expressions = [*event_fields.values(), *count_fields.values()]
    if not expressions:
        block_size = 10_000
        for cursor in range(0, source.entries, block_size):
            stop = min(source.entries, cursor + block_size)
            entries = np.arange(cursor, stop, dtype=np.int64)
            writer.append(
                "Events", {
                    "source_file_id": np.full(len(entries), source.file_id, dtype=np.int32),
                    "event_id": source.event_offset + entries,
                    "source_entry": entries
                },
            )
        return
    cursor = 0
    for batch in _iter_batches(tree, expressions, step_size):
        columns = _event_columns(
            batch,
            writer=writer,
            source=source,
            cursor=cursor,
            event_fields=event_fields,
            count_fields=count_fields
        )
        writer.append("Events", columns)
        cursor += len(next(iter(batch.values())))

def _create_dimuons(
    *,
    writer: RootTreeWriter,
    batch: Mapping[str, ak.Array],
    fields: Mapping[str, str],
    source: InputFile,
    cursor: int
) -> None:
    required = {'PT', 'Eta', 'Phi', 'Charge'}
    if not required.issubset(fields):
        return
    pt = batch[fields['PT']]
    eta = batch[fields['Eta']]
    phi = batch[fields['Phi']]
    charge = batch[fields['Charge']]
    object_index = ak.local_index(pt, axis=1)
    payload: dict[str, ak.Array] = {
        "object_index": object_index,
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "charge": charge
    }
    optional = {
        "D0": "d0",
        "DZ": "dz",
        "ErrorD0": "error_d0",
        "ErrorDZ": "error_dz",
        "fUniqueID": "root_uid"
    }
    for attribute, output_name in optional.items():
        if attribute in fields:
            payload[output_name] = batch[fields[attribute]]
    if "Particle" in fields:
        references = batch[fields['Particle']]
        if 'ref' in ak.fields(references):
            payload["particle_ref"] = references["ref"]
    muons = ak.zip(payload)
    pairs = ak.combinations(muons, 2, axis=1, fields=("first", "second"))
    first = pairs.first
    second = pairs.second
    mask = first.charge + second.charge < 0
    first = first[mask]
    second = second[mask]
    positive_first = first.charge > 0
    def choose(field_name: str, *, positive: bool) -> ak.Array:
        if positive:
            return ak.where(positive_first, first[field_name], second[field_name])
        return ak.where(positive_first, second[field_name], first[field_name])
    px = first.pt*np.cos(first.phi) + second.pt*np.cos(second.phi)
    py = first.pt*np.sin(first.phi) + second.pt*np.sin(second.phi)
    pz = first.pt*np.sinh(first.eta) + second.pt*np.sinh(second.eta)
    first_energy = np.sqrt((first.pt*np.cosh(first.eta))**2 + MUON_M_GEV**2)
    second_energy = np.sqrt((second.pt*np.cosh(second.eta))**2 + MUON_M_GEV**2)
    dimuon_e = first_energy + second_energy
    mass = np.sqrt(np.maximum(dimuon_e**2 - px**2 - py**2 - pz**2, 0.0))
    candidate_pt = np.sqrt(px**2 + py**2)
    delta_eta = first.eta - second.eta
    raw_delta_phi = first.phi - second.phi
    wrapped_delta_phi = np.arctan2(np.sin(raw_delta_phi), np.cos(raw_delta_phi))
    source_entries = np.arange(cursor, cursor+len(pt), dtype=np.int64)
    event_ids = source.event_offset + source_entries
    repeated_event_ids = ak.broadcast_arrays(ak.Array(event_ids), mass)[0]
    repeated_entries = ak.broadcast_arrays(ak.Array(source_entries), mass)[0]
    columns: dict[str, np.ndarray] = {
        "source_file_id": np.full(int(ak.sum(ak.num(mass, axis=1))), source.file_id, dtype=np.int32),
        "event_id": _as_np(ak.flatten(repeated_event_ids, axis=1), label="Dimuons.event_id"),
        "source_entry": _as_np(ak.flatten(repeated_entries, axis=1), label="Dimuons.source_entry"),
        "candidate_index": np.asarray(ak.to_numpy(ak.flatten(ak.local_index(mass, axis=1), axis=1)), dtype=np.int32),
        "muplus_index": np.asarray(ak.to_numpy(ak.flatten(choose("object_index", positive=True), axis=1)), dtype=np.int32),
        "muminus_index": np.asarray(ak.to_numpy(ak.flatten(choose("object_index", positive=False), axis=1)), dtype=np.int32),
        "dimuon_mass": _as_np(ak.flatten(mass, axis=1), lable="Dimuons.dimuon_mass"),
        "dimuon_pt": _as_np(ak.flatten(candidate_pt, axis=1), label="Dimuons.dimuon_pt"),
        "delta_r": _as_np(ak.flatten(np.sqrt(delta_eta**2 + wrapped_delta_phi**2), axis=1), label="Dimuons.delta_r")
    }
    for field_name in payload:
        if field_name == "object_index":
            continue
        columns[f"muplus_{field_name}"] = _as_np(ak.flatten(choose(field_name, positive=True), axis=1), label=f"Dimuons.muplus_{field_name}")
        columns[f"muminus_{field_name}"] = _as_np(ak.flatten(choose(field_name, positive=False), axis=1), label=f"Dimuons.muminus_{field_name}")
    writer.append("Dimuons", columns)

def _write_object_registry(
    *,
    writer: RootTreeWriter,
    collection_id: int,
    columns: Mapping[str, np.ndarray]
) -> None:
    if "root_uid" not in columns:
        return
    valid = columns["root_uid"] != 0
    writer.append(
        "objectRegistry", {
            "source_file_id": columns["source_file_id"][valid],
            "event_id": columns["event_id"][valid],
            "collection_id": np.full(np.count_nonzero(valid), collection_id, dtype=np.int32),
            "object_index": columns["object_index"][valid],
            "root_uid": np.asarray(columns["root_uid"][valid], dtype=np.uint64)
        }
    )

def _write_collection(
    *,
    writer: RootTreeWriter,
    tree: Any,
    source: InputFile,
    collection: str,
    fields: Mapping[str, str],
    step_size: str | int,
    create_dimuons: bool,
    object_registry: bool,
    collection_id: int
) -> None:
    if not fields:
        return
    collection_tree = _tree_name(collection)
    expressions = list(fields.values())
    cursor = 0
    for batch in _iter_batches(tree, expressions, step_size):
        reference = batch[expressions[0]]
        event_count = len(reference)
        try:
            object_counts = ak.num(reference, axis=1)
        except (ValueError, np.exceptions.AxisError) as e:
            raise ConversionError(f"Colección {collection!r} no tiene estructura evento -> objetos.") from e
        object_index = ak.local_index(reference, axis=1)
        entries = np.arange(cursor, cursor+event_count, dtype=np.int64)
        repeated_entries = ak.broadcast_arrays(ak.Array(entries), object_index)[0]
        flat_entries = _as_np(ak.flatten(repeated_entries, axis=1), label=f"{collection_tree}.source_entry")
        flat_index = np.asarray(ak.to_numpy(ak.flatten(object_index, axis=1)), dtype=np.int32)
        flat_size = len(flat_index)
        base_columns: dict[str, np.ndarray] = {
            "source_file_id": np.full(flat_size, source.file_id, dtype=np.int32),
            "event_id": source.event_offset+flat_entries,
            "source_entry": flat_entries,
            "object_index": flat_index
        }
        parent_ids = dict(base_columns)
        for attribute, branch_name in fields.items():
            values = batch[branch_name]
            try:
                actual_counts = ak.num(values, axis=1)
            except (ValueError, np.exceptions.AxisError) as e:
                raise ConversionError(f"Rama {branch_name!r} no contiene objetos por evento.") from e
            if not bool(ak.all(object_counts == actual_counts)):
                raise ConversionError(f"Rama {branch_name!r} no tiene la misma multiplicidad que la colección {collection!r}.")
            flattened = ak.flatten(values, axis=1)
            _add_object_field(
                writer=writer,
                collection_tree=collection_tree,
                attribute_path=attribute,
                values=flattened,
                base_columns=base_columns,
                parent_ids=parent_ids
            )
        writer.append(collection_tree, base_columns)
        if object_registry:
            _write_object_registry(
                writer=writer,
                collection_id=collection_id,
                columns=base_columns
            )
        if collection == "Muon" and create_dimuons:
            _create_dimuons(
                writer=writer,
                batch=batch,
                fields=fields,
                source=source,
                cursor=cursor
            )
        cursor += event_count

def _normalize_input_paths(input_files: str | Path | Iterable[str | Path]) -> tuple[Path, ...]:
    if isinstance(input_files, (str, Path)):
        input_files = (input_files)
    paths = tuple(Path(path).expanduser().resolve() for path in input_files)
    if not paths:
        raise ValueError("No se ha indicado el(los) archivo(s) ROOT.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f" - {path}" for path in missing)
        raise FileNotFoundError(f"No se encontraron los archivos ROOT:\n{formatted}")
    return paths

def _normalize_collections(collections: Iterable[str] | str | None) -> tuple[str, ...] | None:
    if collections is None:
        return None
    if isinstance(collections, str):
        collections = collections.split(",")
    return tuple(dict.fromkeys(name.strip() for name in collections if name.strip()))

def _resolve_inputs(paths: Sequence[Path], tree_path: str) -> tuple[InputFile, ...]:
    sources: list[InputFile] = []
    event_offset = 0
    for file_id, path in enumerate(paths):
        with uproot.open(path) as root_file:
            if tree_path not in root_file:
                raise KeyError(f"Archivo {path.name!r} no contiene el árbol {tree_path!r}.")
            entries = int(root_file[tree_path].num_entries)
        sources.append(InputFile(path=path, file_id=file_id, event_offset=event_offset, entries=entries))
        event_offset += entries
    return tuple(sources)

def _selected_collections(
    discovered: Mapping[str, Mapping[str, str]], 
    *,
    requested: tuple[str, ...] | None,
    all_collections: bool,
    include_particles: bool
) -> tuple[str, ...]:
    if all_collections and requested is not None:
        raise ValueError("Usa 'requested' o 'all_collections'.")
    if all_collections:
        names = sorted(name for name in discovered if name != "Event")
    elif requested is not None:
        missing = [name for name in requested if name not in discovered]
        if missing:
            available = ', '.join(sorted(discovered))
            raise ValueError(f"Colecciones no disponibles: {', '.join(missing)}. Disponibles: {available}.")
        names = [name for name in requested if name != "Event"]
    else:
        names = [name for name in DEF_COLLECTIONS if name in discovered]
    if include_particles and "Particle" in discovered and "Particle" not in names:
        names.append("Particle")
    return tuple(dict.fromkeys(names))

def convert_delphes_root(
    input_files: str | Path | Iterable[str | Path],
    output_path: str | Path,
    *,
    tree_path: str = "Delphes",
    collections: Iterable[str] | str | None = None,
    all_collections: bool = False,
    include_particles: bool = False,
    exclude_branches: Iterable[str] = (),
    step_size: str | int = "50 MB",
    create_dimuons: bool = True,
    object_registry: bool = False,
    overwrite: bool = False,
    compression_level: int = 4,
    verbose: bool = True
) -> Path:
    paths = _normalize_input_paths(input_files)
    output = Path(output_path).expanduser().resolve()
    requested = _normalize_collections(collections)
    excluded = {str(name).strip() for name in exclude_branches if str(name).strip()}
    if output in paths:
        raise ValueError("El archivo de salida no puede ser uno de los archivos de entrada.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"El archivo {output} existe. Para reescribirlo 'overwrite=True'.")
    if compression_level not in range(1, 23):
        raise ValueError("'compression_level' de estar entre 1 y 22 para ZSTD.")
    sources = _resolve_inputs(paths, tree_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}.",
        suffix=".partial.root",
        dir=output.parent,
        delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    metadata: dict[str, Any] = {
        'format': 'delphes-pandas-root-v1',
        'tree_path': tree_path,
        'input_files': [str(source.path) for source in sources],
        'collections': [],
        'excluded_branches': sorted(excluded),
        'step_size': step_size,
        'create_dimuons': create_dimuons,
        'object_registry': object_registry,
        'string_encoding': 'utf-8; texto preservado en arboles auxiliares *Utf8',
        'collection_ids': {},
        'source_branch_mapping': {}
    }
    try:
        with uproot.recreate(
            temporary_path,
            compression=uproot.ZSTD(compression_level)
        ) as output_file:
            writer = RootTreeWriter(output_file)
            source_ids = np.arange(len(sources), dtype=np.int32)
            source_columns = {
                "source_file_id": source_ids,
                "entries": np.asarray([source.entries for source in sources], dtype=np.int64),
                "event_offset": np.asarray([source.event_offset for source in sources], dtype=np.int64)
            }
            _write_utf8_column(
                writer=writer,
                tree_name="SourceFiles",
                column_name="path",
                values=np.asarray([str(source.path) for source in sources], dtype=object),
                parent_ids={"source_file_id": source_ids},
                target_columns=source_columns
            )
            writer.append("SourceFiles", source_columns)
            for source in sources:
                with uproot.open(source.path) as root_file:
                    tree = root_file[tree_path]
                    discovered = discover_collections(tree)
                    chosen = _selected_collections(
                        discovered,
                        requested=requested,
                        all_collections=all_collections,
                        include_particles=include_particles
                    )
                    if not chosen:
                        raise ConversionError("No se encontró ninguna colección Delphes para convertir.")
                    if metadata['collections'] and metadata['collections'] != list(chosen):
                        raise ConversionError("Archivos de entrada con diferentes colecciones seleccionadas.")
                    metadata['collections'] = list(chosen)
                    metadata['collection_ids'] = {name: index for index, name in enumerate(chosen)}
                    if verbose:
                        print(f"\nArchivo de entrada: {source.path}\nEventos: {source.entries:,}\nColecciones: {', '.join(chosen)}")
                    _write_events(
                        writer=writer,
                        tree=tree,
                        source=source,
                        discovered=discovered,
                        selected_collections=chosen,
                        excluded=excluded,
                        step_size=step_size
                    )
                    for collection in chosen:
                        fields = {attribute: branch for attribute, branch in discovered[collection].items() if branch not in excluded}
                        if not fields:
                            raise ConversionError(f"Ramas de la colección {collection!r} excluidas.")
                        metadata['source_branch_mapping'][collection] = fields
                        if verbose:
                            print(f" {_tree_name(collection)}: {len(fields)} ramas; bloques de {step_size}")
                        _write_collection(
                            writer=writer,
                            tree=tree,
                            source=source,
                            collection=collection,
                            fields=fields,
                            step_size=step_size,
                            create_dimuons=create_dimuons,
                            object_registry=object_registry,
                            collection_id=metadata['collection_ids'][collection]
                        )
            metadata['tree_rows'] = dict(writer.rows)
            output_file['ConversionMetadata'] = json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True
            )
        if output.exists() and not overwrite:
            raise FileExistsError(f"El archivo de salida ya existe: {output}.")
        os.replace(temporary_path, output)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    if verbose:
        print(f"\nROOT tabular creado: {output}\nÁrboles generados:")
        for tree_name, row_count in metadata['tree_rows'].items():
            print(f' - {tree_name}: {row_count:,} filas')
    return output

def _load_project_config() -> Any:
    try:
        from . import config
        return config
    except ImportError:
        try:
            from UpROOT import config
            return config
        except ImportError as e:
            raise ImportError("No se pudo importar UpROOT.config.") from e

def convert_dataset(
    dataset: str,
    output_path: str | Path | None = None,
    **options: Any
) -> Path:
    config = _load_project_config()
    try:
        files = config.ROOT_FILES[dataset]
    except KeyError as e:
        available = ', '.join(sorted(config.ROOT_FILES))
        raise ValueError(f"Dataset desconocido: {dataset!r}. Disponibles: {available}.") from e
    if output_path is None:
        output_path = Path(config.DATA_DIR) / f"{dataset}_pandas.root"
    return convert_delphes_root(files, output_path, **options)

def list_output_trees(root_path: str | Path) -> pd.DataFrame:
    path = Path(root_path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with uproot.open(Path) as root_file:
        for name, class_name in root_file.classnames().items():
            if "TTree" not in str(class_name):
                continue
            clean = str(name).split(",")[0]
            tree = root_file[clean]
            rows.append({
                "tree": clean,
                "rows": int(tree.num_entries),
                "columns": len(tree.keys())
            })
    return pd.DataFrame(rows)

def load_output_dataframe(
    root_path: str | Path,
    tree_name: str,
    columns: Sequence[str] | None = None
) -> pd.DataFrame:
    path = Path(root_path).expanduser().resolve()
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            available = [str(name).split(";")[0] for name in root_file.keys()]
            raise KeyError(f"No existe el árbol {tree_name!r}. Disponibles: {available}.")
    return root_file[tree_name].arrays(expressions=list(columns) if columns is not None else None, library='pd')

def read_conversion_metadata(root_path: str | Path) -> dict[str, Any]:
    path = Path(root_path).expanduser().resolve()
    with uproot.open(path) as root_file:
        if "ConversionMetadata" not in root_file:
            raise KeyError("El archivo no contiene ConversionMetadata.")
        return json.loads(str(root_file["ConversionMetadata"]))

def restore_text_column(
    root_path: str | Path,
    tree_name: str,
    column_name: str
) -> pd.Series:
    dataframe = load_output_dataframe(root_path, tree_name)
    length_name = f"{column_name}_nbytes"
    if length_name not in dataframe:
        raise KeyError(f"El árbol {tree_name!r} no contiene la columna {length_name!r}.")
    byte_tree = f"{tree_name}__{_clean_name(column_name)}Utf8"
    path = Path(root_path).expanduser().resolve()
    with uproot.open(path) as root_file:
        if byte_tree not in root_file:
            if bool((dataframe[length_name] != 0).any()):
                raise ConversionError(f"Falta el árbol auxiliar {byte_tree!r} para restaurar el texto.")
            result = pd.Series("", index=dataframe.index, dtype='object')
        else:
            byte_frame = root_file[byte_tree].arrays(library='pd')
            identifiers = [name for name in byte_frame.columns if name not in {"byte_index", "utf8_byte"}]
            grouped: dict[tuple[Any, ...], str] = {}
            for key, group in byte_frame.groupby(identifiers, sort=False):
                normalized_key = key if isinstance(key, tuple) else (key,)
                ordered = group.sort_values("byte_index")["utf8_byte"]
                grouped[normalized_key] = bytes(ordered.tolist()).decode("utf-8")
            result = pd.Series(
                [grouped.get(tuple(row), "") for row in dataframe[identifiers].itertuples(index=False, name=None)],
                index=dataframe.index,
                dtype='object'
            )
    valid_name = f"{column_name}_is_valid"
    if valid_name in dataframe:
        result.loc[~dataframe[valid_name].astype(bool)] = pd.NA
    return result.rename(column_name)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROOT Delphes en TTrees planos.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Nombre de dataset definido.")
    source.add_argument("--input", nargs="+", help="Uno o varios archivos ROOT Delphes.")
    parser.add_argument("--output", help="Ruta del nuevo archivo ROOT.")
    parser.add_argument("--tree", default="Delphes", help="Árbol de entrada; por defecto Delphes.")
    parser.add_argument("--collections", help="Colecciones separadas por comas: Muon,Jet,Particle.")
    parser.add_argument("--all-collections", action="store_true", help="Convierte todas las colecciones.")
    parser.add_argument("--include-particles", action="store_true", help="Incluye Particle.*.")
    parser.add_argument("--exclude-branch", action="append", default=[], help="Rama completa a excluir.")
    parser.add_argument("--step-size", default="50 MB", help="Tamaño aproximado del bloque.")
    parser.add_argument("--no-dimuons", action="store_true", help="No genera el árbol Dimuons.")
    parser.add_argument("--object-registry", action="store_true", help="Genera ObjectRegistry.")
    parser.add_argument("--overwrite", action="store_true", help="Reemplaza la salida si ya existe.")
    parser.add_argument("--compression-level", type=int, default=4, help="Compresión ZSTD: 1 a 22.")
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    options = {
        "tree_path": args.tree,
        "collections": args.collections,
        "all_collections": args.all_collections,
        "include_particles": args.include_particles,
        "exclude_branches": args.exclude_branch,
        "step_size": args.step_size,
        "create_dimuons": not args.no_dimuons,
        "object_registry": args.object_registry,
        "overwrite": args.overwrite,
        "compression_level": args.compression_level
    }
    if args.dataset is not None:
        output = convert_dataset(args.dataset, output_path=args.output, **options)
    else:
        if args.output is None:
            first_input = Path(args.input[0]).expanduser().resolve()
            output_path = first_input.with_name(f"{first_input.stem}_pandas.root")
        else:
            output_path = args.output
        output = convert_delphes_root(args.input, output_path, **options)
    print(f"\nSalida: {output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())