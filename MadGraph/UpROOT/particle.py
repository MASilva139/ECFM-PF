from collections import Counter
from collections.abc import Iterable
from operator import index
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote
import argparse
import json
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

ROWS = 100_000
KEY = b'particles.v1'

def _codes(values: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, (int, np.integer)):
        values = (values,)
    codes = tuple(sorted({index(value) for value in values}))
    return codes

def directory(
    dataset: str, 
    pids: int | Iterable[int],
    *,
    statuses: int | Iterable[int] | None = None,
    tree: str | None = None, 
    extension: str = 'parquet'
) -> Path:
    if __package__:
        from .config import PARQUET_DIR
    else:
        from config import PARQUET_DIR
    suffix = extension.removeprefix(".")
    ending = f'.{suffix}'
    base = dataset[:-len(ending)] if dataset.lower().endswith(ending.lower()) else dataset
    if pids is not None:
        if tree not in (None, 'Particles'):
            raise ValueError('Los filtros de PID y Status corresponden al árbol Particles.')
        base += '_pid' + '_'.join(map(str, _codes(pids)))
        if statuses is not None:
            base += '_st' + '_'.join(map(str, _codes(statuses)))
    elif statuses is not None:
        raise ValueError('Indique los PID de la selección.')
    elif tree is not None:
        base += '_' + quote(tree, safe='')
    return PARQUET_DIR / f'{base}.{suffix}'

def _particle_tree(rf, columns):
    if 'Particles' not in rf:
        raise KeyError('Falta el árbol Particles')
    tree = rf['Particles']
    if not isinstance(tree, uproot.TTree):
        raise TypeError('Se requiere el TTree plano Particles')
    missing = set(columns).difference(tree.keys(recursive=False))
    if missing:
        raise KeyError(f"Faltan columnas en Particles: {', '.join(sorted(missing))}")
    return tree

def _select_tables(tree, schema, pids, statuses, batch_rows):
    filters = ['PID'] if statuses is None else ['PID', 'Status']
    for selectors, report in tree.iterate(filters, step_size=batch_rows, library='np', how=dict, report=True):
        mask = np.isin(selectors['PID'], pids)
        if statuses is not None:
            mask &= np.isin(selectors['Status'], statuses)
        rows = np.flatnonzero(mask)
        if not rows.size:
            continue
        # lectura contigua por columnas
        first =  int(rows[0])
        start = report.tree_entry_start + first
        stop = report.tree_entry_start + int(rows[-1]) + 1
        offsets = rows - first
        columns = {}
        for name in schema.names:
            if name in selectors:
                columns[name] = selectors[name][rows]
            else:
                columns[name] = tree[name].array(entry_start=start, entry_stop=stop, library='np')[offsets]
        yield pa.Table.from_pydict(columns, schema=schema)

def _check_exist(path, signa):
    try:
        metadata = pq.read_schema(path).metadata or {}
    except pa.ArrowInvalid:
        metadata = {}
    if metadata.get(KEY) != signa:
        raise FileExistsError(f'{path} ya existe y no corresponde a esta selección o a esta versión del ROOT.')

def export_particles(
    rf: str | Path,
    output: str | Path | None = None,
    *,
    dataset: str | None = None,
    columns: Iterable[str] | str,
    pids: int | Iterable[int],
    statuses: int | Iterable[int] | None = None,
    batch_rows: int = ROWS,
    overwrite: bool = False
) -> Path:
    source = Path(rf).expanduser().resolve(strict=True)
    pids = _codes(pids)
    statuses = None if statuses is None else _codes(statuses)
    if output is None:
        output = directory(source.stem if dataset is None else dataset, pids, statuses=statuses)
    target = Path(output).expanduser().resolve()
    if target.suffix.lower() != '.parquet' or target == source:
        raise ValueError('Escoja un archivo de salida distinto del ROOT.')
    names = tuple(dict.fromkeys(columns,) if isinstance(columns, str) else columns)
    if not names:
        raise ValueError('Indique las columnas que se desean guardar.')
    batch_rows = index(batch_rows)
    stat = source.stat()
    signa = json.dumps({
        'root': str(source), 
        'size': stat.st_size, 
        'mtime_ns': stat.st_mtime_ns,
        'columns': names,
        'pids': pids,
        'statutes': statuses
    }).encode('utf-8')
    if target.exists() and not overwrite:
        _check_exist(target, signa)
        return target
    filters = ('PID',) if statuses is None else ('PID', 'Status')
    with uproot.open(source, array_cache=None) as root:
        tree = _particle_tree(root, (*names, *filters))
        fields = []
        for name in names:
            dtype = tree[name].interpretation.numpy_dtype
            if dtype.kind not in 'biuf':
                raise TypeError(f'Particles/{name} debe ser una columna numérica escalar.')
            fields.append((name, pa.from_numpy_dtype(dtype.newbyteorder('='))))
        schema = pa.schema(fields, metadata={KEY: signa})
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent) as staging:
            temp = Path(staging) / 'selection.parquet'
            with pq.ParquetWriter(temp, schema, compression='zstd') as writer:
                for table in _select_tables(tree, schema, pids, statuses, batch_rows):
                    writer.write_table(table)
            if overwrite:
                os.replace(temp, target)
            else:
                try:
                    if os.name == 'nt':
                        os.rename(temp, target)
                    else:
                        os.link(temp, target)
                except FileExistsError:
                    _check_exist(target, signa)
    return target

def count_particle_codes(rf: str | Path, *, batch_rows: int = ROWS) -> pd.Series:
    counts = Counter()
    with uproot.open(rf, array_cache=None) as root:
        keys = ['PID', 'Status']
        tree = _particle_tree(root, keys)
        for block in tree.iterate(keys, step_size=batch_rows, library='np', how=dict):
            counts.update(pd.DataFrame(block).value_counts(sort=False).to_dict())
    pairs = pd.MultiIndex.from_tuples(list(counts), names=keys)
    totals = pd.Series(list(counts.values()), index=pairs, name='Particles', dtype='int64')
    return totals.sort_values(ascending=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Archivo ROOT plano con el TTree Particles")
    parser.add_argument("output", type=Path, help="Archivo .parquet de salida")
    parser.add_argument("--pid", type=int, nargs="+", required=True, help="Códigos con signo")
    parser.add_argument("--status", type=int, nargs="+", help="Estados aceptados (opcional)")
    parser.add_argument("--columns", nargs="+", required=True, help="Columnas que se guardarán")
    parser.add_argument("--batch-rows", type=int, default=ROWS, help="Filas ROOT por bloque")
    parser.add_argument("--overwrite", action="store_true", help="Regenerar el archivo seleccionado")
    args = parser.parse_args()
    path = export_particles(
        args.root, args.output, columns=args.columns, pids=args.pid,
        statuses=args.status, batch_rows=args.batch_rows, overwrite=args.overwrite,
    )
    print(f"{path} ({pq.read_metadata(path).num_rows:,} partículas)")

if __name__ == "__main__":
    main()