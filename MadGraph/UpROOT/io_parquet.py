from collections.abc import Iterable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
import awkward as ak
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import uproot

from .particle import count_particle_codes, export_particles, directory

__all__ = [
    "parquet", "read_parquet", "iter_parquet", "parquet_rows",
    "export_particles", "count_particle_codes", "directory"
]

def _columnas(nombres: Iterable[str] | str | None) -> list[str] | None:
    if nombres is None:
        return None
    return [nombres] if isinstance(nombres, str) else list(nombres)

def _set_parquet(
    path: str | Path | None,
    dataset: str | None,
    tree: str | None,
    pids: int | Iterable[int] | None,
    statuses: int | Iterable[int] | None
) -> str | Path:
    if path is not None:
        if any(value is not None for value in (dataset, tree, pids, statuses)):
            raise ValueError('Indique la ruta o los datos de la selección.')
        return path
    if dataset is None:
        raise ValueError('Indique una ruta o use dataset=... para leer desde PARQUET_DIR')
    return directory(dataset, pids, statuses=statuses, tree=tree)

def parquet(
    root_path: str | Path,
    tree_name: str,
    parquet_path: str | Path | None = None,
    branches: Iterable[str] | str | None = None,
    *,
    dataset: str | None = None,
    step_size: str | int = "20 MB",
) -> Path:
    if parquet_path is None:
        if dataset is None:
            origen = root_path
            if isinstance(origen, str) and '://' in origen:
                origen = urlsplit(origen).path
            dataset = Path(origen).stem
        parquet_path = directory(dataset, tree=tree_name)
    destino = Path(parquet_path).expanduser()
    if isinstance(root_path, Path) or "://" not in root_path:
        root_path = Path(root_path).expanduser()
        if root_path.resolve() == destino.resolve():
            raise ValueError("El ROOT y el Parquet deben tener rutas distintas.")
    ramas = _columnas(branches)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with uproot.open(root_path, array_cache=None) as archivo:
        arbol = archivo[tree_name]
        # Con cero entradas, Uproot conserva los tipos incluso si el árbol está vacío.
        vacio = arbol.arrays(expressions=ramas, entry_stop=0, library="ak")
        if not vacio.fields:
            raise ValueError(f"La selección de {tree_name!r} debe incluir al menos una rama.")
        esquema = ak.to_arrow_table(vacio, extensionarray=False).schema
        with TemporaryDirectory(dir=destino.parent) as temporal:
            provisional = Path(temporal) / "tabla.parquet"
            with pq.ParquetWriter(provisional, esquema) as escritor:
                for bloque in arbol.iterate(expressions=ramas, step_size=step_size, library="ak"):
                    escritor.write_table(ak.to_arrow_table(bloque, extensionarray=False))
                    del bloque
            provisional.replace(destino)
    return destino

def read_parquet(
    parquet_path: str | Path | None = None,
    columns: Iterable[str] | str | None = None,
    *,
    rows: int | None = None,
    dataset: str | None = None,
    tree: str | None = None,
    pids: int | Iterable[int] | None = None,
    statuses: int | Iterable[int] | None = None
) -> pd.DataFrame:
    archivo = _set_parquet(parquet_path, dataset, tree, pids, statuses)
    tabla = ds.dataset(archivo, format="parquet")
    seleccion = _columnas(columns)
    if rows is None:
        return tabla.to_table(columns=seleccion).to_pandas()
    return tabla.head(rows, columns=seleccion, batch_size=max(1, min(rows, 50_000)), batch_readahead=0).to_pandas()

def parquet_iter(
    parquet_path: str | Path | None = None,
    columns: Iterable[str] | str | None = None,
    *,
    batch_size: int = 50_000,
    dataset: str | None = None,
    tree: str | None = None,
    pids: int | Iterable[int] | None = None,
    statuses: int | Iterable[int] | None = None
) -> Iterator[pd.DataFrame]:
    archivo = _set_parquet(parquet_path, dataset, tree, pids, statuses)
    tabla = ds.dataset(archivo, format="parquet")
    for bloque in tabla.to_batches(columns=_columnas(columns), batch_size=batch_size, batch_readahead=0):
        if bloque.num_rows:
            yield bloque.to_pandas()

def parquet_rows(
    parquet_path: str | Path | None = None,
    *,
    dataset: str | None = None,
    tree: str | None = None,
    pids: int | Iterable[int] | None = None,
    statuses: int | Iterable[int] | None = None
) -> int:
    archivo = _set_parquet(parquet_path, dataset, tree, pids, statuses)
    return pq.read_metadata(archivo).num_rows