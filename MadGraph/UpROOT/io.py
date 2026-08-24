from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal
import numpy as np
import pandas as pd
import awkward_pandas as apd
import awkward as ak
import uproot
import re
from .config import (
    ROOT_FILES, 
    PREFERRED_TREES, 
    TEXT_DIR,
    ANALYSIS_DIR
)

LibraryType = Literal["np", "pd", "ak"]

def _remove_cycles(key: str) -> str:
    return re.sub(r";\d+(?=/|$)", "", key)

def _dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [".".join(str(level) for level in column if str(level) not in {"", "None"}) for column in df.columns]
    return df

def txt_data(
    txt_file: str | Path,
    base_dir: str | Path = ANALYSIS_DIR
) -> list[str]:
    fpath = Path(txt_file)
    if fpath.suffix.lower()!=".txt":
        fpath = Path(f"{fpath}.txt")
    if not fpath.is_absolute():
        fpath = Path(base_dir)/fpath
    fpath = fpath.expanduser().resolve()
    if not fpath.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de ramas\n - {fpath}")
    sbranches: list[str] = []
    seen: set[str] = set()
    with open(str(fpath), mode="r", encoding="utf-8") as txt:
        for line in txt:
            branch = line.strip()
            if not branch or branch.startswith("#"):
                continue
            if branch not in seen:
                sbranches.append(branch)
                seen.add(branch)
    if not sbranches:
        raise ValueError(f"El archivo no contiene ramas válidas: {fpath}")
    print(f"Ramas cargadas desde: {fpath}")
    print(f"Cantidad de ramas: {len(sbranches)}")
    return sbranches

def _find_tree_names(root_file: Any) -> list[str]:
    cnames = root_file.classnames(recursive=True)
    tree_names: list[str] = []
    for key, class_name in cnames.items():
        ttree = 'TTree' in class_name
        rntuple = 'RNTuple' in class_name
        if ttree or rntuple:
            clean_key = _remove_cycles(str(key))
            if clean_key not in tree_names:
                tree_names.append(clean_key)
    return tree_names

def _branch_names(tree: Any) -> list[str]:
    try:
        keys = tree.keys(recursive=True)
    except TypeError:
        keys = tree.keys()
    return list(dict.fromkeys(str(key) for key in keys))

def _normalize_paths(
    root_files: Iterable[str | Path] | str | Path | None = None,
    dataset: str | None = None
) -> tuple[Path, ...]:
    if root_files is not None and dataset is not None:
        raise ValueError("Utiliza solamente 'dataset' o 'root_files'")
    if dataset is not None:
        try:
            cpaths = ROOT_FILES[dataset]
        except KeyError as e:
            available = ", ".join(ROOT_FILES)
            raise ValueError(
                f"Dataset desconocido: {dataset!r}. "
                f"Opciones disponibles: {available}"
            ) from e
        paths = tuple(Path(path).expanduser().resolve() for path in cpaths)
    elif root_files is not None:
        if isinstance(root_files, (str, Path)):
            root_files = (root_files,)
        paths = tuple(Path(path).expanduser().resolve() for path in root_files)
    else:
        raise ValueError("Indicar el 'dataset' o una lista en 'root_files'.")
    if not paths:
        raise ValueError("La lista de archivos ROOT está vacía.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "No se encontraron los siguientes archivos ROOT:\n"
            f"{formatted}\n\n"
        )
    return paths

def _normalize_branches(
    branches: Iterable[str] | str | None
) -> tuple[str, ...] | None:
    if branches is None:
        return None
    if isinstance(branches, str):
        val = branches.strip()
        if val.lower() == "all" or val =="*":
            return None
        branches = (val,)
    norm = tuple(
        dict.fromkeys(
            str(branch).strip() for branch in branches if str(branch).strip()
        )
    )
    if not norm:
        raise ValueError('La selección de ramas está vacía.')
    return norm

def _ak_branches_name(bname: str) -> str:
    return bname.rsplit("/", 1)[-1]

def awkward_fields(array: ak.Array) -> ak.Array:
    renamed_fields: dict[str, ak.Array] = {}
    original_names: dict[str, str] = {}
    for original_name in array.fields:
        sim_name = _ak_branches_name(original_name)
        if sim_name in renamed_fields:
            prev_name = original_name[sim_name]
            raise ValueError(f"Cambios duplicados:\n - {prev_name}\n - {original_name}\nAmbos se convertirían en {sim_name!r}")
        renamed_fields[sim_name] = array[original_name]
        original_names[sim_name] = original_name
    return ak.zip(renamed_fields, depth_limit=1)

def list_trees(
    file_path: str | Path,
    show: bool = True,
) -> list[str]:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'No se encontró el archivo ROOT: {path}')
    with uproot.open(path) as root_file:
        trees = _find_tree_names(root_file)
        if show:
            print(f'\nArchivo: {path.name}')
            if not trees:
                print('No se encontró ningún TTree o RNTuple')
            else:
                print(f'Árboles encontrados ({len(trees)}):')
                for i, tree in enumerate(trees, start=1):
                    print(f'{i}. {tree}')
    return trees

def list_dataset_trees(
    dataset: str | None = None,
    root_files: Iterable[str | Path] | str | Path | None = None
) -> dict[Path, list[str]]:
    paths = _normalize_paths(dataset=dataset, root_files=root_files)
    result = {}
    for path in paths:
        result[path] = list_trees(path, show=True)
    return result

def find_tree(
    root_file: Any,
    pref_trees: Iterable[str] = PREFERRED_TREES,
    show_trees: bool = False,
) -> str:
    available_trees = _find_tree_names(root_file)
    if show_trees:
        print(f'Árboles encontrados ({len(available_trees)}):')
        for i, tree in enumerate(available_trees, start=1):
            print(f'{i}. {tree}')
    if not available_trees:
        available_objects = [_remove_cycles(str(key)) for key in root_file.keys(recursive=True)]
        formatted = '\n'.join(f' - {key}' for key in available_objects)
        raise KeyError(
            'No se encontró ningún TTree o RNTuple.\n'
            'Objetos disponibles:\n'
            f'{formatted}'
        )
    for pref_tree in pref_trees:
        cl_pref = _remove_cycles(str(pref_tree))
        for tree_name in available_trees:
            if (tree_name == cl_pref or tree_name.endswith(f'/{cl_pref}')):
                return tree_name
    if len(available_trees) == 1:
        return available_trees[0]
    formatted = '\n'.join(f' - {tree_name}' for tree_name in available_trees)
    raise ValueError(
        'El archivo contiene varios árboles, seleccione uno automáticamente.\n'
        'Árboles encontrados:\n'
        f'{formatted}\n\n'
    )

def _get_tree(
    root_file: Any,
    tree_path: str | None = None,
    show_trees: bool = False
) -> tuple[str, Any]:
    sel_tree = (_remove_cycles(tree_path) if tree_path is not None else find_tree(root_file, show_trees=show_trees))
    try:
        tree = root_file[sel_tree]
    except KeyError as e:
        av_trees = _find_tree_names(root_file)
        raise KeyError(f'No existe el árbol {sel_tree}.\nÁrboles disponibles: {av_trees}') from e
    return sel_tree, tree

def list_branches(
    dataset: str | None = None,
    root_files: Iterable[str | Path] | str | Path | None = None,
    tree_path: str | None = None,
    show_trees: bool = False,
    show_branches: bool = True,
    txt_path: str | Path | None = None,
) -> list[str]:
    paths = _normalize_paths(dataset=dataset, root_files=root_files)
    f_path = paths[0]
    with uproot.open(f_path) as rf:
        sel_tree, tree = _get_tree(
            root_file=rf,
            tree_path=tree_path,
            show_trees=show_trees
        )
        branches = sorted(_branch_names(tree))
    if show_branches:
        print(f'Archivo: {f_path.name}')
        print(f'Árbol seleccionado: {sel_tree}')
        print(f'Ramas encontradas: {len(branches)}')
        for branch in branches:
            print(f' - {branch}')
    if txt_path is not None:
        txt_path = TEXT_DIR / "branches" / f'{txt_path}.txt'
        out_path = Path(txt_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(
            str(out_path),
            mode="w", 
            encoding="utf-8",
            newline="\n"
        ) as txt_file:
            txt_file.write("\n".join(branches)+"\n")
        print(f'Lista de ramas guardada en: {out_path}')
    return branches

def _prepare_file_tree_map(
    paths: tuple[Path, ...],
    branches: tuple[str, ...] | None,
    tree_path: str | None,
    show_trees: bool = False,
) -> dict[str, str]:
    file_tree_map: dict[str, str] = {}
    for path in paths:
        with uproot.open(path) as rf:
            # print(f'\nArchivo: {path.name}')
            sel_tree, tree = _get_tree(rf, tree_path=tree_path, show_trees=show_trees)
            if branches is not None:
                av_branches = set(_branch_names(tree))
                miss_branch = [branch for branch in branches if branch not in av_branches]
                if miss_branch:
                    formatted = "\n".join(f' - {branch}' for branch in miss_branch)
                    raise KeyError(f'En el archivo {path.name!r} y el árbol {sel_tree!r} no existe estas ramas:\n{formatted}')
            print(f'Archivo: {path.name}')
            print(f'Árbol seleccionado: {sel_tree}')
            file_tree_map[str(path)] = sel_tree
    return file_tree_map

def load_data(
    branches: Iterable[str] | str | None = None,
    dataset: str | None = None,
    root_files: Iterable[str | Path] | str | Path | None = None,
    tree_path: str | None = None,
    cut: str | None = None,
    library: LibraryType = 'np',
    show_trees: bool = False
) -> Any:
    if library not in ('np', 'pd', 'ak'):
        raise ValueError(f"Biblioteca de salida no válida: {library!r}. Opciones válidas: 'np', 'pd', 'ak'.")
    paths = _normalize_paths(dataset=dataset, root_files=root_files)
    req = _normalize_branches(branches)
    # if not req:
    #     raise ValueError('Solicite al menos una rama.')
    file_tree_map = _prepare_file_tree_map(
        paths=paths,
        branches=req,
        tree_path=tree_path,
        show_trees=show_trees
    )
    opt: dict[str, Any] = {
        'files': file_tree_map,
        'expressions': None if req is None else list(req),
        'cut': cut,
        'library': library
    }
    if library == 'np':
        opt['how'] = dict
    return uproot.concatenate(**opt)

def load_arrays(
    branches: Iterable[str] | str | None = None,
    dataset: str | None = None,
    root_files: Iterable[str | Path] | str | Path | None = None,
    tree_path: str | None = None,
    cut: str | None = None,
    show_trees: bool = False,
) -> dict[str, np.ndarray]:
    np_array = load_data(
        branches=branches,
        dataset=dataset,
        root_files=root_files,
        tree_path=tree_path,
        cut=cut,
        library='np',
        show_trees=show_trees
    )
    print(f"Total de evenots: {len(np_array):,} eventos")
    return np_array

def load_dataframe(
    branches: Iterable[str] | str | None = None,
    dataset: str | None = None,
    root_files: Iterable[str | Path] | str | Path | None = None,
    tree_path: str | None = None,
    cut: str | None = None,
    show_trees: bool = False
) -> pd.DataFrame:
    dataframe = load_data(
        branches=branches,
        dataset=dataset,
        root_files=root_files,
        tree_path=tree_path,
        cut=cut,
        library='pd',
        show_trees=show_trees
    )
    print(f"Total de eventos: {len(dataframe):,} eventos")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError('No se ha generado un pd.DataFrame')
    return dataframe

def load_awkward(
    branches: Iterable[str] | str | None = None,
    dataset: str | None = None, 
    root_files: Iterable[str | Path] | str | Path | None = None,
    tree_path: str | None = None,
    cut: str | None = None,
    show_trees: bool = False,
    simplify_names: bool = False
) -> ak.Array:
    ak_array = load_data(
        branches=branches,
        dataset=dataset,
        root_files=root_files,
        tree_path=tree_path,
        cut=cut,
        library='ak',
        show_trees=show_trees
    )
    if simplify_names:
        ak_array = awkward_fields(ak_array)
    print(f"Total de evenots: {len(ak_array):,} eventos")
    return ak_array

def csv(
    output_path: str | Path,
    branches: Iterable[str] | str | None = None,
    dataset: str | None = None,
    root_files: Iterable[str | Path] | str | Path | None = None,
    tree_path: str | None = None,
    cut: str | None = None,
    library: LibraryType = 'pd',
    show_trees: bool = False,
    awkward_how: str = 'outer'
) -> Path:
    output_path = TEXT_DIR / "csv" / f'{output_path}.csv'
    csv_path = Path(output_path).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if library == 'np':
        arrays = load_arrays(
            branches=branches,
            dataset=dataset,
            root_files=root_files,
            tree_path=tree_path,
            cut=cut,
            show_trees=show_trees
        )
        try:
            df = pd.DataFrame(arrays)
        except ValueError as e:
            raise ValueError('Los arreglos Numpy no se pueden convertir directamente a una tabla.') from e
    elif library == 'pd':
        df = load_dataframe(
            branches=branches,
            dataset=dataset,
            root_files=root_files,
            tree_path=tree_path,
            cut=cut,
            show_trees=show_trees
        )
    elif library == 'ak':
        awkward_array = load_awkward(
            branches=branches,
            dataset=dataset,
            root_files=root_files,
            tree_path=tree_path,
            cut=cut,
            show_trees=show_trees
        )
        df = ak.to_dataframe(awkward_array, how=awkward_how).reset_index()
        df = _dataframe_columns(df)
    else:
        raise ValueError("library debe ser 'np', 'pd', o 'ak'")
    with open(
        str(csv_path),
        mode="w",
        encoding="utf-8",
        newline=""
    ) as csv_file:
        df.to_csv(csv_file, index=False)
    print(f"CSV guardado en: {csv_path}")
    print(f"Filas: {len(df):,}    |    Columnas: {len(df.columns)}")
    return csv_path