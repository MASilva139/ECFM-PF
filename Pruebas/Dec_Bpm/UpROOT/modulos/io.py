from collections.abc import Iterable
from pathlib import Path

import numpy as np
import uproot

from .config import ROOT_FILES, TREE_PATH


def _normalize_paths(root_files: Iterable[str | Path] | None) -> tuple[Path, ...]:
    paths = ROOT_FILES if root_files is None else tuple(Path(path) for path in root_files)
    if not paths:
        raise ValueError("La lista de archivos ROOT está vacía.")

    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "No se encontraron los siguientes archivos ROOT:\n"
            f"{formatted}\n"
            "Verifica que los datos estén dentro de la carpeta data del repositorio."
        )
    return paths


def get_tree(root_file: str | Path, tree_path: str = TREE_PATH):
    """Abre un archivo y devuelve el árbol configurado."""
    file_handle = uproot.open(Path(root_file))
    try:
        return file_handle[tree_path]
    except KeyError as error:
        available = ", ".join(file_handle.keys())
        file_handle.close()
        raise KeyError(
            f"No existe el árbol '{tree_path}' en {root_file}. "
            f"Objetos disponibles: {available}"
        ) from error


def list_branches(
    root_file: str | Path | None = None,
    tree_path: str = TREE_PATH,
) -> list[str]:
    """Lista las ramas del primer archivo o del archivo indicado."""
    path = _normalize_paths([root_file] if root_file is not None else None)[0]
    with uproot.open(path) as file_handle:
        tree = file_handle[tree_path]
        return sorted(str(branch) for branch in tree.keys())


def load_arrays(
    branches: Iterable[str],
    root_files: Iterable[str | Path] | None = None,
    tree_path: str = TREE_PATH,
) -> dict[str, np.ndarray]:
    """Concatena ramas escalares de varios árboles como arreglos NumPy."""
    paths = _normalize_paths(root_files)
    requested = tuple(dict.fromkeys(branches))
    if not requested:
        raise ValueError("Debes solicitar al menos una rama.")

    with uproot.open(paths[0]) as file_handle:
        tree = file_handle[tree_path]
        available = set(tree.keys())
        missing_branches = [branch for branch in requested if branch not in available]
    if missing_branches:
        raise KeyError(
            "No existen estas ramas en el árbol: " + ", ".join(missing_branches)
        )

    file_tree_map = {path: tree_path for path in paths}
    return uproot.concatenate(
        file_tree_map,
        expressions=list(requested),
        library="np",
        how=dict,
    )

