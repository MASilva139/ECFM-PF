from collections.abc import Mapping

import numpy as np


def in_window(values, lower: float, upper: float) -> np.ndarray:
    """Máscara para un intervalo abierto: lower < value < upper."""
    array = np.asarray(values)
    return (array > lower) & (array < upper)


def build_cutflow(
    total_entries: int,
    selections: Mapping[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    """Construye un resumen de cortes expresado respecto del total inicial."""
    rows: list[dict[str, float | int | str]] = [
        {
            "selection": "Inicial",
            "count": int(total_entries),
            "efficiency": 100.0 if total_entries else 0.0,
        }
    ]
    for name, mask in selections.items():
        count = int(np.count_nonzero(mask))
        efficiency = 100.0 * count / total_entries if total_entries else 0.0
        rows.append(
            {"selection": name, "count": count, "efficiency": efficiency}
        )
    return rows


def print_cutflow(rows: list[dict[str, float | int | str]]) -> None:
    for row in rows:
        print(
            f"{str(row['selection']):31s}: "
            f"{int(row['count']):10,d} "
            f"({float(row['efficiency']):6.2f} % del total)"
        )


def estimate_sideband_background(
    masses,
    base_mask: np.ndarray | None = None,
    signal_window: tuple[float, float] = (5240, 5320),
    left_band: tuple[float, float] = (5100, 5200),
    right_band: tuple[float, float] = (5400, 5500),
) -> dict[str, float | int]:
    """Estima el fondo de la región de señal escalando las bandas por anchura."""
    masses = np.asarray(masses)
    if base_mask is None:
        base_mask = np.ones(len(masses), dtype=bool)

    signal_mask = base_mask & in_window(masses, *signal_window)
    left_mask = base_mask & in_window(masses, *left_band)
    right_mask = base_mask & in_window(masses, *right_band)

    n_signal_region = int(np.count_nonzero(signal_mask))
    n_left = int(np.count_nonzero(left_mask))
    n_right = int(np.count_nonzero(right_mask))

    signal_width = signal_window[1] - signal_window[0]
    sideband_width = (
        left_band[1] - left_band[0] + right_band[1] - right_band[0]
    )
    background = (n_left + n_right) * signal_width / sideband_width

    return {
        "n_signal_region": n_signal_region,
        "n_left": n_left,
        "n_right": n_right,
        "background_in_signal": float(background),
        "signal_after_subtraction": float(n_signal_region - background),
    }

