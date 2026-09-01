# Físicas de archivo B2HHH
import numpy as np
import pandas as pd
from .common import invariant_mass

MOMENTUM_UNIT = "MeV"

def calc_B_mass(
    df: pd.DataFrame,
    masses: tuple[float, float, float],
) -> pd.DataFrame:
    result = df.copy()
    for index, mass in zip((1, 2, 3), masses):
        particle = f"H{index}"
        px = result[f"{particle}_PX"]
        py = result[f"{particle}_PY"]
        pz = result[f"{particle}_PZ"]
        result[f"{particle}_P"] = np.sqrt(px**2 + py**2 + pz**2)
        result[f"{particle}_E"] = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    result["B_E"] = (result["H1_E"] + result["H2_E"] + result["H3_E"])
    result["B_PX"] = (result["H1_PX"] + result["H2_PX"] + result["H3_PX"])
    result["B_PY"] = (result["H1_PY"] + result["H2_PY"] + result["H3_PY"])
    result["B_PZ"] = (result["H1_PZ"] + result["H2_PZ"] + result["H3_PZ"])
    result["B_M"] = invariant_mass(result["B_E"], result["B_PX"], result["B_PY"], result["B_PZ"])
    result["B_Charge"] = (result["H1_Charge"] + result["H2_Charge"] + result["H3_Charge"])
    return result

def calc_dalitz_vars(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["m2_12"] = ((result["H1_E"] + result["H2_E"])**2 - (result["H1_PX"] + result["H2_PX"])**2 - (result["H1_PY"] + result["H2_PY"])**2 - (result["H1_PZ"] + result["H2_PZ"])**2)
    result["m2_13"] = ((result["H1_E"] + result["H3_E"])**2 - (result["H1_PX"] + result["H3_PX"])**2 - (result["H1_PY"] + result["H3_PY"])**2 - (result["H1_PZ"] + result["H3_PZ"])**2)
    result["R0low"] = result[["m2_12", "m2_13"]].min(axis=1)
    result["R0high"] = result[["m2_12", "m2_13"]].max(axis=1)
    return result

def compute_acp(n_positive: int, n_negative: int) -> dict[str, float | int]:
    total = n_positive + n_negative
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
    return {
        "A": asymmetry,
        "sigma": sigma,
        "significance": (asymmetry / sigma if sigma > 0 else np.nan), 
        "N_positive": n_positive, 
        "N_negative": n_negative
    }

def dalitz_data(
    df: pd.DataFrame,
    mass_window: tuple[float, float] = (5194, 5364),
    charm_veto: bool = True,
    charm_window: tuple[float, float] = (1800, 2000),
    bins: int = 15,
    range_x: tuple[float, float] = (0.0, 15.0),
    range_y: tuple[float, float] = (0.0, 30.0)
) -> dict:
    if mass_window[0] >= mass_window[1]:
        raise ValueError(f"mass_window[0] ({mass_window[0]}) debe ser menor que mass_window[1] ({mass_window[1]}).")
    if bins <= 0:
        raise ValueError(f"bins ({bins}) debe ser un entero positivo.")
    req_col = {
        'B_M', 'B_Charge', 
        'H1_E', 'H1_PX', 'H1_PY', 'H1_PZ',
        'H2_E', 'H2_PX', 'H2_PY', 'H2_PZ', 
        'H3_E', 'H3_PX', 'H3_PY', 'H3_PZ'
    }
    miss_col = req_col.difference(df.columns)
    if miss_col:
        raise KeyError(f'Faltan columnas requeridas para el análisis Dalitz: {miss_col}')
    df_sig = df.query(f'B_M > {mass_window[0]} & B_M < {mass_window[1]}').copy()
    # df_sig = calc_dalitz_vars(df_sig)
    if charm_veto:
        charm_min = charm_window[0]**2
        charm_max = charm_window[1]**2
        charm_mask = (
            (
                (df_sig['m2_12'] < df_sig['m2_13']) & ((df_sig['m2_12'] < charm_min) | (df_sig['m2_12'] > charm_max))
            ) | (
                (df_sig['m2_12'] > df_sig['m2_13']) & ((df_sig['m2_13'] < charm_min) | (df_sig['m2_13'] > charm_max))
            )
        )
        df_sig = df_sig[charm_mask]
        print(f'Tras charm veto: {len(df_sig):,} eventos')
    Bp_sig = df_sig[df_sig['B_Charge'] ==  1]
    Bm_sig = df_sig[df_sig['B_Charge'] == -1]
    Bp_low = Bp_sig['R0low']/1e6
    Bp_high = Bp_sig['R0high']/1e6
    Bm_low = Bm_sig['R0low']/1e6
    Bm_high = Bm_sig['R0high']/1e6
    hBp, xb, yb = np.histogram2d(
        Bp_low, 
        Bp_high,
        bins=bins,
        range=[range_x, range_y]
    )
    hBm, _, _ = np.histogram2d(
        Bm_low, 
        Bm_high, 
        bins=bins, 
        range=[range_x, range_y]
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        tot = hBp + hBm
        A_map = np.where(tot > 0, (hBm - hBp)/tot, np.nan)
        sA_map = np.where(tot > 0, np.sqrt((1 - A_map**2)/tot), np.nan)
        S_map = np.where(sA_map > 0, A_map/sA_map, np.nan)
    return {
        'df_sig': df_sig,
        'Bp_sig': Bp_sig,
        'Bm_sig': Bm_sig,
        'hBp': hBp,
        'hBm': hBm,
        'total': tot,
        'A_map': A_map,         # asymmetry map
        'sA_map': sA_map,       # uncertainty of asymmetry
        'S_map': S_map,         # significance
        'xb': xb,
        'yb': yb,
        'charm_veto': charm_veto,
        'charm_window': charm_window,
        'bins': bins,
        'range_x': range_x,
        'range_y': range_y
    }

def slarge_CPV_region(
    df: pd.DataFrame,
    *,
    charm_veto: bool = True,
    charm_window: tuple[float, float, float] = (1e6, 2e6, 16e6)
) -> pd.DataFrame:
    if not charm_veto:
        return df.copy()
    required = {"m2_12", "m2_13"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {sorted(missing)}. Ejecuta calc_dalitz_vars() primero.")
    low, high, upper_bound = charm_window
    mask = (
        (df["m2_12"] < df["m2_13"]) & (df["m2_12"] > low) & (df["m2_12"] < high) & (df["m2_12"] < upper_bound)
    ) | (
        (df["m2_12"] > df["m2_13"]) & (df["m2_13"] > low) & (df["m2_13"] < high) & (df["m2_13"] < upper_bound)
    )
    selected = df.loc[mask].copy()
    print(f"Región de gran asimetría CP: {len(selected):,} eventos")
    return selected