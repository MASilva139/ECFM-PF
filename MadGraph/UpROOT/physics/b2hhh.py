import numpy as np
import pandas as pd
from ..common import invariant_mass

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