import pandas as pd

MOMENTUM_UNIT = "MeV"

def select_jpsi(
    df: pd.DataFrame,
    *,
    mass_window: tuple[float, float] = (3030.0, 3150.0),
) -> pd.DataFrame:
    if "J_psi_1S_M" not in df.columns:
        raise KeyError("No existe la columna 'J_psi_1S_M'.")
    minimum, maximum = mass_window
    return df.loc[df["J_psi_1S_M"].between(minimum, maximum)].copy()

def select_bplus(
    df: pd.DataFrame,
    *,
    mass_window: tuple[float, float] = (5100.0,5500.0)
) -> pd.DataFrame:
    if "Bplus_M" not in df.columns:
        raise KeyError("No existe la columna 'Bplus_M'.")
    minimum, maximum = mass_window
    return df.loc[df["Bplus_M"].between(minimum, maximum)].copy()

def select_candidates(
    df: pd.DataFrame,
    *,
    b_mass_window: tuple[float, float] = (5100.0, 5500.0),
    jpsi_mass_window: tuple[float, float] = (3030.0, 3150.0)
) -> pd.DataFrame:
    selected = select_jpsi(df, mass_window=jpsi_mass_window)
    return select_bplus(selected, mass_window=b_mass_window)