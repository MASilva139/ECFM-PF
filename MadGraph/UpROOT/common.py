from collections.abc import Mapping
from enum import Enum
import numpy as np
import pandas as pd

MUON_M_GEV = 0.1056583745
ELECTRON_M_GEV = 0.0005109989461
Z_M_GEV = 91.1876

class Schema(str, Enum):
    DELPHES = 'delphes'
    DVNTUPLE = 'dvntuple'
    B2HHH = 'b2hhh'

def energy(px, py, pz, mass):
    return np.sqrt(px**2 + py**2 + pz**2 + mass**2)

def invariant_mass(e, px, py, pz):
    mass_squared = (e**2 - px**2 - py**2 -pz**2)
    return np.maximum(mass_squared, 0.0)

def delta_phi(phi1, phi2):
    difference = phi1 - phi2
    return np.arctan2(np.sin(difference),np.cos(difference))

def delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.hypot(deta, dphi)

def detect_schema(data: pd.DataFrame | Mapping[str, pd.DataFrame]) -> Schema:
    if isinstance(data, Mapping):
        names = {str(name).lower() for name in data}
        if {'events', 'muons'}.issubset(names):
            return Schema.DELPHES
        raise ValueError(f"No fue posible reconocer el esquema mediante las tablas disponibles: {sorted(names)}.")
    columns = set(map(str, data.columns))
    # Para archivos B2HHH
    if {'H1_PX', 'H2_PX', 'H3_PX'}.issubset(columns):
        return Schema.B2HHH
    # Para archivos dvntuple B+ -> J/psi K+
    if {'Bplus_M', 'J_psi_1S_M'}.issubset(columns):
        return Schema.DVNTUPLE
    # Árbol Muons convertido
    delphes_object = {'event_id', 'object_index', 'PT', 'Eta', 'Phi'}
    # Árbol Dimuons convertido
    delphes_dimuon = {'event_id', 'muplus_pt', 'muminus_pt'}
    if (delphes_object.issubset(columns) or delphes_dimuon.issubset(columns)):
        return Schema.DELPHES
    preview = ', '.join(sorted(columns)[:15])
    raise ValueError(f"No fue posible detectar el esquema del DataFrame. Primeras columnas: {preview}.")