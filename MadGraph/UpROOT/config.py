from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TEXT_DIR = ROOT_DIR / "MadGraph/assset/txt"
ANALYSIS_DIR = ROOT_DIR / "MadGraph/analysis"
FIG_DIR = ROOT_DIR / "MadGraph/assset/images"
FIG_DIR_01 = FIG_DIR / "b2hhh"
FIG_DIR_02 = FIG_DIR / "delphes"

B2HHH_DATA = FIG_DIR_01 / "Data"
B2HHH_SIM = FIG_DIR_01 / "Sim"
DNVTUPLE = FIG_DIR / "dnvtuple"
DELPHES_01 = FIG_DIR_02 / "c01"

# Masa invariante B2HHH [MeV/c2]
mK = 493.677
mPi = 139.570

MASS_MIN = 5100.0   # [MeV/c2]
MASS_MAX = 5500.0   # [MeV/c2]
N_BINS = 80
BIN_WIDTH = (MASS_MAX - MASS_MIN)/N_BINS
MASS_CENTER = (MASS_MAX + MASS_MIN)/2.0

DARK_BACKGROUND = "#111111"
LIGHT_TEXT = "#F2F2F2"
GOLD01 = "#D4AF37"
BLUE01 = "#4C78A8"
GREEN01 = "#59A14F"
RED00 = "#E15759"
RED01 = "#B80A0A"

ROOT_DATASETS = {
    "tt": (
        "MG5_pp_tt_10k_n1GeV.root",
    ),
    "mpmm_10k_n1": (
        "MG5_pp_mpmm_10k_n1GeV.root",
    ),
    "mpmm_10k_n10": (
        "MG5_pp_mpmm_10k_n10GeV.root",
    ),
    "mpmm_10k_n70": (
        "MG5_pp_mpmm_10k_n70GeV.root",
    ),
    "mpmm_10k_70": (
        "MG5_pp_mpmm_10k_70GeV.root",
    ),
    "dvntuple_01": (
        "00334560_00000001_1.dvntuple.root",
        "00334560_00000002_1.dvntuple.root",
        "00334564_00000001_1.dvntuple.root",
        "00334565_00000001_1.dvntuple.root",
        "00334566_00000001_1.dvntuple.root",
    ),
    "b2hhh": (
        "B2HHH_MagnetUp.root",
        "B2HHH_MagnetDown.root",
    ),
    "mpmm_10k_70_p": (
        "mpmm_10k_70_pandas.root",
    ),
}
ROOT_FILES = {
    dataset: tuple(
        DATA_DIR / filename
        for filename in filenames
    )
    for dataset, filenames in ROOT_DATASETS.items()
}

PREFERRED_TREES = (
    'Btree/DecayTree',
    'Delphes',
    'DecayTree'
)