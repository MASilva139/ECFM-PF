from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TEXT_DIR = ROOT_DIR / "MadGraph/assset/txt"
ANALYSIS_DIR = ROOT_DIR / "MadGraph/analysis"
FIG_DIR_01 = ROOT_DIR / "MadGraph/assset/images_01"
FIG_DIR_02 = ROOT_DIR / "MadGraph/assset/images_02"
FIG_DIR_03 = ROOT_DIR / "MadGraph/assset/images_03"
FIG_DIR_01.mkdir(parents=True, exist_ok=True)
FIG_DIR_02.mkdir(parents=True, exist_ok=True)
FIG_DIR_03.mkdir(parents=True, exist_ok=True)

ROOT_DATASETS = {
    "tt": (
        "MG5_pp_tt_10k_1GeV.root",
    ),
    "mpmm_10k_1": (
        "MG5_pp_mpmm_10k_1GeV.root",
    ),
    "mpmm_10k_10": (
        "MG5_pp_mpmm_10k_10GeV.root",
    ),
    "mpmm_10k_70": (
        "MG5_pp_mpmm_10k_10GeV.root",
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