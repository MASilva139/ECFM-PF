from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = PACKAGE_DIR.parent
REPO_ROOT = ANALYSIS_DIR.parents[2]

DATA_DIR = REPO_ROOT / "data"
FIG_DIR = ANALYSIS_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TREE_PATH = "Btree/DecayTree"

ROOT_FILENAMES = (
    "00334560_00000001_1.dvntuple.root",
    "00334560_00000002_1.dvntuple.root",
    "00334564_00000001_1.dvntuple.root",
    "00334565_00000001_1.dvntuple.root",
    "00334566_00000001_1.dvntuple.root",
)
ROOT_FILES = tuple(DATA_DIR / filename for filename in ROOT_FILENAMES)

