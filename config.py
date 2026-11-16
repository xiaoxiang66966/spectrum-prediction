"""Central configuration for the GSL-GCRNN reproduction.

Every hyper-parameter used by the experiment lives here, so the CLI and the
modules share a single source of truth.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# config.py lives in <root>/gsl_gcrnn/, so parents[1] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
TRAIN_DATA = DATA_ROOT / "trainData"
TEST_DATA = DATA_ROOT / "testData"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
# Four Electrosense sensors, 600-700 MHz, 2 MHz resolution -> 101 bands.
SENSORS = ["test_yago", "test_rpi4", "rack_2", "bcn_L"]

# trainData/ already holds the first 80% of the time series and testData/ the
# remaining 20%, so no additional split is applied (train_ratio is kept for the
# single-file fallback in main.py).
TRAIN_RATIO = 0.8

N_BANDS = 101
WINDOW = 16     # tau: number of historical steps fed to the model
HORIZON = 1     # delta: number of future steps to predict

# --------------------------------------------------------------------------- #
# Graph construction (Section III-B, eqs. 1-2)
# --------------------------------------------------------------------------- #
# "pearson" / "spearman" build the adjacency from train data on the fly.
# "load_spearman" loads the provided 0.5-spearman adjacency CSVs (v2).
GRAPH_METHOD = "pearson"
TOP_K = 5       # top-k neighbours kept per node (eq. 2 and the estimator's S_v)

# Combination coefficient of static A and learned A_Q (eq. 7).
MU = 0.8

# --------------------------------------------------------------------------- #
# Model (Section III-C/D/E)
# --------------------------------------------------------------------------- #
GCN_HIDDEN = 64
GCN_LAYERS = 2
EMBED_DIM = 16   # d: per-node embedding dimension produced by the GCN
GRU_HIDDEN = 64  # paper reports 64 hidden units
GRU_LAYERS = 1

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
BATCH_SIZE = 32
EPOCHS = 3000
LR = 1e-4
WEIGHT_DECAY = 0.0
SEED = 42

LOG_EVERY = 200     # print training loss every N epochs
EVAL_EVERY = 200    # evaluate on the test set every N epochs

DEVICE = "auto"     # "auto" -> cuda if available, else cpu


def get_device() -> str:
    if DEVICE == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE
