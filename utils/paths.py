"""Project-wide path configuration for data and output directories."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_OUTPUTS = DATA_DIR / "outputs"
# One folder per experiment, written by simulations/run_gbm_experiment.py / run_btc_bootstrap_experiment.py
GBM_SIMULATIONS = DATA_OUTPUTS / "GBM_simulations"
BOOTSTRAP_SIMULATIONS = DATA_OUTPUTS / "BTC_bootstrap_simulations"

# Output directories
FIGURES_DIR = DATA_OUTPUTS / "figures"
RESULTS_DIR = DATA_OUTPUTS / "results"

# Ensure directories exist
DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
