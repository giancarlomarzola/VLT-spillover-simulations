"""
GBM experiment: the UP/DOWN products on raw GBM paths, one per volatility, against every orderbook.

Each iteration takes a new seed and runs every (volatility, orderbook) pair on it. Within an
iteration every path shares the seed, so they differ only in volatility. Results are saved one file
per seed in data/outputs/GBM_simulations/<experiment>/iterations/, and summary_table.xlsx in the same
folder averages them across seeds (see core.experiment_runner).

n_iterations is the target total: running this script again runs only the seeds in
0..n_iterations-1 not saved yet, then rebuilds the summary tables. Raise n_iterations to add seeds.
Use a new `experiment` name (or delete its folder) whenever the setup changes; a folder refuses a
setup that differs from its config.json.

IMPORTANT NOTE:
SIMULATIONS ASSUME MARKETS RUNNING 24/7
SIMULATIONS ASSUME ALL RESETS ARE DONE IMMEDIATELY TO THE FULL EXTENT
"""

import os

from core.experiment_runner import run_experiment
from core.gbm_price_generator import GBMPriceGenerator
from simulations.products_and_orderbooks import orderbook_sensitivities, vletp_pair
from utils.paths import GBM_SIMULATIONS

experiment = "1s_2kappas"
experiment_dir = GBM_SIMULATIONS / experiment

price_generator = GBMPriceGenerator(
    volatilities=[0.2, 0.4, 0.6, 0.8, 1.0],
    n_prices=144_000,
    start_price=100,
    mu=0.0,
    time_step="1s",
    master_seed=20210519,
)
vletps = vletp_pair("GBM")
orderbooks = orderbook_sensitivities()

n_iterations = 1_000
n_workers = max(1, os.cpu_count() - 1)  # iterations run in parallel, one per process; 1 = no parallelism

# The guard keeps the worker processes, which re-import this file, from starting runs of their own
if __name__ == "__main__":
    run_experiment(experiment_dir, vletps, orderbooks, price_generator, n_iterations, n_workers)


# To inspect one iteration in full, run this in an interactive window. It rebuilds the setup from the
# experiment's config.json (the seed makes it reproducible):
"""
from core.experiment_runner import load_setup, simulate_seed
from utils.paths import GBM_SIMULATIONS
from utils.plotting import plot_prices, plot_leverage, plot_orderbook_depth

seed = 42  # the N in data/outputs/GBM_simulations/<experiment>/iterations/seed_0000NN.parquet
setup = load_setup(GBM_SIMULATIONS / "1s_2kappas")
summary, results = simulate_seed(seed, **setup, keep_results=True)

r = results["vol_60", "baseline"]
plot_prices(r).show()
plot_leverage(r, setup["vletps"]).show()
"""
