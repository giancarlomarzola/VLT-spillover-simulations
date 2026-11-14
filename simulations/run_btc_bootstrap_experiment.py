"""
BTC bootstrap experiment: the UP/DOWN products on price paths block-bootstrapped from the BTC
returns of 2021-05-19, against every orderbook.

Each iteration takes a new seed, draws one bootstrapped path from it and runs every orderbook on it.
The seeds are the iteration numbers 0..n_iterations-1, and iteration i draws from child i of
SeedSequence(master_seed) (see core.seeding), so the whole experiment is reproducible from master_seed.
Results are saved one file per seed in data/outputs/BTC_bootstrap_simulations/<experiment>/iterations/,
next to the pool of returns they were drawn from (returns.parquet), and summary_table.xlsx in the
same folder averages them across seeds (see core.experiment_runner).

n_iterations is the target total: running this script again runs only the seeds in
0..n_iterations-1 not saved yet, then rebuilds the summary tables. Raise n_iterations to add seeds.
Use a new `experiment` name (or delete its folder) whenever the setup changes; a folder refuses a
setup that differs from its config.json.

IMPORTANT NOTE:
SIMULATIONS ASSUME MARKETS RUNNING 24/7
SIMULATIONS ASSUME ALL RESETS ARE DONE IMMEDIATELY TO THE FULL EXTENT
"""

import os

from core.bootstrap_price_generator import BootstrapPriceGenerator
from core.experiment_runner import run_experiment
from simulations.products_and_orderbooks import orderbook_sensitivities, vletp_pair
from utils.paths import BOOTSTRAP_SIMULATIONS

experiment = "100ms_1h_block60s"
experiment_dir = BOOTSTRAP_SIMULATIONS / experiment

# Historical returns: BTC perpetual trade prices on 2021-05-19, resampled to time_step (last trade per
# interval, forward-filled), as log returns. Returns inside the two data outages (no ticks from 13:57:06
# to 15:10:18 and from 19:27:16 to 19:33:36), rounded out to the minute, are dropped.
#
# Each path is a moving block bootstrap of those returns. Blocks of consecutive returns are kept intact,
# so the negative autocorrelation of the returns (bid/ask bounce, lag-1 ~ -0.26 at 100ms, plus mean
# reversion out to about a minute) is preserved instead of inflating the volatility of the paths, as an
# i.i.d. bootstrap would. No block spans an outage.
# Block length check (200 paths each; empirical 1h-scaled vol is 6.39% from 1s returns and 4.08%
# from 1min returns):
#   block   vol from 1s   vol from 1min
#   0.1s       8.44%          8.32%      (= i.i.d.)
#   1s         6.48%          6.38%
#   5s         6.29%          5.84%
#   10s        6.39%          5.50%
#   60s        6.18%          4.07%      <- matches both horizons
price_generator = BootstrapPriceGenerator(
    currency="btc",
    time_step="1s",
    n_prices=144_000,  # 40h at 1s
    block="60s",  # length of each resampled block of returns
    start_price=40_000,
    outages=[
        ("2021-05-19 13:57", "2021-05-19 15:11"),
        ("2021-05-19 19:27", "2021-05-19 19:34"),
    ],
    returns_path=experiment_dir / "returns.parquet",  # built from the tick data on the first run
    master_seed=20210519,  # every iteration's random stream is derived from it (saved to config.json)
)
vletps = vletp_pair("BTC")
orderbooks = orderbook_sensitivities()

n_iterations = 1_000
n_workers = max(1, os.cpu_count() - 1)  # iterations run in parallel, one per process; 1 = no parallelism

# The guard keeps the worker processes, which re-import this file, from starting runs of their own
if __name__ == "__main__":
    run_experiment(experiment_dir, vletps, orderbooks, price_generator, n_iterations, n_workers)


# To inspect one iteration in full, run this in an interactive window. It rebuilds the setup from the
# experiment's config.json (the seed makes it reproducible; the script must have been run once so that
# returns.parquet exists):
"""
from core.experiment_runner import load_setup, simulate_seed
from utils.paths import BOOTSTRAP_SIMULATIONS
from utils.plotting import plot_prices, plot_leverage, plot_orderbook_depth

seed = 42  # the N in data/outputs/BTC_bootstrap_simulations/<experiment>/iterations/seed_0000NN.parquet
setup = load_setup(BOOTSTRAP_SIMULATIONS / "100ms_1h_block60s")
summary, results = simulate_seed(seed, **setup, keep_results=True)

r = results["btc_bootstrap_60s", "baseline"]
plot_prices(r).show()
plot_leverage(r, setup["vletps"]).show()
"""
