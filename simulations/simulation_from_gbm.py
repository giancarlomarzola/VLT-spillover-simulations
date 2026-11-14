import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from core.gbm_iterations import run_iteration, saved_seeds, simulate_seed  # noqa: F401 (interactive use)
from core.orderbook import Orderbook
from core.summary_tables import build_summary_tables
from core.variable_leverage_product import VLETP
from utils.paths import GBM_SIMULATIONS
from utils.plotting import plot_leverage, plot_orderbook_depth, plot_prices  # noqa: F401 (interactive use)

# IMPORTANT NOTE: 
# SIMULATIONS ASSUME MARKETS RUNNING 24/7
# SIMULATIONS ASSUME ALL RESETS ARE DONE IMMEDIATELY TO THE FULL EXTENT

# Products
gbm_up = VLETP( # in line with total for products on TSLA/NVDA etc
    name="GBM_UP",
    start_nav=3_000_000_000, 
    start_leverage=2,
    leverage_lower_bound=1.25,
    leverage_upper_bound=3,
    leverage_target=None, # None: nearest boundary; number: fixed target; (lower, upper): nearest of the two
)

gbm_down  = VLETP( # in line with total for products on TSLA/NVDA etc
    name="GBM_DOWN",
    start_nav=100_000_000, 
    start_leverage=-2,
    leverage_lower_bound=-1.25,
    leverage_upper_bound=-3,
    leverage_target=None,
)

# TSLA shares 400 each; 

# Orderbook
baseline_depth = 250_000_000
baseline_width = 0.01
baseline_spread = 0.005 / 100
baseline_k = 0.5
baseline_resilience = pd.Timedelta("5min")  # half-life of orderbook depth recovery; 0=instant

baseline_orderbook = Orderbook(
    name="baseline",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

depth_orderbook = Orderbook(
    name="depth 0.5x",
    depth=0.5 * baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

width_orderbook = Orderbook(
    name="width 2x",
    depth=baseline_depth,
    width=2 * baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

spread_orderbook = Orderbook(
    name="spread 2x",
    depth=baseline_depth,
    width=baseline_width,
    spread=2 * baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

shape1_orderbook = Orderbook(
    name="shape k1",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=1,
    resilience=baseline_resilience,
)

shape2_orderbook = Orderbook(
    name="shape k1_5",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=1.5,
    resilience=baseline_resilience,
)

resilience_orderbook = Orderbook(
    name="resilience 30min",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=pd.Timedelta("30min"),
)

# None = no orderbook: rebalances have no price impact (the reference run)
orderbooks = [
    None,
    baseline_orderbook,
    depth_orderbook,
    width_orderbook,
    spread_orderbook,
    shape1_orderbook,
    shape2_orderbook,
    resilience_orderbook,
]


# Price process: raw GBM paths, indexed by time elapsed since the start.
volatilities = [0.2, 0.4, 0.6, 0.8, 1.0]
n_prices = 144_000
start_price = 100
mu = 0.0
time_step = "1s"

# Iterations: each one takes a new seed and runs every (volatility, orderbook) pair on it.
# Within an iteration every path shares the seed, so they differ only in volatility.
# Results are saved one file per seed in data/outputs/GBM_simulations/<experiment>/iterations/.
# n_iterations is the target total: running this script again runs only the seeds in
# 0..n_iterations-1 that are not saved yet (none once the target is reached), then rebuilds
# the summary tables from every saved seed. Raise n_iterations to add more seeds.
# To rebuild them without running anything: core.summary_tables.build_summary_tables(experiment_dir).
experiment = "1s_2kappas"  # use a new name/delete the folder when the setup above changes
n_iterations = 1_000
n_workers = max(1, os.cpu_count() - 1)  # iterations run in parallel, one per process; 1 = no parallelism

# Everything simulate_seed needs besides the seed
setup = {
    "vletps": [gbm_up, gbm_down],
    "orderbooks": orderbooks,
    "volatilities": volatilities,
    "n_prices": n_prices,
    "start_price": start_price,
    "mu": mu,
    "time_step": time_step,
}

# The guard keeps the worker processes, which re-import this file, from starting runs of their own
if __name__ == "__main__":
    experiment_dir = GBM_SIMULATIONS / experiment  # config.json and the summary table
    results_dir = experiment_dir / "iterations"  # one parquet file per seed
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup the saved results belong to. Runs with a different setup must not be averaged
    # together, so the folder is tied to the setup of the first run saved in it.
    config = {
        "products": {
            product.name: {
                name: getattr(product, name)
                for name in [
                    "start_nav",
                    "start_leverage",
                    "direction",
                    "leverage_lower_bound",
                    "leverage_upper_bound",
                    "leverage_target",
                ]
            }
            for product in ([setup["vletps"]] if isinstance(setup["vletps"], VLETP) else setup["vletps"])
        },
        "orderbooks": {
            orderbook.name: {
                name: getattr(orderbook, name)
                for name in [
                    "depth_bid_full",
                    "depth_ask_full",
                    "width_bid",
                    "width_ask",
                    "spread_bid",
                    "spread_ask",
                    "k_bid",
                    "k_ask",
                    "resilience_bid",
                    "resilience_ask",
                ]
            }
            for orderbook in orderbooks
            if orderbook is not None
        },
        "price_process": {
            "volatilities": volatilities,
            "n_prices": n_prices,
            "start_price": start_price,
            "mu": mu,
            "time_step": time_step,
        },
    }
    config = json.loads(json.dumps(config))  # normalise (tuples -> lists) so it compares equal to the saved copy
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(
                f"The setup differs from the one already saved in {experiment_dir}. "
                "Change `experiment` to start a new set of results, or delete that folder."
            )
    else:
        config_path.write_text(json.dumps(config, indent=2))

    # Seeds 0..n_iterations-1 not saved yet, so an interrupted run is completed by running again
    existing_seeds = saved_seeds(results_dir)
    seeds = [seed for seed in range(n_iterations) if seed not in existing_seeds]

    if not seeds:
        print(f"All {n_iterations} iterations already saved in {results_dir}; nothing to run")
    else:
        n_processes = min(n_workers, len(seeds))
        runs_per_iteration = len(volatilities) * len(orderbooks)
        print(
            f"Running {len(seeds)} iterations (seeds {seeds[0]} to {seeds[-1]}) on {n_processes} processes\n"
            f"  Each iteration: {len(volatilities)} volatilities x {len(orderbooks)} orderbooks = "
            f"{runs_per_iteration} runs of {n_prices:,} prices ({len(seeds) * runs_per_iteration:,} runs in total)\n"
            f"  Already saved: {len(existing_seeds)} iterations in {results_dir}\n"
            "  The first results can take a while to appear, as iterations finish in batches",
            flush=True,
        )

        started = time.perf_counter()
        pool = ProcessPoolExecutor(max_workers=n_processes)
        try:
            futures = [pool.submit(run_iteration, seed, results_dir, **setup) for seed in seeds]
            for n_done, future in enumerate(as_completed(futures), start=1):
                seed, seconds = future.result()
                print(
                    f"Seed {seed} done in {seconds:.0f}s ({n_done}/{len(seeds)} this run, "
                    f"{len(existing_seeds) + n_done} saved in total)"
                )
        finally:
            pool.shutdown(cancel_futures=True)  # on an error or interrupt, do not start the remaining seeds
        print(f"{len(seeds)} iterations in {time.perf_counter() - started:.0f}s")

    # Averages and standard deviations across every saved seed, saved to summary_table.xlsx
    build_summary_tables(experiment_dir)

# To inspect one iteration in full, re-run 
# the following block in interactive winodw (the seed makes it reproducible):
""" 
from simulations.simulation_from_gbm import setup, gbm_up, gbm_down
from core.gbm_iterations import simulate_seed
from utils.plotting import plot_prices, plot_leverage, plot_orderbook_depth

seed = 42  # the N in data/outputs/<experiment>/iterations/seed_0000NN.parquet
products = [gbm_up, gbm_down]

summary, results = simulate_seed(seed, **{**setup, "vletps": products}, keep_results=True)

r = results["vol_60", "shape k3"]
plot_prices(r).show()
plot_leverage(r, products).show()
"""
