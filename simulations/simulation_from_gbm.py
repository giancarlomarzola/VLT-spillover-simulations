import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd

from core.gbm_iterations import run_iteration, saved_seeds, simulate_seed  # noqa: F401 (interactive use)
from core.orderbook import Orderbook
from core.price_process import generate_gbm_price_series
from core.variable_leverage_product import VLETP
from utils.paths import RESULTS_DIR
from utils.plotting import plot_leverage, plot_orderbook_depth, plot_prices  # noqa: F401 (interactive use)

# IMPORTANT NOTE: SIMULATIONS ASSUME MARKETS RUNNING 24/7

# Products
gbm_up = VLETP( # in line with total for products on TSLA/NVDA etc
    name="GBM_UP",
    start_nav=3_000_000_000, 
    start_leverage=2,
    leverage_lower_bound=1,
    leverage_upper_bound=3,
    leverage_target=None, # None: nearest boundary; number: fixed target; (lower, upper): nearest of the two
)

gbm_down  = VLETP( # in line with total for products on TSLA/NVDA etc
    name="GBM_DOWN",
    start_nav=100_000_000, 
    start_leverage=-2,
    leverage_lower_bound=-1,
    leverage_upper_bound=-3,
    leverage_target=None,
)

# TSLA shares 400 each; 

# Orderbook
baseline_depth = 250_000_000
baseline_width = 0.01
baseline_spread = 0.005 / 100
baseline_k = 0.3
baseline_resilience = 0  # time to fully replenish the orderbook

baseline_orderbook = Orderbook(
    name="baseline",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

depth_orderbook = Orderbook(
    name="depth",
    depth=0.5 * baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

width_orderbook = Orderbook(
    name="width",
    depth=baseline_depth,
    width=2 * baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

spread_orderbook = Orderbook(
    name="spread",
    depth=baseline_depth,
    width=baseline_width,
    spread=2 * baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

shape_orderbook = Orderbook(
    name="shape",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=3,
    resilience=baseline_resilience,
)

resilience_orderbook = Orderbook(
    name="resilience",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=pd.Timedelta("2h"),
)

# None = no orderbook: rebalances have no price impact (the reference run)
orderbooks = [
    None,
    baseline_orderbook,
    depth_orderbook,
    width_orderbook,
    spread_orderbook,
    shape_orderbook,
    resilience_orderbook,
]


# Price process: raw GBM paths, indexed by time elapsed since the start.
volatilities = [0.2, 0.4, 0.6, 0.8, 1.0]
n_prices = 100_000
start_price = 50_000
mu = 0.0
time_step = "5min"

# Iterations: each one takes a new seed and runs every (volatility, orderbook) pair on it.
# Within an iteration every path shares the seed, so they differ only in volatility.
# Results are saved one file per seed in data/outputs/results/<experiment>/, and running
# this script again uses the next unused seeds, adding to what is already there.
# simulation_from_gbm_table.py builds the table of averages from those files.
experiment = "simulation_from_gbm"  # use a new name (or delete the folder) when the setup above changes
n_iterations = 10
n_workers = max(1, os.cpu_count() - 1)  # iterations run in parallel, one per process; 1 = no parallelism

# Everything simulate_seed needs besides the seed
setup = {
    "vletps": gbm_up,
    "orderbooks": orderbooks,
    "volatilities": volatilities,
    "n_prices": n_prices,
    "start_price": start_price,
    "mu": mu,
    "time_step": time_step,
}

# The guard keeps the worker processes, which re-import this file, from starting runs of their own
if __name__ == "__main__":
    results_dir = RESULTS_DIR / experiment
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup the saved results belong to. Runs with a different setup must not be averaged
    # together, so the folder is tied to the setup of the first run saved in it.
    config = {
        "product": {
            name: getattr(gbm_up, name)
            for name in [
                "name",
                "start_nav",
                "start_leverage",
                "direction",
                "leverage_lower_bound",
                "leverage_upper_bound",
                "leverage_target",
            ]
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
    config_path = results_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(
                f"The setup differs from the one already saved in {results_dir}. "
                "Change `experiment` to start a new set of results, or delete that folder."
            )
    else:
        config_path.write_text(json.dumps(config, indent=2))

    # The lowest seeds not saved yet, so an interrupted run leaves no permanent gaps
    existing_seeds = saved_seeds(results_dir)
    seeds = list(itertools.islice((s for s in itertools.count() if s not in existing_seeds), n_iterations))

    n_processes = min(n_workers, n_iterations)
    runs_per_iteration = len(volatilities) * len(orderbooks)
    print(
        f"Running {n_iterations} iterations (seeds {seeds[0]} to {seeds[-1]}) on {n_processes} processes\n"
        f"  Each iteration: {len(volatilities)} volatilities x {len(orderbooks)} orderbooks = "
        f"{runs_per_iteration} runs of {n_prices:,} prices ({n_iterations * runs_per_iteration:,} runs in total)\n"
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
                f"Seed {seed} done in {seconds:.0f}s ({n_done}/{n_iterations} this run, "
                f"{len(existing_seeds) + n_done} saved in total)"
            )
    finally:
        pool.shutdown(cancel_futures=True)  # on an error or interrupt, do not start the remaining seeds
    print(f"{n_iterations} iterations in {time.perf_counter() - started:.0f}s")

    # Raw vol_60 path of the last seed of this run, as a check
    vol = 0.6
    plot_series = generate_gbm_price_series(
        n_prices=n_prices, start_price=start_price, mu=mu, sigma=vol, time_step=time_step, seed=seeds[-1]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_series.index.total_seconds() / 86_400, plot_series, linewidth=0.8)
    ax.set_title(f"GBM price path, {round(vol * 100)}% volatility, seed={seeds[-1]}")
    ax.set_xlabel("Time elapsed (days)")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)
    plt.show()

    # To inspect one iteration in full, re-run it here (the seed makes it reproducible):
    # summary, results = simulate_seed(seeds[-1], **setup, keep_results=True)
    # plot_prices(results["vol_60", "depth"])
    # plot_leverage(results["vol_100", "depth"], gbm_up)
