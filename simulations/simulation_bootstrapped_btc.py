import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from core.bootstrap_iterations import run_iteration, saved_seeds, simulate_seed  # noqa: F401 (interactive use)
from core.orderbook import Orderbook
from core.summary_tables import build_summary_tables
from core.variable_leverage_product import VLETP
from utils.paths import BOOTSTRAP_SIMULATIONS
from utils.plotting import plot_leverage, plot_orderbook_depth, plot_prices  # noqa: F401 (interactive use)

# Same setup as simulation_from_gbm.py, but the price paths are block-bootstrapped from the
# BTC returns of 2021-05-19 instead of drawn from a GBM.

# IMPORTANT NOTE:
# SIMULATIONS ASSUME MARKETS RUNNING 24/7
# SIMULATIONS ASSUME ALL RESETS ARE DONE IMMEDIATELY TO THE FULL EXTENT

# Products
btc_up = VLETP(
    name="BTC_UP",
    start_nav=3_000_000_000,
    start_leverage=2,
    leverage_lower_bound=1.25,
    leverage_upper_bound=3,
    leverage_target=None,  # None: nearest boundary; number: fixed target; (lower, upper): nearest of the two
)

btc_down = VLETP(
    name="BTC_DOWN",
    start_nav=100_000_000,
    start_leverage=-2,
    leverage_lower_bound=-1.25,
    leverage_upper_bound=-3,
    leverage_target=None,
)

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


# Historical returns: BTC perpetual trade prices on 2021-05-19, resampled to time_step (last trade
# per interval, forward-filled), as log returns. Returns inside the two data outages (no ticks from
# 13:57:06 to 15:10:18 and from 19:27:16 to 19:33:36), rounded out to the minute, are dropped.
currency = "btc"
time_step = "100ms"
outages = [
    ("2021-05-19 13:57", "2021-05-19 15:11"),
    ("2021-05-19 19:27", "2021-05-19 19:34"),
]

# Price process: each path is a moving block bootstrap of those returns, indexed by time elapsed
# since the start. Blocks of consecutive returns are kept intact, so the negative autocorrelation
# of the returns (bid/ask bounce, lag-1 ~ -0.26 at 100ms, plus mean reversion out to about a minute)
# is preserved instead of inflating the volatility of the paths, as an i.i.d. bootstrap would.
# No block spans an outage.
# Block length check (200 paths each; empirical 1h-scaled vol is 6.39% from 1s returns and 4.08%
# from 1min returns):
#   block   vol from 1s   vol from 1min
#   0.1s       8.44%          8.32%      (= i.i.d.)
#   1s         6.48%          6.38%
#   5s         6.29%          5.84%
#   10s        6.39%          5.50%
#   60s        6.18%          4.07%      <- matches both horizons
start_price = 40_000
path_length = "1h"
block = "60s"  # length of each resampled block of returns
n_prices = int(pd.Timedelta(path_length) / pd.Timedelta(time_step)) + 1  # including the start price
block_length = int(pd.Timedelta(block) / pd.Timedelta(time_step))  # returns per block
price_label = f"btc_bootstrap_{block}"

# Iterations: each one takes a new seed, draws one bootstrapped path from it and runs every
# orderbook on it. Results are saved one file per seed in
# data/outputs/BTC_bootstrap_simulations/<experiment>/iterations/, next to the pool of returns
# they were drawn from (returns.parquet). n_iterations is the target total: running this script
# again runs only the seeds in 0..n_iterations-1 that are not saved yet, then rebuilds the summary
# tables from every saved seed. Raise n_iterations to add more seeds.
experiment = "100ms_1h_block60s"  # use a new name/delete the folder when the setup above changes
n_iterations = 1_000
n_workers = max(1, os.cpu_count() - 1)  # iterations run in parallel, one per process; 1 = no parallelism

experiment_dir = BOOTSTRAP_SIMULATIONS / experiment  # config.json, returns.parquet and the summary table
returns_path = experiment_dir / "returns.parquet"

# Everything simulate_seed needs besides the seed
setup = {
    "vletps": [btc_up, btc_down],
    "orderbooks": orderbooks,
    "returns_path": str(returns_path),
    "price_label": price_label,
    "n_prices": n_prices,
    "start_price": start_price,
    "block_length": block_length,
    "time_step": time_step,
}


def save_returns():
    """Build the pool of historical returns from the tick data and save it to returns_path."""
    from utils.data_loader import log_returns  # only the main process needs the tick data

    returns = log_returns(currency, time_step, exclude=outages)
    returns.to_frame().to_parquet(returns_path)
    print(
        f"{len(returns):,} {time_step} returns saved to {returns_path} "
        f"({returns.index[0]} to {returns.index[-1]}, outages excluded: {outages})"
    )


# The guard keeps the worker processes, which re-import this file, from starting runs of their own
if __name__ == "__main__":
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
            for product in setup["vletps"]
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
            "type": "moving block bootstrap",
            "currency": currency,
            "time_step": time_step,
            "outages": outages,
            "start_price": start_price,
            "path_length": path_length,
            "n_prices": n_prices,
            "block": block,
            "block_length": block_length,
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

    # The returns are built once per experiment (the config check above ties them to its setup)
    if not returns_path.exists():
        save_returns()

    # Seeds 0..n_iterations-1 not saved yet, so an interrupted run is completed by running again
    existing_seeds = saved_seeds(results_dir)
    seeds = [seed for seed in range(n_iterations) if seed not in existing_seeds]

    if not seeds:
        print(f"All {n_iterations} iterations already saved in {results_dir}; nothing to run")
    else:
        n_processes = min(n_workers, len(seeds))
        print(
            f"Running {len(seeds)} iterations (seeds {seeds[0]} to {seeds[-1]}) on {n_processes} processes\n"
            f"  Each iteration: one {path_length} bootstrapped path ({n_prices:,} prices at {time_step}, "
            f"blocks of {block}) x {len(orderbooks)} orderbooks ({len(seeds) * len(orderbooks):,} runs in total)\n"
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
# the following block in interactive window (the seed makes it reproducible; the script must
# have been run once so that returns.parquet exists):
"""
from simulations.simulation_bootstrapped_btc import setup, btc_up, btc_down, price_label
from core.bootstrap_iterations import simulate_seed
from utils.plotting import plot_prices, plot_leverage, plot_orderbook_depth

seed = 42  # the N in data/outputs/BTC_bootstrap_simulations/<experiment>/iterations/seed_0000NN.parquet
products = [btc_up, btc_down]

summary, results = simulate_seed(seed, **{**setup, "vletps": products}, keep_results=True)

r = results[price_label, "baseline"]
plot_prices(r).show()
plot_leverage(r, products).show()
"""
