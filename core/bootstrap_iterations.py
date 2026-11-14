"""
Repeated simulations on block-bootstrapped historical returns: one iteration = one seed = one
resampled price path, run against every orderbook.

The counterpart of core.gbm_iterations for simulation_bootstrapped_btc.py. The pool of historical
returns is read from a parquet file (saved by the script next to its results) rather than passed
to every iteration, so each worker process reads it once instead of receiving a copy per seed.
"""

import time
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.gbm_iterations import saved_seeds, seed_path  # noqa: F401 (re-exported for the script)
from core.orderbook import Orderbook
from core.price_process import block_starts, generate_block_bootstrap_price_series
from core.simulation import run_simulation, summarise_run
from core.variable_leverage_product import VLETP


@cache
def load_returns_pool(returns_path: str, block_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Historical log returns saved at returns_path (a "log_return" column indexed by timestamp), and the
    positions a block of block_length returns can start at without spanning a gap.

    Cached, so each process reads the file once.
    """
    returns = pd.read_parquet(returns_path)["log_return"]
    return returns.to_numpy(dtype=float), block_starts(returns.index, block_length)


def simulate_seed(
    seed: int,
    vletps: VLETP | list[VLETP],
    orderbooks: list[Orderbook | None],
    returns_path: str,
    price_label: str,
    n_prices: int,
    start_price: float,
    block_length: int,
    time_step: str,
    keep_results: bool = False,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """
    Run one iteration: a block-bootstrapped price path drawn from `seed`, simulated against every
    orderbook (None = no orderbook).

    keep_results: also return the full output of every run (large; for inspecting one iteration).

    Returns (summary, results), as core.gbm_iterations.simulate_seed:
        summary: one row of summarise_run outcomes per orderbook, with `seed`, `price_series`
            (= price_label) and `orderbook` columns in front.
        results: run_simulation outputs keyed by (price_label, orderbook). Empty unless keep_results.
    """
    returns, starts = load_returns_pool(str(returns_path), block_length)
    prices = generate_block_bootstrap_price_series(
        returns,
        starts,
        n_prices=n_prices,
        block_length=block_length,
        start_price=start_price,
        time_step=time_step,
        seed=seed,
    )

    rows = []
    results: dict[tuple[str, str], pd.DataFrame] = {}
    for orderbook in orderbooks:
        orderbook_label = "no_orderbook" if orderbook is None else orderbook.name
        result = run_simulation(vletps, orderbook, prices)
        if keep_results:
            results[price_label, orderbook_label] = result
        rows.append(
            {
                "seed": seed,
                "price_series": price_label,
                "orderbook": orderbook_label,
                **summarise_run(result, vletps),
            }
        )

    summary = pd.DataFrame(rows)
    # All-NaT columns would otherwise be stored as datetimes rather than elapsed times
    for column in summary.columns:
        if column.startswith("wipeout_time_"):
            summary[column] = pd.to_timedelta(summary[column].astype(object))
    return summary, results


def run_iteration(seed: int, results_dir: Path, **setup) -> tuple[int, float]:
    """
    Run simulate_seed(seed, **setup) and save its summary to results_dir.

    Returns (seed, seconds taken). Safe to run in several processes at once: each seed writes its
    own file, and a file only appears once it is complete.
    """
    started = time.perf_counter()
    summary, _ = simulate_seed(seed, **setup)
    path = seed_path(results_dir, seed)
    temporary_path = path.with_suffix(".tmp")
    summary.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
    return seed, time.perf_counter() - started
