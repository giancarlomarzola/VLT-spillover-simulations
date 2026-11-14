"""
Repeated GBM simulations: one iteration = one seed, run on every (volatility, orderbook) pair.

These functions live in a module (rather than in the script that uses them) so that worker
processes can import them, which is what lets iterations run in parallel.
"""

import time
from pathlib import Path

import pandas as pd

from core.orderbook import Orderbook
from core.price_process import generate_gbm_price_series
from core.simulation import run_simulation, summarise_run
from core.variable_leverage_product import VLETP


def seed_path(results_dir: Path, seed: int) -> Path:
    """File holding the summary of one iteration."""
    return Path(results_dir) / f"seed_{seed:06d}.parquet"


def saved_seeds(results_dir: Path) -> set[int]:
    """Seeds that already have a saved iteration in results_dir."""
    return {int(path.stem.removeprefix("seed_")) for path in Path(results_dir).glob("seed_*.parquet")}


def simulate_seed(
    seed: int,
    vletps: VLETP | list[VLETP],
    orderbooks: list[Orderbook | None],
    volatilities: list[float],
    n_prices: int,
    start_price: float,
    mu: float,
    time_step: str,
    keep_results: bool = False,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """
    Run one iteration: a GBM path per volatility, all drawn from `seed` so they differ only
    in volatility, each simulated against every orderbook (None = no orderbook).

    keep_results: also return the full output of every run. Off by default because each
        one is large; turn it on to inspect or plot a single iteration.

    Returns (summary, results):
        summary: one row of summarise_run outcomes per (price series, orderbook) run, with
            `seed`, `price_series` (e.g. "vol_60") and `orderbook` columns in front.
        results: run_simulation outputs keyed by (price series, orderbook), e.g.
            results["vol_60", "depth"]. Empty unless keep_results.
    """
    rows = []
    results: dict[tuple[str, str], pd.DataFrame] = {}
    for sigma in volatilities:
        price_label = f"vol_{round(sigma * 100)}"
        prices = generate_gbm_price_series(
            n_prices=n_prices, start_price=start_price, mu=mu, sigma=sigma, time_step=time_step, seed=seed
        )
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

    Returns (seed, seconds taken). Safe to run in several processes at once: each seed
    writes its own file, and a file only appears once it is complete.
    """
    started = time.perf_counter()
    summary, _ = simulate_seed(seed, **setup)
    path = seed_path(results_dir, seed)
    temporary_path = path.with_suffix(".tmp")
    summary.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
    return seed, time.perf_counter() - started
