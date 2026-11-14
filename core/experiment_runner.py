"""
The shared engine behind the simulations/run_*_experiment.py scripts.

Running an experiment: many iterations (seeds) of the same setup, saved one file per seed, then
summarised across seeds.

An experiment lives in one folder:
    config.json          the setup (products, orderbooks, price process) the results belong to
    iterations/          one parquet file of run summaries per seed
    summary_table.xlsx   averages and standard deviations across seeds (core.summary_tables)
    (returns.parquet     the pool of historical returns, for a bootstrapped price process)

Results from different setups must not be averaged together, so the folder is tied to the setup
of the first run saved in it, and load_setup rebuilds that setup to re-run any one seed in full.

One iteration = one seed: the seed draws the iteration's price paths from the price generator
(core.gbm_price_generator, core.bootstrap_price_generator), and each path is simulated against every
orderbook. These functions live in a module (rather than in the scripts that use them) so that worker
processes can import them, which is what lets iterations run in parallel.
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from core.bootstrap_price_generator import BootstrapPriceGenerator
from core.gbm_price_generator import GBMPriceGenerator
from core.orderbook import Orderbook
from core.simulation import run_simulation, summarise_run
from core.summary_tables import build_summary_tables
from core.variable_leverage_product import VLETP

PRODUCT_FIELDS = [
    "start_nav",
    "start_leverage",
    "direction",
    "leverage_lower_bound",
    "leverage_upper_bound",
    "leverage_target",
]
ORDERBOOK_FIELDS = [
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


# ========================================
# One iteration
# ========================================


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
    price_generator,
    keep_results: bool = False,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """
    Run one iteration: every price path price_generator draws from `seed` (a GBMPriceGenerator,
    a BootstrapPriceGenerator, ...), each simulated against every orderbook (None = no orderbook).

    keep_results: also return the full output of every run. Off by default because each
        one is large; turn it on to inspect or plot a single iteration.

    Returns (summary, results):
        summary: one row of summarise_run outcomes per (price series, orderbook) run, with
            `seed`, `price_series` (e.g. "vol_60") and `orderbook` columns in front.
        results: run_simulation outputs keyed by (price series, orderbook), e.g.
            results["vol_60", "baseline"]. Empty unless keep_results.
    """
    rows = []
    results: dict[tuple[str, str], pd.DataFrame] = {}
    for price_label, prices in price_generator.paths(seed):
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


# ========================================
# A whole experiment
# ========================================


def as_list(vletps: VLETP | list[VLETP]) -> list[VLETP]:
    return [vletps] if isinstance(vletps, VLETP) else list(vletps)


def experiment_config(vletps: VLETP | list[VLETP], orderbooks: list[Orderbook | None], price_generator) -> dict:
    """The setup of an experiment, as saved to its config.json (normalised the way JSON reads it back)."""
    config = {
        "products": {
            product.name: {name: getattr(product, name) for name in PRODUCT_FIELDS} for product in as_list(vletps)
        },
        "orderbooks": {
            orderbook.name: {name: getattr(orderbook, name) for name in ORDERBOOK_FIELDS}
            for orderbook in orderbooks
            if orderbook is not None
        },
        "price_process": price_generator.config(),
    }
    return json.loads(json.dumps(config))  # tuples -> lists, so it compares equal to the saved copy


def check_or_write_config(experiment_dir: Path, config: dict) -> None:
    """Save config to experiment_dir/config.json, or check it matches the one already saved there."""
    config_path = Path(experiment_dir) / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(
                f"The setup differs from the one already saved in {experiment_dir}. "
                "Change `experiment` to start a new set of results, or delete that folder."
            )
    else:
        config_path.write_text(json.dumps(config, indent=2))


def run_experiment(
    experiment_dir: Path,
    vletps: VLETP | list[VLETP],
    orderbooks: list[Orderbook | None],
    price_generator,
    n_iterations: int,
    n_workers: int = 1,
) -> None:
    """
    Run seeds 0..n_iterations-1 that are not saved yet in experiment_dir, in parallel on n_workers
    processes, then rebuild the summary tables from every saved seed.

    n_iterations is the target total: running again completes an interrupted run, or adds seeds
    after n_iterations is raised. Must be called under `if __name__ == "__main__":`, as the worker
    processes re-import the calling script.
    """
    experiment_dir = Path(experiment_dir)
    results_dir = experiment_dir / "iterations"  # one parquet file per seed
    results_dir.mkdir(parents=True, exist_ok=True)

    check_or_write_config(experiment_dir, experiment_config(vletps, orderbooks, price_generator))
    price_generator.prepare()  # e.g. the pool of returns; the config check above ties it to this setup

    existing_seeds = saved_seeds(results_dir)
    seeds = [seed for seed in range(n_iterations) if seed not in existing_seeds]
    setup = {"vletps": vletps, "orderbooks": orderbooks, "price_generator": price_generator}

    if not seeds:
        print(f"All {n_iterations} iterations already saved in {results_dir}; nothing to run")
    else:
        n_processes = max(1, min(n_workers, len(seeds)))
        runs_per_iteration = price_generator.n_paths * len(orderbooks)
        print(
            f"Running {len(seeds)} iterations (seeds {seeds[0]} to {seeds[-1]}) on {n_processes} processes\n"
            f"  Each iteration: {price_generator.describe()} x {len(orderbooks)} orderbooks = "
            f"{runs_per_iteration} runs ({len(seeds) * runs_per_iteration:,} runs in total)\n"
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


def load_setup(experiment_dir: Path, include_no_orderbook: bool = True) -> dict:
    """
    Rebuild an experiment's setup from its config.json, as keyword arguments for
    simulate_seed: {"vletps", "orderbooks", "price_generator"}.

    Lets any saved seed be re-run in full without the script that ran it, e.g.
        summary, results = simulate_seed(42, **load_setup(experiment_dir), keep_results=True)

    include_no_orderbook: put the reference run without an orderbook (None) first in the list of
        orderbooks. config.json lists only the real orderbooks.
    """
    experiment_dir = Path(experiment_dir)
    config = json.loads((experiment_dir / "config.json").read_text())

    vletps = []
    for name, p in config["products"].items():
        direction = p["direction"]
        target = p["leverage_target"]
        if isinstance(target, list):
            target = tuple(direction * t for t in target)
        elif target is not None:
            target = direction * target
        vletps.append(
            VLETP(
                name=name,
                start_nav=p["start_nav"],
                start_leverage=p["start_leverage"],
                leverage_lower_bound=direction * p["leverage_lower_bound"],
                leverage_upper_bound=direction * p["leverage_upper_bound"],
                leverage_target=target,
            )
        )

    orderbooks: list[Orderbook | None] = [None] if include_no_orderbook else []
    for name, o in config["orderbooks"].items():
        orderbooks.append(
            Orderbook(
                name=name,
                depth=(o["depth_bid_full"], o["depth_ask_full"]),
                width=(o["width_bid"], o["width_ask"]),
                spread=(o["spread_bid"], o["spread_ask"]),
                k=(o["k_bid"], o["k_ask"]),
                resilience=(o["resilience_bid"], o["resilience_ask"]),  # half-lives in seconds
            )
        )

    process = dict(config["price_process"])
    if process.get("type") == "moving block bootstrap":
        price_generator = BootstrapPriceGenerator(
            currency=process["currency"],
            time_step=process["time_step"],
            path_length=process["path_length"],
            block=process["block"],
            start_price=process["start_price"],
            outages=[tuple(window) for window in process["outages"]],
            returns_path=experiment_dir / "returns.parquet",
        )
    else:
        price_generator = GBMPriceGenerator(
            volatilities=process["volatilities"],
            n_prices=process["n_prices"],
            start_price=process["start_price"],
            mu=process["mu"],
            time_step=process["time_step"],
        )
    return {"vletps": vletps, "orderbooks": orderbooks, "price_generator": price_generator}
