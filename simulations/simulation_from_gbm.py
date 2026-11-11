import json
import time

import pandas as pd
import plotly.graph_objects as go

from core.orderbook import Orderbook
from core.price_process import generate_gbm_price_series
from core.simulation import run_simulation, summarise_run
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

# Iterations: each one draws a new seed and runs every (volatility, orderbook) pair on it.
# Within an iteration every path shares the seed, so they differ only in volatility.
# Results are saved one file per seed in data/outputs/results/<experiment>/, and running
# this script again continues from the next unused seed, adding to what is already there.
# simulation_from_gbm_table.py builds the table of averages from those files.
experiment = "simulation_from_gbm"  # use a new name (or delete the folder) when the setup above changes
n_iterations = 10

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

existing_seeds = [int(path.stem.removeprefix("seed_")) for path in results_dir.glob("seed_*.parquet")]
first_seed = max(existing_seeds) + 1 if existing_seeds else 0

for seed in range(first_seed, first_seed + n_iterations):
    started = time.perf_counter()
    price_series = {
        f"vol_{round(sigma * 100)}": generate_gbm_price_series(
            n_prices=n_prices,
            start_price=start_price,
            mu=mu,
            sigma=sigma,
            time_step=time_step,
            seed=seed,
        )
        for sigma in volatilities
    }

    # One row of headline outcomes per (price series, orderbook) run
    summary_rows = []
    results = {}  # full output of every run of the latest iteration, e.g. results["vol_60", "depth"]
    for price_label, prices in price_series.items():
        for orderbook in orderbooks:
            orderbook_label = "no_orderbook" if orderbook is None else orderbook.name
            result = run_simulation(gbm_up, orderbook, prices)
            results[price_label, orderbook_label] = result
            summary_rows.append(
                {
                    "seed": seed,
                    "price_series": price_label,
                    "orderbook": orderbook_label,
                    **summarise_run(result, gbm_up),
                }
            )

    iteration_summary = pd.DataFrame(summary_rows)
    # All-NaT columns would otherwise be saved as datetimes rather than elapsed times
    for column in iteration_summary.columns:
        if column.startswith("wipeout_time_"):
            iteration_summary[column] = pd.to_timedelta(iteration_summary[column].astype(object))
    iteration_summary.to_parquet(results_dir / f"seed_{seed:06d}.parquet", index=False)
    print(
        f"Seed {seed} done in {time.perf_counter() - started:.0f}s "
        f"({seed - first_seed + 1}/{n_iterations} this run, "
        f"{len(existing_seeds) + seed - first_seed + 1} saved in total)"
    )

# Raw vol_60 path of the latest iteration, as a check
vol = 0.6
plot_series = price_series[f"vol_{round(100 * vol)}"]
fig = go.Figure(go.Scatter(x=plot_series.index.total_seconds() / 86_400, y=plot_series, mode="lines", name="vol_60"))
fig.update_layout(
    title=f"GBM price path, {round(vol * 100)}% volatility, seed={seed}",
    xaxis_title="Time elapsed (days)",
    yaxis_title="Price",
    template="plotly_white",
)
fig.show()

# To use util plots for the latest iteration's results in interactive window:
# plot_prices(results["vol_60", "depth"])
# plot_leverage(results["vol_100", "depth"], gbm_up)
