import pandas as pd
from IPython.display import display

from core.orderbook import Orderbook
from core.price_process import generate_gbm_price_series
from core.simulation import run_simulation, summarise_run
from core.variable_leverage_product import VLETP
from utils.simulation_plots import plot_prices

# Products
gbm_up = VLETP(
    name="GBM_UP",
    start_nav=10_000_000,
    start_leverage=2,
    leverage_lower_bound=1.25,
    leverage_upper_bound=4,
    leverage_target=None,
)

# Orderbook
baseline_depth = 50_000_000
baseline_width = 0.01
baseline_spread = 0.005 / 100
baseline_k = 0.3
baseline_resilience = 0.2

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
    width=0.5 * baseline_width,
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
    resilience=0.1 * baseline_resilience,
)

orderbooks = [
    baseline_orderbook,
    depth_orderbook,
    width_orderbook,
    spread_orderbook,
    shape_orderbook,
    resilience_orderbook,
]


# Price process: raw GBM paths, indexed by time elapsed since the start.
# Same seed for every path, so they differ only in volatility.
seed = 42
volatilities = [0.2, 0.4, 0.6, 0.8, 1.0]

price_series = {
    f"vol_{round(sigma * 100)}": generate_gbm_price_series(
        n_prices=100_000,
        start_price=50_000,
        mu=0.0,
        sigma=sigma,
        time_step="5min",
        seed=seed,
    )
    for sigma in volatilities
}

# Summary: one row of headline outcomes per (price series, orderbook) run
summary_rows = []
for price_label, prices in price_series.items():
    for orderbook in orderbooks:
        print(f"Running {price_label} / {orderbook.name}")
        result = run_simulation(gbm_up, orderbook, prices)
        summary_rows.append({"price_series": price_label, "orderbook": orderbook.name, **summarise_run(result)})

summary = pd.DataFrame(summary_rows).set_index(["price_series", "orderbook"])

# Sizes and NAVs in quote currency, price impacts in bps, vols in %
formats = {column: "{:,.0f}" for column in summary.columns}
formats |= {column: (lambda x: f"{x * 1e4:.2f} bps") for column in summary.columns if column.endswith("_impact")}
formats |= {"realised_vol_raw": "{:.2%}", "realised_vol_simulated": "{:.2%}", "realised_vol_ratio": "{:.5f}"}
display(summary.style.format(formats, na_rep="-"))

# Simulation: results hold both raw_price (the GBM path) and simulated_price (with VLETP impact)
# results = {baseline_orderbook.name: run_simulation(gbm_up, baseline_orderbook, price_series["vol_100"])}

# Plots
# plot_resample_freq = None  # e.g. "30s"; None = no resampling

# plot_prices(results, resample_freq=plot_resample_freq, title_prefix="GBM").show()
