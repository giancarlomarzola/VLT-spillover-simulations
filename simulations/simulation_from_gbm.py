import pandas as pd
import plotly.graph_objects as go
from IPython.display import display

from core.orderbook import Orderbook
from core.price_process import generate_gbm_price_series
from core.simulation import run_simulation, summarise_run
from core.variable_leverage_product import VLETP
from utils.simulation_plots import plot_leverage, plot_orderbook_depth, plot_prices  # noqa: F401 (interactive use)

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
# Same seed for every path, so they differ only in volatility.
seed = 64
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

# Raw vol_60 path, as a check before running the simulations
vol = 0.6
plot_series = price_series[f"vol_{round(100*vol)}"]
fig = go.Figure(go.Scatter(x=plot_series.index.total_seconds() / 86_400, y=plot_series, mode="lines", name="vol_60"))
fig.update_layout(
    title=f"GBM price path, {round(vol*100)}% volatility, seed={seed}",
    xaxis_title="Time elapsed (days)",
    yaxis_title="Price",
    template="plotly_white",
)
fig.show()

# Summary: one row of headline outcomes per (price series, orderbook) run
summary_rows = []
results = {}  # full output of every run, keyed by (price series, orderbook), e.g. results["vol_60", "depth"]
for price_label, prices in price_series.items():
    for orderbook in orderbooks:
        orderbook_label = "no_orderbook" if orderbook is None else orderbook.name
        print(f"Running {price_label} / {orderbook_label}")
        result = run_simulation(gbm_up, orderbook, prices)
        results[price_label, orderbook_label] = result
        summary_rows.append(
            {"price_series": price_label, "orderbook": orderbook_label, **summarise_run(result, gbm_up)}
        )

summary = pd.DataFrame(summary_rows).set_index(["price_series", "orderbook"])

# Sizes, depths and NAVs in quote currency, price impacts and deviations in bps, vols in %
formats = {column: "{:,.0f}" for column in summary.columns}
formats |= {
    column: (lambda x: f"{x * 1e4:.2f} bps")
    for column in summary.columns
    if column.endswith(("_impact", "_price_deviation"))
}
formats |= {column: str for column in summary.columns if column.startswith("wipeout_time_")}
formats |= {
    "fill_ratio": "{:.2%}",
    "realised_vol_raw": "{:.2%}",
    "realised_vol_simulated": "{:.2%}",
    "realised_vol_ratio": "{:.5f}",
}
display(summary.style.format(formats, na_rep="-"))


# To use util plots for results in interactive window:
# plot_prices(results["vol_60", "depth"])
# plot_leverage(results["vol_100", "depth"], gbm_up)
