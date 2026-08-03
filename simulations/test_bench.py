# Imports
import pandas as pd

from core.orderbook import Orderbook
from core.price_spillover_simulations import run_simulation
from utils.paths import DATA_PROCESSED
from utils.plotting import plot_orderbook_depth, plot_results

# Parameters
currency = "btc"
frequency = "30s"

lambda_target = None  # None = boundary rebalancing, float = target rebalancing
lambda_upper = 4  # upper boundary
lambda_lower = 1.25  # lower boundary

k = 0.3
resilience = 0.2

# Display options
show_hover_info = True
show_markers = False
plot_resample_freq = None
leverage_timing = "Before Rebalance"  # 'Before Rebalance'


# Orderbooks
# width and spread in decimal form
orderbooks = [
    None,
    Orderbook(
        name="Deep Narrow Tight",
        depth_bid=50_000_000,
        depth_ask=50_000_000,
        width_bid=0.01,
        width_ask=0.01,
        spread_bid=0.005 / 100,
        spread_ask=0.005 / 100,
        k_bid=k,
        k_ask=k,
        resilience_ask=resilience,
        resilience_bid=resilience,
    ),
    Orderbook(
        name="Deep Narrow Broad",
        depth_bid=50_000_000,
        depth_ask=50_000_000,
        width_bid=0.01,
        width_ask=0.01,
        spread_bid=0.05 / 100,
        spread_ask=0.05 / 100,
        k_bid=k,
        k_ask=k,
        resilience_ask=resilience,
        resilience_bid=resilience,
    ),
    Orderbook(
        name="Deep Wide Tight",
        depth_bid=50_000_000,
        depth_ask=50_000_000,
        width_bid=0.1,
        width_ask=0.1,
        spread_bid=0.005 / 100,
        spread_ask=0.005 / 100,
        k_bid=k,
        k_ask=k,
        resilience_ask=resilience,
        resilience_bid=resilience,
    ),
    Orderbook(
        name="Shallow Narrow Tight",
        depth_bid=5_000_000,
        depth_ask=5_000_000,
        width_bid=0.01,
        width_ask=0.01,
        spread_bid=0.005 / 100,
        spread_ask=0.005 / 100,
        k_bid=k,
        k_ask=k,
        resilience_ask=resilience,
        resilience_bid=resilience,
    ),
    Orderbook(
        name="Asymmetrical Depth",
        depth_bid=30_000_000,
        depth_ask=50_000_000,
        width_bid=0.01,
        width_ask=0.01,
        spread_bid=0.005 / 100,
        spread_ask=0.005 / 100,
        k_bid=k,
        k_ask=k,
        resilience_ask=resilience,
        resilience_bid=resilience,
    ),
]


# Load pre-processed data
try:
    print(f"Loading {currency.upper()} at {frequency} frequency...")
    binance_data = pd.read_parquet(DATA_PROCESSED / f"{currency}_{frequency}_processed.parquet")
    binance_data["timestamp"] = pd.to_datetime(binance_data["timestamp"], utc=True)
    print(f"Loaded {len(binance_data)} rows")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Pre-processed data not found for {currency} at {frequency} frequency.\n"
        f"Run data_processing.py to prepare the data."
    )

# Simulation
results = {}

for orderbook in orderbooks:
    orderbook_name = orderbook.name if orderbook is not None else "No Orderbook"
    print(f"Running simulation for {orderbook_name}")
    result = run_simulation(
        lambda_target=lambda_target,
        lambda_upper=lambda_upper,
        lambda_lower=lambda_lower,
        orderbook=orderbook,
        prepared_data=binance_data,
        timestamps=binance_data["timestamp"].values,
    )
    results[orderbook_name] = result

# Plot all simulations
print("Plotting the data")
title_freq = frequency.replace("min", "min ")
title_prefix = f"{currency.upper()} {title_freq}"

price_fig, leverage_fig = plot_results(
    results,
    market_price=binance_data["price"].values,
    x_axis=binance_data["timestamp"].values,
    lambda_upper=lambda_upper,
    lambda_lower=lambda_lower,
    show_hover=show_hover_info,
    show_markers=show_markers,
    leverage_timing=leverage_timing,
    title_prefix=title_prefix,
    currency=currency.upper(),
    resample_freq=plot_resample_freq,
)

price_fig.show()
leverage_fig.show()

# Plot orderbook depth over time (if orderbooks were used)
print("\nPlotting orderbook depth")
if any(name != "No Orderbook" for name in results):
    depth_fig = plot_orderbook_depth(
        results,
        x_axis=binance_data["timestamp"].values,
        show_hover=show_hover_info,
        title_prefix=title_prefix,
        resample_freq=plot_resample_freq,
    )
    depth_fig.show()
