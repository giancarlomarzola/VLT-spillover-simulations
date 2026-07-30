import math
import random
from datetime import UTC, datetime

import pandas as pd

from functions.orderbook import Orderbook
from functions.plot_utils import plot_results
from functions.price_spillover_simulations import run_simulation

lambda_target = None  # None = boundary rebalancing, float = target rebalancing
lambda_upper = 4  # upper boundary
lambda_lower = 1.25  # lower boundary

start_nav_up = 30_000_000
start_exposure_up = 90_000_000

start_nav_down = 20_000_000
start_exposure_down = -60_000_000

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
        k_bid=0.5,
        k_ask=0.5,
    ),
    Orderbook(
        name="Deep Narrow Broad",
        depth_bid=50_000_000,
        depth_ask=50_000_000,
        width_bid=0.01,
        width_ask=0.01,
        spread_bid=0.05 / 100,
        spread_ask=0.05 / 100,
        k_bid=0.5,
        k_ask=0.5,
    ),
]

# Display options
show_hover_info = True
show_markers = False
plot_resample_freq = "30s"  # None


# Random price series - to be substituted later
def generate_price_series(
    start_price=100, start_time=None, end_time=None, freq="1min", volatility=0.01
):
    """
    Generate a price series with timestamps.

    Args:
        start_price: Initial price
        start_time: Start datetime (e.g., datetime(2024, 1, 1))
        end_time: End datetime (e.g., datetime(2024, 1, 10))
        freq: Pandas frequency string (e.g., "1min", "5min", "1H")
        volatility: Annual volatility

    Returns:
        tuple: (timestamps, prices) where both are lists
    """

    if start_time is None or end_time is None:
        raise ValueError("start_time and end_time must be provided")

    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq).tolist()

    # Convert pandas frequency to seconds
    time_delta = pd.Timedelta(freq)
    time_step_seconds = time_delta.total_seconds()

    # Convert to years (assuming 252 trading days per year)
    time_step_years = time_step_seconds / (252 * 24 * 60 * 60)

    # Adjust volatility from annual to frequency
    adjusted_volatility = volatility * math.sqrt(time_step_years)

    prices = [start_price]

    for _ in range(len(timestamps) - 1):
        ret = random.uniform(-adjusted_volatility, adjusted_volatility)
        prices.append(round(prices[-1] * (1 + ret), 2))

    return timestamps, prices


timestamps, price_series = generate_price_series(
    start_price=50_000,
    start_time=datetime(2024, 1, 18, 12, 0, tzinfo=UTC),
    end_time=datetime(2024, 1, 18, 14, 0, tzinfo=UTC),
    freq="500ms",
    volatility=50,
)


# TODO: Introduce eta parameter - manual shocks to otherwise stable price series


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
        price_series=price_series,
        start_nav_up=start_nav_up,
        start_exposure_up=start_exposure_up,
        start_nav_down=start_nav_down,
        start_exposure_down=start_exposure_down,
        timestamps=timestamps,
    )
    results[orderbook_name] = result


# Plot results
price_fig, leverage_fig = plot_results(
    results,
    x_axis=timestamps,
    lambda_upper=lambda_upper,
    lambda_lower=lambda_lower,
    resample_freq=plot_resample_freq,
)
price_fig.show()
leverage_fig.show()
