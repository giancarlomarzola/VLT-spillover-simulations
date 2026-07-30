import random
from datetime import UTC, datetime

import pandas as pd

from new_code.plot_utils import plot_results
from new_code.price_spillover_simulations import run_simulation


# Random price series - to be substituted later
def generate_price_series(start_price=100, start_time=None, end_time=None, freq="1min", volatility=0.001):
    """
    Generate a price series with timestamps.

    Args:
        start_price: Initial price
        start_time: Start datetime (e.g., datetime(2024, 1, 1))
        end_time: End datetime (e.g., datetime(2024, 1, 10))
        freq: Pandas frequency string (e.g., "1min", "5min", "1H")
        volatility: Random return volatility

    Returns:
        tuple: (timestamps, prices) where both are lists
    """

    if start_time is None or end_time is None:
        raise ValueError("start_time and end_time must be provided")

    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq).tolist()
    prices = [start_price]

    for _ in range(len(timestamps) - 1):
        ret = random.uniform(-volatility, volatility)
        prices.append(round(prices[-1] * (1+ret), 2))

    return timestamps, prices


timestamps, price_series = generate_price_series(
    start_price=50_000,
    start_time=datetime(2024, 1, 18, 12, 0, tzinfo=UTC),
    end_time=datetime(2024, 1, 18, 14, 0, tzinfo=UTC),
    freq="50ms"
)


lambda_target = None  # None = boundary rebalancing, float = target rebalancing
lambda_upper = 4  # upper boundary
lambda_lower = 1.25  # lower boundary

start_nav_up = 30_000_000
start_exposure_up = 90_000_000

start_nav_down = 20_000_000
start_exposure_down = -60_000_000

# Orderbooks
# width and spread in %
orderbook_formula = "curved"
k = 0.5

orderbooks = {
    "No Orderbook": None,
    "Deep Narrow Tight"     : {"depth":50_000_000, "width":1,  "spread":0.005},
    "Deep Narrow Broad"     : {"depth":50_000_000, "width":1,  "spread":0.05},
}

# Display options
show_hover_info = True
show_markers = False
plot_resample_freq = "30s"  # None

# Simulation
results = {}

for orderbook_name, orderbook in orderbooks.items():
    print(f"Running simulation for {orderbook_name}")
    result = run_simulation(
        lambda_target=lambda_target,
        lambda_upper=lambda_upper,
        lambda_lower=lambda_lower,
        orderbook=orderbook,
        orderbook_formula=orderbook_formula,
        k=k,
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